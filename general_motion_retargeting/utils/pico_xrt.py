"""Loader for Pico XR-Toolkit body-tracking recordings (``xrobot_*.jsonl``).

Each line of the recording is one frame emitted by the HoloMotion teleop node
(``holomotion_teleop_node.py``). We consume ``body_data``: a dict of 24
SMPL-ordered joints, each ``[position(xyz, meters), quaternion(xyzw)]``, given
as GLOBAL poses in a Z-up world frame (head above feet along +Z).

Why this is a POSITION-BASED (keypoint) loader
----------------------------------------------
Pico's body stream is a 3-point (head + two controllers) full-body estimate.
Its per-joint POSITIONS are reliable, but its per-joint ORIENTATION quaternions
drift badly relative to the positions: the pelvis/spine/knee "up" axis wanders
tens of degrees over a take while the joint POSITIONS keep the subject upright
(verified: pelvis-height-above-foot stays 0.92-0.94 m and knees stay ~straight
for the whole recording, yet the raw pelvis quaternion pitches 0->58 deg). Naively
feeding those quaternions makes the retargeted robot slowly topple forward.

So we ignore the unreliable joint quaternions entirely and drive everything
from POSITIONS:

* Torso + legs: joint POSITION targets (spine3, knee, foot) -- reliable.
* Pelvis heading: RECONSTRUCTED yaw-only from the hip line, base kept upright
  (``reconstruct_pelvis_quat``); the waist joint handles real torso lean via
  the spine3 position target.
* Arms: driven by DIRECTION, not position. A single position scale cannot match
  both the shoulder location and the (~2x shorter) robot arm length, so an
  absolute elbow/wrist position target is unreachable and the arm flails.
  Instead we reconstruct the upper-arm and forearm DIRECTIONS from positions
  (``arm_frames``) and feed them as orientation targets; the config drives the
  arms orientation-only (zero position cost) with a high base rot cost, so a
  short 4-DOF arm tracks the gesture without dragging the torso.

The teleop node's own online path (Pico -> SMPL FK -> SMPL-X IK config) trusts
those quaternions; that is exactly why it drifts. This loader deliberately does
not round-trip through SMPL (no smpl_sim dependency) and uses the true joint
positions instead of SMPL template-skeleton positions.

Output frame format matches the rest of GMR:
``{human_body_name: [position(3,), quaternion_wxyz(4,)]}``.
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

# 24 SMPL joints, in the exact order the recording stores ``body_data`` keys.
PICO_JOINT_ORDER = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
    "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
    "Left_Hand", "Right_Hand",
]
_JI = {name: i for i, name in enumerate(PICO_JOINT_ORDER)}

# Everything is driven by DIRECTIONS reconstructed from reliable positions
# (mirrors the proven unitree_g1 config, which orientation-drives every
# segment). Pico's per-joint quaternions drift, so we never use them.
#
# ARM chains: for each (shoulder, elbow, wrist) we build two frames via
# ``limb_frames`` -- the shoulder frame's x-axis is the upper-arm direction, the
# elbow frame's x-axis is the forearm direction, both sharing a z-axis = the
# elbow-bend-plane normal (well-defined because the elbow is usually flexed).
# Fed as ORIENTATION targets so the arm follows overhead raises regardless of
# the ~2x shorter, wristless robot arm.
_ARM_CHAINS = (
    ("left_shoulder", "left_elbow", ("Left_Shoulder", "Left_Elbow", "Left_Wrist")),
    ("right_shoulder", "right_elbow", ("Right_Shoulder", "Right_Elbow", "Right_Wrist")),
)

# LEGS are driven by POSITIONS, not the limb-frame orientation: a standing leg
# is nearly straight, so the knee-bend plane is degenerate and its normal (the
# limb-frame twist axis) is noise -> orientation-driving the hip would spin
# hip_yaw. Instead we pin knee + ankle + toe POSITIONS (3 points fully determine
# the leg) and add a light hip_yaw/roll regularizer (motion_retarget.py) to keep
# the legs facing forward.
_LEG_POSITION_TARGETS = {
    "left_knee": "Left_Knee",
    "right_knee": "Right_Knee",
    "left_ankle": "Left_Ankle",
    "right_ankle": "Right_Ankle",
}

# Foot: toe POSITION + flat-foot orientation (forward = ankle->toe, up = +Z).
_FOOT_CHAINS = (
    ("left_foot", ("Left_Ankle", "Left_Foot")),
    ("right_foot", ("Right_Ankle", "Right_Foot")),
)

_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
_DEFAULT_HUMAN_HEIGHT = 1.6


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


_WORLD_UP = np.array([0.0, 0.0, 1.0])


def reconstruct_pelvis_quat(positions, prev_quat_wxyz=None):
    """Pelvis global orientation (wxyz) from reliable joint POSITIONS.

    Yaw-only: the base is kept UPRIGHT (up = world +Z) and only its heading
    (yaw) follows the hip line (left = R_hip->L_hip). Frame axes in the robot's
    own convention: forward = left x up (+x), left (+y), up (+z).

    Rationale: the subject stands upright for the whole capture (pelvis height
    above foot stays ~constant); forcing an upright base is robust and avoids
    the base double-counting genuine upper-torso lean, which the waist joint
    already captures via the spine3 POSITION target. Falls back to
    ``prev_quat_wxyz`` (or identity) if the hip line is degenerate.
    """
    up = _WORLD_UP
    lat = _unit(positions[_JI["Left_Hip"]] - positions[_JI["Right_Hip"]])
    if lat is None:
        return prev_quat_wxyz if prev_quat_wxyz is not None else _IDENTITY_WXYZ.copy()
    lat = _unit(lat - np.dot(lat, up) * up)  # project hip line onto horizontal
    if lat is None:
        return prev_quat_wxyz if prev_quat_wxyz is not None else _IDENTITY_WXYZ.copy()
    fwd = np.cross(lat, up)
    mat = np.stack([fwd, lat, up], axis=1)  # columns: forward(+x), left(+y), up(+z)
    return R.from_matrix(mat).as_quat(scalar_first=True)


def _unit_or(v, fallback):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else fallback


def reconstruct_chest_quat(positions):
    """Chest/torso global orientation (wxyz) from reliable joint POSITIONS.

    up = pelvis->spine3 (torso pitch/lean), left = R_shoulder->L_shoulder
    (torso twist/yaw), forward = left x up. Columns [forward, left, up] in the
    robot convention (forward=+x, left=+y, up=+z). Drives the waist joint so it
    follows the real torso -- and, being strongly weighted, stops the arm
    orientation targets (which sit downstream of the waist) from bending the
    waist to satisfy themselves.
    """
    up = _unit_or(positions[_JI["Spine3"]] - positions[_JI["Pelvis"]], _WORLD_UP)
    lat = _unit_or(positions[_JI["Left_Shoulder"]] - positions[_JI["Right_Shoulder"]],
                   np.array([0.0, 1.0, 0.0]))
    fwd = _unit_or(np.cross(lat, up), np.array([1.0, 0.0, 0.0]))
    lat = np.cross(up, fwd)
    mat = np.stack([fwd, lat, up], axis=1)
    return R.from_matrix(mat).as_quat(scalar_first=True)


def limb_frames(proximal, mid, distal):
    """(T_proximal, T_mid) 3x3 rotation matrices from a 3-joint limb chain.

    Works for an arm (shoulder, elbow, wrist) or a leg (hip, knee, ankle):
      T_proximal x-axis = proximal-segment direction (proximal->mid);
      T_mid      x-axis = distal-segment direction   (mid->distal);
      shared z-axis     = joint-bend-plane normal (proximal_seg x distal_seg).
    Robust to a straight (degenerate-plane) limb.
    """
    a = _unit_or(mid - proximal, np.array([0.0, 0.0, -1.0]))
    b = _unit_or(distal - mid, a)
    n = _unit_or(np.cross(a, b), None)
    if n is None:  # straight limb: any axis perpendicular to the proximal seg
        n = _unit_or(np.cross(a, np.array([0.0, 0.0, 1.0])), np.array([0.0, 1.0, 0.0]))
    y_a = _unit_or(np.cross(n, a), np.array([0.0, 1.0, 0.0]))
    y_b = _unit_or(np.cross(n, b), np.array([0.0, 1.0, 0.0]))
    return np.stack([a, y_a, n], axis=1), np.stack([b, y_b, n], axis=1)


# Back-compat alias (arms are just a limb chain).
arm_frames = limb_frames


def foot_frame(ankle, toe):
    """Flat-foot global orientation matrix: forward = ankle->toe (horizontal),
    up = world +Z, columns [forward, left, up]."""
    fwd = _unit_or(toe - ankle, np.array([1.0, 0.0, 0.0]))
    fwd = _unit_or(fwd - np.dot(fwd, _WORLD_UP) * _WORLD_UP, np.array([1.0, 0.0, 0.0]))
    left = np.cross(_WORLD_UP, fwd)
    return np.stack([fwd, left, _WORLD_UP], axis=1)


def _positions_to_frame(positions, prev_pelvis_quat):
    def q(mat):
        return R.from_matrix(mat).as_quat(scalar_first=True)

    pelvis_quat = reconstruct_pelvis_quat(positions, prev_pelvis_quat)
    frame = {
        "pelvis": [positions[_JI["Pelvis"]].astype(np.float64), pelvis_quat],
        "spine3": [positions[_JI["Spine3"]].astype(np.float64),
                   reconstruct_chest_quat(positions)],
    }
    for human_name, pico_name in _LEG_POSITION_TARGETS.items():
        frame[human_name] = [positions[_JI[pico_name]].astype(np.float64),
                             _IDENTITY_WXYZ.copy()]
    for foot_name, (ankle_j, toe_j) in _FOOT_CHAINS:
        # position target is the TOE (Pico Left_Foot == toe joint), orientation
        # is the flat-foot frame.
        frame[foot_name] = [positions[_JI[toe_j]].astype(np.float64),
                            q(foot_frame(positions[_JI[ankle_j]], positions[_JI[toe_j]]))]
    for prox_name, mid_name, (p_j, m_j, d_j) in _ARM_CHAINS:
        t_prox, t_mid = limb_frames(
            positions[_JI[p_j]], positions[_JI[m_j]], positions[_JI[d_j]]
        )
        frame[prox_name] = [positions[_JI[p_j]].astype(np.float64), q(t_prox)]
        frame[mid_name] = [positions[_JI[m_j]].astype(np.float64), q(t_mid)]
    return frame, pelvis_quat


def _resample_positions(positions, timestamps_s, tgt_fps):
    """Linearly resample (T,24,3) positions onto a uniform ``tgt_fps`` grid."""
    t0, t1 = timestamps_s[0], timestamps_s[-1]
    duration = t1 - t0
    if duration <= 0 or tgt_fps <= 0:
        return positions, len(positions) / max(duration, 1e-9)
    n_out = int(np.floor(duration * tgt_fps)) + 1
    target_t = t0 + np.arange(n_out) / tgt_fps
    out = np.empty((n_out,) + positions.shape[1:], dtype=positions.dtype)
    for axis in range(positions.shape[1]):
        for c in range(3):
            out[:, axis, c] = np.interp(target_t, timestamps_s, positions[:, axis, c])
    return out, float(tgt_fps)


def load_pico_xrt_file(jsonl_file, tgt_fps=None):
    """Load a Pico XRT ``.jsonl`` recording for GMR position-based retargeting.

    Args:
        jsonl_file: path to ``xrobot_*.jsonl``.
        tgt_fps: if given, resample the (variable-rate) capture onto a uniform
            ``tgt_fps`` grid. If ``None``, keep native frames and report the
            measured mean capture rate.

    Returns:
        (frames, human_height, fps)
        frames: list of ``{human_body_name: [pos(3,), quat_wxyz(4,)]}``, where
            only ``pelvis`` carries a meaningful (reconstructed) orientation.
        human_height: meters (from the recording's ``actual_human_height``).
        fps: frames-per-second of the returned sequence.
    """
    positions = []
    timestamps_ns = []
    human_height = _DEFAULT_HUMAN_HEIGHT
    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            body = rec.get("body_data")
            if not body or "Pelvis" not in body:
                continue
            positions.append([np.asarray(body[n][0], dtype=np.float64)
                              for n in PICO_JOINT_ORDER])
            timestamps_ns.append(int(rec.get("wall_time_ns", 0)))
            if "actual_human_height" in rec:
                human_height = float(rec["actual_human_height"])

    if not positions:
        raise ValueError(f"No usable body_data frames found in {jsonl_file}")
    positions = np.asarray(positions)  # (T, 24, 3)

    timestamps_s = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    if (len(timestamps_s) > 1 and np.all(np.diff(timestamps_s) >= 0)
            and timestamps_s[-1] > timestamps_s[0]):
        native_fps = (len(timestamps_s) - 1) / (timestamps_s[-1] - timestamps_s[0])
    else:
        native_fps = 30.0
        timestamps_s = np.arange(len(positions)) / native_fps

    if tgt_fps is not None:
        positions, fps = _resample_positions(positions, timestamps_s, tgt_fps)
    else:
        fps = native_fps

    frames = []
    prev_pelvis_quat = None
    for k in range(len(positions)):
        frame, prev_pelvis_quat = _positions_to_frame(positions[k], prev_pelvis_quat)
        frames.append(frame)

    return frames, human_height, fps
