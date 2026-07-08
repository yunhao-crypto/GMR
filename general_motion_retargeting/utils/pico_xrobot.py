"""Pico jsonl loader with TWIST2/XRobotStreamer-compatible semantics.

TWIST2 uses ``XRobotStreamer.get_processed_body_data()`` from upstream GMR:
raw XR body poses are converted from Unity coordinates to a right-handed frame,
and the original per-joint quaternions are passed to GMR as global body frames.

This loader provides the same frame shape for offline ``xrobot_*.jsonl``
recordings. Some recordings in this repository were already written after a
teleop-side coordinate transform, so ``coordinate_mode="stored"`` is the
default. Use ``coordinate_mode="unity"`` for raw XRoboToolkit SDK logs.
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from ..rot_utils import quat_mul_np
from .pico_xrt import PICO_JOINT_ORDER


_DEFAULT_HUMAN_HEIGHT = 1.6
_UNITY_TO_XROBOT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_UNITY_TO_XROBOT_QUAT = R.from_matrix(_UNITY_TO_XROBOT).as_quat(
    scalar_first=True
)


def coordinate_transform_unity_data(frame):
    """Match upstream ``XRobotStreamer.coordinate_transform_unity_data``."""
    out = {}
    for body_name, value in frame.items():
        x, y, z = value[0]
        qw, qx, qy, qz = value[1]
        orientation = quat_mul_np(
            _UNITY_TO_XROBOT_QUAT,
            np.array([qw, qx, qy, qz], dtype=np.float64),
            scalar_first=True,
        )
        position = np.array([x, y, z], dtype=np.float64) @ _UNITY_TO_XROBOT.T
        out[body_name] = [position, orientation]
    return out


def _read_arrays(jsonl_file):
    positions = []
    quats_xyzw = []
    timestamps_ns = []
    human_height = _DEFAULT_HUMAN_HEIGHT

    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            body = rec.get("body_data")
            raw_body_poses = rec.get("body_poses")
            if body and "Pelvis" in body:
                positions.append(
                    [np.asarray(body[name][0], dtype=np.float64) for name in PICO_JOINT_ORDER]
                )
                quats_xyzw.append(
                    [np.asarray(body[name][1], dtype=np.float64) for name in PICO_JOINT_ORDER]
                )
            elif raw_body_poses and len(raw_body_poses) >= len(PICO_JOINT_ORDER):
                positions.append(
                    [np.asarray(raw_body_poses[i][:3], dtype=np.float64) for i in range(len(PICO_JOINT_ORDER))]
                )
                quats_xyzw.append(
                    [np.asarray(raw_body_poses[i][3:7], dtype=np.float64) for i in range(len(PICO_JOINT_ORDER))]
                )
            else:
                continue
            timestamps_ns.append(int(rec.get("wall_time_ns", 0)))
            if "actual_human_height" in rec:
                human_height = float(rec["actual_human_height"])

    if not positions:
        raise ValueError(f"No usable body_data frames found in {jsonl_file}")

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(quats_xyzw, dtype=np.float64),
        np.asarray(timestamps_ns, dtype=np.float64) / 1e9,
        human_height,
    )


def _resample(positions, quats_xyzw, timestamps_s, tgt_fps):
    if (
        len(timestamps_s) < 2
        or np.any(np.diff(timestamps_s) < 0.0)
        or timestamps_s[-1] <= timestamps_s[0]
    ):
        native_fps = 30.0
        timestamps_s = np.arange(len(positions), dtype=np.float64) / native_fps
    else:
        native_fps = (len(timestamps_s) - 1) / (timestamps_s[-1] - timestamps_s[0])

    if tgt_fps is None:
        return positions, quats_xyzw, float(native_fps)

    t0, t1 = timestamps_s[0], timestamps_s[-1]
    target_t = t0 + np.arange(int(np.floor((t1 - t0) * tgt_fps)) + 1) / tgt_fps

    pos_out = np.empty((len(target_t), len(PICO_JOINT_ORDER), 3), dtype=np.float64)
    for joint_idx in range(len(PICO_JOINT_ORDER)):
        for axis in range(3):
            pos_out[:, joint_idx, axis] = np.interp(
                target_t, timestamps_s, positions[:, joint_idx, axis]
            )

    quat_out = np.empty((len(target_t), len(PICO_JOINT_ORDER), 4), dtype=np.float64)
    for joint_idx in range(len(PICO_JOINT_ORDER)):
        q = quats_xyzw[:, joint_idx, :]
        norms = np.linalg.norm(q, axis=1)
        good = norms > 1e-8
        if np.count_nonzero(good) < 2:
            quat_out[:, joint_idx, :] = np.array([0.0, 0.0, 0.0, 1.0])
            continue
        q = q[good] / norms[good, None]
        t = timestamps_s[good]
        uniq_t, uniq_idx = np.unique(t, return_index=True)
        quat_out[:, joint_idx, :] = Slerp(uniq_t, R.from_quat(q[uniq_idx]))(
            target_t
        ).as_quat()

    return pos_out, quat_out, float(tgt_fps)


def _arrays_to_frames(positions, quats_xyzw, coordinate_mode):
    frames = []
    for frame_idx in range(len(positions)):
        frame = {}
        for joint_idx, name in enumerate(PICO_JOINT_ORDER):
            qx, qy, qz, qw = quats_xyzw[frame_idx, joint_idx]
            frame[name] = [
                positions[frame_idx, joint_idx].copy(),
                np.array([qw, qx, qy, qz], dtype=np.float64),
            ]
        if coordinate_mode == "unity":
            frame = coordinate_transform_unity_data(frame)
        elif coordinate_mode != "stored":
            raise ValueError(
                "coordinate_mode must be 'stored' or 'unity', "
                f"got {coordinate_mode!r}"
            )
        frames.append(frame)
    return frames


def load_pico_xrobot_file(jsonl_file, tgt_fps=None, coordinate_mode="stored"):
    """Load a Pico jsonl as TWIST2/XRobot-style GMR frames.

    Args:
        jsonl_file: Pico/XRobot recording with ``body_data`` entries.
        tgt_fps: optional resampling fps.
        coordinate_mode: ``"stored"`` for already-transformed recordings,
            ``"unity"`` to apply upstream XRobotStreamer's Unity conversion.

    Returns:
        ``(frames, human_height, fps)`` where each frame uses Pico/XRobot body
        names such as ``"Pelvis"`` and global quaternions in wxyz order.
    """
    positions, quats_xyzw, timestamps_s, human_height = _read_arrays(jsonl_file)
    positions, quats_xyzw, fps = _resample(
        positions, quats_xyzw, timestamps_s, tgt_fps
    )
    frames = _arrays_to_frames(positions, quats_xyzw, coordinate_mode)
    return frames, human_height, fps
