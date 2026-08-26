import mujoco as mj
import numpy as np
from scipy.signal import butter, filtfilt

from ._geom_utils import collect_geom_ids_by_token, geom_min_z


_VT_HUMAN_POSTPROCESS_SPEC = {
    "butterworth_cutoff_hz": 2.0,
    "butterworth_order": 4,
    "filter_joint_names": (
        "waist_yaw_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ),
    "standing_segment": {
        "body_names": ("base_link", "left_toe_link", "right_toe_link"),
        "base_xy_step_max": 0.004,
        "toe_xy_step_max": 0.003,
        "min_run_len": 12,
        "butterworth_cutoff_hz": 1.5,
        "root_position_qpos_indices": (0, 1),
        "joint_names": (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
    },
    "ground_geom_name_token": "foot",
    "ground_clearance": 0.0,
    "stance_ground_snap": {
        "enabled": True,
        "max_height": 0.08,
        "max_vertical_speed": 0.20,
        "max_snap_delta": 0.08,
        "offset_smooth_cutoff_hz": 6.0,
    },
}

# Vita02A: identical structure to the VT spec (same joint naming, same
# base_link/toe standing bodies, same collision_*_foot* geom token) — only the
# global filter set grows by the six wrist DOFs vt_human lacks.
_VITA02A_POSTPROCESS_SPEC = {
    **_VT_HUMAN_POSTPROCESS_SPEC,
    "filter_joint_names": (
        *_VT_HUMAN_POSTPROCESS_SPEC["filter_joint_names"],
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    ),
}

_SEQUENCE_POSTPROCESS_SPECS = {
    "vt_human": _VT_HUMAN_POSTPROCESS_SPEC,
    "vt_human_v2": _VT_HUMAN_POSTPROCESS_SPEC,
    "vita02a": _VITA02A_POSTPROCESS_SPEC,
}


def has_sequence_postprocess(robot_name):
    return robot_name in _SEQUENCE_POSTPROCESS_SPECS


def _butterworth_lowpass_smooth_time(array, fps, cutoff_hz, order):
    num_frames = array.shape[0]
    if num_frames < 3 or fps <= 0.0 or cutoff_hz <= 0.0:
        return array.astype(np.float32, copy=True)

    nyquist = 0.5 * float(fps)
    wn = float(cutoff_hz) / nyquist
    wn = min(max(wn, 1.0e-6), 0.999)

    flat = array.astype(np.float64, copy=False).reshape(num_frames, -1)
    b, a = butter(int(order), wn, btype="low", analog=False)
    maxlen = max(len(b), len(a))
    padlen_required = max(3 * (maxlen - 1), 3 * maxlen)
    if num_frames <= padlen_required:
        return array.astype(np.float32, copy=True)

    filtered = filtfilt(b, a, flat, axis=0, method="pad")
    return filtered.reshape(array.shape).astype(np.float32, copy=False)


def _quat_normalize(quat):
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (quat / norm).astype(np.float32, copy=False)


def _quat_hemisphere_align(quat):
    if quat.shape[0] == 0:
        return quat
    aligned = quat.astype(np.float32, copy=True)
    prev = aligned[0]
    for frame_idx in range(1, aligned.shape[0]):
        if float(np.dot(prev, aligned[frame_idx])) < 0.0:
            aligned[frame_idx] = -aligned[frame_idx]
        prev = aligned[frame_idx]
    return aligned


def _collect_body_ids(model, body_names):
    body_ids = []
    missing = []
    for body_name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            missing.append(body_name)
        else:
            body_ids.append(body_id)
    return body_ids, missing


def _collect_body_positions(retargeter, qpos_sequence, body_ids):
    positions = np.zeros((len(qpos_sequence), len(body_ids), 3), dtype=np.float32)
    for frame_idx, qpos in enumerate(qpos_sequence):
        retargeter.configuration.data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, retargeter.configuration.data)
        for body_list_idx, body_id in enumerate(body_ids):
            positions[frame_idx, body_list_idx] = retargeter.configuration.data.xpos[
                body_id
            ]
    return positions


def _build_standing_mask(body_positions, base_xy_step_max, toe_xy_step_max, min_run_len):
    if body_positions.shape[0] < 2:
        return np.zeros(body_positions.shape[0], dtype=bool)

    base_xy_step = np.linalg.norm(
        np.diff(body_positions[:, 0, :2], axis=0), axis=1
    )
    left_toe_xy_step = np.linalg.norm(
        np.diff(body_positions[:, 1, :2], axis=0), axis=1
    )
    right_toe_xy_step = np.linalg.norm(
        np.diff(body_positions[:, 2, :2], axis=0), axis=1
    )
    edge_mask = (
        (base_xy_step <= float(base_xy_step_max))
        & (left_toe_xy_step <= float(toe_xy_step_max))
        & (right_toe_xy_step <= float(toe_xy_step_max))
    )

    frame_mask = np.zeros(body_positions.shape[0], dtype=bool)
    frame_mask[1:] = edge_mask
    frame_mask[:-1] |= edge_mask

    standing_mask = np.zeros_like(frame_mask)
    run_start = None
    for frame_idx, active in enumerate(frame_mask):
        if active and run_start is None:
            run_start = frame_idx
        elif (not active) and run_start is not None:
            if frame_idx - run_start >= int(min_run_len):
                standing_mask[run_start:frame_idx] = True
            run_start = None
    if run_start is not None and len(frame_mask) - run_start >= int(min_run_len):
        standing_mask[run_start:] = True
    return standing_mask


def _apply_segment_joint_filter(
    qpos_sequence,
    qpos_indices,
    standing_mask,
    motion_fps,
    cutoff_hz,
    butterworth_order,
):
    if not qpos_indices or not np.any(standing_mask):
        return qpos_sequence

    filtered = qpos_sequence.copy()
    run_start = None
    for frame_idx, active in enumerate(standing_mask):
        if active and run_start is None:
            run_start = frame_idx
        elif (not active) and run_start is not None:
            filtered[run_start:frame_idx, qpos_indices] = _butterworth_lowpass_smooth_time(
                filtered[run_start:frame_idx, qpos_indices],
                motion_fps,
                cutoff_hz,
                butterworth_order,
            )
            run_start = None
    if run_start is not None:
        filtered[run_start:, qpos_indices] = _butterworth_lowpass_smooth_time(
            filtered[run_start:, qpos_indices],
            motion_fps,
            cutoff_hz,
            butterworth_order,
        )
    return filtered


def _apply_segment_root_quat_filter(
    qpos_sequence,
    quat_slice,
    standing_mask,
    motion_fps,
    cutoff_hz,
    butterworth_order,
):
    if quat_slice is None or not np.any(standing_mask):
        return qpos_sequence

    start_idx, end_idx = quat_slice
    if end_idx - start_idx != 4:
        return qpos_sequence

    filtered = qpos_sequence.copy()
    run_start = None
    for frame_idx, active in enumerate(standing_mask):
        if active and run_start is None:
            run_start = frame_idx
        elif (not active) and run_start is not None:
            quat_segment = filtered[run_start:frame_idx, start_idx:end_idx]
            quat_segment = _quat_hemisphere_align(_quat_normalize(quat_segment))
            quat_segment = _butterworth_lowpass_smooth_time(
                quat_segment,
                motion_fps,
                cutoff_hz,
                butterworth_order,
            )
            filtered[run_start:frame_idx, start_idx:end_idx] = _quat_normalize(
                quat_segment
            )
            run_start = None
    if run_start is not None:
        quat_segment = filtered[run_start:, start_idx:end_idx]
        quat_segment = _quat_hemisphere_align(_quat_normalize(quat_segment))
        quat_segment = _butterworth_lowpass_smooth_time(
            quat_segment,
            motion_fps,
            cutoff_hz,
            butterworth_order,
        )
        filtered[run_start:, start_idx:end_idx] = _quat_normalize(quat_segment)
    return filtered


def _resolve_qpos_indices(model, joint_names):
    """qpos start-index per named joint; missing joints are dropped silently."""
    indices = []
    for joint_name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            continue
        indices.append(int(model.jnt_qposadr[joint_id]))
    return indices


def _apply_global_joint_filter(qpos_sequence, retargeter, spec, motion_fps):
    """Step 1: full-sequence Butterworth on the spec's filter_joint_names."""
    cutoff_hz = float(spec["butterworth_cutoff_hz"])
    if cutoff_hz <= 0.0:
        return qpos_sequence
    qpos_indices = _resolve_qpos_indices(retargeter.model, spec["filter_joint_names"])
    if not qpos_indices:
        return qpos_sequence
    qpos_sequence[:, qpos_indices] = _butterworth_lowpass_smooth_time(
        qpos_sequence[:, qpos_indices],
        motion_fps,
        cutoff_hz,
        int(spec["butterworth_order"]),
    )
    return qpos_sequence


def _apply_standing_segment_filters(
    qpos_sequence, retargeter, standing_spec, motion_fps, butterworth_order
):
    """Step 2: detect standing segments via toe/base XY motion, then filter root_xy / root_quat / leg joints inside those segments."""
    body_ids, missing_bodies = _collect_body_ids(
        retargeter.model, standing_spec["body_names"]
    )
    if missing_bodies or len(body_ids) != 3:
        return qpos_sequence

    body_positions = _collect_body_positions(retargeter, qpos_sequence, body_ids)
    standing_mask = _build_standing_mask(
        body_positions,
        standing_spec["base_xy_step_max"],
        standing_spec["toe_xy_step_max"],
        standing_spec["min_run_len"],
    )
    cutoff_hz = float(standing_spec["butterworth_cutoff_hz"])

    qpos_sequence = _apply_segment_joint_filter(
        qpos_sequence,
        list(standing_spec.get("root_position_qpos_indices", ())),
        standing_mask,
        motion_fps,
        cutoff_hz,
        butterworth_order,
    )
    qpos_sequence = _apply_segment_root_quat_filter(
        qpos_sequence,
        standing_spec.get("root_quaternion_qpos_slice", (3, 7)),
        standing_mask,
        motion_fps,
        cutoff_hz,
        butterworth_order,
    )
    qpos_sequence = _apply_segment_joint_filter(
        qpos_sequence,
        _resolve_qpos_indices(retargeter.model, standing_spec["joint_names"]),
        standing_mask,
        motion_fps,
        cutoff_hz,
        butterworth_order,
    )
    return qpos_sequence


def _apply_global_ground_snap(qpos_sequence, retargeter, spec):
    """Step 3: shift the sequence up only if any foot geom penetrates the floor."""
    geom_ids = collect_geom_ids_by_token(
        retargeter.model, spec["ground_geom_name_token"]
    )
    if not geom_ids:
        return qpos_sequence

    min_z = np.inf
    for qpos in qpos_sequence:
        retargeter.configuration.data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, retargeter.configuration.data)
        frame_min_z = min(
            geom_min_z(retargeter.model, retargeter.configuration.data, gid)
            for gid in geom_ids
        )
        min_z = min(min_z, frame_min_z)
    if min_z < float(spec["ground_clearance"]):
        qpos_sequence[:, 2] += float(spec["ground_clearance"]) - float(min_z)
    return qpos_sequence


def _compute_frame_foot_min_z(retargeter, qpos_sequence, geom_ids):
    min_z = np.zeros(qpos_sequence.shape[0], dtype=np.float32)
    for frame_idx, qpos in enumerate(qpos_sequence):
        retargeter.configuration.data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, retargeter.configuration.data)
        min_z[frame_idx] = min(
            geom_min_z(retargeter.model, retargeter.configuration.data, gid)
            for gid in geom_ids
        )
    return min_z


def _apply_stance_ground_snap(qpos_sequence, retargeter, spec, motion_fps):
    """Snap likely support frames to the floor by shifting only root z."""
    snap_spec = spec.get("stance_ground_snap")
    if not snap_spec or not bool(snap_spec.get("enabled", False)):
        return qpos_sequence

    geom_ids = collect_geom_ids_by_token(
        retargeter.model, spec["ground_geom_name_token"]
    )
    if not geom_ids or qpos_sequence.shape[0] == 0:
        return qpos_sequence

    foot_min_z = _compute_frame_foot_min_z(retargeter, qpos_sequence, geom_ids)
    fps = float(motion_fps)
    if fps > 0.0 and foot_min_z.shape[0] >= 2:
        vertical_speed = np.abs(np.gradient(foot_min_z, 1.0 / fps))
    else:
        vertical_speed = np.zeros_like(foot_min_z)

    clearance = float(spec["ground_clearance"])
    raw_delta = clearance - foot_min_z
    support_mask = (
        (foot_min_z <= clearance + float(snap_spec["max_height"]))
        & (vertical_speed <= float(snap_spec["max_vertical_speed"]))
        & (np.abs(raw_delta) <= float(snap_spec["max_snap_delta"]))
    )
    if not np.any(support_mask):
        return qpos_sequence

    snap_delta = np.zeros_like(foot_min_z)
    snap_delta[support_mask] = raw_delta[support_mask]

    cutoff_hz = float(snap_spec.get("offset_smooth_cutoff_hz", 0.0))
    if cutoff_hz > 0.0:
        snap_delta = _butterworth_lowpass_smooth_time(
            snap_delta[:, None],
            motion_fps,
            cutoff_hz,
            2,
        )[:, 0]
        snap_delta[~support_mask] = 0.0

    snapped = qpos_sequence.copy()
    snapped[:, 2] += snap_delta
    return snapped


def postprocess_qpos_sequence(retargeter, qpos_sequence, motion_fps):
    spec = _SEQUENCE_POSTPROCESS_SPECS.get(retargeter.tgt_robot)
    if spec is None:
        return qpos_sequence

    qpos_sequence = np.asarray(qpos_sequence, dtype=np.float32).copy()
    if qpos_sequence.ndim != 2 or qpos_sequence.shape[1] < 8:
        raise ValueError(
            f"Expected qpos sequence with shape [T, qpos_dim], got {qpos_sequence.shape}"
        )

    qpos_sequence = _apply_global_joint_filter(
        qpos_sequence, retargeter, spec, motion_fps
    )

    standing_spec = spec.get("standing_segment")
    if standing_spec is not None:
        qpos_sequence = _apply_standing_segment_filters(
            qpos_sequence,
            retargeter,
            standing_spec,
            motion_fps,
            int(spec["butterworth_order"]),
        )

    qpos_sequence = _apply_stance_ground_snap(
        qpos_sequence, retargeter, spec, motion_fps
    )
    qpos_sequence = _apply_global_ground_snap(qpos_sequence, retargeter, spec)
    return qpos_sequence
