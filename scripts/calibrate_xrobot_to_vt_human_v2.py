#!/usr/bin/env python3
"""Calibrate raw XRobot -> VT_human_v2 orientation offsets.

The online retargeter must stay a single ``xrobot -> vt_human_v2`` GMR
instance.  This script only uses the historical ``xrobot -> unitree_g1`` path
offline to estimate static VT ``rot_offset`` values from a Pico/XRobot JSONL
recording.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.pico_xrobot import (
    body_data_to_pico_xrobot_frame,
)


ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

ARM_TASKS = {
    "left_shoulder_yaw_link": ("Left_Shoulder", "shoulder"),
    "left_elbow_link": ("Left_Elbow", "elbow"),
    "left_wrist_task": ("Left_Wrist", "wrist"),
    "right_shoulder_yaw_link": ("Right_Shoulder", "shoulder"),
    "right_elbow_link": ("Right_Elbow", "elbow"),
    "right_wrist_task": ("Right_Wrist", "wrist"),
}

BODY_ORIENTATION_TASKS = {
    "base_link": ("Pelvis", "base"),
    "waist_pitch_link": ("Spine3", "waist"),
    "left_toe_link": ("Left_Foot", "toe"),
    "right_toe_link": ("Right_Foot", "toe"),
}


def _load_records(path: Path, max_frames: int | None, frame_stride: int):
    records = []
    human_height = 1.6
    with path.open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % frame_stride != 0:
                continue
            rec = json.loads(line)
            body = rec.get("body_data")
            if not body or "Pelvis" not in body:
                continue
            records.append(body)
            human_height = float(rec.get("actual_human_height", human_height))
            if max_frames is not None and len(records) >= max_frames:
                break
    if not records:
        raise ValueError(f"No usable body_data frames found in {path}")
    return records, human_height


def _as_numpy_body(body):
    return {
        name: [
            np.asarray(value[0], dtype=np.float64),
            np.asarray(value[1], dtype=np.float64),
        ]
        for name, value in body.items()
    }


def _joint_qpos_index(model: mj.MjModel, joint_name: str) -> int:
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(f"Missing joint in model: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def _copy_arm_qpos_by_name(
    source_model: mj.MjModel,
    target_model: mj.MjModel,
    source_qpos: np.ndarray,
    target_qpos: np.ndarray,
):
    out = target_qpos.copy()
    for joint_name in ARM_JOINTS:
        src_idx = _joint_qpos_index(source_model, joint_name)
        dst_idx = _joint_qpos_index(target_model, joint_name)
        dst_joint_id = mj.mj_name2id(target_model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        value = float(source_qpos[src_idx])
        if target_model.jnt_limited[dst_joint_id]:
            low, high = target_model.jnt_range[dst_joint_id]
            value = float(np.clip(value, low, high))
        out[dst_idx] = value
    return out


def _mean_quat_wxyz(rotations: list[R]) -> list[float]:
    quat = R.concatenate(rotations).mean().as_quat(scalar_first=True)
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return [float(x) for x in quat]


def _estimate_offsets(records, human_height: float):
    g1 = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=False,
    )
    vt_base = GMR(
        src_human="pico_xrobot",
        tgt_robot="vt_human_v2",
        actual_human_height=human_height,
        verbose=False,
    )

    g1_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    vt_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["vt_human_v2"]))
    vt_data = mj.MjData(vt_model)
    calibrated_tasks = {**BODY_ORIENTATION_TASKS, **ARM_TASKS}
    task_body_ids = {
        body_name: mj.mj_name2id(vt_model, mj.mjtObj.mjOBJ_BODY, body_name)
        for body_name in calibrated_tasks
    }
    missing = [name for name, body_id in task_body_ids.items() if body_id == -1]
    if missing:
        raise ValueError(f"VT model is missing calibration bodies: {missing}")

    per_task_rotations: dict[str, list[R]] = {name: [] for name in calibrated_tasks}
    prev_pelvis_quat = None
    prev_arm_normals = None

    for frame_idx, body in enumerate(records):
        xrobot_frame = _as_numpy_body(body)
        pico_frame, prev_pelvis_quat, prev_arm_normals = body_data_to_pico_xrobot_frame(
            copy.deepcopy(xrobot_frame),
            prev_pelvis_quat,
            prev_arm_normals,
        )
        g1_qpos = g1.retarget(copy.deepcopy(xrobot_frame), offset_to_ground=True)
        vt_qpos = vt_base.retarget(pico_frame, offset_to_ground=False)
        arm_ref_qpos = _copy_arm_qpos_by_name(g1_model, vt_model, g1_qpos, vt_qpos)

        vt_data.qpos[:] = vt_qpos
        mj.mj_forward(vt_model, vt_data)
        for task_body, (human_body, _kind) in BODY_ORIENTATION_TASKS.items():
            human_rot = R.from_quat(xrobot_frame[human_body][1], scalar_first=True)
            vt_rot = R.from_matrix(
                vt_data.xmat[task_body_ids[task_body]].reshape(3, 3)
            )
            per_task_rotations[task_body].append(human_rot.inv() * vt_rot)

        vt_data.qpos[:] = arm_ref_qpos
        mj.mj_forward(vt_model, vt_data)
        for task_body, (human_body, _kind) in ARM_TASKS.items():
            human_rot = R.from_quat(xrobot_frame[human_body][1], scalar_first=True)
            vt_rot = R.from_matrix(
                vt_data.xmat[task_body_ids[task_body]].reshape(3, 3)
            )
            per_task_rotations[task_body].append(human_rot.inv() * vt_rot)

        if frame_idx == 0 or (frame_idx + 1) % 100 == 0:
            print(f"[calibrate] processed {frame_idx + 1}/{len(records)} frames")

    return {
        task_body: _mean_quat_wxyz(rotations)
        for task_body, rotations in per_task_rotations.items()
    }


def _update_config(
    template_path: Path,
    output_path: Path,
    offsets: dict[str, list[float]],
    *,
    arm_orientation_cost: float,
    wrist_orientation_cost: float,
    base_orientation_cost: float,
    waist_orientation_cost: float,
    toe_orientation_cost: float,
    dry_run: bool,
):
    config = json.loads(template_path.read_text())
    config["_note"] = (
        "TWIST2/HoloMotion raw XRobot -> vt_human_v2. Base, waist, feet, "
        "and arms use raw XRobot orientations with static VT offsets "
        "calibrated from Pico recordings against the G1 xrobot reference. "
        "Generated by scripts/calibrate_xrobot_to_vt_human_v2.py."
    )
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config[table_name]
        for task_body, (_human_body, kind) in BODY_ORIENTATION_TASKS.items():
            orientation_cost_by_kind = {
                "base": base_orientation_cost,
                "waist": waist_orientation_cost,
                "toe": toe_orientation_cost,
            }
            table[task_body][2] = orientation_cost_by_kind[kind]
            table[task_body][4] = offsets[task_body]
        for task_body, (human_body, kind) in ARM_TASKS.items():
            orientation_cost = (
                wrist_orientation_cost if kind == "wrist" else arm_orientation_cost
            )
            table[task_body] = [
                human_body,
                0,
                orientation_cost,
                [0, 0, 0],
                offsets[task_body],
            ]

    payload = json.dumps(config, indent=4) + "\n"
    if dry_run:
        print(payload)
    else:
        output_path.write_text(payload)
        print(f"[calibrate] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pico-file", default="outputs/test.jsonl")
    parser.add_argument(
        "--template-config",
        default="general_motion_retargeting/ik_configs/xrobot_to_vt_human_v2.json",
    )
    parser.add_argument(
        "--output-config",
        default="general_motion_retargeting/ik_configs/xrobot_to_vt_human_v2.json",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--arm-orientation-cost", type=float, default=10.0)
    parser.add_argument("--wrist-orientation-cost", type=float, default=5.0)
    parser.add_argument("--base-orientation-cost", type=float, default=40.0)
    parser.add_argument("--waist-orientation-cost", type=float, default=40.0)
    parser.add_argument("--toe-orientation-cost", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records, human_height = _load_records(
        Path(args.pico_file),
        max_frames=args.max_frames if args.max_frames > 0 else None,
        frame_stride=max(1, args.frame_stride),
    )
    print(
        f"[calibrate] loaded {len(records)} frames from {args.pico_file}, "
        f"human_height={human_height:.3f}"
    )
    offsets = _estimate_offsets(records, human_height)
    for task_body, quat in offsets.items():
        print(f"[calibrate] {task_body}: {quat}")

    _update_config(
        Path(args.template_config),
        Path(args.output_config),
        offsets,
        arm_orientation_cost=float(args.arm_orientation_cost),
        wrist_orientation_cost=float(args.wrist_orientation_cost),
        base_orientation_cost=float(args.base_orientation_cost),
        waist_orientation_cost=float(args.waist_orientation_cost),
        toe_orientation_cost=float(args.toe_orientation_cost),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
