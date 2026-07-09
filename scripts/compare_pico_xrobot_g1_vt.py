#!/usr/bin/env python3
"""Offline compare TWIST2 G1 XR retargeting against VT_human_v2.

The input is a live Pico/XRobot JSONL recorded by ``live_pico_xrobot_viewer``.
Its ``body_data`` field is already in XRobot/GMR coordinates with wxyz
quaternions.  G1 uses the historical TWIST2 ``xrobot -> unitree_g1`` IK config
directly; VT's calibrated ``xrobot`` path consumes the same raw XRobot frame
with VT-specific static offsets.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT
from general_motion_retargeting.utils.pico_xrobot import (
    body_data_to_pico_xrobot_frame,
)


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


def _retarget_g1_and_vt(
    records,
    human_height,
    *,
    g1_offset_to_ground,
    vt_offset_to_ground,
    vt_source,
):
    print("[compare] initializing G1 xrobot retargeter")
    g1 = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=False,
    )
    print(f"[compare] initializing VT {vt_source} retargeter")
    vt = GMR(
        src_human=vt_source,
        tgt_robot="vt_human_v2",
        actual_human_height=human_height,
        verbose=False,
    )

    g1_qpos = []
    vt_qpos = []
    g1_human = []
    vt_human = []
    prev_pelvis_quat = None
    prev_arm_normals = None
    for idx, body in enumerate(records):
        xrobot_frame = _as_numpy_body(body)
        if vt_source == "xrobot":
            vt_frame = copy.deepcopy(xrobot_frame)
        else:
            vt_frame, prev_pelvis_quat, prev_arm_normals = body_data_to_pico_xrobot_frame(
                copy.deepcopy(xrobot_frame),
                prev_pelvis_quat,
                prev_arm_normals,
            )
        g1_q = g1.retarget(
            copy.deepcopy(xrobot_frame),
            offset_to_ground=g1_offset_to_ground,
        )
        vt_q = vt.retarget(
            vt_frame,
            offset_to_ground=vt_offset_to_ground,
        )
        g1_qpos.append(g1_q.copy())
        vt_qpos.append(vt_q.copy())
        g1_human.append(copy.deepcopy(g1.scaled_human_data))
        vt_human.append(copy.deepcopy(vt.scaled_human_data))
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"[compare] retargeted {idx + 1}/{len(records)} frames")

    return (
        np.asarray(g1_qpos, dtype=np.float32),
        np.asarray(vt_qpos, dtype=np.float32),
        g1_human,
        vt_human,
    )


class _OffscreenRobot:
    def __init__(self, robot: str, width: int, height: int, azimuth: float):
        self.robot = robot
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, width=width, height=height)
        self.camera = mj.MjvCamera()
        self.camera.distance = 2.0
        self.camera.elevation = -10.0
        self.camera.azimuth = float(azimuth)
        self.base_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, ROBOT_BASE_DICT[robot]
        )
        if self.base_body_id == -1:
            self.base_body_id = 0

    def render(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.xpos[self.base_body_id]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def _body_position_trace(robot: str, qpos: np.ndarray, body_names: tuple[str, ...]):
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ROBOT_BASE_DICT[robot])
    body_ids = {}
    missing = []
    for body_name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            missing.append(body_name)
        else:
            body_ids[body_name] = body_id
    if missing:
        raise ValueError(f"{robot} is missing metric bodies: {missing}")

    traces = {name: np.zeros((len(qpos), 3), dtype=np.float32) for name in body_ids}
    for frame_idx, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        root = data.xpos[base_id].copy()
        for name, body_id in body_ids.items():
            traces[name][frame_idx] = data.xpos[body_id] - root
    return traces


def _write_metrics(g1_qpos: np.ndarray, vt_qpos: np.ndarray, path: Path):
    body_pairs = {
        "torso": ("torso_link", "waist_pitch_link"),
        "left_toe": ("left_toe_link", "left_toe_link"),
        "right_toe": ("right_toe_link", "right_toe_link"),
        "left_shoulder": ("left_shoulder_yaw_link", "left_shoulder_yaw_link"),
        "right_shoulder": ("right_shoulder_yaw_link", "right_shoulder_yaw_link"),
        "left_elbow": ("left_elbow_link", "left_elbow_link"),
        "right_elbow": ("right_elbow_link", "right_elbow_link"),
        "left_wrist": ("left_wrist_yaw_link", "left_wrist_task"),
        "right_wrist": ("right_wrist_yaw_link", "right_wrist_task"),
    }
    g1_bodies = tuple(pair[0] for pair in body_pairs.values())
    vt_bodies = tuple(pair[1] for pair in body_pairs.values())
    g1_trace = _body_position_trace("unitree_g1", g1_qpos, g1_bodies)
    vt_trace = _body_position_trace("vt_human_v2", vt_qpos, vt_bodies)

    metrics = {}
    for metric_name, (g1_body, vt_body) in body_pairs.items():
        delta = g1_trace[g1_body] - vt_trace[vt_body]
        norm = np.linalg.norm(delta, axis=1)
        metrics[metric_name] = {
            "g1_body": g1_body,
            "vt_body": vt_body,
            "rms_m": float(np.sqrt(np.mean(norm * norm))),
            "mean_m": float(np.mean(norm)),
            "max_m": float(np.max(norm)),
        }
    avg_rms = float(np.mean([item["rms_m"] for item in metrics.values()]))

    segment_pairs = {
        "left_upper_arm": (
            ("left_shoulder_yaw_link", "left_elbow_link"),
            ("left_shoulder_yaw_link", "left_elbow_link"),
        ),
        "left_forearm": (
            ("left_elbow_link", "left_wrist_yaw_link"),
            ("left_elbow_link", "left_wrist_task"),
        ),
        "right_upper_arm": (
            ("right_shoulder_yaw_link", "right_elbow_link"),
            ("right_shoulder_yaw_link", "right_elbow_link"),
        ),
        "right_forearm": (
            ("right_elbow_link", "right_wrist_yaw_link"),
            ("right_elbow_link", "right_wrist_task"),
        ),
    }
    g1_segment_bodies = tuple(
        dict.fromkeys(body for pair, _ in segment_pairs.values() for body in pair)
    )
    vt_segment_bodies = tuple(
        dict.fromkeys(body for _, pair in segment_pairs.values() for body in pair)
    )
    g1_segment_trace = _body_position_trace("unitree_g1", g1_qpos, g1_segment_bodies)
    vt_segment_trace = _body_position_trace("vt_human_v2", vt_qpos, vt_segment_bodies)
    direction_metrics = {}
    for metric_name, (g1_segment, vt_segment) in segment_pairs.items():
        g1_vec = g1_segment_trace[g1_segment[1]] - g1_segment_trace[g1_segment[0]]
        vt_vec = vt_segment_trace[vt_segment[1]] - vt_segment_trace[vt_segment[0]]
        g1_vec = g1_vec / np.maximum(
            np.linalg.norm(g1_vec, axis=1, keepdims=True),
            1e-9,
        )
        vt_vec = vt_vec / np.maximum(
            np.linalg.norm(vt_vec, axis=1, keepdims=True),
            1e-9,
        )
        angle_deg = np.degrees(
            np.arccos(np.clip(np.sum(g1_vec * vt_vec, axis=1), -1.0, 1.0))
        )
        direction_metrics[metric_name] = {
            "g1_segment": g1_segment,
            "vt_segment": vt_segment,
            "mean_deg": float(np.mean(angle_deg)),
            "rms_deg": float(np.sqrt(np.mean(angle_deg * angle_deg))),
            "max_deg": float(np.max(angle_deg)),
        }
        for frame_idx, label in ((270, "frame_270_9s_deg"), (540, "frame_540_18s_deg")):
            if frame_idx < len(angle_deg):
                direction_metrics[metric_name][label] = float(angle_deg[frame_idx])

    payload = {
        "frame_count": int(min(len(g1_qpos), len(vt_qpos))),
        "body_position_relative_to_base": metrics,
        "average_rms_m": avg_rms,
        "arm_segment_direction_error_deg": direction_metrics,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[compare] saved metrics: {path}")
    print(f"[compare] average relative-link RMS: {avg_rms:.4f} m")


def _write_side_by_side_video(
    g1_qpos,
    vt_qpos,
    fps,
    path: Path,
    width: int,
    height: int,
    g1_azimuth: float,
    vt_azimuth: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[compare] rendering side-by-side video: {path}")
    g1_view = _OffscreenRobot("unitree_g1", width, height, g1_azimuth)
    vt_view = _OffscreenRobot("vt_human_v2", width, height, vt_azimuth)
    try:
        with imageio.get_writer(path, fps=fps) as writer:
            for idx, (g1_q, vt_q) in enumerate(zip(g1_qpos, vt_qpos)):
                g1_img = g1_view.render(g1_q)
                vt_img = vt_view.render(vt_q)
                divider = np.zeros((height, 16, 3), dtype=np.uint8)
                frame = np.concatenate([g1_img, divider, vt_img], axis=1)
                writer.append_data(frame)
                if idx == 0 or (idx + 1) % 100 == 0:
                    print(f"[compare] rendered {idx + 1}/{len(g1_qpos)} frames")
    finally:
        g1_view.close()
        vt_view.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pico-file", default="outputs/test.jsonl")
    parser.add_argument("--qpos-npz", default="",
                        help="Reuse a previously saved g1_vt_qpos_compare.npz and only render video.")
    parser.add_argument("--output-dir", default="outputs/g1_vt_compare")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--g1-camera-azimuth", type=float, default=180.0)
    parser.add_argument("--vt-camera-azimuth", type=float, default=0.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--vt-source", choices=["xrobot", "pico_xrobot"],
                        default="xrobot",
                        help="VT retarget source. xrobot mirrors TWIST2/G1 raw XR retargeting.")
    parser.add_argument("--g1-no-offset-to-ground", action="store_true")
    parser.add_argument("--vt-offset-to-ground", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.qpos_npz:
        qpos_path = Path(args.qpos_npz)
        data = np.load(qpos_path)
        g1_qpos = data["g1_qpos"]
        vt_qpos = data["vt_qpos"]
        args.fps = float(data["fps"])
        print(f"[compare] loaded qpos arrays: {qpos_path}")
    else:
        pico_file = Path(args.pico_file)
        records, human_height = _load_records(
            pico_file,
            max_frames=args.max_frames if args.max_frames > 0 else None,
            frame_stride=max(1, args.frame_stride),
        )
        print(
            f"[compare] loaded {len(records)} frames from {pico_file}, "
            f"human_height={human_height:.3f}"
        )

        g1_qpos, vt_qpos, _g1_human, _vt_human = _retarget_g1_and_vt(
            records,
            human_height,
            g1_offset_to_ground=not args.g1_no_offset_to_ground,
            vt_offset_to_ground=args.vt_offset_to_ground,
            vt_source=args.vt_source,
        )

        qpos_path = output_dir / "g1_vt_qpos_compare.npz"
        np.savez_compressed(
            qpos_path,
            fps=np.float32(args.fps),
            g1_qpos=g1_qpos,
            vt_qpos=vt_qpos,
            source_file=str(pico_file),
        )
        print(f"[compare] saved qpos arrays: {qpos_path}")
    print(
        f"[compare] qpos shapes: g1={g1_qpos.shape}, vt_human_v2={vt_qpos.shape}"
    )
    _write_metrics(g1_qpos, vt_qpos, output_dir / "metrics.json")

    if not args.no_video:
        _write_side_by_side_video(
            g1_qpos,
            vt_qpos,
            args.fps,
            output_dir / "g1_vs_vt_human_v2.mp4",
            args.video_width,
            args.video_height,
            args.g1_camera_azimuth,
            args.vt_camera_azimuth,
        )


if __name__ == "__main__":
    main()
