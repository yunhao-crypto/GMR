#!/usr/bin/env python3
"""Live Pico XRoboToolkit/TWIST2-style to vt_human_v2 MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.aorta_lowcmd_publisher import (
    AortaLowCmdPublisher,
    DEFAULT_TOPIC as DEFAULT_AORTA_TOPIC,
    LOWCMD_LAYOUTS,
)
from general_motion_retargeting.utils.pico_xrobot import (
    body_data_to_pico_xrobot_frame,
)
from general_motion_retargeting.utils.xrobot_streamer import XRobotStreamer


def _jsonable_body_data(body_data):
    return {
        name: [
            np.asarray(value[0], dtype=float).tolist(),
            np.asarray(value[1], dtype=float).tolist(),
        ]
        for name, value in body_data.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="vt_human_v2", choices=["vt_human_v2"])
    parser.add_argument("--actual-human-height", type=float, default=1.6)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--record-path", default="")
    parser.add_argument("--show-human", action="store_true")
    parser.add_argument("--human-point-scale", type=float, default=0.08)
    parser.add_argument("--publish-aorta", action="store_true",
                        help="Publish qpos[7:] as Aorta LowCmd to locomotion.")
    parser.add_argument("--aorta-topic", default=DEFAULT_AORTA_TOPIC)
    parser.add_argument("--aorta-group", default=os.environ.get("AORTA_GROUP", "default"))
    parser.add_argument("--aorta-schema-module", default="",
                        help="Override low_cmd schema-meta module name.")
    parser.add_argument("--aorta-node-name", default="gmr_pico_xrobot_lowcmd_pub")
    parser.add_argument("--motor-mode", type=int, default=2,
                        help="LowCmd MotorCmd.mode, default 2=RL_REALTIME.")
    parser.add_argument("--lowcmd-layout", default="extended44",
                        choices=sorted(LOWCMD_LAYOUTS),
                        help="LowCmd slot layout; extended44 reserves wrist/finger/head slots.")
    parser.add_argument("--aorta-wait-ms", type=int, default=500,
                        help="Wait for a matching Aorta subscriber before streaming.")
    parser.add_argument("--src-human", choices=["pico_xrobot", "xrobot"],
                        default="xrobot",
                        help="Retarget source semantics. xrobot uses calibrated raw XRobot orientation offsets.")
    args = parser.parse_args()

    print("[live-xrobot] initializing XRobotStreamer")
    streamer = XRobotStreamer()
    print(f"[live-xrobot] initializing GMR {args.src_human} -> vt_human_v2")
    retargeter = GMR(
        src_human=args.src_human,
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
    )
    print("[live-xrobot] opening MuJoCo viewer")
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.fps,
        transparent_robot=0,
    )

    record_file = None
    if args.record_path:
        record_path = Path(args.record_path).expanduser()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_file = record_path.open("w", buffering=1)
        print(f"[live-xrobot] recording {record_path}")

    aorta_pub = None
    if args.publish_aorta:
        print(f"[live-xrobot] initializing Aorta LowCmd publisher topic={args.aorta_topic}")
        aorta_pub = AortaLowCmdPublisher(
            topic=args.aorta_topic,
            group=args.aorta_group,
            schema_module=args.aorta_schema_module or None,
            node_name=args.aorta_node_name,
            motor_mode=args.motor_mode,
            layout=args.lowcmd_layout,
            wait_for_matching_ms=args.aorta_wait_ms,
        )

    frames = 0
    no_body = 0
    prev_pelvis_quat = None
    prev_arm_normals = None
    last_log = time.time()

    try:
        while viewer.viewer.is_running():
            body_data, controller, headset = streamer.get_current_frame()
            now = time.time()
            if body_data is None:
                no_body += 1
                if now - last_log >= 2.0:
                    ctrl_ts = controller.get("timestamp") if isinstance(controller, dict) else None
                    print(
                        f"[live-xrobot] waiting body_data no_body={no_body} "
                        f"controller_ts={ctrl_ts} headset={headset is not None}",
                        flush=True,
                    )
                    last_log = now
                time.sleep(0.02)
                continue

            if args.src_human == "xrobot":
                frame = body_data
            else:
                frame, prev_pelvis_quat, prev_arm_normals = body_data_to_pico_xrobot_frame(
                    body_data,
                    prev_pelvis_quat,
                    prev_arm_normals,
                )
            qpos = retargeter.retarget(frame, offset_to_ground=False)
            frames += 1

            if aorta_pub is not None:
                aorta_pub.publish_qpos(qpos)

            if record_file is not None:
                record_file.write(
                    json.dumps(
                        {
                            "wall_time_ns": time.time_ns(),
                            "robot": args.robot,
                            "actual_human_height": args.actual_human_height,
                            "body_data": _jsonable_body_data(body_data),
                            "frame_mode": "position",
                            "qpos_pico_xrobot_wxyz": qpos.tolist(),
                        }
                    )
                    + "\n"
                )

            if frames == 1 or now - last_log >= 2.0:
                dof = qpos[7:]
                print(
                    f"[live-xrobot] frames={frames} root={np.round(qpos[:3], 4).tolist()} "
                    f"dof_minmax=({float(np.min(dof)):.3f},{float(np.max(dof)):.3f}) "
                    f"nan={int(np.isnan(qpos).sum())}",
                    flush=True,
                )
                last_log = now

            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data if args.show_human else None,
                human_point_scale=args.human_point_scale,
                rate_limit=True,
                follow_camera=True,
            )
    finally:
        if record_file is not None:
            print(f"[live-xrobot] closed recording after frames={frames}")
            record_file.close()
        if aorta_pub is not None:
            aorta_pub.close()
        viewer.close()


if __name__ == "__main__":
    main()
