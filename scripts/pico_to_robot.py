"""Retarget a Pico XR-Toolkit body-tracking recording to a robot.

Pipeline:  xrobot_*.jsonl  ->  utils.pico_xrt.load_pico_xrt_file
           ->  GMR(src_human="pico_xrt")  ->  (optional) sequence postprocess
           ->  viewer / video / saved motion pickle.

Example:
    python scripts/pico_to_robot.py \
        --pico_file /path/xrobot_vt_human_v2_20260707_213441.jsonl \
        --robot vt_human_v2 --tgt_fps 30 \
        --save_path out/pico_take.pkl --record_video --video_path out/pico_take.mp4
"""
import argparse
import copy
import os
import pathlib
import time

import numpy as np
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.pico_xrt import load_pico_xrt_file
from general_motion_retargeting.reference_postprocess import (
    has_sequence_postprocess,
    postprocess_qpos_sequence,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pico_file", type=str, required=True,
                        help="Pico XRT recording (xrobot_*.jsonl).")
    parser.add_argument("--robot", choices=["vt_human_v2"], default="vt_human_v2")
    parser.add_argument("--tgt_fps", type=int, default=30,
                        help="Resample the capture to this fps (0 = keep native).")
    parser.add_argument("--save_path", default=None, help="Output motion .pkl.")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--no_postprocess", action="store_true", default=False,
                        help="Disable the sequence-level butterworth / ground-snap.")
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--show_human_labels", action="store_true", default=False)
    parser.add_argument("--human_point_scale", type=float, default=0.08)
    args = parser.parse_args()

    tgt_fps = None if args.tgt_fps in (0, None) else args.tgt_fps
    frames, human_height, fps = load_pico_xrt_file(args.pico_file, tgt_fps=tgt_fps)
    print(f"[pico] loaded {len(frames)} frames @ {fps:.2f} fps, human_height={human_height:.2f} m")

    retarget = GMR(
        actual_human_height=human_height,
        src_human="pico_xrt",
        tgt_robot=args.robot,
    )

    # Retarget the full sequence (stateful: prev-posture smoothing / first frame).
    qpos_frames = []
    human_frames = []
    for frame in frames:
        qpos = retarget.retarget(frame, offset_to_ground=False)
        qpos_frames.append(qpos.copy())
        human_frames.append(copy.deepcopy(retarget.scaled_human_data))

    use_pp = has_sequence_postprocess(args.robot) and not args.no_postprocess
    if use_pp:
        print(f"[pico] sequence postprocess (butterworth + ground snap)...")
        qpos_frames = postprocess_qpos_sequence(retarget, qpos_frames, fps)
    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        import pickle
        motion_data = {
            "fps": fps,
            "root_pos": qpos_frames[:, :3],
            "root_rot": qpos_frames[:, 3:7][:, [1, 2, 3, 0]],  # wxyz -> xyzw
            "dof_pos": qpos_frames[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[pico] saved motion to {args.save_path}")

    if args.record_video or not args.save_path:
        default_video = f"videos/{args.robot}_{pathlib.Path(args.pico_file).stem}.mp4"
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path or default_video,
        )
        i = 0
        while True:
            if args.loop:
                i = (i + 1) % len(qpos_frames)
            else:
                i += 1
                if i >= len(qpos_frames):
                    break
            q = qpos_frames[i]
            viewer.step(
                root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:],
                human_motion_data=human_frames[i],
                show_human_body_name=args.show_human_labels,
                human_point_scale=args.human_point_scale,
                rate_limit=args.rate_limit,
            )
        viewer.close()


if __name__ == "__main__":
    main()
