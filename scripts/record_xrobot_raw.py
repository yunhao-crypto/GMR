"""Record raw XRoboToolkit SDK streams for offline retargeting comparison.

This intentionally logs the data before GMR-specific conversion, so the same
take can later be replayed through both the current Pico loader and a
TWIST2/XRobotStreamer-compatible loader.
"""

import argparse
import json
import os
import time

try:
    import xrobotoolkit_sdk as xrt
except ImportError as exc:  # pragma: no cover - only available on teleop machine.
    xrt = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _tolist(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_tolist(v) for v in value]
    if isinstance(value, dict):
        return {k: _tolist(v) for k, v in value.items()}
    return value


def _safe_call(name, default=None):
    fn = getattr(xrt, name, None)
    if fn is None:
        return default
    try:
        return _tolist(fn())
    except Exception as exc:
        return {"error": str(exc)}


def read_frame(actual_human_height):
    body_available = bool(_safe_call("is_body_data_available", False))
    rec = {
        "wall_time_ns": time.time_ns(),
        "actual_human_height": actual_human_height,
        "body_available": body_available,
        "body_poses": None,
        "body_velocities": None,
        "body_accelerations": None,
        "body_joint_timestamps": None,
        "body_timestamp_ns": None,
        "headset_pose": _safe_call("get_headset_pose"),
        "left_controller_pose": _safe_call("get_left_controller_pose"),
        "right_controller_pose": _safe_call("get_right_controller_pose"),
        "controller_inputs": {
            "left_trigger": _safe_call("get_left_trigger", 0.0),
            "right_trigger": _safe_call("get_right_trigger", 0.0),
            "left_grip": _safe_call("get_left_grip", 0.0),
            "right_grip": _safe_call("get_right_grip", 0.0),
            "A": _safe_call("get_A_button", False),
            "B": _safe_call("get_B_button", False),
            "X": _safe_call("get_X_button", False),
            "Y": _safe_call("get_Y_button", False),
            "left_axis": _safe_call("get_left_axis"),
            "right_axis": _safe_call("get_right_axis"),
            "left_axis_click": _safe_call("get_left_axis_click", False),
            "right_axis_click": _safe_call("get_right_axis_click", False),
            "timestamp_ns": _safe_call("get_time_stamp_ns"),
        },
    }

    if body_available:
        rec.update(
            {
                "body_poses": _safe_call("get_body_joints_pose"),
                "body_velocities": _safe_call("get_body_joints_velocity"),
                "body_accelerations": _safe_call("get_body_joints_acceleration"),
                "body_joint_timestamps": _safe_call("get_body_joints_timestamp"),
                "body_timestamp_ns": _safe_call("get_body_timestamp_ns"),
            }
        )

    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output raw jsonl path.")
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl+C.")
    args = parser.parse_args()

    if xrt is None:
        raise SystemExit(
            "xrobotoolkit_sdk is not available in this environment: "
            f"{_IMPORT_ERROR}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    xrt.init()

    dt = 1.0 / args.fps if args.fps > 0 else 0.0
    deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
    count = 0
    print(f"[xrobot_raw] writing {args.out}; Ctrl+C to stop")

    with open(args.out, "w") as f:
        try:
            while deadline is None or time.monotonic() < deadline:
                t0 = time.monotonic()
                f.write(json.dumps(read_frame(args.actual_human_height)) + "\n")
                count += 1
                sleep_s = dt - (time.monotonic() - t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            pass

    print(f"[xrobot_raw] wrote {count} frames to {args.out}")


if __name__ == "__main__":
    main()
