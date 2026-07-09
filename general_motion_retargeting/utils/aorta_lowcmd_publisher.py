"""Aorta LowCmd publisher for VT_HUMAN_V2 teleop streams.

The live Pico retargeting path produces a MuJoCo ``qpos``. For ``vt_human_v2``,
``qpos[7:]`` is the current 22-DOF joint vector. The default Aorta LowCmd layout
publishes a 44-slot vector so wrist/finger/head/B-sample joints have stable
reserved indices; missing retarget outputs are filled with zero.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Iterable

import numpy as np


DEFAULT_TOPIC = "/locomotion/external_command"
DEFAULT_SCHEMA_MODULE_CANDIDATES = (
    "low_cmd_schema_meta",
    "aorta_msgs.low_cmd_schema_meta",
    "aorta_msgs.lowlevel.low_cmd_schema_meta",
    "lowlevel.low_cmd_schema_meta",
)
VT_HUMAN_V2_RETARGET_JOINT_ORDER = (
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
)
VT_HUMAN_V2_EXTENDED_LOWCMD_JOINT_ORDER = (
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",  # 1
    "left_hip_yaw_joint",  # 2
    "left_knee_joint",  # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",  # 7
    "right_hip_yaw_joint",  # 8
    "right_knee_joint",  # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",  # 12
    "waist_pitch_joint",  # 13
    "left_shoulder_pitch_joint",  # 14
    "left_shoulder_roll_joint",  # 15
    "left_shoulder_yaw_joint",  # 16
    "left_elbow_joint",  # 17
    "right_shoulder_pitch_joint",  # 18
    "right_shoulder_roll_joint",  # 19
    "right_shoulder_yaw_joint",  # 20
    "right_elbow_joint",  # 21
    "left_wrist_roll_joint",  # 22
    "left_wrist_pitch_joint",  # 23
    "left_wrist_yaw_joint",  # 24
    "left_thumb_pitch_joint",  # 25
    "left_thumb_roll_joint",  # 26
    "left_index_finger_pitch_joint",  # 27
    "left_middle_finger_pitch_joint",  # 28
    "left_ring_finger_pitch_joint",  # 29
    "left_little_finger_pitch_joint",  # 30
    "right_wrist_roll_joint",  # 31
    "right_wrist_pitch_joint",  # 32
    "right_wrist_yaw_joint",  # 33
    "right_thumb_pitch_joint",  # 34
    "right_thumb_roll_joint",  # 35
    "right_index_finger_pitch_joint",  # 36
    "right_middle_finger_pitch_joint",  # 37
    "right_ring_finger_pitch_joint",  # 38
    "right_little_finger_pitch_joint",  # 39
    "head_pitch_joint",  # 40
    "head_yaw_joint",  # 41
    "head_roll_joint",  # 42, B-sample slot
    "waist_roll_joint",  # 43, B-sample slot
)
VT_HUMAN_V2_LOWCMD_JOINT_ORDER = VT_HUMAN_V2_EXTENDED_LOWCMD_JOINT_ORDER
LOWCMD_LAYOUTS = {
    "extended44": VT_HUMAN_V2_EXTENDED_LOWCMD_JOINT_ORDER,
    "legacy22": VT_HUMAN_V2_RETARGET_JOINT_ORDER,
}
RETARGET_INDEX_BY_NAME = {
    name: idx for idx, name in enumerate(VT_HUMAN_V2_RETARGET_JOINT_ORDER)
}


def _load_schema_module(schema_module: str | None):
    candidates = (schema_module,) if schema_module else DEFAULT_SCHEMA_MODULE_CANDIDATES
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "Could not import low_cmd schema metadata module. Install aorta_msgs "
        "or pass --aorta-schema-module. Tried: " + "; ".join(errors)
    )


def _resolve_attr(module, name: str):
    if hasattr(module, name):
        return getattr(module, name)
    low_cmd = getattr(module, "LowCmd", None)
    if low_cmd is not None and hasattr(low_cmd, name):
        return getattr(low_cmd, name)
    root_type = getattr(module, "ROOT_TYPE_NAME", "")
    namespace = root_type.rsplit(".", 1)[0] if "." in root_type else ""
    if namespace and name == "CreateMotorCmd":
        motor_cmd_module = importlib.import_module(f"{namespace}.MotorCmd")
        if hasattr(motor_cmd_module, name):
            return getattr(motor_cmd_module, name)
    raise AttributeError(f"{module.__name__} does not expose {name}")


def retarget_dof_to_lowcmd_slots(
    dof_pos: Iterable[float],
    *,
    layout: str = "extended44",
) -> np.ndarray:
    """Map current VT_HUMAN_V2 retarget DOFs into a LowCmd slot layout."""
    if layout not in LOWCMD_LAYOUTS:
        raise ValueError(
            f"Unknown LowCmd layout {layout!r}; choose one of "
            f"{sorted(LOWCMD_LAYOUTS)}"
        )
    lowcmd_joint_order = LOWCMD_LAYOUTS[layout]
    dof_q = np.asarray(dof_pos, dtype=np.float32).reshape(-1)
    if dof_q.shape[0] < len(VT_HUMAN_V2_RETARGET_JOINT_ORDER):
        raise ValueError(
            f"Expected at least {len(VT_HUMAN_V2_RETARGET_JOINT_ORDER)} "
            f"VT_HUMAN_V2 retarget joints, got {dof_q.shape[0]}"
        )
    dof_q = dof_q[: len(VT_HUMAN_V2_RETARGET_JOINT_ORDER)]
    if not np.all(np.isfinite(dof_q)):
        raise ValueError("Cannot publish LowCmd with NaN/Inf joint targets")

    lowcmd_q = np.zeros(len(lowcmd_joint_order), dtype=np.float32)
    for dst_idx, name in enumerate(lowcmd_joint_order):
        src_idx = RETARGET_INDEX_BY_NAME.get(name)
        if src_idx is not None:
            lowcmd_q[dst_idx] = dof_q[src_idx]
    return lowcmd_q


class AortaLowCmdPublisher:
    """Publish VT_HUMAN_V2 joint targets as Aorta LowCmd messages."""

    def __init__(
        self,
        *,
        topic: str = DEFAULT_TOPIC,
        node_name: str = "gmr_pico_lowcmd_pub",
        group: str = "default",
        schema_module: str | None = None,
        motor_mode: int = 2,
        layout: str = "extended44",
        wait_for_matching_ms: int = 500,
        initial_size: int = 4096,
    ) -> None:
        os.environ["AORTA_GROUP"] = group
        import aorta

        self._aorta = aorta
        self._schema = _load_schema_module(schema_module)
        self._topic = topic
        self._motor_mode = int(motor_mode)
        self._initial_size = int(initial_size)
        if layout not in LOWCMD_LAYOUTS:
            raise ValueError(
                f"Unknown LowCmd layout {layout!r}; choose one of "
                f"{sorted(LOWCMD_LAYOUTS)}"
            )
        self._layout = layout
        self._lowcmd_joint_order = LOWCMD_LAYOUTS[layout]

        self._create_motor_cmd = _resolve_attr(self._schema, "CreateMotorCmd")
        self._start_motor_cmd_vector = _resolve_attr(
            self._schema, "LowCmdStartMotorCmdVector"
        )
        self._start = _resolve_attr(self._schema, "LowCmdStart")
        self._add_header = _resolve_attr(self._schema, "LowCmdAddAortaHeader")
        self._add_motor_cmd = _resolve_attr(self._schema, "LowCmdAddMotorCmd")
        self._end = _resolve_attr(self._schema, "LowCmdEnd")

        self._node = aorta.Node(node_name, group=group)
        self._pub = self._node.create_publisher_typed(
            self._schema,
            topic,
            qos=aorta.QoS.realtime_control(),
        )
        if wait_for_matching_ms > 0:
            self._pub.wait_for_matching(int(wait_for_matching_ms))
        # Give Zenoh discovery a small grace period even if no subscriber is up
        # yet; the stream is continuous, so missing the first frame is fine.
        time.sleep(0.05)

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def lowcmd_joint_order(self) -> tuple[str, ...]:
        return self._lowcmd_joint_order

    def publish_dof(self, dof_pos: Iterable[float]) -> None:
        lowcmd_q = retarget_dof_to_lowcmd_slots(dof_pos, layout=self._layout)
        self.publish_lowcmd_positions(lowcmd_q)

    def publish_lowcmd_positions(self, lowcmd_pos: Iterable[float]) -> None:
        q = np.asarray(lowcmd_pos, dtype=np.float32).reshape(-1)
        if q.shape[0] != len(self._lowcmd_joint_order):
            raise ValueError(
                f"Expected {len(self._lowcmd_joint_order)} LowCmd slots for "
                f"{self._layout}, got {q.shape[0]}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("Cannot publish LowCmd with NaN/Inf joint targets")
        mode = self._motor_mode

        def fill(builder, header_off):
            self._start_motor_cmd_vector(builder, len(q))
            for value in reversed(q):
                self._create_motor_cmd(
                    builder,
                    mode,
                    float(value),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            motor_cmd_off = builder.EndVector()

            self._start(builder)
            self._add_header(builder, header_off)
            self._add_motor_cmd(builder, motor_cmd_off)
            return self._end(builder)

        self._pub.publish_typed(fill, initial_size=self._initial_size)

    def publish_qpos(self, qpos: Iterable[float]) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos_arr.shape[0] < 7 + len(VT_HUMAN_V2_RETARGET_JOINT_ORDER):
            raise ValueError(
                "Expected qpos with floating-base prefix plus 22 VT_HUMAN_V2 DOFs"
            )
        self.publish_dof(qpos_arr[7:])

    def close(self) -> None:
        self._node.close()

    def __enter__(self) -> "AortaLowCmdPublisher":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
