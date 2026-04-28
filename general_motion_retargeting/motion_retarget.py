
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from ._geom_utils import collect_geom_ids_by_token, geom_min_z
from rich import print


_WAIST_REGULARIZER_SPECS = {
    "vt_human": {"joints": ("waist_yaw_joint", "waist_pitch_joint"), "cost": 20.0},
}

_PREV_POSTURE_SMOOTHING_SPECS = {
    "vt_human": {
        "default_cost": 1e-3,
        "joint_costs": {
            "waist_yaw_joint": 1e-1,
            "waist_pitch_joint": 1e-1,
            "left_shoulder_pitch_joint": 5e-2,
            "left_shoulder_roll_joint": 5e-2,
            "left_shoulder_yaw_joint": 5e-2,
            "left_elbow_joint": 5e-2,
            "right_shoulder_pitch_joint": 5e-2,
            "right_shoulder_roll_joint": 5e-2,
            "right_shoulder_yaw_joint": 5e-2,
            "right_elbow_joint": 5e-2,
        },
    },
}

_GROUND_SNAP_SPECS = {
    "vt_human": {"geom_name_token": "foot", "clearance": 0.0},
}

_QPOS_SMOOTHING_SPECS = {
    "vt_human": {
        "alpha": 0.35,
        "joints": (
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
    },
}


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:
        self.tgt_robot = tgt_robot

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0

        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping
        self.first_frame_damping = max(float(damping), 2.0)
        self.first_frame_max_iter = max(int(self.max_iter), 10)
        self._is_first_frame = True

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self._arm_task_original_orientation_costs = {}
        self._first_frame_arm_orientation_cost = 1.0

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        self.setup_retarget_configuration()

        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self._default_qpos = self.configuration.data.qpos.copy()
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
        self.posture_task.set_target(self._default_qpos)
        self.prev_posture_task = mink.PostureTask(
            self.model,
            cost=self._build_prev_posture_cost(),
        )
        self.prev_posture_task.set_target(self._default_qpos)
        self.waist_posture_task = self._build_waist_regularizer_task()
        self.ground_snap_geom_ids, self.ground_snap_clearance = self._build_ground_snap()
        self.qpos_smoothing_alpha, self.qpos_smoothing_indices = (
            self._build_qpos_smoothing_spec()
        )

        self.tasks1 = []
        self.tasks2 = []

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1.setdefault(body_name, []).append(task)
                self.pos_offsets1.setdefault(body_name, []).append(
                    np.array(pos_offset) - self.ground
                )
                self.rot_offsets1.setdefault(body_name, []).append(
                    R.from_quat(rot_offset, scalar_first=True)
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
                if self._is_arm_body(body_name):
                    self._arm_task_original_orientation_costs[task] = float(
                        rot_weight
                    )

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2.setdefault(body_name, []).append(task)
                self.pos_offsets2.setdefault(body_name, []).append(
                    np.array(pos_offset) - self.ground
                )
                self.rot_offsets2.setdefault(body_name, []).append(
                    R.from_quat(rot_offset, scalar_first=True)
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []
                if self._is_arm_body(body_name):
                    self._arm_task_original_orientation_costs[task] = float(
                        rot_weight
                    )

    def _build_waist_regularizer_task(self):
        """Per-DOF posture task that pins regularizer joints to default qpos."""
        spec = _WAIST_REGULARIZER_SPECS.get(self.tgt_robot)
        if spec is None:
            return None

        cost_vec = np.zeros(self.model.nv)
        missing = []
        for joint_name in spec["joints"]:
            dof_idx = self.robot_dof_names.get(joint_name)
            if dof_idx is None:
                missing.append(joint_name)
            else:
                cost_vec[dof_idx] = spec["cost"]

        if missing:
            print(
                f"[GMR] warning: waist regularizer skipped — joints not found "
                f"in {self.tgt_robot}: {missing}"
            )
            return None

        task = mink.PostureTask(self.model, cost=cost_vec)
        task.set_target(self._default_qpos)
        return task

    def _build_prev_posture_cost(self):
        spec = _PREV_POSTURE_SMOOTHING_SPECS.get(self.tgt_robot)
        if spec is None:
            return 1e-3

        cost_vec = np.full(self.model.nv, float(spec["default_cost"]))
        missing = []
        for joint_name, joint_cost in spec["joint_costs"].items():
            dof_idx = self.robot_dof_names.get(joint_name)
            if dof_idx is None:
                missing.append(joint_name)
            else:
                cost_vec[dof_idx] = float(joint_cost)
        if missing:
            print(
                f"[GMR] warning: prev_posture per-joint cost ignored — joints "
                f"not found in {self.tgt_robot}: {missing}"
            )
        return cost_vec

    def _build_ground_snap(self):
        """Returns (geom_ids, clearance) for foot-snapping; ([], 0.0) if disabled."""
        spec = _GROUND_SNAP_SPECS.get(self.tgt_robot)
        if spec is None:
            return [], 0.0

        token = spec["geom_name_token"]
        geom_ids = collect_geom_ids_by_token(self.model, token)
        if not geom_ids:
            print(
                f"[GMR] warning: ground snap disabled — no geom name contains "
                f"'{token}' in {self.tgt_robot}"
            )
        return geom_ids, float(spec["clearance"])

    def _build_qpos_smoothing_spec(self):
        spec = _QPOS_SMOOTHING_SPECS.get(self.tgt_robot)
        if spec is None:
            return None, ()

        qpos_indices = []
        missing = []
        for joint_name in spec["joints"]:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                missing.append(joint_name)
                continue
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))

        if missing:
            print(
                f"[GMR] warning: qpos smoothing skipped — joints not found "
                f"in {self.tgt_robot}: {missing}"
            )
            return None, ()
        return float(spec["alpha"]), tuple(qpos_indices)

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            self._update_table_targets(
                self.human_body_to_task1,
                self.pos_offsets1,
                self.rot_offsets1,
                human_data,
            )
        if self.use_ik_match_table2:
            self._update_table_targets(
                self.human_body_to_task2,
                self.pos_offsets2,
                self.rot_offsets2,
                human_data,
            )

    @staticmethod
    def _update_table_targets(body_to_tasks, pos_offsets, rot_offsets, human_data):
        for body_name, tasks in body_to_tasks.items():
            pos, rot = human_data[body_name]
            for task, pos_offset, rot_offset in zip(
                tasks, pos_offsets[body_name], rot_offsets[body_name]
            ):
                updated_quat = (
                    R.from_quat(rot, scalar_first=True) * rot_offset
                ).as_quat(scalar_first=True)
                global_pos_offset = R.from_quat(
                    updated_quat, scalar_first=True
                ).apply(pos_offset)
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(updated_quat), pos + global_pos_offset
                    )
                )


    @staticmethod
    def _is_arm_body(body_name):
        return any(
            token in body_name
            for token in (
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
            )
        )

    def _set_first_frame_arm_task_costs(self, enabled):
        for task, original_orientation_cost in (
            self._arm_task_original_orientation_costs.items()
        ):
            if enabled and original_orientation_cost > 0.0:
                orientation_cost = self._first_frame_arm_orientation_cost
            else:
                orientation_cost = original_orientation_cost
            task.set_orientation_cost(orientation_cost)

    def _apply_ground_snap(self):
        if not self.ground_snap_geom_ids:
            return
        mj.mj_forward(self.model, self.configuration.data)
        min_z = min(
            geom_min_z(self.model, self.configuration.data, gid)
            for gid in self.ground_snap_geom_ids
        )
        if min_z < self.ground_snap_clearance:
            self.configuration.data.qpos[2] += self.ground_snap_clearance - min_z
            mj.mj_forward(self.model, self.configuration.data)

    def _apply_qpos_smoothing(self, prev_q):
        if self.qpos_smoothing_alpha is None or not self.qpos_smoothing_indices:
            return
        alpha = self.qpos_smoothing_alpha
        for qpos_idx in self.qpos_smoothing_indices:
            current = self.configuration.data.qpos[qpos_idx]
            self.configuration.data.qpos[qpos_idx] = prev_q[qpos_idx] + alpha * (
                current - prev_q[qpos_idx]
            )
        mj.mj_forward(self.model, self.configuration.data)

    def _solve_task_group(
        self,
        tasks,
        error_fn,
        *,
        damping,
        max_iter,
        include_posture,
        include_prev_posture,
    ):
        solve_tasks = list(tasks)
        if include_posture:
            solve_tasks.append(self.posture_task)
        if include_prev_posture:
            solve_tasks.append(self.prev_posture_task)
        if self.waist_posture_task is not None:
            solve_tasks.append(self.waist_posture_task)

        curr_error = error_fn()
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration,
            solve_tasks,
            dt,
            self.solver,
            damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(vel, dt)
        next_error = error_fn()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < max_iter:
            curr_error = next_error
            dt = self.configuration.model.opt.timestep
            vel = mink.solve_ik(
                self.configuration,
                solve_tasks,
                dt,
                self.solver,
                damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel, dt)
            next_error = error_fn()
            num_iter += 1

    def retarget(
        self,
        human_data,
        offset_to_ground=False,
        apply_ground_snap=True,
        apply_qpos_smoothing=True,
    ):
        is_first_frame = self._is_first_frame
        prev_q = self.configuration.data.qpos.copy()
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)
        include_posture = is_first_frame
        include_prev_posture = True
        solve_damping = (
            self.first_frame_damping if is_first_frame else self.damping
        )
        solve_max_iter = (
            self.first_frame_max_iter if is_first_frame else self.max_iter
        )
        self.prev_posture_task.set_target(prev_q)
        if is_first_frame:
            self._set_first_frame_arm_task_costs(True)

        if self.use_ik_match_table1:
            self._solve_task_group(
                self.tasks1,
                self.error1,
                damping=solve_damping,
                max_iter=solve_max_iter,
                include_posture=include_posture,
                include_prev_posture=include_prev_posture,
            )

        if self.use_ik_match_table2:
            self._solve_task_group(
                self.tasks2,
                self.error2,
                damping=solve_damping,
                max_iter=solve_max_iter,
                include_posture=include_posture,
                include_prev_posture=include_prev_posture,
            )

        if is_first_frame:
            self._set_first_frame_arm_task_costs(False)
        self._is_first_frame = False
        if apply_ground_snap:
            self._apply_ground_snap()
        # Skip EMA on the very first frame: prev_q is the home pose, so EMA
        # would pull the initial pose toward T-pose for one rendered frame.
        if apply_qpos_smoothing and not is_first_frame:
            self._apply_qpos_smoothing(prev_q)
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name] = [pos - np.array([0, 0, self.ground_offset]), quat]
        return human_data
