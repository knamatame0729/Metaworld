"""Base classes for Panda robot environments."""

from __future__ import annotations

import copy
from functools import cached_property
from typing import Any, Callable, Literal

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Space
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias

from metaworld.types import XYZ, EnvironmentStateDict, ObservationDict, Task
from metaworld.utils import reward_utils

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"


class PandaMocapBase(mjenv_gym):
    """Base class for Panda Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 80,
    }

    @cached_property
    def panda_observation_space(self) -> Space:
        raise NotImplementedError

    def __init__(
        self,
        model_name: str,
        frame_skip: int = 5,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        mjenv_gym.__init__(
            self,
            model_name,
            frame_skip=frame_skip,
            observation_space=self.panda_observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

    def get_endeff_pos(self) -> npt.NDArray[Any]:
        """Returns the position of the end effector (hand)."""
        return self.data.body("hand").xpos

    @property
    def tcp_center(self) -> npt.NDArray[Any]:
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """
        # Panda uses left_finger and right_finger body names
        right_finger_pos = self.data.body("right_finger").xpos
        left_finger_pos = self.data.body("left_finger").xpos
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def get_env_state(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get the environment state.

        Returns:
            A tuple of (qpos, qvel).
        """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return copy.deepcopy((qpos, qvel))

    def set_env_state(
        self, state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> None:
        """Set the environment state.

        Args:
            state: A tuple of (qpos, qvel).
        """
        mocap_pos, mocap_quat = state
        self.set_state(mocap_pos, mocap_quat)

    def __getstate__(self) -> EnvironmentStateDict:
        state = self.__dict__.copy()
        return {"state": state, "mjb": self.model_name, "mocap": self.get_env_state()}

    def __setstate__(self, state: EnvironmentStateDict) -> None:
        self.__dict__ = state["state"]
        mjenv_gym.__init__(
            self,
            state["mjb"],
            frame_skip=self.frame_skip,
            observation_space=self.panda_observation_space,
        )
        self.set_env_state(state["mocap"])

    def reset_mocap_welds(self) -> None:
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                    )


class PandaXYZEnv(PandaMocapBase, EzPickle):
    """The base environment for Panda robot envs that use mocap for XYZ control."""

    _HAND_SPACE = Box(
        np.array([-0.525, 0.348, -0.0525]),
        np.array([+0.525, 1.025, 0.7]),
        dtype=np.float64,
    )

    max_path_length: int = 500
    TARGET_RADIUS: float = 0.05

    class _Decorators:
        @classmethod
        def assert_task_is_set(cls, func: Callable) -> Callable:
            def inner(*args, **kwargs) -> Any:
                env = args[0]
                if not env._set_task_called:
                    raise RuntimeError(
                        "You must call env.set_task before using env." + func.__name__
                    )
                return func(*args, **kwargs)
            return inner

    def __init__(
        self,
        frame_skip: int = 5,
        hand_low: XYZ = (-0.2, 0.55, 0.05),
        hand_high: XYZ = (0.2, 0.75, 0.3),
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        action_scale: float = 1.0 / 100,
        action_rot_scale: float = 1.0,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_name: str | None = None,
        reward_function_version: str | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.num_resets: int = 0
        self.current_seed: int | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        self.width = width
        self.height = height

        self.discrete_goal_space: Box | None = None
        self.discrete_goals: list = []
        self.active_discrete_goal: int | None = None

        self._partially_observable: bool = True

        self.task_name = self.__class__.__name__

        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )

        mujoco.mj_forward(self.model, self.data)

        self._did_see_sim_exception: bool = False
        
        # Panda uses left_finger/right_finger instead of leftpad/rightpad
        self.init_left_pad: npt.NDArray[Any] = self.get_body_com("left_finger")
        self.init_right_pad: npt.NDArray[Any] = self.get_body_com("right_finger")

        # Panda action space: 3D position + 1D gripper
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float32,
        )
        self._obs_obj_max_len: int = 14
        self._set_task_called: bool = False
        self.hand_init_pos: npt.NDArray[Any] | None = None
        self._target_pos: npt.NDArray[Any] | None = None
        self._random_reset_space: Box | None = None
        self.goal_space: Box | None = None
        self._last_stable_obs: npt.NDArray[np.float64] | None = None

        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = self._get_curr_obs_combined_no_goal()

        self.task_name = self.__class__.__name__

        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
        )

    def seed(self, seed: int) -> list[int]:
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        assert self.goal_space
        self.goal_space.seed(seed)
        return [seed]

    @staticmethod
    def _set_task_inner() -> None:
        pass

    def set_task(self, task: Task) -> None:
        self._set_task_called = True
        data = pickle.loads(task.data) if isinstance(task.data, bytes) else task.data
        assert isinstance(data, dict)
        self._last_rand_vec = data["rand_vec"]
        self._freeze_rand_vec = True
        self._last_stable_obs = data.get("_last_stable_obs", None)

    @cached_property
    def panda_observation_space(self) -> Box:
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        goal_low = np.zeros(3, dtype=np.float64) + np.array(
            [-0.1, 0.85, 0.0], dtype=np.float64
        )
        goal_high = np.zeros(3, dtype=np.float64) + np.array(
            [0.1, 0.9 + 1e-7, 0.4], dtype=np.float64
        )
        gripper_low = -1.0
        gripper_high = +1.0
        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        raise NotImplementedError

    def _set_pos_site(self, name: str, pos: npt.NDArray[Any]) -> None:
        self.data.site(name).xpos = pos

    def _get_pos_site(self, name: str) -> npt.NDArray[Any]:
        return self.data.site(name).xpos.copy()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flatten().copy()
        qvel = self.data.qvel.flatten().copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_state_rand_vec(self) -> npt.NDArray[Any]:
        if self._freeze_rand_vec and self._last_rand_vec is not None:
            return self._last_rand_vec
        elif self.seeded_rand_vec:
            rand_vec = self.np_random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            )
            self._last_rand_vec = rand_vec
            return rand_vec
        else:
            rand_vec = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            )
            self._last_rand_vec = rand_vec
            return rand_vec

    def _get_site_pos(self, siteName: str) -> npt.NDArray[Any]:
        return self.data.site(siteName).xpos.copy()

    def _set_pos_ctrl(self, pos: npt.NDArray[Any]) -> None:
        self.data.mocap_pos[0] = pos

    def _set_gripper_ctrl(self, gripper_ctrl: float) -> None:
        # Panda gripper control - single value for both fingers
        self.data.ctrl[0] = gripper_ctrl

    def _reset_hand(self, steps: int = 50) -> None:
        assert self.hand_init_pos is not None
        for _ in range(steps):
            self._set_pos_ctrl(self.hand_init_pos)
            self._set_gripper_ctrl(0.0)
            mujoco.mj_step(self.model, self.data)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict]:
        self.curr_path_length = 0
        self._did_see_sim_exception = False
        return super().reset(seed=seed, options=options)

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:
        """Get current observation without goal."""
        pos_hand = self.get_endeff_pos()

        # Use left_finger and right_finger for Panda
        finger_right = self.data.body("right_finger")
        finger_left = self.data.body("left_finger")

        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))

    def _get_obs(self) -> npt.NDArray[np.float64]:
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_obs_dict(self) -> ObservationDict:
        obs = self._get_obs()
        return ObservationDict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[4:7],
        )

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        raise NotImplementedError

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        raise NotImplementedError

    def _get_pos_goal(self) -> npt.NDArray[Any]:
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    @property
    def touching_main_object(self) -> bool:
        return self.touching_object(self._get_id_main_object())

    def touching_object(self, object_geom_id: int) -> bool:
        """Check if gripper is touching the object."""
        # Get finger collision geoms - we'll use any collision with fingers
        contacts_with_object = []
        for contact in self.data.contact:
            if object_geom_id in (contact.geom1, contact.geom2):
                contacts_with_object.append(contact)
        return len(contacts_with_object) > 0

    def _get_id_main_object(self) -> int:
        return self.data.geom("objGeom").id

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        assert len(action) == 4, f"Actions should be 4D, got {len(action)}"

        self.curr_path_length += 1

        try:
            # XYZ position control
            pos_delta = action[:3] * self.action_scale
            new_mocap_pos = self.data.mocap_pos[0] + pos_delta
            new_mocap_pos = np.clip(new_mocap_pos, self.mocap_low, self.mocap_high)
            self._set_pos_ctrl(new_mocap_pos)

            # Gripper control (action[3])
            gripper_ctrl = action[3]
            # Map from [-1, 1] to gripper control range [0, 255]
            gripper_ctrl = (gripper_ctrl + 1) / 2 * 255
            self._set_gripper_ctrl(gripper_ctrl)

            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

        except mujoco.MujocoException as err:
            print(f"MuJoCo simulation error: {err}")
            self._did_see_sim_exception = True

        obs = self._get_obs()

        reward, info = self.evaluate_state(obs, action)
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[Any],
        obj_pos: npt.NDArray[Any],
        object_reach_radius: float = 0.01,
        obj_radius: float = 0.015,
        pad_success_thresh: float = 0.05,
        xz_thresh: float = 0.005,
        desired_gripper_effort: float = 0.7,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        """Reward for caging the object with gripper."""
        pad_success_margin = pad_success_thresh - obj_radius
        object_reach_margin = object_reach_radius - obj_radius

        x_z_success_margin = xz_thresh - obj_radius
        tcp = self.tcp_center

        # Use left_finger and right_finger for Panda
        left_pad = self.get_body_com("left_finger")
        right_pad = self.get_body_com("right_finger")
        
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, x_z_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, x_z_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_xz = obj_pos + np.array([0.0, -obj_pos[1], 0.0])
        x_z_margin = np.linalg.norm(self.hand_init_pos - obj_pos) + object_reach_margin
        x_z_caging = reward_utils.tolerance(
            float(np.linalg.norm(tcp_xz - obj_xz)),
            bounds=(0, x_z_success_margin),
            margin=x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = min(max(0, action[-1]), 1)
        assert y_caging >= 0 and y_caging <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if high_density:
            caging_and_gripping = reward_utils.hamacher_product(caging, gripper_closed)
            caging_and_gripping = (caging_and_gripping + caging) / 2
            return caging_and_gripping
        elif medium_density:
            return 0.5 * caging + 0.5 * gripper_closed
        else:
            return caging


# Import pickle for set_task
import pickle
from typing import SupportsFloat
