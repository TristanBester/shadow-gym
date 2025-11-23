import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from src.config import base_config, camera_config, robot_config, sign_config_r


class ShadowEnv(gym.Env):
    """Shadow environment."""

    MJ_STEPS_PER_ACTION = 4
    DISTANCE_WEIGHT = 10.0
    SMOOTHNESS_WEIGHT = 0.5
    VELOCITY_WEIGHT = 0.1
    ENERGY_WEIGHT = 0.1

    def __init__(
        self,
        render_mode: str = "human",
    ):
        """Initialise the environment."""
        super().__init__()
        self.render_mode = render_mode
        self.base_config = base_config
        self.camera_config = camera_config
        self.robot_config = robot_config
        self.sign_config = sign_config_r

        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(20,),
            dtype="float32",
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(48,),
            dtype="float32",
        )

        self._init_scene()
        self._init_renderer()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""
        self.steps = 0
        self._prev_action = np.zeros(self.action_space.shape)

        self._reset_simulation()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment."""
        self._apply_action(action)
        self._step_simulation()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        truncated = self.steps >= 100
        terminated = False

        self.steps += 1
        self._prev_action = action
        return obs, reward, terminated, truncated, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.renderer.render("human")
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _init_scene(self):
        """Initialise the MuJoCo model from the given scene path."""
        self.model = mujoco.MjModel.from_xml_path(self.base_config.scene_path)
        self.data = mujoco.MjData(self.model)

        # Used to reset the environment
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def _init_renderer(self):
        """Initialise the MuJoCo renderer."""
        renderer_config = {
            "distance": self.camera_config.distance,
            "azimuth": self.camera_config.azimuth,
            "elevation": self.camera_config.elevation,
            "lookat": self.camera_config.lookat,
        }
        self.renderer = MujocoRenderer(
            model=self.model,
            data=self.data,
            default_cam_config=renderer_config,
        )
        self.renderer.width = self.camera_config.width
        self.renderer.height = self.camera_config.height

    def _reset_simulation(self):
        """Reset the simulation."""
        # Reset buffers for joint states, warm-start, control buffers etc.
        mujoco.mj_resetData(self.model, self.data)

        # Restore initial simulation time and joint state from saved snapshot
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)

        # Clear actuator activations (if any) to avoid residual control input
        if self.model.na != 0:
            self.data.act[:] = None

        # Recompute all derived quantities after manual state changes
        mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, action: np.ndarray):
        """Apply the action to the environment."""
        clipped_action = np.clip(action, -1, 1)

        # Get the control ranges for each actuator in the model
        control_ranges = self.model.actuator_ctrlrange

        # Compute the half the range of actuation for each actuator
        half_actuation_ranges = (control_ranges[:, 1] - control_ranges[:, 0]) / 2.0

        # Compute the center of the actuation for each actuator
        actuation_centers = (control_ranges[:, 1] + control_ranges[:, 0]) / 2.0

        # Compute the control signal associated with the provided action
        control = actuation_centers + clipped_action * half_actuation_ranges
        self.data.ctrl[:] = control

    def _step_simulation(self):
        """Step the simulation."""
        for _ in range(self.MJ_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        """Returns all joint positions and velocities associated with a robot."""
        joint_positions = np.array(
            [
                self._get_joint_qpos(self.model, self.data, name)
                for name in self.robot_config.joint_names
            ]
        ).squeeze()
        joint_velocities = np.array(
            [
                self._get_joint_qvel(self.model, self.data, name)
                for name in self.robot_config.joint_names
            ]
        ).squeeze()
        return np.concatenate([joint_positions, joint_velocities])

    def _get_joint_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, name: str):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addr = model.jnt_qposadr[joint_id]

        # All joints are either HINGE or SLIDE
        ndim = 1
        start_idx = joint_addr
        end_idx = joint_addr + ndim
        qpos = data.qpos[start_idx:end_idx].copy()
        return qpos

    def _get_joint_qvel(self, model: mujoco.MjModel, data: mujoco.MjData, name: str):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addr = model.jnt_dofadr[joint_id]

        # All joints are either HINGE or SLIDE
        ndim = 1
        start_idx = joint_addr
        end_idx = joint_addr + ndim
        qvel = data.qvel[start_idx:end_idx].copy()
        return qvel

    def _compute_reward(self, action: np.ndarray):
        # 1. Distance from goal position

        target_values = []
        for joint_name in self.robot_config.joint_names:
            joint_name = joint_name.split(":")[1].lower()
            target_value = getattr(self.sign_config, f"joint_{joint_name}")
            target_values.append(target_value)

        actual_values = self.data.qpos[np.array(self.robot_config.joint_qpos_addrs)]

        distances = np.linalg.norm(target_values - actual_values)
        distance_reward = -1.0 * distances

        # 2. Smoothness
        if self.steps > 0:
            action_difference = np.linalg.norm(action - self._prev_action)
            smoothness_reward = -1.0 * action_difference
        else:
            smoothness_reward = 0.0

        # 3. Velocity penalty
        velocity_penalty = -1.0 * np.linalg.norm(self.data.qvel)

        # 4. Energy penalty
        energy_penalty = -1.0 * np.linalg.norm(action)

        # Total reward
        reward = (
            self.DISTANCE_WEIGHT * distance_reward
            + self.SMOOTHNESS_WEIGHT * smoothness_reward
            + self.VELOCITY_WEIGHT * velocity_penalty
            + self.ENERGY_WEIGHT * energy_penalty
        )
        return float(reward)
