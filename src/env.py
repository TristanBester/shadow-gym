import gymnasium as gym
import mujoco
import numpy as np
from omegaconf import DictConfig

from src.render import ShadowRenderer


class ShadowEnv(gym.Env):
    # TODO: Fix the arguments that are passed in to make sense
    def __init__(self, config: DictConfig):
        self.scene_path = config.scene_path
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # TODO: Name these better
        self.fingertip_sites = config.sites.fingertip
        self.finger_sites = config.sites.finger
        self.target_sites = config.sites.target

        self.config = config

        self.renderer = ShadowRenderer(
            model=self.model,
            data=self.data,
            camera_config=config.camera,
            fingertip_sites=self.fingertip_sites,
            finger_sites=self.finger_sites,
            target_sites=self.target_sites,
            goal=config.goal,
        )

        # Reset stuff
        # TODO: Clean this up
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_simulation()

        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._apply_action(action)

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        reward = self._compute_reward()
        truncated = self.steps >= 200
        terminated = truncated

        self.steps += 1
        return obs, reward, terminated, truncated, {}

    def render(self):
        self.renderer.render()

    def close(self):
        pass

    def _get_obs(self):
        fingertip_positions = []
        for site_name in self.fingertip_sites:
            site_position = self._get_site_position(site_name)
            fingertip_positions.append(site_position)

        fingertip_positions = np.array(fingertip_positions)
        return fingertip_positions

    def _get_site_position(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id]

    def _apply_action(self, action):
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

    def _compute_reward(self):
        goal_positions = np.array([np.array(i) for i in self.config.goal.values()])
        actual_positions = self._get_obs()

        distances = np.linalg.norm(goal_positions - actual_positions)

        reward = -distances
        return reward

    def _reset_simulation(self):
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
