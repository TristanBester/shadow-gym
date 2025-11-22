import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from src.render import ShadowRenderer


# TODO: Remove this trash
def extract_mj_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj):
    if obj_type == mujoco.mjtObj.mjOBJ_BODY:
        name_addr = model.name_bodyadr
        n_obj = model.nbody

    elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        name_addr = model.name_jntadr
        n_obj = model.njnt

    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        name_addr = model.name_geomadr
        n_obj = model.ngeom

    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        name_addr = model.name_siteadr
        n_obj = model.nsite

    elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
        name_addr = model.name_lightadr
        n_obj = model.nlight

    elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
        name_addr = model.name_camadr
        n_obj = model.ncam

    elif obj_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        name_addr = model.name_actuatoradr
        n_obj = model.nu

    elif obj_type == mujoco.mjtObj.mjOBJ_SENSOR:
        name_addr = model.name_sensoradr
        n_obj = model.nsensor

    elif obj_type == mujoco.mjtObj.mjOBJ_TENDON:
        name_addr = model.name_tendonadr
        n_obj = model.ntendon

    elif obj_type == mujoco.mjtObj.mjOBJ_MESH:
        name_addr = model.name_meshadr
        n_obj = model.nmesh
    else:
        raise ValueError(
            "`{}` was passed as the MuJoCo model object type. The MuJoCo model object type can only be of the following mjtObj enum types: {}.".format(
                obj_type, MJ_OBJ_TYPES
            )
        )

    id2name = {i: None for i in range(n_obj)}
    name2id = {}
    for addr in name_addr:
        name = model.names[addr:].split(b"\x00")[0].decode()
        if name:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

    return tuple(id2name[id] for id in sorted(name2id.values())), name2id, id2name


class ShadowEnv(gym.Env):
    # TODO: Fix the arguments that are passed in to make sense
    def __init__(self, config: DictConfig, render_mode: str = "human"):
        self.scene_path = config.scene_path
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.config = config

        # # TODO: Remove this trash and put it in config
        # self.joint_names, _, _ = extract_mj_names(self.model, mujoco.mjtObj.mjOBJ_JOINT)

        # TODO: Move this into config
        self.joint_names = []
        self.joint_qpos_addrs = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self.joint_names.append(joint_name)
                self.joint_qpos_addrs.append(self.model.jnt_qposadr[i])
            else:
                print(f"Warning: Joint {i} has no name")

        # TODO: Name these better
        self.fingertip_sites = config.sites.fingertip
        self.finger_sites = config.sites.finger
        self.target_sites = config.sites.target

        self.config = config
        self.render_mode = render_mode

        self.renderer = ShadowRenderer(
            model=self.model,
            data=self.data,
            camera_config=config.camera,
            fingertip_sites=self.fingertip_sites,
            finger_sites=self.finger_sites,
            target_sites=self.target_sites,
            goal=config.goal,
            render_mode=render_mode,
        )

        # Reset stuff
        # TODO: Clean this up
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

        # TODO: Make nicer
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_simulation()

        self._solved_counter = 0
        self._prev_action = np.zeros(self.action_space.shape)

        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._apply_action(action)

        for i in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        reward = self._compute_reward(action)
        truncated = self.steps >= 100

        # if reward > -0.025:
        #     self._solved_counter += 1
        #     print(f"Solved counter: {self._solved_counter}")
        # else:
        #     self._solved_counter = 0

        # terminated = self._solved_counter >= 50

        # if terminated:
        #     reward = 10

        terminated = False

        self.steps += 1
        self._prev_action = action

        return obs, reward, terminated, truncated, {}

    def render(self):
        return self.renderer.render()

    def close(self):
        pass

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

    def _get_obs(self):
        """Returns all joint positions and velocities associated with a robot."""
        joint_positions = np.array(
            [
                self._get_joint_qpos(self.model, self.data, name)
                for name in self.joint_names
            ]
        ).squeeze()
        joint_velocities = np.array(
            [
                self._get_joint_qvel(self.model, self.data, name)
                for name in self.joint_names
            ]
        ).squeeze()
        return np.concatenate([joint_positions, joint_velocities])

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

    def _compute_reward(self, action: np.ndarray):
        #### 1) Distance from goal position

        target_values = []
        for joint_name in self.joint_names:
            joint_name_clipped = joint_name.split(":")[1]
            target_value = self.config.goal.sign_h[joint_name_clipped]
            target_values.append(target_value)
        target_values = np.array(target_values)

        actual_values = self.data.qpos[np.array(self.joint_qpos_addrs)]

        distances = np.linalg.norm(target_values - actual_values)
        distance_reward = -1.0 * distances

        #### 2) Smoothness
        if self.steps > 0:
            action_difference = np.linalg.norm(action - self._prev_action)
            smoothness_reward = -1.0 * action_difference
        else:
            smoothness_reward = 0.0

        #### 3) Velocity penalty
        velocity_penalty = -1.0 * np.linalg.norm(self.data.qvel)

        #### 4) Energy penalty
        energy_penalty = -1.0 * np.linalg.norm(action)

        #### Total reward
        reward = (
            self.config.distance_weight * distance_reward
            + 0.5 * smoothness_reward
            + 0.1 * velocity_penalty
            + 0.1 * energy_penalty
        )
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
