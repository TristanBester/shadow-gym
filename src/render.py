import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


class ShadowRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_config: dict,
        fingertip_sites: list,
        finger_sites: list,
        target_sites: list,
        goal: dict,
    ):
        self.model = model
        self.data = data
        self.renderer = MujocoRenderer(
            model=model,
            data=data,
            default_cam_config=camera_config,
        )
        self.fingertip_sites = fingertip_sites
        self.finger_sites = finger_sites
        self.target_sites = target_sites
        self.goal = goal

        self._set_goal_site_positions()

    def render(self):
        # TODO: Name this better
        self._set_goal_site_positions()
        self._set_finger_site_positions()

        self.renderer.render("human")

    def _set_goal_site_positions(self):
        site_offset = self.data.site_xpos - self.model.site_pos

        for target_site_name in self.target_sites:
            target_site_id = self._get_site_id(target_site_name)

            goal_position_global = np.array(self.goal[target_site_name])
            goal_position_local = goal_position_global - site_offset[target_site_id]
            self.model.site_pos[target_site_id] = goal_position_local

    def _set_finger_site_positions(self):
        # Get fingertip positions
        fingertip_global_positions = []
        for site_name in self.fingertip_sites:
            site_id = self._get_site_id(site_name)
            global_position = self.data.site_xpos[site_id]
            fingertip_global_positions.append(global_position)

        # Compute global position offsets
        site_offset = self.data.site_xpos - self.model.site_pos

        # Visualise fingertips
        for site_name, global_position in zip(
            self.finger_sites, fingertip_global_positions
        ):
            site_id = self._get_site_id(site_name)
            # Convert the global position to a local position for the visualisation site
            local_position = global_position - site_offset[site_id]
            self.model.site_pos[site_id] = local_position

    def _get_site_id(self, site_name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
