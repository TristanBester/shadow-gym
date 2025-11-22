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
        render_mode: str = "human",
    ):
        self.model = model
        self.data = data

        # Extract width and height from camera_config (they're not MuJoCo camera attributes)
        width = camera_config.get("width", 640)
        height = camera_config.get("height", 480)

        # Create camera config without width/height for MuJoCo camera
        cam_config = {
            k: v for k, v in camera_config.items() if k not in ["width", "height"]
        }

        self.renderer = MujocoRenderer(
            model=model,
            data=data,
            default_cam_config=cam_config,
        )

        # For rgb_array mode, explicitly set width and height on the renderer
        if render_mode == "rgb_array":
            self.renderer.width = width
            self.renderer.height = height

        self.fingertip_sites = fingertip_sites
        self.finger_sites = finger_sites
        self.target_sites = target_sites
        self.goal = goal
        self.render_mode = render_mode

        # self._set_goal_site_positions()

    def render(self):
        # TODO: Name this better
        # self._set_goal_site_positions()
        # self._set_finger_site_positions()

        if self.render_mode == "human":
            self.renderer.render("human")
        elif self.render_mode == "rgb_array":
            return self.renderer.render("rgb_array")
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    # def _set_goal_site_positions(self):
    #     site_offset = self.data.site_xpos - self.model.site_pos

    #     for target_site_name in self.target_sites:
    #         target_site_id = self._get_site_id(target_site_name)

    #         goal_position_global = np.array(self.goal[target_site_name])
    #         goal_position_local = goal_position_global - site_offset[target_site_id]
    #         self.model.site_pos[target_site_id] = goal_position_local

    # def _set_finger_site_positions(self):
    #     # Get fingertip positions
    #     fingertip_global_positions = []
    #     for site_name in self.fingertip_sites:
    #         site_id = self._get_site_id(site_name)
    #         global_position = self.data.site_xpos[site_id]
    #         fingertip_global_positions.append(global_position)

    #     # Compute global position offsets
    #     site_offset = self.data.site_xpos - self.model.site_pos

    #     # Visualise fingertips
    #     for site_name, global_position in zip(
    #         self.finger_sites, fingertip_global_positions
    #     ):
    #         site_id = self._get_site_id(site_name)
    #         # Convert the global position to a local position for the visualisation site
    #         local_position = global_position - site_offset[site_id]
    #         self.model.site_pos[site_id] = local_position

    # def _get_site_id(self, site_name: str) -> int:
    #     return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
