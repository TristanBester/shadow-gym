import time

import hydra
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from omegaconf import DictConfig


def sample_action():
    action = np.random.uniform(-1, 1, size=(20,))
    return np.ones_like(action)


def set_action(action, model, data):
    clipped_action = np.clip(action, -1, 1)

    # Get the control ranges for each actuator in the model
    control_ranges = model.actuator_ctrlrange

    # Compute the half the range of actuation for each actuator
    half_actuation_ranges = (control_ranges[:, 1] - control_ranges[:, 0]) / 2.0

    # Compute the center of the actuation for each actuator
    actuation_centers = (control_ranges[:, 1] + control_ranges[:, 0]) / 2.0

    # Compute the control signal associated with the provided action
    control = actuation_centers + clipped_action * half_actuation_ranges

    data.ctrl[:] = control
    return data


class Visualiser:
    def __init__(self, position_sites, target_sites, fingertip_sites, goal):
        self.position_sites = position_sites
        self.target_sites = target_sites
        self.fingertip_sites = fingertip_sites
        self.goal = goal

    def __call__(self, model, data):
        model = self._visualise_targets(model, data)
        model = self._visualise_fingertips(model, data)
        return model

    def _visualise_targets(self, model, data):
        site_offset = data.site_xpos - model.site_pos

        for target_site_name in self.target_sites:
            target_site_id = self._get_site_id(model, target_site_name)

            goal_position_global = np.array(self.goal[target_site_name])
            goal_position_local = goal_position_global - site_offset[target_site_id]
            model.site_pos[target_site_id] = goal_position_local

        return model

    def _visualise_fingertips(self, model, data):
        # Get fingertip positions
        fingertip_global_positions = []
        for site_name in self.position_sites:
            site_id = self._get_site_id(model, site_name)
            global_position = data.site_xpos[site_id]
            fingertip_global_positions.append(global_position)

        # Compute global position offsets
        site_offset = data.site_xpos - model.site_pos

        # Visualise fingertips
        for site_name, global_position in zip(
            self.fingertip_sites, fingertip_global_positions
        ):
            site_id = self._get_site_id(model, site_name)
            # Convert the global position to a local position for the visualisation site
            local_position = global_position - site_offset[site_id]
            model.site_pos[site_id] = local_position
        return model

    def _get_site_id(self, model, site_name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    model = mujoco.MjModel.from_xml_path(config.scene_path)
    data = mujoco.MjData(model)

    renderer = MujocoRenderer(model=model, data=data, default_cam_config=config.camera)
    visualiser = Visualiser(
        config.sites.fingertip, config.sites.target, config.sites.finger, config.goal
    )

    while True:
        action = sample_action()
        # data = set_action(action, model, data)
        model = visualiser(model, data)

        mujoco.mj_step(model, data)
        renderer.render("human")
        time.sleep(0.01)


if __name__ == "__main__":
    main()
