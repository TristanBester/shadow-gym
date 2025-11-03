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


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    model = mujoco.MjModel.from_xml_path(config.scene_path)
    data = mujoco.MjData(model)

    renderer = MujocoRenderer(model=model, data=data, default_cam_config=config.camera)

    for i, site_name in enumerate(config.sites.target):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        model.site_pos[site_id] = [0.0, -0.3 + i * 0.025, 0.1]

    # for i, site_name in enumerate(config.sites.finger):
    #     site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    #     model.site_pos[site_id] = [0.0, -0.3 + i * 0.025, 0.1]

    while True:
        site_offset = (data.site_xpos - model.site_pos).copy()

        # Get current fingertip positions
        fingertip_positions = []
        for site_name in config.sites.fingertip:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            site_position = data.site_xpos[site_id]
            fingertip_positions.append(site_position)

        fingertip_positions = np.array(fingertip_positions).flatten().reshape(5, 3)

        # Visulise fingetip positions
        for i, site_name in enumerate(config.sites.finger):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            model.site_pos[site_id] = fingertip_positions[i] - site_offset[site_id]

        action = sample_action()
        data = set_action(action, model, data)

        mujoco.mj_step(model, data)
        renderer.render("human")
        time.sleep(0.01)


if __name__ == "__main__":
    main()
