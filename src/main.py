import time

import hydra
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from omegaconf import DictConfig


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    model = mujoco.MjModel.from_xml_path(config.scene_path)
    data = mujoco.MjData(model)

    renderer = MujocoRenderer(model=model, data=data, default_cam_config=config.camera)

    while True:
        mujoco.mj_step(model, data)
        renderer.render("human")
        time.sleep(0.01)


if __name__ == "__main__":
    main()
