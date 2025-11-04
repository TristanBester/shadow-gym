import time

import hydra
import numpy as np
from omegaconf import DictConfig

from src.env import ShadowEnv


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    env = ShadowEnv(config)

    obs, _ = env.reset()

    action = np.zeros(20)

    while True:
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
