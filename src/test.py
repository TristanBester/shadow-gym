import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.env import ShadowEnv

ACTION = np.array(
    [
        -1.000,
        +0.176,
        -1.000,
        -1.000,
        -1.000,
        -1.000,
        -1.000,
        -1.000,
        -1.000,
        +1.000,
        +1.000,
        -0.490,
        -1.000,
        +1.000,
        +1.000,
        +0.720,
        +0.750,
        +1.000,
        -1.000,
        -1.000,
    ]
)


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config):
    env = ShadowEnv(config)

    obs, _ = env.reset()
    rewards = []

    for i in range(200):
        action = ACTION

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        rewards.append(reward)

        if terminated:
            print("Task Solved!!")
            break

    plt.plot(rewards)
    plt.show()

    print("Episode return: ", np.sum(rewards))


if __name__ == "__main__":
    main()
