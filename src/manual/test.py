import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.env import ShadowEnv

ACTION = np.array(
    [
        0.555,
        0.176,
        0.000,
        -1.000,
        0.280,
        0.000,
        -1.000,
        0.290,
        0.000,
        -1.000,
        0.280,
        -1.000,
        0.000,
        -1.000,
        0.290,
        0.000,
        -1.000,
        0.000,
        0.000,
        1.000,
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

    for i in range(100):
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
