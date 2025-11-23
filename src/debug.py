import matplotlib.pyplot as plt
import numpy as np

from src.config import sign_config_r
from src.core import ShadowEnv


def main():
    """Main function."""
    env = ShadowEnv(render_mode="human")

    _, _ = env.reset()
    rewards = []

    for _ in range(200):
        action = np.array(sign_config_r.action)

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
