import time

import matplotlib.pyplot as plt
import numpy as np

from src.config import sign_config_a, sign_config_l, sign_config_r
from src.core import cross_product_factory
from src.core.ground import SignLanguageGroundEnvironment
from src.core.label import SignLanguageLF, Symbol


def main():
    """Main function."""
    env = cross_product_factory(render_mode="human")

    obs, _ = env.reset()
    rewards = []

    for step in range(500):
        if env.u == 0:
            action = sign_config_r.action
        elif env.u == 1:
            action = sign_config_a.action
        else:
            action = sign_config_l.action

        next_obs, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
        env.render()

        if terminated:
            print(f"Terminated!! at step {step}")
            break

        # time.sleep(0.1)

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    main()
