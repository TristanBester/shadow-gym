import time

import matplotlib.pyplot as plt
import numpy as np

from src.config import sign_config_a, sign_config_i, sign_config_l, sign_config_r
from src.core import cross_product_factory
from src.core.ground import SignLanguageGroundEnvironment
from src.core.label import SignLanguageLF, Symbol


def main():
    """Main function."""
    env = SignLanguageGroundEnvironment(render_mode="human")
    lf = SignLanguageLF()

    obs, _ = env.reset()

    for _ in range(200):
        action = sign_config_a.action
        next_obs, reward, terminated, _, _ = env.step(action)
        props = lf(obs, action, next_obs)

        print(props)

        env.render()

        obs = next_obs


if __name__ == "__main__":
    main()
