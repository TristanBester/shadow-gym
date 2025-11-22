import hydra
import matplotlib.pyplot as plt
import numpy as np

from src.core import ShadowEnv


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config):
    env = ShadowEnv(
        scene_path=config.scene_path,
        camera_config=config.camera,
        sign_config=config.signs.r,
        render_mode="human",
    )

    obs, _ = env.reset()
    rewards = []

    for i in range(200):
        action = np.array(config.signs.l.action)

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
