import os
import time
from glob import glob

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.sac import SAC

from src.env import ShadowEnv


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint based on step number."""
    checkpoint_files = glob(os.path.join(checkpoint_dir, "sac_shadow_*_steps.zip"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Extract step numbers and find the maximum
    step_numbers = []
    for file in checkpoint_files:
        basename = os.path.basename(file)
        # Extract number from filename like "sac_shadow_3000000_steps.zip"
        try:
            steps = int(basename.split("_")[2])
            step_numbers.append((steps, file))
        except (IndexError, ValueError):
            continue

    if not step_numbers:
        raise ValueError("Could not parse step numbers from checkpoint files")

    # Sort by step number and return the file with the highest steps
    step_numbers.sort(reverse=True)
    latest_checkpoint = step_numbers[0][1]

    print(
        f"Loading checkpoint: {os.path.basename(latest_checkpoint)} ({step_numbers[0][0]:,} steps)"
    )
    return latest_checkpoint


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    # Get the project root directory (go up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, "checkpoints")

    # Load the latest checkpoint
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    print("Loading model...")
    model = SAC.load(latest_checkpoint_path)
    print("Model loaded successfully!")

    # Create environment with human rendering
    env = ShadowEnv(config, render_mode="human")

    # Run for 10 episodes
    num_episodes = 10

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        terminated = False
        truncated = False

        all_actions = []

        while not (terminated or truncated):
            # Get deterministic action from the model
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Render the environment
            env.render()
            time.sleep(0.1)
            all_actions.append(action)

        all_actions = np.array(all_actions)

        # Create subplots for each action dimension in a grid layout
        num_actions = all_actions.shape[1]

        # Calculate grid dimensions (prefer roughly square or slightly wider)
        n_cols = int(np.ceil(np.sqrt(num_actions)))
        n_rows = int(np.ceil(num_actions / n_cols))

        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

        # # Flatten axes array for easy iteration
        # if num_actions == 1:
        #     axes = [axes]
        # else:
        #     axes = axes.flatten()

        # for i in range(num_actions):
        #     axes[i].plot(all_actions[:, i])
        #     axes[i].set_xlabel("Timestep")
        #     axes[i].set_ylim(-1, 1)
        #     axes[i].grid(True, alpha=0.3)

        # # Hide any unused subplots
        # for i in range(num_actions, len(axes)):
        #     axes[i].set_visible(False)

        # fig.suptitle(f"Episode {episode + 1} Actions", fontsize=14)
        # plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        # plt.show()

        print(f"Episode finished after {step_count} steps")
        print(f"Episode reward: {episode_reward:.4f}")

    print(f"\n{'='*50}")
    print("Demo completed!")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()
