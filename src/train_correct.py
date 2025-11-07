import uuid

import hydra
import wandb
from omegaconf import DictConfig
from stable_baselines3.sac import SAC

from src.env import ShadowEnv


@hydra.main(
    config_path="../config",
    config_name="config.yaml",
    version_base=None,
)
def main(config: DictConfig):
    wandb.init(
        project="shadow-gym",
        name="SAC_correct",
        sync_tensorboard=True,
    )

    env = ShadowEnv(config, use_correct_action_space=True)
    agent = SAC(
        policy="MlpPolicy",
        env=env,
        tensorboard_log="logs/",
        verbose=1,
        buffer_size=1_000_000,
        batch_size=2_500,
        device="cuda",
    )e

    agent.learn(
        total_timesteps=10_000_000,
        log_interval=1,
    )
    agent.save(f"models/SAC_correct_{uuid.uuid4()}.pkl")


if __name__ == "__main__":
    main()
