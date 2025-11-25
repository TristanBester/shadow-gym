import os

os.environ["MUJOCO_GL"] = "egl"
from glob import glob

import wandb
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.sac import SAC

from src.core import cross_product_factory


class VideoUploadCallback(BaseCallback):
    """Callback to upload videos to WandB when they are created."""

    def __init__(self, video_folder: str, run, verbose: int = 0):
        """Initialize the callback."""
        super().__init__(verbose)
        self.video_folder = video_folder
        self.run = run
        self.uploaded_videos = set()

    def _on_step(self) -> bool:
        """On step callback."""
        if self.num_timesteps > 10 and self.num_timesteps % 50_000 != 0:
            return True

        video_files = glob(os.path.join(self.video_folder, "*.mp4"))

        for video_path in video_files:
            if video_path not in self.uploaded_videos:
                # Upload the video to wandb
                video_name = os.path.basename(video_path)
                if self.verbose > 0:
                    print(f"Uploading video: {video_name}")

                self.run.log(
                    {
                        "video": wandb.Video(video_path, fps=4, format="mp4"),
                        "global_step": self.num_timesteps,
                    }
                )

                self.uploaded_videos.add(video_path)

        return True


def main():
    """Main function."""
    run = wandb.init(
        project="shadow-gym",
        name="SAC",
        sync_tensorboard=True,
    )

    env = cross_product_factory(render_mode="rgb_array")

    # Wrap with DummyVecEnv for vectorization
    # The environment needs render_mode="rgb_array" for video recording
    # Monitor wrapper tracks episode returns and lengths for TensorBoard logging
    env = DummyVecEnv([lambda: Monitor(env)])

    # Wrap with VecVideoRecorder for video recording
    env = VecVideoRecorder(
        env,
        f"./runs/{run.id}/videos",
        record_video_trigger=lambda step: step % 100_000 == 0,
        video_length=1000,
        name_prefix="sac-shadow",
    )

    agent = SAC(
        policy="MlpPolicy",
        env=env,
        tensorboard_log="logs/",
        verbose=1,
        buffer_size=1_000_000,
        batch_size=2_500,
        device="cuda",
    )

    # Create checkpoint callback to save model every 500k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=f"./runs/{run.id}/models",
        name_prefix="sac_shadow",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    video_upload_callback = VideoUploadCallback(
        video_folder=f"./runs/{run.id}/videos",
        run=run,
        verbose=1,
    )
    callbacks = [checkpoint_callback, video_upload_callback]

    agent.learn(
        total_timesteps=10_000_000,
        log_interval=1,
        callback=callbacks,
    )


if __name__ == "__main__":
    main()
