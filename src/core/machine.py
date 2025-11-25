from typing import Callable

import numpy as np
from pycrm.automaton import RewardMachine

from src.config import (
    base_config,
    sign_config_a,
    sign_config_i,
    sign_config_l,
    sign_config_r,
)
from src.core.label import Symbol


class SignLanguageRM(RewardMachine):
    """Reward machine for the sign language environment."""

    def __init__(self):
        """Initialise the reward machine."""
        super().__init__(env_prop_enum=Symbol)

    @property
    def u_0(self) -> int:
        """Initial state."""
        return 0

    @property
    def encoded_configuration_size(self) -> int:
        """Encoded configuration size."""
        return 2

    def _get_state_transition_function(self) -> dict:
        return {
            0: {
                "R": 1,
                "NOT R": 0,
            },
            1: {
                "A": 2,
                "NOT A": 1,
            },
            2: {
                "I": 3,
                "NOT I": 2,
            },
            3: {
                "L": -1,
                "NOT L": 3,
            },
        }

    def _get_reward_transition_function(self) -> dict:
        return {
            0: {
                "R": 1000,
                "NOT R": self._create_r_reward(),
            },
            1: {
                "A": 1000,
                "NOT A": self._create_a_reward(),
            },
            2: {
                "I": 1000,
                "NOT I": self._create_i_reward(),
            },
            3: {
                "L": 1000,
                "NOT L": self._create_l_reward(),
            },
        }

    def _create_r_reward(self) -> Callable:
        """Create the reward function for the R sign."""

        def sign_r_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            """Reward for the R sign."""
            del obs

            # Distance component
            current_qpos = next_obs[:24]
            target_qpos = sign_config_r.qpos
            distance = np.linalg.norm(current_qpos - target_qpos)
            distance_reward = -1.0 * distance

            # Velocity component
            velocity_reward = -1.0 * np.linalg.norm(next_obs[24:])

            # Energy component
            energy_reward = -1.0 * np.linalg.norm(action)

            # Total reward
            reward = (
                base_config.distance_weight * distance_reward
                + base_config.velocity_weight * velocity_reward
                + base_config.energy_weight * energy_reward
            )
            # Add tiered reward
            reward = reward - 100.0
            return float(reward)

        return sign_r_reward

    def _create_a_reward(self) -> Callable:
        """Create the reward function for the A sign."""

        def sign_a_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            """Reward for the A sign."""
            del obs

            # Distance component
            current_qpos = next_obs[:24]
            target_qpos = sign_config_a.qpos
            distance = np.linalg.norm(current_qpos - target_qpos)
            distance_reward = -1.0 * distance

            # Velocity component
            velocity_reward = -1.0 * np.linalg.norm(next_obs[24:])

            # Energy component
            energy_reward = -1.0 * np.linalg.norm(action)

            # Total reward
            reward = (
                base_config.distance_weight * distance_reward
                + base_config.velocity_weight * velocity_reward
                + base_config.energy_weight * energy_reward
            )
            # Add tiered reward
            reward = reward - 50.0
            return float(reward)

        return sign_a_reward

    def _create_i_reward(self) -> Callable:
        """Create the reward function for the I sign."""

        def sign_i_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            """Reward for the I sign."""
            del obs

            # Distance component
            current_qpos = next_obs[:24]
            target_qpos = sign_config_i.qpos
            distance = np.linalg.norm(current_qpos - target_qpos)
            distance_reward = -1.0 * distance

            # Velocity component
            velocity_reward = -1.0 * np.linalg.norm(next_obs[24:])

            # Energy component
            energy_reward = -1.0 * np.linalg.norm(action)

            # Total reward
            reward = (
                base_config.distance_weight * distance_reward
                + base_config.velocity_weight * velocity_reward
                + base_config.energy_weight * energy_reward
            )
            # Add tiered reward
            reward = reward - 10.0
            return float(reward)

        return sign_i_reward

    def _create_l_reward(self) -> Callable:
        """Create the reward function for the L sign."""

        def sign_l_reward(
            obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
        ) -> float:
            """Reward for the L sign."""
            del obs

            # Distance component
            current_qpos = next_obs[:24]
            target_qpos = sign_config_l.qpos
            distance = np.linalg.norm(current_qpos - target_qpos)
            distance_reward = -1.0 * distance

            # Velocity component
            velocity_reward = -1.0 * np.linalg.norm(next_obs[24:])

            # Energy component
            energy_reward = -1.0 * np.linalg.norm(action)

            # Total reward
            reward = (
                base_config.distance_weight * distance_reward
                + base_config.velocity_weight * velocity_reward
                + base_config.energy_weight * energy_reward
            )
            return float(reward)

        return sign_l_reward
