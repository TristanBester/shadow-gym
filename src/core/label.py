from enum import Enum, auto

import numpy as np
from pycrm.label import LabellingFunction

from src.config import (
    base_config,
    sign_config_a,
    sign_config_i,
    sign_config_l,
    sign_config_r,
)


class Symbol(Enum):
    """Symbols in the SignLanguageEnvironment."""

    R = auto()
    A = auto()
    I = auto()
    L = auto()


class SignLanguageLF(LabellingFunction):
    """Labelling function for the SignLanguageEnvironment."""

    @LabellingFunction.event
    def test_sign_r(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """Test if shadow hand is in the correct position for the R sign."""
        del obs, action
        current_qpos = next_obs[:24]
        target_qpos = sign_config_r.qpos
        distance = np.linalg.norm(current_qpos - target_qpos)

        if distance < base_config.labelling_function_tol:
            return Symbol.R
        else:
            return None

    @LabellingFunction.event
    def test_sign_a(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """Test if shadow hand is in the correct position for the A sign."""
        del obs, action
        current_qpos = next_obs[:24]
        target_qpos = sign_config_a.qpos
        distance = np.linalg.norm(current_qpos - target_qpos)

        if distance < base_config.labelling_function_tol:
            return Symbol.A
        else:
            return None

    @LabellingFunction.event
    def test_sign_i(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """Test if shadow hand is in the correct position for the I sign."""
        del obs, action
        current_qpos = next_obs[:24]
        target_qpos = sign_config_i.qpos
        distance = np.linalg.norm(current_qpos - target_qpos)

        if distance < base_config.labelling_function_tol:
            return Symbol.I
        else:
            return None

    @LabellingFunction.event
    def test_sign_l(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """Test if shadow hand is in the correct position for the L sign."""
        del obs, action
        current_qpos = next_obs[:24]
        target_qpos = sign_config_l.qpos
        distance = np.linalg.norm(current_qpos - target_qpos)

        if distance < base_config.labelling_function_tol:
            return Symbol.L
        else:
            return None
