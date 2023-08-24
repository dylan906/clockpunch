"""Reward wrappers."""
# %% Imports
# Standard Library Imports
from abc import ABC, ABCMeta, abstractmethod
from typing import final

# Third Party Imports
from gymnasium import Env, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete

# Punch Clock Imports
from punchclock.environment.env import SSAScheduler


class RewardBase(ABC, Wrapper):
    """ABC for reward scheme wrapper.

    Handles the basic overhead work of reward schemes and ensures that variables
    other-than reward are not modified.

    Each subclass of RewardBase must define its own calcReward method. The method
    step() should NOT be overridden. This this is the core of the RewardBase ABC;
    the class defines only a reward scheme, it does not modify other inputs/outputs of step.
    """

    def __init__(self, env: Env):
        """Initialize base class."""
        super().__init__(env)

    @final
    def step(self, action):
        """Step environment forward. Do not modify."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        rewards = self.calcReward(
            observations, rewards, terminations, truncations, infos
        )

        return (observations, rewards, terminations, truncations, infos)

    @abstractmethod
    def calcReward(
        self,
        obs,
        reward: float | int,
        termination: bool,
        truncation: bool,
        info,
    ) -> float | int:
        """Subclasses of RewardBase must define their own calcReward.

        Args:
            obs: See Gymnasium documentation.
            reward (float | int): Unwrapped environment reward.
            termination (bool): See Gymnasium documentation.
            truncation (bool): See Gymnasium documentation.
            info: See Gymnasium documentation.

        Returns:
            float | int: Wrapped reward.
        """
        return reward


# %% Binary Reward
class BinaryReward(RewardBase):
    def __init__(self, env: Env):
        super().__init__(env)

    def calcReward(self, obs, reward, termination, truncation, info):
        return 1
