"""Reward wrappers."""
# %% Imports
# Standard Library Imports
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Union, final

# Third Party Imports
from gymnasium import Env, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import multiply, ndarray, ones, sum

# Punch Clock Imports
from punchclock.common.utilities import actionSpace2Array
from punchclock.environment.env import SSAScheduler


class RewardBase(ABC, Wrapper):
    """ABC for reward scheme wrapper.

    Handles the basic overhead work of reward schemes and ensures that variables
    other-than reward are not modified.

    Each subclass of RewardBase must define its own calcReward method. The method
    step() should NOT be overridden. This this is the core of the RewardBase ABC;
    the class defines only a reward scheme, it does not modify other inputs/outputs
    of step.
    """

    def __init__(self, env: Env):
        """Initialize base class."""
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a Dict."
        super().__init__(env)
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a MultiDiscrete."

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
            observations, rewards, terminations, truncations, infos, action
        )

        return (observations, rewards, terminations, truncations, infos)

    @abstractmethod
    def calcReward(
        self,
        obs: OrderedDict,
        reward: Union[float, int],
        termination: bool,
        truncation: bool,
        info: dict,
        action: ndarray[int],
    ) -> Union[float, int]:
        """Subclasses of RewardBase must define their own calcReward.

        Args:
            obs: See Gymnasium documentation.
            reward (float | int): Unwrapped environment reward.
            termination (bool): See Gymnasium documentation.
            truncation (bool): See Gymnasium documentation.
            info: See Gymnasium documentation.
            action: See Gymnasium documentation.

        Returns:
            float | int: Wrapped reward.
        """
        return reward


# %% Binary Reward
class BinaryReward(RewardBase):
    """Grants a constant reward per sensor assigned to valid target.

    Nomenclature:
        M: Number of sensors.
        N: Number of targets.

    Example:
        # for 3 sensors, 2 targets
        vis_map = array([[1, 1, 1],
                         [0, 0, 1]])
        action = array([0, 1, 2])
        reward = 1 + 0 + 0 = 1

        Sensor 0 earns 1 reward because it tasked a visible target.
        Sensor 1 earns 0 reward because it tasked a non-visible target.
        Sensor 2 earns 0 reward because it tasked null-action.
    """

    def __init__(self, env: Env, vis_map_key: str, reward: float = 1):
        """Wrap environment.

        Args:
            env (Env): See RewardBase for requirements.
            vis_map_key (str): Key corresponding to vis status map in observation
                space. Value associated with vis_map_key must be (N, M) binary
                array where a 1 indicates the sensor-target pair have access to
                each other (the pairing is a valid action).
            reward (float, optional): Reward generated per valid sensor-target
                assignment. Defaults to 1.
        """
        super().__init__(env)
        assert (
            vis_map_key in env.observation_space.spaces
        ), f"'{vis_map_key}' not in env.observation_space."
        assert isinstance(
            env.observation_space.spaces[vis_map_key], MultiBinary
        ), f"env.observation_space['{vis_map_key}'] must be MultiBinary."

        self.vis_map_key = vis_map_key
        self.reward_per_valid = reward
        self.action_converter = partial(
            actionSpace2Array,
            num_sensors=len(env.action_space),
            num_targets=env.action_space.nvec[0] - 1,
        )

    def calcReward(
        self,
        obs: OrderedDict,
        reward: Any,
        termination: Any,
        truncationAny: Any,
        info: Any,
        action: ndarray[int],
    ) -> float:
        """Calculate binary reward.

        Args:
            obs (OrderedDict): Must have vis_map_key in it.
            reward, termination, truncation, info: Unused.
            action (ndarray[int]): A (N,) array of ints where the i-th value is
                the i-th sensor and the value denotes the target number (0 to N-1);
                a value of N denotes null action.

        Returns:
            float: Total reward for step.
        """
        action_2d = self.action_converter(action)
        action_2d_nonulls = action_2d[:-1, :]
        vis_map = obs[self.vis_map_key]
        reward_mat = multiply(
            self.reward_per_valid * vis_map, action_2d_nonulls
        )
        reward = sum(reward_mat)

        return reward
