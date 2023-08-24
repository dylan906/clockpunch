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
        assert all(
            env.action_space.nvec == env.action_space.nvec[0]
        ), "All values in action_space.nvec must be same."

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
class VismaskViolationReward(RewardBase):
    """Grants a constant reward per sensor assigned to valid (or invalid) action.

    Use to reward tasking sensors to valid targets or penalize for tasking to invalid
    ones.

    Nomenclature:
        M: Number of sensors.
        N: Number of targets.

    Example:
        # for 3 sensors, 2 targets, reward valid actions
        wrapped_env = VismaskViolationReward(env, "mask")
        vis_mask = array([[1, 1, 1],
                         [0, 0, 1]])
        action = array([0, 1, 2])
        # reward = 1 + 0 + 0 = 1

        Sensor 0 earns 1 reward because it tasked a visible target.
        Sensor 1 earns 0 reward because it tasked a non-visible target.
        Sensor 2 earns 0 reward because it tasked null-action.

    Example:
        # for 3 sensors, 2 targets, penalize invalid actions
        wrapped_env = VismaskViolationReward(env, "mask", reward=-1,
            reward_valid_actions=False)
        vis_mask = array([[1, 1, 1],
                         [0, 0, 1]])
        action = array([0, 1, 2])
        # reward = 0 + -1 + 0 = -1
    """

    def __init__(
        self,
        env: Env,
        action_mask_key: str,
        reward: float = 1,
        reward_valid_actions: bool = True,
    ):
        """Wrap environment.

        Args:
            env (Env): See RewardBase for requirements.
            action_mask_key (str): Key corresponding to action mask in observation
                space. Value associated with action_mask_key must be (N, M) binary
                array where a 1 indicates the sensor-target the pairing is a valid
                action).
            reward (float, optional): Reward generated per (in)valid sensor-target
                assignment. Defaults to 1.
            reward_valid_actions (bool, optional): If True, valid actions are rewarded.
                If False, invalid actions are reward. Defaults to True.
        """
        super().__init__(env)
        assert (
            action_mask_key in env.observation_space.spaces
        ), f"'{action_mask_key}' not in env.observation_space."
        assert isinstance(
            env.observation_space.spaces[action_mask_key], MultiBinary
        ), f"env.observation_space['{action_mask_key}'] must be MultiBinary."

        self.action_mask_key = action_mask_key
        self.reward_per_valid = reward
        self.reward_valid_actions = reward_valid_actions
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
        truncation: Any,
        info: Any,
        action: ndarray[int],
    ) -> float:
        """Calculate binary reward.

        Args:
            obs (OrderedDict): Must have action_mask_key in it.
            reward, termination, truncation, info: Unused.
            action (ndarray[int]): A (N,) array of ints where the i-th value is
                the i-th sensor and the value denotes the target number (0 to N-1);
                a value of N denotes null action.

        Returns:
            float: Total reward for step.
        """
        action_2d = self.action_converter(action)
        action_2d_nonulls = action_2d[:-1, :]
        vis_mask = obs[self.action_mask_key]

        if self.reward_valid_actions is True:
            # Reward valid actions
            reward_mat = multiply(
                self.reward_per_valid * vis_mask, action_2d_nonulls
            )
        else:
            # Reward invalid actions
            reward_mat = multiply(
                self.reward_per_valid * (1 - vis_mask), action_2d_nonulls
            )

        reward = sum(reward_mat)

        return reward


# %% Reward Null Action
class NullActionReward(RewardBase):
    """Rewards selection (or non-selection) of null actions.

    The null action is the max value allowed in a MultiDiscrete action space. All
        values in action space must be identical.

    Example:
        action_space = MultiDiscrete([3, 3, 3]) # 3 is the null action
        wrapped_env = NullActionReward(env)
        action = array([0, 1, 3])
        reward = 0 + 0 + 1 = 1

    Example:
        action_space = MultiDiscrete([3, 3, 3]) # 3 is the null action
        wrapped_env = NullActionReward(env, reward_null_actions=False)
        action = array([0, 1, 3])
        reward = 1 + 1 + 0 = 2

    """

    def __init__(
        self, env: Env, reward: float = 1, reward_null_actions: bool = True
    ):
        """Wrap environment.

        Args:
            env (Env): See RewardBase for requirements.
            reward (float, optional): Reward generated per (non-)null action assignment.
                Defaults to 1.
            reward_null_actions (bool, optional): If True, reward is assigned for
                null actions. If False, reward is assigned for non-null actions.
                Defaults to True.
        """
        super().__init__(env)
        self.reward_per_null_action = reward
        self.reward_null_actions = reward_null_actions
        self.null_action_index = env.action_space.nvec[0] - 1

    def calcReward(
        self,
        obs: Any,
        reward: Any,
        termination: Any,
        truncationAny: Any,
        info: Any,
        action: ndarray[int],
    ) -> float:
        """Calculate null action reward.

        Args:
            obs, reward, termination, truncation, info: Unused.
            action (ndarray[int]): A (N,) array of ints where the i-th value is
                the i-th sensor and the value denotes the target number (0 to N-1);
                a value of N denotes null action.

        Returns:
            float: Total reward for step.
        """
        if self.reward_null_actions is True:
            # Reward null actions
            reward = (
                self.reward_per_null_action
                * (action == self.null_action_index).sum()
            )
        else:
            # reward non-null actions (aka active actions)
            reward = (
                self.reward_per_null_action
                * (action != self.null_action_index).sum()
            )
        return reward
