"""Reward wrappers."""
# %% Imports
# Standard Library Imports
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, final

# Third Party Imports
from gymnasium import Env, RewardWrapper, Wrapper
from gymnasium.spaces import Dict, MultiDiscrete
from gymnasium.wrappers import TransformReward
from numpy import float32, int8, int_, ndarray

# Punch Clock Imports
from punchclock.common.math import logistic
from punchclock.common.utilities import getInequalityFunc


# %% Base class for reward configuration wrappers
class RewardBase(ABC, Wrapper):
    """ABC for reward scheme wrapper.

    Handles the basic overhead work of reward schemes and ensures that variables
    other-than reward are not modified.

    Each subclass of RewardBase must define its own calcReward method. The method
    step() should NOT be overridden. This this is the core of the RewardBase ABC;
    the class defines only a reward scheme, it does not modify other inputs/outputs
    of step.

    Overwrites input reward with output of calcReward().
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
            unwrapped_rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        # add unwrapped reward to new reward
        rewards = self.calcReward(
            observations,
            unwrapped_rewards,
            terminations,
            truncations,
            infos,
            action,
        )

        return (observations, rewards, terminations, truncations, infos)

    @abstractmethod
    def calcReward(
        self,
        obs: OrderedDict,
        reward: float | int,
        termination: bool,
        truncation: bool,
        info: dict,
        action: ndarray[int],
    ) -> float | int:
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


# %% Assign ObsInfo Base Class
class AssignThingToReward(RewardBase):
    """Assign an value from observation or info to reward.

    Overwrites existing reward.

    Required Dict observation space. By Gymnasium Env API, info is already a dict.
    """

    def __init__(self, env: Env, key: str, info_or_obs: str):
        """Wrap environment with AssignObsToReward.

        Args:
            env (Env): See RewardBase for requirements.
            key (str): Key corresponding to value in observation space or info
                dict that will be assigned to reward.
            info_or_obs (str): ["info" | "obs"] Whether to pull from the info or
                observation.
        """
        super().__init__(env)
        if info_or_obs == "obs":
            spaces = env.observation_space.spaces
        elif info_or_obs == "info":
            _, spaces = deepcopy(env).reset()

        assert key in spaces, f"{key} must be in observation_space.spaces."
        assert spaces[key].shape in [
            (1,),
            (),
        ], f"observation_space['{key}'] must be a (1,)- or ()-sized space."
        assert spaces[key].dtype in (
            float,
            int,
            float32,
            int_,
            int8,
        ), f"{key} must correspond to a scalar value."

        self.key = key
        self.info_or_obs = info_or_obs

    def calcReward(
        self,
        obs: OrderedDict,
        reward: Any,
        termination: Any,
        truncationAny: Any,
        info: Any,
        action: Any,
    ):
        """Assign item to reward.

        Args:
            obs (OrderedDict): Observation from environment.
            reward, termination, truncation: Unused.
            info (dict): Info from environment.


        Returns:
            _type_: _description_
        """
        if self.info_or_obs == "info":
            io = deepcopy(info)
        elif self.info_or_obs == "obs":
            io = deepcopy(obs)

        # overwrite reward
        if io[self.key].shape == (1,):
            reward = io[self.key][0]
        elif io[self.key].shape == ():
            reward = io[self.key]

        return reward


# %% Assign observation space variable to reward
class AssignObsToReward(AssignThingToReward):
    """Get an item in the observation space and assign reward to the value.

    The value gotten from the observation must have shape (1,) or (). Does not modify
    observation.
    """

    def __init__(self, env: Env, key: str):
        """Wrap environment with AssignObsToReward.

        Args:
            env (Env): See RewardBase for requirements.
            key (str): Key corresponding to value in observation space
                that will be assigned to reward.
        """
        super().__init__(env=env, key=key, info_or_obs="obs")


# %% AssignInfoToReward
class AssignInfoToReward(AssignThingToReward):
    """Get an item in the info and assign reward to the value.

    The value gotten from the info must have shape (1,) or (). Does not modify
    info.
    """

    def __init__(self, env: Env, key: str):
        """Wrap environment with AssignItemToReward.

        Args:
            env (Env): See RewardBase for requirements.
            key (str): Key corresponding to value in item space
                that will be assigned to reward.
        """
        super().__init__(env, key=key, info_or_obs="info")


# %% Threshold Reward
class ThresholdReward(RewardWrapper):
    """Gives a reward if unwrapped reward meets an inequality operation.

    Overrides unwrapped reward.

    If unwrapped reward is <= (by default) the threshold, then a reward is granted.
        The reward per step is set on instantiation and does not change.
        The inequality is set on instantiation (can be <=, >=, <, or >) and does
        not change.
    """

    def __init__(
        self,
        env: Env,
        unwrapped_reward_threshold: float | int,
        reward: float = 1,
        inequality: str = "<=",
    ):
        """Wrap environment with ThresholdReward.

        Args:
            env (Env): A Gymnasium environment.
            unwrapped_reward_threshold (float | int): Threshold to evaluate
                unwrapped reward against.
            reward (float, optional): Reward generated per step that inequality
                evaluates to True. Defaults to 1.
            inequality (str, optional): String representation of inequality operator
                to use in threshold operation. Must be one of ['<=', '>=', '<', '>'].
                Defaults to "<=".
        """
        super().__init__(env)

        self.reward_per_step = reward
        # getInequalityFunc checks arg type
        self.inequalityFunc = getInequalityFunc(inequality)
        self.threshold = unwrapped_reward_threshold

    def reward(self, reward: float) -> float:
        """Calculate threshold reward.

        Returns:
            float: Either 0 (if inequality evaluates to False) or self.reward_per_step
                (if inequality evaluates to True) specified on instantiation.
        """
        inbounds = self.inequalityFunc(reward, self.threshold)
        # inequalityFunc returns numpy bool, which needs to be compared with "=="
        # instead of "is"
        if inbounds == True:  # noqa
            new_reward = self.reward_per_step
        elif inbounds == False:  # noqa
            new_reward = 0
        else:
            TypeError("inbounds is neither True nor False")

        return new_reward


# %% ZeroReward
class ZeroReward(TransformReward):
    """Makes environment return 0 reward for all states."""

    def __init__(self, env: Env):
        """Wrap env with ZeroReward."""
        super().__init__(env, self.return0)

    def return0(self, reward: float):
        """Returns 0."""
        return 0


# %% Logistic transform
class LogisticTransformReward(TransformReward):
    """Transform reward through logistic function."""

    def __init__(
        self,
        env: Env,
        x0: float = 0.0,
        k: float = 1.0,
        L: float = 1.0,
    ):
        """Wrap environment with LogisticTransformReward.

        Args:
            env (Env): A Gymnasium environment.
            x0 (float, optional): Value of x at sigmoid's midpoint. Defaults to 0.0.
            k (float, optional): Steepness parameter. Defaults to 1.0.
            L (float, optional): Max value of output. Defaults to 1.0.
        """
        logisticPartial = partial(logistic, x0=x0, k=k, L=L)

        super().__init__(env, logisticPartial)
