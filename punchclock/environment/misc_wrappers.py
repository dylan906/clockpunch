"""Misc wrappers."""
# %% Imports
# Standard Library Imports
import operator as op
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from inspect import signature
from typing import Any, Callable, Tuple
from warnings import warn

# Third Party Imports
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Dict, Space

# Punch Clock Imports
from punchclock.common.utilities import getInfo
from punchclock.environment.wrapper_utils import OperatorFuncBuilder
from punchclock.policies.policy_builder import buildSpace


# %% RandomInfo
class RandomInfo(Wrapper):
    """Appends items to an env's info.

    Used for development.
    """

    def __init__(self, env, info_space: Dict = None):
        """Wrap an env."""
        super().__init__(env)
        if info_space is None:
            info_space = Dict({0: Box(0, 1)})

        assert isinstance(info_space, Dict)
        assert all([isinstance(i, Space) for i in info_space.values()])
        self.info_space = info_space

    def step(self, action):
        """Step an env."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        new_info = self.info_space.sample()
        infos.update(new_info)
        return (observations, rewards, terminations, truncations, infos)

    def reset(self, seed: int | None = None, options=None):
        """Reset me bro."""
        obs = self.observation_space.sample()
        info = self.info_space.sample()

        return obs, info


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    # NOTE: SimRunner is hugely dependent on this class. Be careful about modifying
    # it.

    def __init__(self, env: Env, id: Any = None):  # noqa
        """Wrap environment with IdentityWrapper.

        Args:
            env (Env): A Gymnasium environment.
            id (Any, optional): Mainly used to distinguish between multiple instances
                of IdentityWrapper. Defaults to None.
        """
        super().__init__(env)

        self.id = id
        return

    def observation(self, obs):
        """Pass-through observation."""
        return obs


# %% Base class: ModifyObsOrInfo
class ModifyObsOrInfo(Wrapper):
    """Base wrapper used for wrapper that modify either info or obs."""

    def __init__(self, env: Env, obs_info: str):
        """Initialize wrapper.

        Args:
            env (Env): Must have Dict observation space.
            obs_info (str): ["obs" | "info"] The wrapper modifies either the
                observation or the info (not both).
        """
        super().__init__(env)
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert obs_info in [
            "info",
            "obs",
        ], "obs_info must be either 'obs' or 'info'."

        self.obs_info = obs_info

    @abstractmethod
    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Child classes require this function.

        Args:
            obs (OrderedDict): Unwrapped observation.
            info (dict): Unwrapped info.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", this is the modified
                observation. Otherwise, same as input.
            new_info (dict): If self.obs_info == "info", this is the modified
                info. Otherwise, same as input.
        """
        return new_obs, new_info  # noqa

    def reset(
        self, seed: int | None = None, options=None
    ) -> Tuple[OrderedDict, dict]:
        """Reset env."""
        obs, info = super().reset(seed=seed, options=options)
        new_obs, new_info = self.modifyOI(
            obs=deepcopy(obs), info=deepcopy(info)
        )
        self.info = deepcopy(info)
        return new_obs, new_info

    def step(self, action: Any) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Step environment."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        new_obs, new_info = self.modifyOI(obs=observations, info=infos)
        self.info = deepcopy(new_info)
        return (new_obs, rewards, terminations, truncations, new_info)

    def observation(self, observation: OrderedDict) -> OrderedDict:
        """Modify observation only if obs_info == "obs"."""
        if self.obs_info == "obs":
            info = deepcopy(self.info)
            new_obs, _ = self.modifyOI(obs=observation, info=info)
        else:
            new_obs = observation

        return new_obs


# %% CopyObsInfoItem
class CopyObsInfoItem(Wrapper):
    """Copy an item from and obs/info to obs/info.

    Overwrites existing item, if it already exists.

    Environment must have Dict observation space.
    """

    def __init__(
        self,
        env: Env,
        copy_from: str,
        copy_to: str,
        from_key: str,
        to_key: str = None,
        info_space_config: dict = None,
    ):
        """Wrap environment with CopyObsInfoItem wrapper.

        Args:
            env (Env): Gymnasium environment with Dict observation space.
            copy_from (str): ["obs" | "info"] The source of the item.
            copy_to (str): ["obs" | "info"] The place to copy the source item to.
            from_key (str): Must be in copy_from associated dict.
            to_key (str, optional): Key assigned to copied value. If None, from_key
                will be used. Defaults to None.
            info_space_config (dict, optional): Required if copy_from/copy_to
                are "info"/"obs". Ignored otherwise. Config to build a space
                corresponding to source item in info. See buildSpace for details.
                Format is:
                    {
                        "space": Name of space class (e.g. "MultiDiscrete"),
                        kwargs: kwargs used for the desired space,
                    }
        """
        super().__init__(env)
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        for k in [copy_from, copy_to]:
            assert k in ["info", "obs"]

        if copy_from == "info":
            copy_source = getInfo(env)
        elif copy_from == "obs":
            copy_source = deepcopy(env.observation_space.spaces)

        if copy_to == "info":
            copy_destination = getInfo(env)
        elif copy_to == "obs":
            copy_destination = deepcopy(env.observation_space)

        assert (
            from_key in copy_source
        ), f"{from_key} is not a key in {copy_from}."

        if to_key is None:
            # default key
            to_key = from_key

        if to_key in copy_destination:
            warn(
                f"{to_key} is already in {copy_destination}. Value will be overwritten."
            )

        if copy_from == "info" and copy_to == "obs":
            assert (
                info_space_config is not None
            ), "info_space_config is required if copy from info into obs."

        self.copy_from = copy_from
        self.copy_to = copy_to
        self.from_key = from_key
        self.to_key = to_key

        # build new observation space, if required
        if self.copy_to == "obs":
            new_obs_space = deepcopy(env.observation_space)
            if self.copy_from == "info":
                self.info_item_space = buildSpace(
                    space_config=info_space_config
                )
                new_obs_space[self.to_key] = self.info_item_space
            elif self.copy_from == "obs":
                new_obs_space[self.to_key] = copy_source[self.from_key]

            self.observation_space = new_obs_space

    def reset(
        self, seed: int | None = None, options=None
    ) -> Tuple[OrderedDict, dict]:
        """Copy of info is stored on call (as self.info).

        Args:
            seed (int | None, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            obs (OrderedDict): Observation.
            info (dict): Info.
        """
        obs, info = super().reset(seed=seed, options=options)
        new_item = self._getSourceItem(info=info, obs=obs)
        info_new, obs_new = self._copySourceToDestinationItem(
            info=info,
            obs=obs,
            source_item=new_item,
        )

        # store info for use in self.observation()
        self.info = deepcopy(info_new)

        return obs_new, info_new

    def step(self, action: Any) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Copy entry from unwrapped info to wrapped observation.

        self.info is updated on step.

        Argument unused in this method.
        """
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        self.info = deepcopy(infos)

        new_item = self._getSourceItem(info=infos, obs=observations)
        info_new, obs_new = self._copySourceToDestinationItem(
            info=infos,
            obs=observations,
            source_item=new_item,
        )

        return (obs_new, rewards, terminations, truncations, info_new)

    def observation(self, observation: OrderedDict) -> OrderedDict:
        """Append item to unwrapped observation.

        Args:
            observation (OrderedDict): Unwrapped observation.

        Returns:
            OrderedDict: Same as input, but with item from info, as saved at last
                reset() or step() call.
        """
        if self.copy_to == "obs":
            # only update observation if wrapper modified obs
            info = deepcopy(self.info)
            new_item = self._getSourceItem(info=info, obs=observation)
            _, new_obs = self._copySourceToDestinationItem(
                info=info,
                obs=observation,
                source_item=new_item,
            )

        else:
            new_obs = observation

        return new_obs

    def _getSourceItem(self, info: dict, obs: dict) -> dict:
        """Copy an item from obs or info."""
        if self.copy_from == "info":
            source = deepcopy(info)
        elif self.copy_from == "obs":
            source = deepcopy(obs)

        source_value = source[self.from_key]
        new_item = {self.from_key: source_value}

        return new_item

    def _copySourceToDestinationItem(
        self,
        info: dict,
        obs: dict,
        source_item: dict,
    ) -> Tuple[dict, dict]:
        """Copy source item to either info or dict, return updated info/dict.

        One of the returns (info or obs) is modified. The other return is same
        as input.

        Returns:
            info (dict): info
            obs (dict): observation
        """
        destination_item = {self.to_key: source_item[self.from_key]}
        if self.copy_to == "info":
            info = deepcopy(info)
            info.update(destination_item)
        elif self.copy_to == "obs":
            obs = deepcopy(obs)
            obs.update(destination_item)

        return info, obs


# %% OperatorWrapper
class OperatorWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        obs_or_info: str,
        func_str: str,
        key: str,
        copy_key: str = None,
        a: Any = None,
        b: Any = None,
    ):
        super().__init__(env)
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."

        self.func = OperatorFuncBuilder(func_str=func_str, a=a, b=b)

        self.obs_or_info = obs_or_info
        self.key = key
        self.copy_key = copy_key

    def reset(
        self, seed: int | None = None, options=None
    ) -> Tuple[OrderedDict, dict]:
        obs, info = super().reset(seed=seed, options=options)

        return obs, info
