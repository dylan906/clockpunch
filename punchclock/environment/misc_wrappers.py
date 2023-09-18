"""Misc wrappers."""
# %% Imports
# Standard Library Imports
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Tuple
from warnings import warn

# Third Party Imports
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Dict, Space

# Punch Clock Imports
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


# %% AppendInfoItemToObs
class AppendInfoItemToObs(Wrapper):
    """Append an item from info to observation.

    Copy an item from `info` (as returned from env.step() or env.reset()) into
    `observation`. Overwrites existing observation item, if it already exists.

    Environment must have Dict observation space.
    """

    def __init__(
        self,
        env: Env,
        info_key: str,
        info_space_config: dict,
        obs_key: str = None,
    ):
        """Wrap environment with AppendInfoItemToObs wrapper.

        Args:
            env (Env): Gymnasium environment with Dict observation space.
            info_key (str): Must be in info returned by env.step() and env.reset().
            info_space_config (dict): Config to build a space corresponding to
                the desired item in info. See buildSpace for details. Format is:
                    {
                        "space": Name of space class (e.g. "MultiDiscrete"),
                        kwargs: kwargs used for the desired space,
                    }
            obs_key (str, optional): Key assigned to value copied from info to
                observation. If None, info_key will be used. Defaults to None.
        """
        super().__init__(env)
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."

        env_copy = deepcopy(env)
        _, info = env_copy.reset()
        assert info_key in info, f"{info_key} is not a key in `info`."

        if obs_key is None:
            # default key
            obs_key = info_key

        if obs_key in env.observation_space.spaces:
            warn(
                f"{obs_key} is already in observation space. Value will be overwritten."
            )

        self.info_key = info_key
        self.obs_key = obs_key

        # build new observation space
        self.info_item_space = buildSpace(space_config=info_space_config)
        new_obs_space = deepcopy(env.observation_space)
        new_obs_space[self.obs_key] = self.info_item_space
        self.observation_space = new_obs_space

    def reset(
        self, seed: int | None = None, options=None
    ) -> Tuple[OrderedDict, dict]:
        """Reset env."""
        obs, info = super().reset(seed=seed, options=options)
        new_item = self._getAppendInfoItem(info)
        obs = deepcopy(obs)
        obs.update(new_item)
        return obs, info

    def step(self, action: Any) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Copy entry from unwrapped info to wrapped observation.

        Argument unused in this method.
        """
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        new_obs = self._getAppendInfoItem(infos)
        observations = deepcopy(observations)
        observations.update(new_obs)

        return (observations, rewards, terminations, truncations, infos)

    def _getAppendInfoItem(self, info: dict) -> dict:
        """Copy the item (specified at instantiation) from info."""
        info = deepcopy(info)
        value = deepcopy(info[self.info_key])
        obs_item = {self.info_key: value}
        return obs_item
