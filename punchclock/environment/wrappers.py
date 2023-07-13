"""SSAScheduler wrappers module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import partial

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete, flatten_space, unflatten
from gymnasium.spaces.utils import flatten
from numpy import (
    append,
    float32,
    inf,
    int64,
    multiply,
    ndarray,
    ones,
    reshape,
    split,
    zeros,
)
from sklearn.preprocessing import MinMaxScaler

# Punch Clock Imports
from punchclock.environment.env import SSAScheduler
from punchclock.reward_funcs.reward_utils import cropArray


# %% Wrappers
class FloatObs(gym.ObservationWrapper):
    """Convert any ints in the observation space to floats."""

    def __init__(self, env: SSAScheduler):
        super().__init__(env)
        self.observation_space = self._recursiveConvertDictSpace(
            env.observation_space
        )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Transform obs returned by base env before passing out from wrapped env."""
        obs_new = self._recursiveConvertDict(obs)

        return obs_new

    def _recursiveConvertDictSpace(
        self, obs_space: gym.spaces.Dict
    ) -> OrderedDict:
        """Loop through a `dict` and convert all `Box` values that have
        dtype == `int` into `Box`es with dtype = `float`."""

        obs_space_new = gym.spaces.Dict({})
        for k, v in obs_space.items():
            # recurse if entry is a Dict
            if isinstance(v, gym.spaces.Dict):
                obs_space_new[k] = self._recursiveConvertDictSpace(v)
            # assign as-is if entry already is already a float
            elif isinstance(v, gym.spaces.Box) and v.dtype == float32:
                obs_space_new[k] = v
            # replace entry with new Box w/ dtype = float
            else:
                list_of_attrs = ["low", "high", "shape"]
                kwargs = {key: getattr(v, key) for key in list_of_attrs}
                kwargs["dtype"] = float32
                obs_space_new[k] = gym.spaces.Box(**kwargs)

        return obs_space_new

    def _recursiveConvertDict(self, obs: OrderedDict) -> OrderedDict:
        """Convert obs with `int`s to `float`s."""
        obs_new = OrderedDict({})
        for k, v in obs.items():
            # recurse if entry is a Dict
            if isinstance(v, OrderedDict):
                obs_new[k] = self._recursiveConvertDict(v)
            # assign as-is if entry is already a float
            elif v.dtype == float32:
                obs_new[k] = v
            # replace array of ints with floats
            else:
                obs_new[k] = v.astype(float32)

        return obs_new


# %% Wrapper for action mask
class ActionMask(gym.ObservationWrapper):
    """Mask invalid actions based on estimated sensor-target visibility.

    Observation space is an `OrderedDict` with the following structure:
        {
            "observations" (`gym.spaces.Dict`): Same space as
                env.observation_space["observations"]. Includes "vis_map_est",
                which is a (N, M) array.
            "action_mask" (`gym.spaces.Box`): A flattened version of
                env.observation_space["observations"]["vis_map_est"] with shape
                ( (N+1) * M, ). This is also the same as flatten_space(env.action_space),
                assuming action_space is `MultiDiscrete`.
        }
    """

    def __init__(
        self,
        env: SSAScheduler,
        action_mask_on: bool = True,
    ):
        """Wrapped observation space is a dict with "observations" and "action_mask" keys.

        Args:
            env (`SSAScheduler`): Unwrapped environment with Dict observation
                space that includes (at a minimum) "vis_map_est".
            action_mask_on (`bool`, optional): Whether or not to mask actions.
                Defaults to True.

        Set `action_mask_on` to False to keep all actions unmasked.
        """
        super().__init__(env)
        self.action_mask_on = action_mask_on
        self.mask_space = flatten_space(self.action_space)
        self.observation_space = gym.spaces.Dict(
            {
                "observations": env.observation_space,
                "action_mask": self.mask_space,
            }
        )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Convert unwrapped observation to `ActionMask` observation.

        Args:
            obs (`OrderedDict`): Unwrapped observation `dict`. Must contain
                "vis_map_est" key. Value of "vis_map_est" is a (N, M) array.

        Returns:
            `OrderedDict`: Output obs is
                {
                    "observations" (`OrderedDict`): The same as the input obs,
                    "action_mask" (`ndarray[int]`): obs["vis_map_est"] with
                        shape ( (N+1) * M, ). Values are 0 or 1.
                }
                If ActionMask.action_mask_on == False, all "action_mask" values
                    are 1.
        """
        # Append row of ones to mask to account for inaction (which is never masked).
        # Then transpose the mask, then flatten. Transpose _before_ flattening is
        # necessary to play nice with MultiDiscrete action space.
        mask = obs["vis_map_est"]
        m = mask.shape[1]
        mask = append(mask, ones(shape=(1, m), dtype=int64), axis=0)
        mask_flat = gym.spaces.flatten(self.mask_space, mask.transpose())

        if self.action_mask_on is True:
            obs_mask = mask_flat
        else:
            # Get pass-thru action mask (no actions are masked)
            obs_mask = ones(shape=mask_flat.shape, dtype=int64)

        obs_new = OrderedDict(
            {
                "observations": obs,
                "action_mask": obs_mask,
            }
        )

        return obs_new


# %% Wrapper for flattening part of observation space
class FlatDict(gym.ObservationWrapper):
    """Flatten entries of a Dict observation space, leaving the top level unaffected.

    Unwrapped environment must have a Dict observation space.

    Can selectively flatten entries in a Dict observation space with the keys
        arg.
    """

    def __init__(
        self,
        env: gym.Env,
        keys: list[str] = [],
    ):
        """Flatten sub-levels of a Dict observation space.

        Args:
            env (gym.Env): An environment with a Dict observation space.
            keys (list[str], optional): List of sub-levels to flatten. If empty,
                all sub-levels are flattened. Defaults to [].
        """
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), f"""The input environment to FlatDict() must have a `gym.spaces.Dict` 
        observation space."""
        assert isinstance(keys, list), f"keys must be a list."
        super().__init__(env)

        # replace empty list of keys with all keys by default
        if len(keys) == 0:
            keys = list(env.observation_space.spaces.keys())

        # Each space is flattened by the same basic function, gym.spaces.utils.flatten,
        # but with a different arg for "space". So make a unique partial function
        # for each space. This way, only the new observation (space) needs to be
        # passed into the flatten function each time, instead of both the observation
        # and the original space.
        relevant_spaces = [env.observation_space.spaces[k] for k in keys]
        flatten_funcs = [partial(flatten, s) for s in relevant_spaces]
        self.preprocessor = SelectiveDictProcessor(flatten_funcs, keys)

        # Redefine the observation space with new flattened spaces.
        space_preprocessor = SelectiveDictProcessor([flatten_space], keys)
        new_obs_space = space_preprocessor.applyFunc(
            env.observation_space.spaces
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Flatten items in Dict observation space, leaving top level intact.

        Args:
            obs (`OrderedDict`): Observation

        Returns:
            `gym.spaces.Dict`: All entries below top level are flattened.
        """
        obs_new = self.preprocessor.applyFunc(obs)
        obs_new = OrderedDict(obs_new)

        return obs_new


class MakeDict(gym.ObservationWrapper):
    """Converts a non-dict observation to a dict."""

    # Mostly useful for tests.

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        obs_dict = {"obs": env.observation_space}
        self.observation_space = gym.spaces.Dict(**obs_dict)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        obs_new = {"obs": obs}

        return obs_new


class FlattenMultiDiscrete(gym.ActionWrapper):
    """Convert `Box` action space to `MultiDiscrete`.

    Converts `Box` action to `MultiDiscrete` action before passing to base environment.
    Input action must be shape (A * B,) `ndarray` of 0s and 1s and conform to format
    of a flattened `MultiDiscrete` space, where A is the number of possible actions
    in a group, and B is the number of groups. Only a single 1 in each group of
    A entries is allowed.

    Examples:
        A, B |      input_act     | output_act
        -----|--------------------|------------
        2, 2 | [0, 1, 1, 0]       | [1, 0]
        2, 3 | [1, 0, 1, 0, 0, 1] | [0, 0, 1]
        3, 2 | [1, 0, 0, 0, 0, 1] | [0, 2]
    """

    def __init__(self, env):
        """Environment must have `MultiDiscrete` action space."""
        super().__init__(env)
        assert isinstance(env.action_space, MultiDiscrete)
        self.action_space = flatten_space(env.action_space)

    def action(self, act: ndarray[int]) -> ndarray[int]:
        """Convert `Box` action to `MultiDiscrete` action.

        Args:
            act (`ndarray[int]`): Action in `Box` space.

        Returns:
            `ndarray[int]`: Action in `MultiDiscrete` space.
        """
        try:
            x = unflatten(self.env.action_space, act)
        except ValueError:
            # print("Error in unflattening Box action to MultiDiscrete")
            raise (
                ValueError("Error in unflattening Box action to MultiDiscrete")
            )

        return x


# %% Rescale obs
class LinScaleDictObs(gym.ObservationWrapper):
    """Rescale selected entries in a Dict observation space.

    Items in unwrapped observation that have common keys with rescale_config are
    multiplied by values in rescale_config.

    Example:
        # state_a_wrapped = 1e-4 * state_a_unwrapped

        unwrapped_obs = {
            "state_a": 2.0,
            "state_b": 2.0
        }

        rescale_config = {
            "state_a": 1e-4
        }

        wrapped_obs = {
            "state_a": 2e-4,
            "state_b": 2.0
        }
    """

    def __init__(
        self,
        env: gym.Env,
        rescale_config: dict = {},
    ):
        """Wrap an environment with LinScaleDictObs.

        Args:
            env (`gym.Env`): Observation space must be a `gymDict`.
            rescale_config (`dict`, optional): Keys must be a subset of unwrapped
                observation space keys. Values must be `float`s. If empty, wrapped
                observation space is same as unwrapped. Defaults to {}.
        """
        assert isinstance(env.observation_space, gym.spaces.Dict), (
            f"The input environment to LinScaleDictObs() must have a `gym.spaces.Dict`"
            f" observation space."
        )

        super().__init__(env)

        self.rescale_config = rescale_config

        # Loop through all items in observation_space, check if they are specified
        # by rescale_config, and then rescale the limits of the space. This only
        # works for Box environments, so check if the entries in observation_space
        # are Box before changing them. Leave all items in observation_space that
        # are NOT specified in rescale_config as defaults.
        for key, space in env.observation_space.items():
            if key in rescale_config.keys():
                assert isinstance(
                    space, Box
                ), f"LinScaleDictObs only works with Dict[Box] spaces."

                new_low = space.low * rescale_config[key]
                new_high = space.high * rescale_config[key]

                self.observation_space[key] = Box(
                    low=new_low,
                    high=new_high,
                )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Get a scaled observation.

        Args:
            obs (`OrderedDict`): Unwrapped observation.

        Returns:
            `OrderedDict`: Rescaled input observations, as specified by
                self.rescale_config.
        """
        new_obs = deepcopy(obs)
        for key, val in obs.items():
            if key in self.rescale_config.keys():
                new_obs[key] = val * self.rescale_config[key]

        return new_obs


class MinMaxScaleDictObs(gym.ObservationWrapper):
    """MinMax scale entries in a dict observation space.

    Each value in the observation space is scaled by
        X_scaled = X_std * (max - min) + min.

    See sklearn.preprocessing.MinMaxScaler for algorithm details.
    """

    def __init__(self, env: gym.Env):
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), f"""The input environment to MinMaxScaleDictObs() must have a `gym.spaces.Dict` 
        observation space."""

        for space in env.observation_space.spaces.values():
            assert isinstance(
                space, gym.spaces.Box
            ), f"""All spaces in Dict observation space must be a `gym.spaces.Box`."""

        super().__init__(env)

        # Update wrapper observation_shape. Set all lows/highs to 0/1.
        # NOTE: Any subspace with dtype==int will be changed to float.
        new_obs_space = {}
        for k, space in env.observation_space.spaces.items():
            new_space = Box(
                low=zeros(space.low.shape),
                high=ones(space.high.shape),
                shape=space.shape,
            )
            new_obs_space.update({k: new_space})
        self.observation_space = Dict(new_obs_space)
        return

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Rescale each entry in obs by MinMax algorithm.

        Args:
            obs (OrderedDict): Values must be arrays.

        Returns:
            OrderedDict: Scaled version of obs. Keys are same.
        """
        # MinMaxScaler scales along the 0th dimension (vertical). Dict values are
        # not guaranteed to be 2d or, if 1d, vertical. So need to flip horizontal
        # arrays prior to transforming via MinMaxScaler.
        # If a 1d array is handed in, need to convert to 2s for MinMaxScaler to
        # work. Then need to convert back to 1d before passing back out.
        # Convert dtypes to float32 to match observation_space (float32 is default
        # dtype for gym Box spaces).

        new_obs = {}
        for k, v in obs.items():
            v, reshaped = self.make2d(v)
            v, flip = self.transposeHorizontalArray(v)
            scaler = MinMaxScaler().fit(v)
            new_v = self.unTransposeArray(scaler.transform(v), flip)
            new_v = self.make1d(new_v, reshaped)
            new_v = new_v.astype(float32)
            new_obs[k] = new_v

        # check that new observation is in bounds
        assert self.observation_space.contains(new_obs)

        return new_obs

    def make2d(self, x: ndarray) -> tuple[ndarray, bool]:
        """Make a 1d array into a 2d array with a singleton 0th dimension.

        Do nothing to 2d arrays.
        """
        reshaped = False
        if x.ndim == 1:
            x = x.reshape((-1, 1))
            reshaped = True

        return x, reshaped

    def make1d(self, x: ndarray, reshaped: bool) -> ndarray:
        """Make an array 1d if reshaped == True, do nothing otherwise."""
        if reshaped is True:
            x = x.reshape((-1))

        return x

    def transposeHorizontalArray(self, x: ndarray) -> tuple[ndarray, bool]:
        """Transpose 1d horizontal array, do nothing otherwise.

        Returns a tuple where the first value is the the array, and the 2nd value
        is a flag that is True if the input array was transposed.
        """
        transposed = False
        if x.shape[0] == 1:
            x = x.transpose()
            transposed = True
        return x, transposed

    def unTransposeArray(self, x: ndarray, trans: bool) -> ndarray:
        """Transpose x if trans is True; return x."""
        if trans is True:
            x = x.transpose()
        return x


class SplitArrayObs(gym.ObservationWrapper):
    """Split array entries in a Dict observation space into multiple entries.

    Example:
        wrapped_env = SplitArrayObs(env, keys=["state_a"],
            new_keys=["state_a1", "state_a2",],
            indices_or_sections=[2],
            )

        unwrapped_obs = {
            "state_a": array([1, 2, 3, 4])
        }

        wrapped_obs = wrapped_env.observation(unwrapped_obs)

        # output
        wrapped_obs = {
            "state_a1": array([1, 2]),
            "state_a2": array([3, 4])
        }

    See numpy.split for details on indices_or_sections and axes args.
    """

    def __init__(
        self,
        env: gym.Env,
        keys: list[str],
        new_keys: list[list[str]],
        indices_or_sections: list[int | ndarray],
        axes: list[int] = [0],
    ):
        """Initialize SplitArrayObs wrapper.

        Args:
            env (gym.Env): Must have a Dict observation space.
            keys (list[str]): (A-long) List of keys in unwrapped observation to
                be replaced with new_keys.
            new_keys (list[list[str]]): (A-long) Nested list of new keys to replace
                original keys with. Outer level must be A-long, inner level lengths
                must be consistent with indices_or_sections.
            indices_or_sections (list[int  |  ndarray]): (A-long | 1-long) Number
                of segments to split arrays into. See numpy.split for details.
                If 1-long, all arrays will be split into same number of segments.
            axes (list[int], optional): (A-long | 1-long) Axes to perform array
                splits on. See numpy.split for details. If 1-long, all arrays will
                be split on same axis. Defaults to [0].
        """
        # Type and size checking
        assert isinstance(
            env.observation_space, Dict
        ), f"""Input environment must have a `gym.spaces.Dict` observation space."""
        assert len(keys) == len(new_keys), "len(keys) must equal len(new_keys)"
        assert (len(indices_or_sections) == len(keys)) or (
            len(indices_or_sections) == 1
        ), "len(indices_or_sections) must be 1 or same as len(keys)"
        assert (len(axes) == len(keys)) or (
            len(axes) == 1
        ), """len(axes) must be 1 or same as len(keys)"""
        assert all(
            [k in env.observation_space.spaces for k in keys]
        ), """All entries in keys must be in unwrapped observation space."""
        relevant_spaces = [env.observation_space.spaces[k] for k in keys]
        assert all(
            [isinstance(space, Box) for space in relevant_spaces]
        ), """All spaces specified in keys must be Box spaces in unwrapped environment."""

        # Defaults
        if len(indices_or_sections) == 1:
            indices_or_sections = [
                indices_or_sections[0] for i in range(len(keys))
            ]
        if len(axes) == 1:
            axes = [axes[0] for i in range(len(keys))]

        # %% Set attributes
        super().__init__(env)
        self.keys = keys
        self.new_keys = new_keys
        self.indices_or_sections = indices_or_sections
        self.axes = axes
        self.key_map = {k: v for (k, v) in zip(keys, new_keys)}

        # Redefine env.observation_space
        self.observation_space = self.buildNewObsSpace(
            self.key_map,
            env.observation_space,
            indices_or_sections,
            axes,
        )

    def buildNewObsSpace(
        self,
        key_map: dict,
        obs_space: gym.spaces.Dict,
        indices_or_sections: list[int | ndarray],
        axes: list[int],
    ) -> Dict:
        """Redefine wrapped env observation space."""
        # New observation space will have more items than old (unwrapped) observation
        # space. Need to add new items and make sure they are the correct size
        # (which will be different than unwrapped).
        new_obs_space = Dict(
            {k: deepcopy(v) for (k, v) in self.observation_space.items()}
        )
        for (key, new_keys), iors, ax in zip(
            key_map.items(),
            indices_or_sections,
            axes,
        ):
            original_subspace = obs_space[key]
            original_high = original_subspace.high[0, 0]
            original_low = original_subspace.low[0, 0]
            original_dtype = original_subspace.dtype
            split_subspaces = split(
                original_subspace.sample(),
                indices_or_sections=indices_or_sections,
                axis=ax,
            )
            for space, nk in zip(split_subspaces, new_keys):
                space_shape = space.shape
                new_obs_space[nk] = Box(
                    low=original_low,
                    high=original_high,
                    shape=space_shape,
                    dtype=original_dtype,
                )
            new_obs_space.spaces.pop(key)

        return new_obs_space

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Get wrapped observation.

        Args:
            obs (OrderedDict): Has original (unwrapped) keys.

        Returns:
            OrderedDict: Replaces some unwrapped keys (as specified during instantiation)
                with new keys (also as specified during instantiation).
        """
        # Deepcopy is needed so original obs isn't modified
        new_obs = deepcopy(obs)
        for (key, new_keys), i, axis in zip(
            self.key_map.items(),
            self.indices_or_sections,
            self.axes,
        ):
            # After splitting observation, the ob is now a list of multiple arrays.
            # So we assign each array to a new item in the new obs dict. Also have
            # to delete the replaced key.
            split_ob = split(obs[key], indices_or_sections=i, axis=axis)
            ob = {k: v for (k, v) in zip(new_keys, split_ob)}
            new_obs.pop(key)
            new_obs.update(ob)

        return new_obs


# %% Utils
def getNumWrappers(env: gym.Env, num: int = 0) -> int:
    """Get the number of wrappers around a Gym environment.

    Recursively looks through a wrapped environment, then reports out the number
    of times it recursed, which is the number of wrappers around the base environment.
    Returns 0 if base environment is passed in.

    Args:
        env (`gym.Env`): Can be wrapped or a bse environment.
        num (`int`, optional): Number of wrappers above input env (usually set
            to 0 on initial call). Defaults to 0.
    """
    if hasattr(env, "env"):
        num += 1
        num = getNumWrappers(env.env, num)

    return num


def getWrapperList(env: gym.Env, wrappers: list = None) -> list:
    """Get list of wrappers from a multi-wrapped environment."""
    if wrappers is None:
        wrappers = []

    if isinstance(env, gym.Wrapper):
        wrappers.append(type(env))
        wrappers = getWrapperList(env.env, wrappers)

    return wrappers


class SelectiveDictProcessor:
    """Apply functions to a subset of items in a dict.

    Attributes:
        keys (list): All the keys that are operated on when self.applyFunc()
            is called. Each function is applied to the value associated with the
            key (not the key itself).
        funcs (list[Callable]): The functions that are applied to the items specified
            by keys.
        key_func_map (dict): The mapping of keys to funcs. Funcs are mapped to
            keys by the input order of the lists used at instantiation.
    """

    def __init__(self, funcs: list[Callable], keys: list[str]):
        """Initialize SelectiveDictProcessor.

        Args:
            funcs (list[Callable]): Entries are paired with keys and applied
                correspondingly.
            keys (list[str]): List of keys in input dict to self.applyFunc().

        The i-th func is applied to the i-th key value.

        If a funcs has only 1 entry, then the same function is applied to all
            keys.
        """
        assert isinstance(funcs, list)
        assert isinstance(keys, list)
        assert all(isinstance(key, str) for key in keys)
        if len(funcs) == 1:
            funcs = [funcs[0] for i in range(len(keys))]

        self.keys = keys
        self.funcs = funcs
        self.key_func_map = {k: f for (k, f) in zip(keys, funcs)}

    def applyFunc(self, in_dict: dict, **kwargs) -> dict:
        """Applies function to previously defined items in in_dict.

        Args:
            in_dict (dict): Must include keys specified when SelectiveDictProcessor
                was instantiated.

        Returns:
            dict: Same keys as in_dict. Some/all values (as specified at instantiation)
                have self.funcs applied to them.
        """
        processed_dict = {}
        for k, v in in_dict.items():
            if k in self.keys:
                # processed_dict[k] = self.funcs(v, **kwargs)
                processed_dict[k] = self.key_func_map[k](v, **kwargs)

        out_dict = deepcopy(in_dict)
        out_dict.update(processed_dict)

        return out_dict
