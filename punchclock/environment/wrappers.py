"""SSAScheduler wrappers module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary, flatten_space
from gymnasium.spaces.utils import flatten
from numpy import (
    all,
    append,
    array,
    diag,
    float32,
    inf,
    int8,
    int64,
    multiply,
    ndarray,
    ones,
    split,
    sum,
    zeros,
)
from sklearn.preprocessing import MinMaxScaler

# Punch Clock Imports
from punchclock.common.custody_tracker import CustodyTracker
from punchclock.environment.wrapper_utils import (
    SelectiveDictProcessor,
    checkDictSpaceContains,
)


# %% Wrappers
class FloatObs(gym.ObservationWrapper):
    """Convert any ints in the Dict observation space to floats.

    Observation space must be a Dict. Recursively searches through observation
    space for int dtypes.
    """

    def __init__(self, env: gym.Env):
        """Wrap environment."""
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.Dict."

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
        """Change all Box dtypes to floats.

        Loop through a dict and convert all Box values that have
        dtype == int into Boxes with dtype = float.
        """
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
        """Convert obs with ints to floats."""
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
        env: gym.Env,
        action_mask_on: bool = True,
    ):
        """Wrapped observation space has "observations" and "action_mask" keys.

        Args:
            env (`gym.Env`): Unwrapped environment with Dict observation
                space that includes (at a minimum) "vis_map_est".
            action_mask_on (`bool`, optional): Whether or not to mask actions.
                Defaults to True.

        Set `action_mask_on` to False to keep all actions unmasked.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.Dict."
        assert (
            "vis_map_est" in env.observation_space.spaces
        ), "vis_map_est must be an item in observation_space"

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


class ActionMaskv2(gym.ObservationWrapper):
    """Add a new item to a Dict observation space called "mask".

    New item is the existing (unwrapped) item provided on instantiation. The copied
        item must be a MultiBinary space.

    Example:
        masker = ActionMaskv2(env, key="a")

        unwrapped_obs = {
            "a": MultiBinary(2),
            "b": Box()
        }

        wrapped_obs = {
            "a": MultiBinary(2),
            "b": Box(),
            "mask": MultiBinary(2)
        }

    """

    def __init__(self, env: gym.Env, key: Any):
        """Wrap env observation space.

        Args:
            env (gym.Env): Must have gym.Dict observation space.
            key (Any): A key in unwrapped observation space.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "Environment observation space is not a Dict."
        assert (
            key in env.observation_space.spaces
        ), f"{key} not in env.observation_space.spaces."
        assert isinstance(
            env.observation_space.spaces[key], (MultiBinary)
        ), f"observation_space[{key}] must be MultiBinary."

        super().__init__(env)
        self.key = key
        self.observation_space = Dict(
            {
                **env.observation_space.spaces,
                "mask": env.observation_space.spaces[key],
            }
        )

    def observation(self, obs: dict) -> dict:
        """Add "mask" to existing keys of obs that is a copy of another item.

        Args:
            obs (dict): Must contain self.key.

        Returns:
            dict: Same as input, but with appended item "mask", which is a copy
                of another item in obs, as specified on class instantiation.
        """
        new_obs = deepcopy(obs)
        new_obs.update({"mask": obs[self.key]})
        return new_obs


class IntersectMask(gym.ObservationWrapper):
    """Layer the unwrapped action mask with another mask from the obs space.

    Specify a key from the unwrapped obs space that contains the new mask to layer
        over observation_space.spaces["action_mask"]. The specified key value must
        be a MultiBinary space of the same shape as "action_mask".
    """

    def __init__(self, env: gym.Env, key: str):
        """Wrap environment with IntersectMask ObservationWrapper."""
        assert isinstance(
            env.observation_space, Dict
        ), "Environment observation space is not a Dict."
        assert (
            "action_mask" in env.observation_space.spaces
        ), "Environment observation space does not contain 'action_mask'."
        assert (
            "observations" in env.observation_space.spaces
        ), "Environment observation space does not contain 'observations'."
        assert (
            key in env.observation_space.spaces["observations"].spaces
        ), f"'{key}' is not in env.observation_space.spaces['observations'].spaces."
        assert isinstance(
            env.observation_space.spaces["observations"].spaces[key],
            MultiBinary,
        ), f"['observations']['{key}'] must be a gym.spaces.MultiBinary."
        mask_shape = env.observation_space.spaces["action_mask"].shape
        layeredmask_shape = (
            env.observation_space.spaces["observations"].spaces[key].shape
        )
        assert (
            mask_shape == layeredmask_shape
        ), f"['observations']['{key}'] must be same shape as 'action_mask'"

        super().__init__(env)
        self.imask_key = key

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Pass unwrapped action mask through another, pre-specified, mask.

        Args:
            obs (OrderedDict): Must be structured like
                {
                    "observations": {
                        "key": binary_array1,
                        ...
                    }
                    "action_mask": binary_array2,
                }

        Returns:
            OrderedDict: Same keys as obs.
        """
        x_mask = obs["action_mask"]
        y_mask = obs["observations"][self.imask_key]
        new_mask = multiply(x_mask, y_mask)

        new_obs = deepcopy(obs)
        new_obs["action_mask"] = new_mask
        return new_obs


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
        keys: list = None,
    ):
        """Flatten sub-levels of a Dict observation space.

        Args:
            env (gym.Env): An environment with a Dict observation space.
            keys (list, optional): List of sub-levels to flatten. If empty,
                all sub-levels are flattened. Defaults to [].
        """
        if keys is None:
            keys = []

        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "env.observation_space must be a gymnasium.Dict."
        assert isinstance(keys, list), "keys must be a list-like."

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
    """Wraps the observation space in a 1-item Dict.

    wrapped_obs_space = Dict({
        "obs": unwrapped_obs_space
    })

    """

    # Mostly useful for tests.

    def __init__(
        self,
        env: gym.Env,
    ):
        """Wrap env."""
        super().__init__(env)
        obs_dict = {"obs": env.observation_space}
        self.observation_space = gym.spaces.Dict(**obs_dict)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Get wrapped observation."""
        obs_new = {"obs": obs}

        return obs_new


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
        rescale_config: dict = None,
    ):
        """Wrap an environment with LinScaleDictObs.

        Args:
            env (gym.Env): Observation space must be a gymDict.
            rescale_config (dict, optional): Keys must be a subset of unwrapped
                observation space keys. Values must be floats. If empty, wrapped
                observation space is same as unwrapped. Defaults to {}.
        """
        # default config
        if rescale_config is None:
            rescale_config = {}
        self.rescale_config = rescale_config

        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), """The input environment to LinScaleDictObs() must have a `gym.spaces.Dict`
             observation space."""
        assert all(
            [
                k in list(env.observation_space.keys())
                for k in rescale_config.keys()
            ]
        ), "Keys of rescale_config must be a subset of keys in env.observation_space."
        assert all(
            [
                isinstance(env.observation_space[space], Box)
                for space in rescale_config.keys()
            ]
        ), """All spaces in env.observation_space that are specified by rescale_config
        must be Box type."""

        super().__init__(env)

        mult_funcs = [
            partial(self.multWrap, mult=m) for m in rescale_config.values()
        ]

        self.processor = SelectiveDictProcessor(
            funcs=mult_funcs, keys=list(rescale_config.keys())
        )

        # Loop through all items in observation_space, check if they are specified
        # by rescale_config, and then rescale the limits of the space. Leave all
        # items in observation_space that are NOT specified in rescale_config as
        # defaults.
        for key, space in env.observation_space.items():
            if key in rescale_config.keys():
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
        new_obs = self.processor.applyFunc(deepcopy(obs))
        return new_obs

    def multWrap(self, x: float, mult: float) -> float:
        """Wrapper for multiplcation."""
        return x * mult


class MinMaxScaleDictObs(gym.ObservationWrapper):
    """MinMax scale entries in a dict observation space.

    Each value in the observation space is scaled by
        X_scaled = X_std * (max - min) + min.

    See sklearn.preprocessing.MinMaxScaler for algorithm details.
    """

    def __init__(self, env: gym.Env):
        """Wrap environment that has a Dict observation space."""
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), """The input environment to MinMaxScaleDictObs() must have a `gym.spaces.Dict`
         observation space."""

        for space in env.observation_space.spaces.values():
            assert isinstance(
                space, gym.spaces.Box
            ), "All spaces in Dict observation space must be a `gym.spaces.Box`."

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
            # set clip=True to prevent occasional out-of-bounds returns from
            # fit_transformation.
            scaled_data = MinMaxScaler(clip=True).fit_transform(v)
            new_v = self.unTransposeArray(scaled_data, flip)
            new_v = self.make1d(new_v, reshaped)
            new_v = new_v.astype(float32)
            new_obs[k] = new_v

        # check that new observation is in bounds and ID key with problem
        if self.observation_space.contains(new_obs) is False:
            contains_report = checkDictSpaceContains(
                self.observation_space, new_obs
            )
            bad_keys = [k for (k, v) in contains_report.items() if v is False]
            raise Exception(
                f"Observation not in observation space. Check keys: \n {bad_keys}"
            )

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
        axes: list[int] = None,
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
        # Defaults
        if len(indices_or_sections) == 1:
            indices_or_sections = [
                indices_or_sections[0] for i in range(len(keys))
            ]
        if axes is None:
            axes = [0]
        if len(axes) == 1:
            axes = [axes[0] for i in range(len(keys))]

        # Type and size checking
        assert isinstance(
            env.observation_space, Dict
        ), "Input environment must have a `gym.spaces.Dict` observation space."
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
        ), "All spaces specified in keys must be Box spaces in unwrapped environment."

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
                indices_or_sections=iors,
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


# %% Classes built from SelectiveDictObsWrapper
class SelectiveDictObsWrapper(gym.ObservationWrapper):
    """Base class for wrappers that apply a function to a selection of Dict entries."""

    def __init__(
        self,
        env: gym.Env,
        funcs: list[Callable],
        keys: list[str],
        new_obs_space: gym.spaces.Space,
    ):
        """Initialize base class and check observation space for correctness.

        Args:
            env (gym.Env): Must have Dict observation space.
            funcs (list[Callable]): List of functions to be paired with list of
                keys. Paired functions will be applied to keys.
            keys (list[str]): List of keys to be operated on by funcs.
            new_obs_space (gym.spaces.Space): Observation space of wrapped env.
        """
        assert isinstance(
            env.observation_space, Dict
        ), """Observation space must be a gym.spaces.Dict."""

        super().__init__(env)
        self.observation_space = new_obs_space

        self.processor = SelectiveDictProcessor(funcs, keys)

        assert self.checkObsSpace(), """Observation not contained in new observation
        space. Check your observation space and/or observation."""

        return

    def observation(self, obs: OrderedDict) -> dict:
        """Get wrapped observation from a Dict observation space."""
        new_obs = self.processor.applyFunc(obs)
        return new_obs

    def checkObsSpace(self) -> bool:
        """Check input observation space for consistency.

        Use this method on instantiation of SelectiveDictObsWrapper to make sure
            that the correct-shaped observation space was input to be compatible
            with wrapped observations.

        Returns:
            bool: True if wrapped observation is contained in wrapped observation
                space.
        """
        obs = self.env.observation_space.sample()
        wrapped_obs = self.observation(obs)
        check_result = self.observation_space.contains(wrapped_obs)
        return check_result


class SumArrayWrapper(SelectiveDictObsWrapper):
    """Sum array(s) along a given dimension for items in a Dict observation space."""

    def __init__(
        self,
        env: gym.Env,
        keys: list[str],
        axis: int | None = None,
    ):
        """Initialize wrapper.

        Args:
            env (gym.Env): A gym environment with a Dict observation space.
            keys (list[str]): Keys whose values will be summed.
            axis (int | None, optional): Axis along which to sum key values. If
                None, all elements of array will be summed. Defaults to None.
        """
        assert all(
            [isinstance(env.observation_space.spaces[k], Box) for k in keys]
        ), "Keys must correspond to Box spaces in env.observation_space."

        funcs = [partial(self.wrapSum, axis=axis)]
        obs_space = Dict(
            {k: deepcopy(v) for (k, v) in env.observation_space.items()}
        )
        for k in keys:
            v = obs_space[k]
            if axis is None:
                # corner case for sum of all elements of array
                new_shape = (1,)
            else:
                # Get new shape by deleting axis-th index of v.shape
                new_shape = list(v.shape)
                del new_shape[axis]
                new_shape = array(new_shape)

            # Even if unwrapped space has bounded lows/highs, summing means that
            # wrapped obs space will have unbounded lows/highs
            obs_space[k] = Box(
                low=-inf,
                high=inf,
                shape=new_shape,
            )

        super().__init__(
            env=env, funcs=funcs, keys=keys, new_obs_space=obs_space
        )

    def wrapSum(self, x: ndarray, axis: int | None) -> ndarray:
        """Wrapper around numpy sum to handle corner case.

        Args:
            x (ndarray): Array to be summed.
            axis (int | None): Axis on which to sum x. If None, all entries of
                x are summed.

        Returns:
            ndarray: Dimensions depends on value of axis. If axis == None, then
                return a (1,) ndarray.
        """
        if axis is None:
            sum_out = sum(x, axis).reshape((1,))
        else:
            sum_out = sum(x, axis)
        return sum_out


class CustodyWrapper(gym.ObservationWrapper):
    """Add 'custody' as an item to a Dict observation space.

    Custody entry entry is a MultiBinary space with shape (N,), where N is the
    number of targets. Number of targets is determined by shape of "est_cov"
    in unwrapped observation space. Does not modify other items in observation
    space, just adds "custody".

    Unwrapped observation space is required to have "est_cov" as an item, which
    must be a Box space with shape == (6, N), where the column values are covariance
    matrix diagonals for the n-th target.
    """

    def __init__(self, env: gym.Env, key: Any, config: dict):
        """Wrap environment with CustodyWrapper observation space wrapper.

        Args:
            env (gym.Env): Must have a Dict observation space with key value
                corresponding to covariance diagonals.
            key (Any): A key contained in the observation space. Key's corresponding
                value must be a (6, N) Box space.
            config (dict): See CustodyTracker for details. The wrapper derives
                most of the args to use in CustodyTracker. The only CustodyTracker
                arg that needs to be provided in config is the arg to
                CustodyTracker(config).
        """
        # Type and shape checking
        assert (
            key in env.observation_space.spaces
        ), f"{key} must be in env.observation_space to use wrapper."
        assert isinstance(
            env.observation_space.spaces[key], Box
        ), f"{key} must be a gym.Box space."
        assert (
            len(env.observation_space.spaces[key].shape) == 2
        ), f"{key} should be a 2d space."
        assert (
            env.observation_space.spaces[key].shape[0] == 6
        ), f"The first dimension of {key} should be 6."

        # make wrapper
        super().__init__(env)
        self.key = key
        num_targets = env.num_targets
        target_names = env.target_ids
        self.custody_tracker = CustodyTracker(
            config=config,
            num_targets=num_targets,
            target_names=target_names,
        )

        # update observation space
        new_space = {**env.observation_space}
        new_space.update({"custody": MultiBinary(num_targets)})
        self.observation_space = Dict(new_space)

    def observation(self, obs: OrderedDict) -> dict:
        """Convert unwrapped observation to wrapped observation.

        Args:
            obs (OrderedDict): Must have self.key as item.

        Returns:
            dict: Same as input dict, but with "custody" item appended. Custody
                is a (N,) binary array where 1 indicates the n-th target is in
                custody.
        """
        new_obs = deepcopy(obs)
        est_cov2d = obs[self.key]
        est_cov3d = self.covFlatTo3d(est_cov2d)
        # custody_tracker outputs custody status as a list of bools; convert to
        # a 1d array of ints. Use int8 for dtype-> this is the default dtype of
        # MultiBinary space.
        custody = array(self.custody_tracker.update(est_cov3d)).astype(int8)
        new_obs.update({"custody": custody})

        assert self.observation_space.contains(new_obs)
        return new_obs

    def covFlatTo3d(self, cov2d: ndarray) -> ndarray:
        """Converts an array of covariance diags to 3D array of diagonal matrices.

        Args:
            cov2d (ndarray): (6, N) Each column is the diagonals of the n-th covariance
                matrix, where the first 3 entries are positional (I, J, K).

        Returns
            ndarray: (N, 6, 6) Diagonal matrices where the diagonal values are
                the columns of cov2d. Off-diagonals are 0.
        """
        cov3d = zeros(shape=(cov2d.shape[1], 6, 6))
        for i in range(cov2d.shape[1]):
            cov3d[i, :, :] = diag(cov2d[:, i])

        return cov3d
