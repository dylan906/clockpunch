"""SSAScheduler observation wrappers module."""
# %% Imports
# Standard Library Imports
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, Tuple

# Third Party Imports
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import (
    Box,
    Dict,
    MultiBinary,
    MultiDiscrete,
    Space,
    flatten_space,
)
from gymnasium.spaces.utils import flatten
from numpy import (
    Inf,
    all,
    append,
    array,
    asarray,
    clip,
    concatenate,
    diag,
    diagonal,
    float32,
    inf,
    int8,
    int64,
    int_,
    multiply,
    ndarray,
    ones,
    ravel,
    split,
    squeeze,
    stack,
    sum,
    zeros,
)
from sklearn.preprocessing import MinMaxScaler

# Punch Clock Imports
from punchclock.common.custody_tracker import CustodyTracker
from punchclock.environment.wrapper_utils import (
    SelectiveDictObsWrapper,
    SelectiveDictProcessor,
    checkDictSpaceContains,
    convertBinaryBoxToMultiBinary,
    convertNumpyFuncStrToCallable,
    getSpaceClosestCommonDtype,
    getSpacesRange,
    remakeSpace,
)


# %% FloatObs
class FloatObs(gym.ObservationWrapper):
    """Convert any ints in the Dict observation space to floats.

    Observation space must be a Dict. Recursively searches through observation
    space for int dtypes.
    """

    def __init__(self, env: Env):
        """Wrap environment."""
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."

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
        """Convert fundamental spaces to Box with dtype == float.

        Loop through a dict and convert all Box values that have dtype == int and
        Multibinary values into Boxes with dtype = float.
        """
        obs_space_new = gym.spaces.Dict({})
        for k, v in obs_space.items():
            if isinstance(v, gym.spaces.Dict):
                # recurse if entry is a Dict
                obs_space_new[k] = self._recursiveConvertDictSpace(v)
            elif isinstance(v, gym.spaces.Box) and v.dtype == float32:
                # assign as-is if entry already is already a float
                obs_space_new[k] = v
            elif isinstance(v, MultiBinary):
                # Convert MultiBinary into Box with 0/1 low/high
                obs_space_new[k] = Box(low=0, high=1, shape=v.shape)
            else:
                # replace entry with new Box w/ dtype = float
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


# %% NestObsItems
class NestObsItems(gym.ObservationWrapper):
    """Nest item(s) in a Dict observation space within a Dict.

    Nested keys are placed at end of OrderedDict observation space. Order of nested
    space is maintained from original space.

    Example:
        nest_env = NestObsItems(unwrapped_env, new_key="foo", keys_to_nest=["a", "c"])

        unwrapped_obs_space = OrderedDict({
            "a": Discrete(3),
            "b": Discrete(2),
            "c": Discrete(1)
        })

        wrapped_obs_space = OrderedDict({
            "b": Discrete(2),
            "foo": Dict({
                "a": Discrete(3),
                "c": Discrete(1),
            })
        })
    """

    def __init__(
        self,
        env: Env,
        new_key: Any = None,
        keys_to_nest: list = None,
        reverse: bool = False,
    ):
        """Wrap environment with NestObsItems ObservationWrapper.

        Args:
            env (Env): Must have a Dict observation space.
            new_key (Any, optional): The name of the new item that will be added
                to the top level of the observation space. Defaults to "new_key".
            keys_to_nest (list, optional): Keys of items in unwrapped observation
                space that will be combined into a Dict space under new_key. If
                None, all items in observation space will be nested under single
                item. Defaults to None.
            reverse (bool, optional): If True, keys NOT in keys_to_nest will be
                nested. Items in keys_to_nest will remain at top level of dict.
                Defaults to False.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "Environment observation space is not a Dict."

        if keys_to_nest is None:
            # default to nesting all items
            keys_to_nest = env.observation_space.spaces.keys()

        assert all(
            [k in env.observation_space.spaces.keys() for k in keys_to_nest]
        ), "All entries in keys_to_nest must be in observation space."

        super().__init__(env)

        if new_key is None:
            new_key = "new_key"
        self.new_key = new_key
        self.reverse = reverse

        spaces_in_list, spaces_notin_list = self._getNestedSpaces(
            env=env, keys=keys_to_nest
        )

        if reverse is False:
            self.keys_to_nest = keys_to_nest
            self.observation_space = self._createNewObsSpace(
                unnested_spaces=spaces_notin_list,
                nested_spaces=spaces_in_list,
                nest_key=new_key,
            )
        elif reverse is True:
            # reverse keys_to_nest from arg
            self.keys_to_nest = [k for k in spaces_notin_list.keys()]
            self.observation_space = self._createNewObsSpace(
                unnested_spaces=spaces_in_list,
                nested_spaces=spaces_notin_list,
                nest_key=new_key,
            )

    def _getNestedSpaces(
        self, env: Env, keys: list[str]
    ) -> tuple[OrderedDict[Space], OrderedDict[Space]]:
        """Get Spaces from env observation space that are in and not in keys.

        Args:
            env (Env): A Gym environment with a Dict obs space.
            keys (list[str]): List of keys; all must be in obs space.

        Returns:
            spaces_in_list (OrderedDict[Space]): Spaces from
                env.observation_space.spaces that are in keys.
            spaces_notin_list (OrderedDict[Space]): Spaces from
                env.observation_space.spaces that are NOT in keys.
        """
        spaces_in_list = OrderedDict({})
        for k, v in env.observation_space.spaces.items():
            if k in keys:
                spaces_in_list[k] = v

        spaces_notin_list = OrderedDict({})
        for k, v in env.observation_space.items():
            if k not in keys:
                spaces_notin_list[k] = v

        return spaces_in_list, spaces_notin_list

    def _createNewObsSpace(
        self,
        unnested_spaces: OrderedDict[Space],
        nested_spaces: OrderedDict[Space],
        nest_key: str,
    ):
        """Create a Dict space with a list of nested/unnested spaces."""
        new_obs_space = Dict(
            {
                **unnested_spaces,
                nest_key: Dict(nested_spaces),
            }
        )

        return new_obs_space

    def observation(self, obs: dict) -> OrderedDict:
        """Nest item(s) from obs into a new item, leave other items at top.

        Args:
            obs (dict): Order will be maintained. Must be contained in
                self.observation_space.

        Returns:
            OrderedDict: One or more items from unwrapped observation will be nested
                under a single item, with key specified on class instantiation.
                Un-nested items are first, nested item is last.
        """
        nested_obs = OrderedDict({})
        unnested_obs = OrderedDict({})
        for k, v in obs.items():
            if k in self.keys_to_nest:
                nested_obs[k] = v
            else:
                unnested_obs[k] = v

        new_obs = OrderedDict({**unnested_obs, self.new_key: {**nested_obs}})

        return new_obs


# %% VisMap2ActionMask
class VisMap2ActionMask(gym.ObservationWrapper):
    """Convert visibility map within an observation space into a 2d action mask.

    Append a row of 1's to the bottom of a visibility map.

    Set action_mask_on == False to make the modified observation space item always
    an array of 1s.

    Example (B = 2):
        env.observation_space = {
            "vis_map": MultiBinary((A, B))
        }

        env.action_space = MultiDiscrete([A+1, A+1])

        wrapped_env = VisMap2ActionMask(env,
            vis_map_key="vis_map",
            rename_key="action_mask")

        wrapped_env.observation_space = {
            "action_mask": MultiBinary((A+1, B))
        }

    """

    def __init__(
        self,
        env: Env,
        vis_map_key: str,
        rename_key: str = None,
        action_mask_on: bool = True,
    ):
        """Wrap environment with VisMap2ActionMask.

        Args:
            env (Env): Must have:
                - Dict observation space
                - MultiDiscrete action space
                - vis_map_key must be in observation space
                - observation_space[vis_map_key] must be a 2d MultiBinary
                - Number of columns in observation_space[vis_map_key] must be
                    same as length of action space.
            vis_map_key (str): An item in observation space.
            rename_key (str, optional): Optionally rename vis_map_key in wrapped
                observation space. If None, key will not be changed. Defaults to
                None.
            action_mask_on (bool, optional): If False, all values in wrapped
                observation[rename_key] will be 1. Otherwise, will be copies of
                vis_map_key. Defaults to True.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gym.spaces.Dict."
        assert (
            vis_map_key in env.observation_space.spaces
        ), "vis_map_key must be in observation space."
        assert isinstance(
            env.observation_space.spaces[vis_map_key], MultiBinary
        ), f"observation_space[{vis_map_key}] must be a gym.spaces.MultiBinary."
        assert (
            len(env.observation_space.spaces[vis_map_key].shape) == 2
        ), f"observation_space[{vis_map_key}] must be 2d."
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a gymnasium.spaces.MultiDiscrete."

        vis_map_shape = env.observation_space.spaces[vis_map_key].shape
        assert vis_map_shape[1] == len(
            env.action_space.nvec
        ), """Shape mismatch between action space and selected item in observation
        space. The number of columns in observation_space[vis_map_key] must be equal
        to length of action_space.nvec."""

        super().__init__(env)

        if rename_key is None:
            rename_key = vis_map_key

        self.vis_map_key = vis_map_key
        self.rename_key = rename_key
        self.action_mask_on = action_mask_on
        num_rows, num_cols = env.observation_space[vis_map_key].shape
        self.mask_space = MultiBinary((num_rows + 1, num_cols))

        # Maintain same order of obs dict
        new_obs_space = OrderedDict({})
        for k, space in env.observation_space.items():
            if k == vis_map_key:
                new_obs_space[rename_key] = self.mask_space
            else:
                new_obs_space[k] = space
        self.observation_space = Dict(new_obs_space)

    def observation(self, obs: dict) -> OrderedDict:
        """Generate wrapped observation.

        Args:
            obs (dict): Must have self.vis_map_key in keys.

        Returns:
            OrderedDict: Same as input obs except for modified rename_key (if
                rename_key was provided on instantiation).

        Example (num_sensors = 2, num_targets = 3):
            obs = OrderedDict({"vis_map": array([[1, 0],
                                                 [0, 0],
                                                 [0, 0]])})
            action_mask = VisMap2ActionMask.observation(obs)
            # action_mask = array([[1, 0],
            #                      [0, 0],
            #                      [0, 0],
            #                      [1, 1]])  <- inaction always 1 (valid)
        """
        mask = obs[self.vis_map_key]
        m = mask.shape[1]
        mask = append(mask, ones(shape=(1, m), dtype=int64), axis=0)

        if self.action_mask_on is False:
            # Get pass-thru action mask (no actions are masked)
            mask = ones(shape=mask.shape, dtype=int64)

        # Maintain same order of obs dict
        obs_new = OrderedDict()
        for k, v in obs.items():
            if k == self.vis_map_key:
                obs_new[self.rename_key] = mask
            else:
                obs_new[k] = v

        return obs_new


# %% MultiplyObsItems
class MultiplyObsItems(gym.ObservationWrapper):
    """Element-by-element multiply entries from a Dict observation space.

    Specify n>1 keys in the unwrapped observation space. The values associated
        with the keys will be multiplied element-wise. All keys.values must be
        arrays.

    If specified, the resultant array will be appended to end of (ordered) Dict
        observation space. Otherwise the multiplied array replaced the value associated
        with keys[0].

    Example:
        env.observation_space = Dict(
            {
                "a1": MultiBinary(4),  # has same shape as "a2" even though
                                       # different spaces
                "a2": Box(0, 1, shape=(4,), dtype=int),
            }
        )

        env_wrapped = MultiplyObsItems(env, keys=["a1", "a2"], new_key="foo")

        env_wrapped.observation_space = Dict(
            {
                "a1": MultiBinary(4),
                "a2": Box(0, 1, shape=(4,), dtype=int),
                "foo": Box(0, 1, shape=(4,), dtype=int),
            }
        )
    """

    def __init__(self, env: Env, keys: list, new_key: str = None):
        """Wrap environment observation space.

        Args:
            env (Env): Gym environment.
            keys (list): List of keys. Keys must be in env.observation_space.
            new_key (str, optional): Key of new item in observation space where
                return value will be placed. If None, the value of
                observation_space.spaces[keys[0]] will be overridden. Defaults
                to None.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "Environment observation space is not a Dict."
        assert len(keys) > 1, "keys must have length > 1."
        for k in keys:
            assert (
                k in env.observation_space.spaces
            ), f"'{k}' is not in env.observation_space."

        shapes = [env.observation_space.spaces[k].shape for k in keys]
        assert all(
            [(a == shapes[0]) for a in shapes]
        ), "Shape of observation_space[keys] must be same for all keys."

        if new_key is None:
            new_key = keys[0]

        super().__init__(env)
        self.keys = keys
        self.new_key = new_key

        relevant_spaces = [
            deepcopy(self.observation_space.spaces[key]) for key in keys
        ]
        self.new_space = self.makeSuperSpace(relevant_spaces)

        # Redefine new entry in obs space. Overwrites key if new_key is already
        # in obs space. Otherwise, appends to Dict.
        self.observation_space[new_key] = self.new_space

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Pass unwrapped action mask through another, pre-specified, mask.

        Args:
            obs (OrderedDict): Must contain keys defined at mask instantiation.

        Returns:
            OrderedDict: Same keys as obs, plus optionally new_key (specified
                at instantiation). If new_key is included, it is the last entry
                in the returned OrderedDict.
        """
        # Loop through relevant items and multiply by each other. Make sure to
        # convert dtype to that specified on instantiation. Some dtypes don't
        # elementwise multiply together, so convert dtype after Hadamard product.
        out_arr = ones(obs[self.keys[0]].shape)

        for k in self.keys:
            out_arr = multiply(out_arr, obs[k])
            out_arr = out_arr.astype(self.new_space.dtype)

        # Copy input obs, then replace/append new array
        new_obs = OrderedDict(**deepcopy(obs))
        new_obs[self.new_key] = out_arr
        return new_obs

    def makeSuperSpace(self, spaces: list[Space]) -> Space:
        """Make new space that is the superset of multiple spaces.

        All input spaces must have same shape.

        Args:
            spaces (list[Space]): List of Gymnasium spaces.

        Returns:
            Space: A space that has the most general dtype of the input spaces.
        """
        common_dtypes = []
        sranges = []
        for s1, s2 in zip(spaces[:-1], spaces[1:]):
            common_dtypes.append(getSpaceClosestCommonDtype(s1, s2))
            sranges.append(list(getSpacesRange(s1, s2)))

        # dtype hierarchy: float (broader) -> int (narrower)
        if len(common_dtypes) == 1:
            cd = common_dtypes[0]
        elif float in common_dtypes:
            cd = float
        else:
            cd = int

        # Get lowest low and highest high
        lows = [sr[0] for sr in sranges]
        highs = [sr[1] for sr in sranges]
        new_low = min(lows)
        new_high = max(highs)

        # Because wrapper multiplies arrays together, ranges can vary b/w (-Inf, Inf).
        if new_low < 0:
            new_low = -Inf

        if new_high > 1:
            new_high = Inf

        new_space = Box(
            low=new_low, high=new_high, shape=spaces[0].shape, dtype=cd
        )

        # convert to MultiBinary if necessary
        new_space = convertBinaryBoxToMultiBinary(new_space)

        return new_space


# %% FlatDict
class FlatDict(gym.ObservationWrapper):
    """Flatten entries of a Dict observation space, leaving the top level unaffected.

    Unwrapped environment must have a Dict observation space.

    Can selectively flatten entries in a Dict observation space with the keys
        arg.
    """

    def __init__(
        self,
        env: Env,
        keys: list = None,
    ):
        """Flatten sub-levels of a Dict observation space.

        Args:
            env (Env): An environment with a Dict observation space.
            keys (list, optional): List of sub-levels to flatten. If empty,
                all sub-levels are flattened. Defaults to [].
        """
        if keys is None:
            keys = []

        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
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


# %% MakeDict
class MakeDict(gym.ObservationWrapper):
    """Wrap the observation space in a 1-item Dict.

    wrapped_obs_space = Dict({
        "obs": unwrapped_obs_space
    })

    """

    # Mostly useful for tests.

    def __init__(
        self,
        env: Env,
    ):
        """Wrap env."""
        super().__init__(env)
        obs_dict = {"obs": env.observation_space}
        self.observation_space = gym.spaces.Dict(**obs_dict)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Get wrapped observation."""
        obs_new = {"obs": obs}

        return obs_new


# %% LinScaleDictObs
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
        env: Env,
        rescale_config: dict = None,
    ):
        """Wrap an environment with LinScaleDictObs.

        Args:
            env (Env): Observation space must be a gymDict.
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


# %% MinMaxScaleDictObs
class MinMaxScaleDictObs(gym.ObservationWrapper):
    """MinMax scale entries in a dict observation space.

    Each value in the observation space is scaled by
        X_scaled = X_std * (max - min) + min.

    MultiBinary observations are not scaled (output = input).

    See sklearn.preprocessing.MinMaxScaler for algorithm details.
    """

    def __init__(self, env: Env):
        """Wrap environment that has a Dict observation space."""
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), """The input environment to MinMaxScaleDictObs() must have a `gym.spaces.Dict`
         observation space."""
        for space in env.observation_space.spaces.values():
            assert isinstance(
                space, (Box, MultiBinary)
            ), """
            All sub-spaces in env.observation_space must be one of [Box, MultiBinary]."""
            assert (
                len(space.shape) <= 2
            ), """All sub-spaces in env.observation_space must have <=2 dims."""

        super().__init__(env)

        # Update wrapper observation_shape. Set all lows/highs to 0/1.
        # NOTE: Any subspace with dtype==int will be changed to float.
        new_obs_space = OrderedDict({})
        for k, space in env.observation_space.spaces.items():
            if isinstance(space, MultiBinary):
                # Don't rescale MultiBinary spaces. Handles edge case with all
                # 1s in obs.
                new_obs_space[k] = deepcopy(space)
            else:
                # remake space with same shape, but new low/high/dtype.
                new_obs_space[k] = remakeSpace(
                    space=space, lows=0, highs=1, dtype=float
                )
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
        # If a 1d array is handed in, need to convert to 2d for MinMaxScaler to
        # work. Then need to convert back to 1d before passing back out.
        # Convert dtypes to float32 to match observation_space (float32 is default
        # dtype for gym Box spaces).
        # Skip MultiBinary observations.

        new_obs = {}
        for k, v in obs.items():
            if isinstance(self.observation_space[k], MultiBinary):
                # skip loop early if MultiBinary; values already scaled [0, 1].
                new_obs[k] = v
                continue
            if isinstance(v, list):
                # edge case: Env checkers can pass in 1d arrays as lists, so correct
                # here to satisfy checker.
                v = asarray(v)

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


# %% SplitArrayObs
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
        env: Env,
        keys: list[str],
        new_keys: list[list[str]],
        indices_or_sections: list[int | ndarray],
        axes: list[int] = None,
    ):
        """Initialize SplitArrayObs wrapper.

        Args:
            env (Env): Must have a Dict observation space.
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
        new_keys_flat = list(ravel(new_keys))
        assert all(
            [nk not in env.observation_space.spaces for nk in new_keys_flat]
        ), """Entries in new_keys cannot already be in observation space."""

        super().__init__(env)
        self.key_map = {k: v for (k, v) in zip(keys, new_keys)}
        self.indices_or_sections = indices_or_sections
        self.axes = axes

        # Store partials of split() to apply to selective dict entries using
        # SelectiveDictProcessor (sdp).
        funcs = [
            partial(split, indices_or_sections=iors, axis=ax)
            for iors, ax, in zip(indices_or_sections, axes)
        ]
        self.sdp = SelectiveDictProcessor(funcs, keys)

        # Redefine env.observation_space
        self.observation_space = self.buildNewObsSpace(
            env,
            self.key_map,
            env.observation_space,
            indices_or_sections,
            axes,
        )

    def buildNewObsSpace(
        self,
        env: Env,
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
            {k: deepcopy(v) for (k, v) in env.observation_space.items()}
        )
        for (key, new_keys), iors, ax in zip(
            key_map.items(),
            indices_or_sections,
            axes,
        ):
            original_subspace = obs_space[key]
            split_subspaces = split(
                original_subspace.sample(),
                indices_or_sections=iors,
                axis=ax,
            )
            for arr, nk in zip(split_subspaces, new_keys):
                new_obs_space[nk] = remakeSpace(
                    space=original_subspace, shape=arr.shape
                )
            new_obs_space.spaces.pop(key)

        return new_obs_space

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Wrap an observation with SplitArrayObs.

        Args:
            obs (OrderedDict): Unwrapped observation.

        Returns:
            OrderedDict: Selected keys (determined at instantiation) are deleted
                and replaced with two keys each. The replacement values are split
                arrays. New items are at end of returned OrderedDict.
        """
        new_obs = deepcopy(obs)
        # replace select arrays with lists of arrays
        split_arrays = self.sdp.applyFunc(new_obs)
        # reassign entries of lists of arrays to separate OrderedDict items
        for k, v in self.key_map.items():
            new_obs[v[0]] = split_arrays[k][0]
            new_obs[v[1]] = split_arrays[k][1]
            new_obs.pop(k)

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


# %% SumArrayWrapper
class SumArrayWrapper(SelectiveDictObsWrapper):
    """Sum array(s) along a given dimension for items in a Dict observation space."""

    def __init__(
        self,
        env: Env,
        keys: list[str],
        axis: int | None = None,
    ):
        """Initialize wrapper.

        Args:
            env (Env): A gym environment with a Dict observation space.
            keys (list[str]): Keys whose values will be summed.
            axis (int | None, optional): Axis along which to sum key values. If
                None, all elements of array will be summed. Defaults to None.
        """
        funcs = [partial(self.wrapSum, axis=axis)]
        obs_space = Dict(
            {k: deepcopy(v) for (k, v) in env.observation_space.items()}
        )
        for k in keys:
            v = obs_space[k]
            if axis is None:
                # corner case for sum of all elements of array
                new_shape = [
                    1,
                ]
            else:
                # Get new shape by deleting axis-th index of v.shape
                new_shape = list(v.shape)
                del new_shape[axis]

            # Even if unwrapped space has bounded lows/highs, summing means that
            # wrapped obs space will have unbounded lows/highs.
            # Adding MultiBinaries produces a Box.
            # MultiBinary has dtype == int8; set new space.dtype == int to stay
            # generic and consistent with numpy defaults.
            if isinstance(v, MultiBinary):
                new_dtype = int
            elif isinstance(v, Box):
                new_dtype = float

            obs_space[k] = Box(
                low=-inf,
                high=inf,
                shape=new_shape,
                dtype=new_dtype,
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


# %% CustodyWrapper
class CustodyWrapper(gym.ObservationWrapper):
    """Add 'custody' as an item to a Dict observation space.

    Custody entry is a MultiBinary space with shape (N,), where N is the
    number of targets. Does not modify other items in observation
    space, just adds "custody" to end of (ordered) dict.

    Unwrapped observation space is required to have `key` as an item, which
    must be a 3d Box space.
    """

    def __init__(
        self,
        env: Env,
        key: Any,
        config: dict = None,
        target_names: list = None,
        initial_status: list[bool] = None,
    ):
        """Wrap environment with CustodyWrapper observation space wrapper.

        Args:
            env (Env): Must have a Dict observation space with key in it.
            key (Any): A key contained in the observation space. The value corresponding
                to this key must conform to interface expected in CustodyTracker
                and config.
            config (dict, optional): See CustodyTracker for details. Defaults to None.
            target_names (list, optional): Target names. Used for debugging. Must
                have length == env.action_space.nvec[0]. Defaults to None.
            initial_status (list[bool], optional): See CustodyTracker for details.
                Defaults to None.
        """
        assert (
            key in env.observation_space.spaces
        ), f"{key} must be in env.observation_space."
        assert (
            "custody" not in env.observation_space.spaces
        ), "'custody' is already in env.observation_space."
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "Action space must be MultiDiscrete."

        num_targets = env.action_space.nvec[0] - 1

        if target_names is not None:
            assert (
                len(target_names) == num_targets
            ), "num_targets must equal len(target_names) (if target_names is not None)."

        # make wrapper
        super().__init__(env)
        self.key = key

        self.custody_tracker = CustodyTracker(
            config=config,
            num_targets=num_targets,
            target_names=target_names,
        )

        # Update observation space, maintain order, append "custody" to end.
        new_space = OrderedDict({**env.observation_space})
        new_space["custody"] = MultiBinary(num_targets)
        self.observation_space = Dict(new_space)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Convert unwrapped observation to wrapped observation.

        Args:
            obs (OrderedDict): Must have self.key as item.

        Returns:
            OrderedDict: Same as input dict, but with "custody" item appended.
                Custody is a (N,) binary array where 1 indicates the n-th target
                is in custody.
        """
        new_obs = OrderedDict(deepcopy(obs))

        custody_input = deepcopy(obs[self.key])

        # custody_tracker outputs custody status as a list of bools; convert to
        # a 1d array of ints. Use int8 for dtype-> this is the default dtype of
        # MultiBinary space. Make sure "custody" is added at end of OrderedDict
        # observation.
        custody = array(self.custody_tracker.update(custody_input)).astype(int8)
        new_obs["custody"] = custody

        assert self.observation_space.contains(new_obs)
        return new_obs


# %% Convert2dTo3dObsItems
class Convert2dTo3dObsItems(gym.ObservationWrapper):
    """Convert 2d arrays in a Dict observation space to sparse 3d arrays.

    Order in observation space is maintained. Specified items have shapes changed
        from (A, B) to (A, B, B) or (B, A, A), depending on configuration. Off-diagonals
        are 0s.

    Example:
        env.observation_space = Dict(
            {
                "a": Box(0, 1, shape=(2, 3)),
                "b": Box(0, 1, shape=(6, 2)),
            }

        wrapped_env = Convert2dTo3dObsItems(env, keys=["a"], diag_on_0_or_1=1)

        wrapped_env.observation_space = Dict(
            {
                "a": Box(0, 1, shape=(3, 2, 2)),
                "b": Box(0, 1, shape=(6, 2)),
            }
        )
    """

    def __init__(
        self,
        env: Env,
        keys: list,
        diag_on_0_or_1: list[int] = None,
    ):
        """Wrap environment.

        Args:
            env (Env): Must have Dict observation space.
            keys (list): Items in observation space that will be diagonalized.
                All entries must:
                    - be keys in env.observation_space
                    - be 2d ndarrays
            diag_on_0_or_1 (list[int], optional): Must be 0s or 1s. Determines
                which indices of input arrays will be used to diagonalize 2d arrays.
                If 0, output shape == (A, B, B). If 1, output shape == (B, A, A).
                Must have length == len(keys). Defaults to all 0s.
        """
        # default to all 0s
        if diag_on_0_or_1 is None:
            diag_on_0_or_1 = [0 for i in range(len(keys))]

        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert len(diag_on_0_or_1) == len(
            keys
        ), "diag_on_0_or_1 must have same length as keys."
        for k, d in zip(keys, diag_on_0_or_1):
            assert (
                k in env.observation_space.spaces
            ), f"{k} not in observation_space."
            assert isinstance(
                env.observation_space.spaces[k], Box
            ), f"{k} is not a Box."
            assert (
                len(env.observation_space.spaces[k].shape) == 2
            ), f"{k} is not 2d."
            assert d in [0, 1], "All entries of diag_on_0_or_1 must be 0 or 1."

        super().__init__(env)

        self.keys = keys
        self.first_dims = diag_on_0_or_1
        self.observation_space = deepcopy(env.observation_space)
        for k, d in zip(keys, diag_on_0_or_1):
            # NOTE: Assumes the old space has same highs/lows for all values
            old_space = self.observation_space[k]
            self.observation_space[k] = Box(
                old_space.low[0, 0],
                high=old_space.high[0, 0],
                shape=(
                    old_space.shape[d],
                    old_space.shape[1 - d],
                    old_space.shape[1 - d],
                ),
                dtype=old_space.dtype,
            )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Convert selected 2d array entries of a Dict observation space to 3d.

        Args:
            obs (OrderedDict): Only certain items, specified at instantiation,
                will be modified.

        Returns:
            OrderedDict: Same order as input. Specified items, which were input
                as 2d, are now 3d.
        """
        new_obs = OrderedDict(deepcopy(obs))
        for k, d in zip(self.keys, self.first_dims):
            # Loop through keys. Take old space (shape=AxB) and make new space
            # (shape BxAxA).
            ob = obs[k]
            new_ob = zeros(
                shape=(ob.shape[d], ob.shape[1 - d], ob.shape[1 - d])
            )
            for i in range(new_ob.shape[0]):
                if d == 0:
                    # Diagonalize rows of input obs
                    new_ob[i, :, :] = diag(ob[i, :])
                elif d == 1:
                    # Diagonalize cols of input obs
                    new_ob[i, :, :] = diag(ob[:, i])

            # Convert dtype after filling in all 0-th indexed entries-- not before
            new_obs[k] = new_ob.astype(self.observation_space[k].dtype)

        return new_obs


# %% DiagonalObsItems
class DiagonalObsItems(SelectiveDictObsWrapper):
    """Get diagonals of multidimensional space(s).

    Example:
        unwrapped_env.observation_space = Dict(
            {
                "a": Box(0, 1, shape=[2, 3, 3])
            }

        wrapped_env = DiagonalObsItems(
            unwrapped_env,
            keys=["a"],
            axis1=[1],
            axis2=[2])

        wrapped_env.observation_space = Dict(
            {
                "a": Box(0, 1, shape=[2, 3])
            }

        obs = unwrapped_env.observation_space.sample()
        print(obs)
        # prints
        # [[[1 0 0]
        #   [0 0 1]
        #   [0 1 1]]

        # [[1 0 1]
        #  [1 1 1]
        #  [1 0 0]]]

        print(wrapped_env.observation(obs))
        # prints
        # [[1 0 1]
        #  [1 1 0]]

    See numpy.diagonal for details.
    """

    def __init__(
        self,
        env: Env,
        keys: list[str],
        offset: list[int] = None,
        axis1: list[int] = None,
        axis2: list[int] = None,
    ):
        """Get diagonal elements from multidimensional array.

        See numpy.diagonal for details.

        Args:
            env (Env): Must have Dict observation space.
            keys (list[str]): Must be in observation space. Each corresponding
                value must be a Space with >1 dims.
            offset (list[int], optional): Offset from main diagonal. Defaults to 0.
            axis1 (list[int], optional): Axis to be used as first axis of 2d sub-arrays
                from which the diagonals should be taken. If used, must have same
                length as keys. Defaults to 0.
            axis2 (list[int], optional): Axis to be used as second axis of 2d sub-arrays
                from which the diagonals should be taken. If used, must have same
                length as keys. Defaults to 1.
        """
        if offset is None:
            offset = [0 for i in range(len(keys))]
        if axis1 is None:
            axis1 = [0 for i in range(len(keys))]
        if axis2 is None:
            axis2 = [1 for i in range(len(keys))]

        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert (
            len(keys) == len(offset) == len(axis1) == len(axis2)
        ), """Lengths of keys, offset, axis1, and axis2 must be equal (if non-Nones
        are provided)."""
        assert all(
            [len(env.observation_space.spaces[k].shape) > 1 for k in keys]
        ), """All subspaces specified in keys must have dim > 1."""

        # Loop through keys (and associated args) to recreate spaces with same
        # space type (e.g. Box, MultiBinary), same dtype, same highs/lows, but
        # different shapes.
        new_obs_space = deepcopy(env.observation_space)
        for k, off, ax1, ax2 in zip(keys, offset, axis1, axis2):
            orig_space = env.observation_space[k]
            diag_obs = diagonal(
                orig_space.sample(), offset=off, axis1=ax1, axis2=ax2
            )
            new_space_shape = diag_obs.shape
            new_obs_space[k] = remakeSpace(orig_space, shape=new_space_shape)

        funcs = [
            partial(diagonal, offset=o, axis1=a1, axis2=a2)
            for (o, a1, a2) in zip(offset, axis1, axis2)
        ]

        super().__init__(env, funcs, keys, new_obs_space)


# %% ConvertCustody2ActionMask
class ConvertCustody2ActionMask(gym.ObservationWrapper):
    """Convert a MultiBinary custody array to a MultiBinary action mask.

    Assumes inaction is a valid action.

    Notation:
        M: number of sensors
        N: number of targets
        custody array: N-long, 1d array, where a 1 indicates the n'th target is
            in custody, and 0 otherwise.
        action mask: (N+1, M) binary array, where a 1 indicates the (n, m)'th
            target-sensor pair is a valid action. The extra "+1" entries denote
            inaction, which is always valid.

    Example (M=2):
        env.observation_space = Dict(
            {
                "custody": MultiBinary(3),
            }

        wrapped_env = ConvertCustody2ActionMask(
            env,
            key = "custody",
            )

        wrapped_env.observation_space = Dict(
            {
                "custody": MultiBinary([4, 2]),
            }
        )

    Example (with rename_key) (M=2):
        env.observation_space = Dict(
            {
                "custody": MultiBinary(3),
                "bar": Box(0, 1)
            }

        wrapped_env = ConvertCustody2ActionMask(
            env,
            key = "custody",
            rename_key = "foo"
            )

        wrapped_env.observation_space = Dict(
            {
                "bar": Box(0, 1)
                "foo": MultiBinary([4, 2]),  # new item is at end of Dict
            }
        )


    """

    def __init__(
        self,
        env: Env,
        key: str,
        rename_key: str = None,
    ):
        """Wrap environment.

        Args:
            env (Env): Must have a Dict observation_space.
            key (str): Contained in env.observation_space. Must be a 1d MultiBinary
                space.
            rename_key (str, optional): Unwrapped observation space entry key will
                be renamed to rename_key and put at end of Dict observation space.
                If rename_key == None, then unwrapped key is unmodified and stays
                in same place. Defaults to None.
        """
        if rename_key is None:
            rename_key = key

        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert (
            key in env.observation_space.spaces
        ), f"{key} is not in env.observation_space."
        assert isinstance(
            env.observation_space.spaces[key], MultiBinary
        ), f"env.observation_space[{key}] must be a MultiBinary."
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a gymnasium.spaces.MultiDiscrete."

        super().__init__(env)

        self.key = key
        self.num_sensors = env.action_space.shape[0]
        # convert num_targets from numpy dtype to Python int
        self.num_targets = int(env.observation_space[key].n)
        self.rename_key = rename_key
        self.sdp = SelectiveDictProcessor(
            funcs=[
                partial(
                    self.binary2ActionMask,
                    num_sensors=self.num_sensors,
                )
            ],
            keys=[rename_key],
        )
        self.observation_space = deepcopy(env.observation_space)
        self.mask2d_space = MultiBinary(
            [self.num_targets + 1, self.num_sensors]
        )

        # If original key's value is being overwritten, then maintain the item's
        # position in the OrderedDict. But if the original key is being replaced
        # with a new key, delete the original item and append the new item to end
        # of OrderedDict.
        if rename_key != key:
            del self.observation_space.spaces[key]

        self.observation_space[rename_key] = self.mask2d_space

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Convert custody observation to action mask.

        Args:
            obs (OrderedDict): Must have self.key in keys.

        Returns:
            OrderedDict: Value is converted from custody array to action mask.
                See class description for details. Returned OrderedDict has same
                keys as obs, unless self.rename_key was set on instantiation. If
                self.rename_key != self.key, then returned OrderedDict will not
                have self.key, but will have self.rename_key. Order of returned
                OrderedDict is same as obs. "rename_key", if used, is appended
                to end of OrderedDict.
        """
        new_obs = deepcopy(obs)
        # Copy item to new item (no effect if key == rename_key). Delete original
        # key if key != rename_key
        new_obs[self.rename_key] = new_obs[self.key]
        if self.rename_key != self.key:
            del new_obs[self.key]

        # sdp.applyFunc overwrites new_obs[rename_key], leaves other keys untouched
        new_obs = self.sdp.applyFunc(new_obs)
        return new_obs

    def binary2ActionMask(
        self, custody_array: ndarray, num_sensors: int
    ) -> ndarray:
        """Convert a 1d binary array to a 2d action mask.

        Notation:
            M: number of sensors
            N: number of targets

        Args:
            custody_array (ndarray): (N, ) single-dimensional binary array.

        Returns:
            ndarray: (N+1, M) single-dimensional binary array. Every (N+1)th
                entry corresponds to inaction, and always == 1.


        Example:
            custody = [1, 0, 1]
            action_mask = binary2ActionMask(custody, num_sensors = 2)
            # action_mask = array([[1, 1],
            #                      [0, 0],
            #                      [1, 1],
            #                      [1, 1]])  <- inaction always 1 (valid)

        """
        custody_copies = [custody_array for _ in range(num_sensors)]
        partial_action_mask_2d = stack(custody_copies, axis=1)
        full_action_mask_2d = concatenate(
            [partial_action_mask_2d, ones((1, num_sensors), dtype=int)], axis=0
        )

        assert self.mask2d_space.contains(full_action_mask_2d)

        return full_action_mask_2d


# %% ConvertObsBoxToMultiBinary
class ConvertObsBoxToMultiBinary(gym.ObservationWrapper):
    """Convert a Box space (which is an entry in a Dict space) to MultiBinary.

    Overwrites value in Dict space.

    Example:
        env.observation_space = Dict(
            {
                "foo": Box(low=0, high=1, shape=(2, 2), dtype=float),
            }

        wrapped_env = ConvertObsBoxToMultiBinary(env, key = "foo")

        wrapped_env.observation_space = Dict(
            {
                "foo": MultiBinary(2, 2),
            }
        )
    """

    def __init__(self, env: Env, key: str):
        """Wrap environment.

        Args:
            env (Env): Must have Dict observation space.
            key (str): Must be in env.observation_space. env.observation_space[key]
                must be a Box space with high == 1, low == 0, and dtype != int.
        """
        assert isinstance(
            env.observation_space.spaces[key], Box
        ), f"env.observation_space.spaces[{key}] must be a Box."
        assert (
            key in env.observation_space.spaces
        ), f"{key} must be in env.observation_space."
        assert env.observation_space.spaces[key].dtype in (
            int,
            int_,
            int8,
        ), f"env.observation_space.spaces[{key}].dtype must be int."
        assert all(
            env.observation_space.spaces[key].low == 0
        ), f"env.observation_space.spaces[{key}].low must = 0."
        assert all(
            env.observation_space.spaces[key].high == 1
        ), f"env.observation_space.spaces[{key}].high must = 0."

        super().__init__(env)

        self.key = key
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space[key] = MultiBinary(
            env.observation_space[key].shape
        )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Convert Box observation to MultiBinary observation.

        Output keys are same as input keys. Values not equal to self.key are unaffected.
        Values in obs[self.key] are not changed, but dtype is.
        """
        new_obs = deepcopy(obs)
        new_obs[self.key] = obs[self.key].astype(int)
        return new_obs


# %% SqueezeObsItems
class SqueezeObsItems(SelectiveDictObsWrapper):
    """Squeeze a multidimensional space.

    Supports Box and MultiBinary spaces.

    Example:
        env.observation_space = Dict(
            {
                "a": Box(low=0, high=1, shape=(2, 1, 2)),
                "b": MultiBinary((2, 1, 2)),
            }
        )

        env_wrapped = SqueezeObsItems(env, keys=["a", "b"])

        env_wrapped.observation_space = Dict(
            {
                "a": Box(low=0, high=1, shape=(2, 2)),
                "b": MultiBinary((2, 2)),
            }
        )

    See numpy.squeeze for details.
    """

    def __init__(
        self,
        env: Env,
        keys: list[str],
        axis: list[int] | list[Tuple[int]] = None,
    ):
        """Initialize wrapper.

        Args:
            env (Env): Must have a Dict observation space.
            keys (list[str]): Must be in observation space.
            axis (list[int] | list[Tuple[int]], optional): Axis on which to squeeze.
                See numpy.squeeze for details. If len == 1, then all keys will
                be squeezed on same axis. Defaults to None.
        """
        if (axis is not None) and len(axis) == 1:
            # Copy axis to be same length as keys, if axis was provided and short.
            axis = [axis for i in range(len(keys))]
        elif axis is None:
            # If axis not provided, make list of Nones
            axis = [None for i in range(len(keys))]

        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert all(
            k in env.observation_space.spaces for k in keys
        ), "All keys must be in observation space."
        assert len(axis) == len(keys), "len(axis) != len(keys)."
        for k in keys:
            assert isinstance(
                env.observation_space[k], (Box, MultiBinary)
            ), f"{k} must be one of [Box, MultiBinary]."

        new_obs_space = deepcopy(env.observation_space)
        # Get shape of new observation space by checking length of .shape attr, then
        # removing any 1s.
        for k in keys:
            orig_space = env.observation_space.spaces[k]
            space_shape = orig_space.shape
            new_space_shape = [d for d in space_shape if d > 1]
            lows, highs = self.getLowsHighs(orig_space)

            new_obs_space[k] = remakeSpace(
                orig_space,
                shape=new_space_shape,
            )

        # get list of squeeze functions (inputs for axis may be different)
        funcs = [partial(squeeze, axis=a) for a in axis]

        super().__init__(
            env=env,
            funcs=funcs,
            keys=keys,
            new_obs_space=new_obs_space,
        )

    def getLowsHighs(self, space: gym.Space) -> list[ndarray | int]:
        """Get low and high values from a Gym space."""
        if isinstance(space, Box):
            lows = squeeze(space.low)
            highs = squeeze(space.high)
        elif isinstance(space, MultiBinary):
            lows = 0
            highs = 1

        return lows, highs


# %% WastedActionsMask
class WastedActionsMask(gym.ObservationWrapper):
    """Mask null action if target(s) available to sensor.

    Notation:
        M : Number of sensors.
        N : Number of targets.

    Observation space must be a Dict.

    Observation space must contain sensor-target availability map in the form of
    a (N, M) or (N+1, M) binary array. The +1 row corresponds to the null action.

    Wrapped observation space is same as unwrapped observation space, with an additional
    item: a 2d action mask. The key for the new item is set on instantiation. The
    action mask is a (N+1, M) binary array. The bottom row corresponds to null
    actions.

    For every column in the availability map, if there is a 1 anywhere in the first
    [0:N] entries, the mask value is 0. Otherwise, the value is 1. The (N+1) row
    of the input array, if it exists, is ignored in calculating the wasted action
    mask.
    """

    def __init__(self, env: Env, vis_map_key: ndarray, mask_key: str = None):
        """Wrap environment with WastedActionsMask.

        Args:
            env (Env): Must have a Dict observation space.
            vis_map_key (ndarray): Corresponds to item in observation space.
                Corresponding value must be a (N, M) or (N+1, M) binary array.
            mask_key (str, optional): The key of the new action mask entry in the
                wrapped observation space. Defaults to 'mask'.
        """
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert (
            vis_map_key in env.observation_space.spaces
        ), f"{vis_map_key} must be in observation space."
        assert isinstance(
            env.observation_space.spaces[vis_map_key], MultiBinary
        ), f"env.observation_space[{vis_map_key}] must be a MultiBinary."
        num_sensors = env.observation_space.spaces[vis_map_key].shape[1]
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a gymnasium.spaces.MultiDiscrete."
        assert (
            len(env.action_space.nvec) == num_sensors
        ), f"""Length of action space must match number of columns in
        observation_space[{vis_map_key}]."""
        assert all(
            env.action_space.nvec == env.action_space.nvec[0]
        ), "All values in action_space.nvec must be identical."
        # num_targets = env.observation_space.spaces[vis_map_key].shape[0]
        num_targets = env.action_space.nvec[0] - 1
        assert env.observation_space.spaces[vis_map_key].shape[0] in [
            num_targets,
            num_targets + 1,
        ], f"""Observation and action spaces disagree on number of targets. The
        observation space should have N or N+1 rows where `N+1` is the value of
        all entries in the action space.
        Action space value = {env.action_space[0]},
        Observation space shape = {env.observation_space.spaces[vis_map_key].shape[0]}.
        """

        if mask_key is None:
            assert (
                "mask" not in env.observation_space.spaces
            ), """A new mask key was not provided, but the default name, 'mask'
            shadows an existing observation space key. Either provide an argument
            to mask_key or change the observation space to not have 'mask' in the
            keys."""
            mask_key = "mask"

        super().__init__(env)

        if env.observation_space.spaces[vis_map_key].shape[0] == num_targets:
            # So we know whether or not to remove the bottom row in checking visibility
            self.nullaction_included = False
        else:
            self.nullaction_included = True

        self.vis_map_key = vis_map_key
        self.mask_key = mask_key
        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.observation_space[mask_key] = MultiBinary(
            (self.num_targets + 1, self.num_sensors)
        )

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Mask null-action from sensors that have visible targets.

        Returns:
            OrderedDict: Appends an extra item from the input obs. Key of extra
                item is determined at instantiation.
        """
        mask = ones((self.num_targets + 1, self.num_sensors), dtype=int)

        vis_map = obs[self.vis_map_key]
        if self.nullaction_included is True:
            # crop if null action row is included
            vis_map = vis_map[:-1, :]

        for i, col in enumerate(vis_map.T):
            new_col = deepcopy(col)
            if any(col == 1):
                new_col = append(new_col, 0)
            else:
                new_col = append(new_col, 1)

            mask[:, i] = new_col

        new_obs = deepcopy(obs)
        new_obs[self.mask_key] = mask

        return new_obs


# %% TransformDictObsWithNumpy
class TransformDictObsWithNumpy(SelectiveDictObsWrapper):
    """Transform an entry in a Dict observation space by a numpy function.

    Apply a numpy function to a single entry in a dict observation. Converts
    the corresponding entry in the observation space to a Box with (-Inf, Inf)
    bounds.

    Works only with numpy functions that can be called like numpy.[func](args).

    Overwrites key[value] in observation.

    Examples:
        # env.observation_space = Dict({"a": Box(0, 1, shape=(3, 2))})

        TransformDictObsWithNumpy(env, "mean", "a")
        TransformDictObsWithNumpy(env, "median", "a", axis=1)
    """

    def __init__(
        self,
        env: Env,
        numpy_func_str: str,
        key: str,
        **kwargs: Any,
    ):
        """Wrap environment with TransformDictObsWithNumpy.

        Args:
            env (Env): Must have a Dict observation space.
            numpy_func_str (str): Must be an attribute of numpy (i.e. works by calling
                getattr(numpy, numpy_func_str)).
            key (str): Key of observation space to apply function to.
            kwargs (Any, optional): Any kwargs to be used in numpy function.
        """
        partial_func = convertNumpyFuncStrToCallable(
            numpy_func_str=numpy_func_str, **kwargs
        )

        # Make new observation space by trasnforming sample observation, then getting
        # shape and dtype. Assume bounds are (-Inf, Inf).
        new_obs_space = deepcopy(env.observation_space)
        unwrapped_obs = env.observation_space.sample()[key]
        wrapped_obs = partial_func(unwrapped_obs)

        new_shape = wrapped_obs.shape
        new_dtype = wrapped_obs.dtype

        new_obs_space[key] = Box(
            -Inf,
            Inf,
            shape=new_shape,
            dtype=new_dtype,
        )

        super().__init__(env, [partial_func], [key], new_obs_space)


# %% convertBinaryBoxToMultiBinary
class MakeObsSpaceMultiBinary(SelectiveDictObsWrapper):
    """Convert an entry of the obs space from Box to MultiBinary."""

    def __init__(self, env: Env, key: str):
        """Wrap env.

        Args:
            env (Env): Must have a Dict observation space.
            key (str): Key to an item in the obs space. Corresponds to a Box env.
        """
        assert isinstance(
            env.observation_space.spaces[key], Box
        ), f"env.observation_space.spaces[{key}] must be a Box."

        new_obs_space = deepcopy(env.observation_space)
        mb_space = MultiBinary(env.observation_space.spaces[key].shape)
        new_obs_space[key] = mb_space

        func = self.toIntArray
        super().__init__(
            env=env, funcs=[func], keys=[key], new_obs_space=new_obs_space
        )

        return

    def toIntArray(self, x: ndarray) -> ndarray:
        """Wrapper around numpy .astype method."""
        return clip(x.astype(int), 0, 1)
