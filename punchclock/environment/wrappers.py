"""SSAScheduler wrappers module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, flatten_space, unflatten
from gymnasium.spaces.utils import flatten
from numpy import append, float32, inf, int64, ndarray, ones
from sklearn.preprocessing import MinMaxScaler

# Punch Clock Imports
from punchclock.environment.env import SSAScheduler


# %% Wrapper for covariance diagonals
class FilterCovElements(gym.ObservationWrapper):
    """Hide either position or velocity covariances in SSAScheduler observation.

    Observation is a dict with key "est_cov" being a (6, M) array, where the 6 elements
        are position (:3) and velocity (3:) covariances, respectively, and M is the
        number of covariance estimates.
    """

    def __init__(self, env: SSAScheduler, pos_vel_flag: str):
        """Initialize wrapped environment.

        Args:
            env (`SSAScheduler`): An instance `SSAScheduler` gym environment.
            pos_vel_flag (`str`): ("position" | "velocity") Denotes which elements
            of covariance diagonals to keep in observation.
        """
        # Initialize bare environment
        super().__init__(env)

        # save position/velocity flag
        self.pos_vel_flag = pos_vel_flag

        # Modify observation space (same regardless of `pos_vel_flag` value)
        observation_space = deepcopy(env.observation_space)
        num_targets = observation_space["est_cov"].shape[1]
        observation_space["est_cov"] = gym.spaces.Box(
            -inf, inf, (3, num_targets)
        )
        self.observation_space = observation_space

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Filter covariance elements of observation.

        Args:
            obs (`OrderedDict`): Must contain "est_cov" as a key, with value being a
                (6, M) array, where the 6 elements are position (:3) and velocity
                (3:) covariances, respectively, and M is the number of covariance
                estimates.

        Returns:
            `OrderedDict`: Same as input obs, but with modified `est_cov`.

        Notes:
            - If obs does not contain "est_cov", output is same as input.
        """
        if self.pos_vel_flag == "position":
            obs["est_cov"] = obs["est_cov"][:3, :]
        elif self.pos_vel_flag == "velocity":
            obs["est_cov"] = obs["est_cov"][3:, :]

        return obs


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
            env (`SSAScheduler`): Unwrapped environment with `Dict` observation
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
    """Flatten entries of a `Dict` observation space, leaving the top level unaffected.

    Unwrapped environment must have a `Dict` observation space.
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        """Flatten sub-levels of a `Dict` observation space.

        Args:
            env (`gym.Env`): An environment with a `Dict` observation space.
        """
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), f"""The input environment to FlatDict() must have a `gym.spaces.Dict` 
        observation space."""
        super().__init__(env)

        items = {}
        for key, val in env.observation_space.items():
            items[key] = flatten_space(val)

        self.observation_space = gym.spaces.Dict(**items)

    def observation(self, obs: OrderedDict) -> OrderedDict:
        """Flatten items in `Dict` observation space, leaving top level intact.

        Args:
            obs (`OrderedDict`): Observation

        Returns:
            `gym.spaces.Dict`: All entries below top level are flattened.
        """
        obs_new = {}
        for key, val in obs.items():
            obs_new[key] = flatten(self.env.observation_space[key], val)

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
    """Rescale selected entries in a `Dict` observation space.

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
        new_obs = {}
        for k, v in obs.items():
            v, flip = self.transposeHorizontalArray(v)
            scaler = MinMaxScaler().fit(v)
            new_v = self.unTransposeArray(scaler.transform(v), flip)
            new_obs[k] = new_v

        return new_obs

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
