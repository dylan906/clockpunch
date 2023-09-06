"""Wrapper utilities."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from typing import Tuple

# Third Party Imports
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space
from numpy import all, int8, int64, max, min, ndarray, ravel

# Punch Clock Imports
from punchclock.environment.misc_wrappers import IdentityWrapper


# %% Classes
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
        assert isinstance(funcs, list), "funcs must be a list of Callables."
        assert isinstance(keys, list), "keys must be a list."
        assert all(
            isinstance(key, str) for key in keys
        ), "all entries in keys must be strs"
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
                processed_dict[k] = self.key_func_map[k](v, **kwargs)

        out_dict = deepcopy(in_dict)
        out_dict.update(processed_dict)

        return out_dict


def checkDictSpaceContains(
    space: gym.spaces.Dict, in_dict: dict | OrderedDict
) -> dict:
    """Check the values of a dict for being contained in values of a gym.spaces.Dict.

    Args:
        space (gym.spaces.Dict): Each value is called as value.contains().
        in_dict (dict | OrderedDict): Keys must match space.

    Returns:
        dict: A mapping of True/False, where True indicates the value of in_dict
            is contained in the corresponding value of space.
    """
    report = {}
    for k, v in in_dict.items():
        report[k] = space[k].contains(v)

    return report


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


def remakeSpace(
    space: Space,
    lows: ndarray | int = None,
    highs: ndarray | int = None,
    shape: list = None,
    dtype: type = None,
) -> Space:
    """Remake a space based on an original space.

    Used to reshape a space or assign new lows/highs or dtype.

    Supports Box and MultiBinary spaces.

    Any unused args are replaced with the same value(s) from the original space.

    Args:
        space (Space): Original space. Must be one of [Box, Multibinary].
        lows (ndarray | int, optional): New low value(s). Defaults to None.
        highs (ndarray | int, optional): New high value(s). Defaults to None.
        shape (list, optional): New shape. Defaults to original shape.
        dtype (type, optional): New dtype. Defaults to original dtype.

    Returns:
        gym.Space: Same type of space as original space.
    """
    supported_spaces = (Box, MultiBinary)
    assert isinstance(
        space, supported_spaces
    ), "Original space must be one of supported spaces."

    if lows is None:
        lows = getattr(space, "low", None)
    if highs is None:
        highs = getattr(space, "high", None)
    if shape is None:
        # All gym spaces by definition have shape
        shape = getattr(space, "shape", None)
    if dtype is None:
        # All gym spaces by definition have dtype
        dtype = getattr(space, "dtype", None)

    # If shape is changed, reshape lows/highs to be single int. Assumes that all
    # lows are same and all highs are same.
    # Only applies to Box (MultiBinary has intrinsic lows/highs)
    if shape != space.shape and isinstance(space, Box):
        # lows = lows.flat[0]
        # highs = highs.flat[0]
        lows = ravel(lows)[0]
        highs = ravel(highs)[0]

    if isinstance(space, Box):
        new_space = Box(
            low=lows,
            high=highs,
            shape=shape,
            dtype=dtype,
        )
    elif isinstance(space, MultiBinary):
        new_space = MultiBinary(n=shape)

    return new_space


def getSpaceClosestCommonDtype(
    space1: Box | Discrete | MultiBinary | MultiDiscrete,
    space2: Box | Discrete | MultiBinary | MultiDiscrete,
) -> float | int | int8:
    """Return the common dtype between two Gymnasium spaces.

    Args:
        space1 (Box | Discrete | MultiBinary | MultiDiscrete): A numerical Gymnasium
            space.
        space2 (Box | Discrete | MultiBinary | MultiDiscrete): A numerical Gymnasium
            space.

    Returns:
        float | int | int8: The common dtype between spaces.
    """
    assert isinstance(
        space1, (Box, Discrete, MultiBinary, MultiDiscrete)
    ), f"{space1} must be one of (Box, Discrete, MultiBinary, MultiDiscrete)."
    assert isinstance(
        space2, (Box, Discrete, MultiBinary, MultiDiscrete)
    ), f"{space2} must be one of (Box, Discrete, MultiBinary, MultiDiscrete)."

    dtypes = [space1.dtype, space2.dtype]

    if dtypes[0] == dtypes[1]:
        # returns int8 if both spaces are MultiBinary
        common_dtype = dtypes[0]
    elif float in dtypes:
        common_dtype = float
    elif int64 in dtypes:
        common_dtype = int

    return common_dtype


def getSpacesRange(
    space1: Box | Discrete | MultiBinary | MultiDiscrete,
    space2: Box | Discrete | MultiBinary | MultiDiscrete,
) -> Tuple:
    """Get the min and max ranges between two Gymnasium spaces.

    Args:
        space1 (Box | Discrete | MultiBinary | MultiDiscrete): A numerical Gymnasium
            space.
        space2 (Box | Discrete | MultiBinary | MultiDiscrete): A numerical Gymnasium
            space.

    Returns:
        Tuple: Minimum value, maximum value.
    """
    assert isinstance(
        space1, (Box, Discrete, MultiBinary, MultiDiscrete)
    ), f"{space1} must be one of (Box, Discrete, MultiBinary, MultiDiscrete)."
    assert isinstance(
        space2, (Box, Discrete, MultiBinary, MultiDiscrete)
    ), f"{space2} must be one of (Box, Discrete, MultiBinary, MultiDiscrete)."

    lows = [_getLowHigh(space, "low") for space in [space1, space2]]
    highs = [_getLowHigh(space, "high") for space in [space1, space2]]

    lows = [min(lo) for lo in lows]
    highs = [max(hi) for hi in highs]
    superrange = (min(lows), max(highs))

    return superrange


def _getLowHigh(space: Space, lowhigh: str) -> float | int:
    """Get the min or max value from an arbitrary space.

    Smartly handles MultiBinary, MultiDiscrete, and Discrete spaces.
    """
    if lowhigh == "low":
        if isinstance(space, MultiBinary):
            out = 0
        elif isinstance(space, MultiDiscrete):
            out = min(space.nvec)
        elif isinstance(space, Discrete):
            out = space.start
        else:
            out = space.low
    elif lowhigh == "high":
        if isinstance(space, MultiBinary):
            out = 1
        elif isinstance(space, MultiDiscrete):
            out = max(space.nvec)
        elif isinstance(space, Discrete):
            out = space.n
        else:
            out = space.high

    return out


def convertBinaryBoxToMultiBinary(box_space: Box) -> Box | MultiBinary:
    """Convert a Box space with (low, high) == (0, 1) and dtype == int to MultiBinary.

    Args:
        box_space (Box): A Gymnasium space.

    Returns:
        Box | MultiBinary: Output space is same shape as input space. If input
            space does not have (low, high) == (0, 1) and dtype == int, then output
            is same as input.
    """
    assert isinstance(box_space, Box)

    if (
        (box_space.dtype == int)
        and all(lo == 0 for lo in box_space.low)
        and all(hi == 1 for hi in box_space.high)
    ):
        new_space = MultiBinary(box_space.shape)
    else:
        new_space = box_space
    return new_space


def getIdentityWrapperEnv(env: Env) -> Env:
    """Get the IdentityWrapper level of an env, if one exists.

    Args:
        env (Env): A Gymnasium environment.

    Raises:
        Exception: If there is no IdentityWrapper in the stack of wrappers, raises
            an Exception.

    Returns:
        Env: Returns the environment with IdentityWrapper at the top level. All
            wrappers above IdentityWrapper are discarded.
    """
    env_temp = deepcopy(env)
    while not isinstance(env_temp, IdentityWrapper):
        if env_temp == env_temp.unwrapped:
            raise Exception(f"No IdentityWrapper in {env}")

        env_temp = getattr(env_temp, "env", {})

    return env_temp
