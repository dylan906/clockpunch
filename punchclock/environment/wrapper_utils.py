"""Wrapper utilities."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, MultiBinary, Space
from numpy import ndarray, ravel


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
