"""Reward function utils."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from functools import partial
from typing import Callable

# Third Party Imports
from numpy import max, mean, median, min, ndarray, split, sum, trace

# Punch Clock Imports
from scheduler_testbed.common.math import linear, logistic


# %% Lookup preprocessor
def lookupPreprocessor(func_name: str | dict | Callable) -> Callable:
    """Lookup a preprocessor from a list of recognized functions."""
    preprocessor_map = {
        # Simple functions
        "trace": trace,
        "mean": mean,
        "median": median,
        "max": max,
        "min": min,
        "sum_cols": sumCols,
        "sum_rows": sumRows,
        # Configurable functions
        "crop_array": cropArray,
        "linear": linear,
        "logistic": logistic,
    }

    if isinstance(func_name, str):
        func = preprocessor_map[func_name]
    elif isinstance(func_name, dict):
        func = partial(
            preprocessor_map[func_name["preprocessor"]],
            **func_name["config"],
        )
    elif isinstance(func_name, Callable):
        func = func_name

    return func


# %% Small functions
def sumCols(x):
    """Sum columns of a 2d matrix."""
    return sum(x, axis=0)


def sumRows(x):
    """Sum rows of a 2d matrix."""
    return sum(x, axis=1)


# %% Complex functions
def cropArray(
    x: ndarray,
    indices_or_sections: int | ndarray,
    section_to_keep: int,
    axis: int = 0,
) -> ndarray:
    """Get a sub-array from an input array.

    Uses numpy.split(), but returns a single array, designated by section_to_keep
    instead of a list of arrays.

    Args:
        x (`ndarray`): Array to be divided into sub-arrays.
        indices_or_sections (`int` | `ndarray`):
            If indices_or_sections is an integer, N, the array will be divided
            into N equal arrays along axis. If such a split is not possible, an
            error is raised.
            If indices_or_sections is a 1-D array of sorted integers, the entries
            indicate where along axis the array is split. For example, [2, 3] would,
            for axis=0, result in
                - x[:2]
                - x[2:3]
                - x[3:]
        section_to_keep (`int`): The sub-array of the resultant array split to
            return.
        axis (`int`, optional): The axis along which to split. Defaults to 0.

    Returns:
        `ndarray`: A sub-array of x.
    """
    arrays = split(x, indices_or_sections, axis)
    x_out = arrays[section_to_keep]
    return x_out
