"""Utility functions."""
# %% Imports
# Standard Library Imports
import collections.abc
import json
import os
import os.path
from ast import literal_eval
from copy import deepcopy
from itertools import groupby
from operator import ge, gt, le, lt
from pathlib import Path
from typing import Any, Callable

# Third Party Imports
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete, Space
from gymnasium.spaces.utils import flatten, unflatten
from numpy import (
    Inf,
    abs,
    arange,
    array,
    asarray,
    delete,
    diag,
    fabs,
    finfo,
    float32,
    float64,
    iinfo,
    int32,
    int64,
    ndarray,
    ones,
    pi,
    vstack,
    zeros,
)
from satvis.visibility_func import (
    calcVisAndDerVis,
    isVis,
    visDerivative,
    visibilityFunc,
)

# Punch Clock Imports
from punchclock.common.constants import getConstants

# %% Constants
MAX_FLOAT32 = finfo("float32").max
MAX_FLOAT64 = finfo("float64").max
MAX_INT32 = iinfo("int32").max
MAX_INT64 = iinfo("int64").max
MIN_FLOAT32 = finfo("float32").min
MIN_FLOAT64 = finfo("float64").min
MIN_INT32 = iinfo("int32").min
MIN_INT64 = iinfo("int64").min
RE = getConstants()["earth_radius"]


# %% Functions
def loadJSONFile(file_name: str | Path) -> dict:
    """Load in a JSON file into a Python dictionary.

    Args:
        file_name (str): name of JSON file to load
    Raises:
        FileNotFoundError: helps with debugging bad filenames

        json.decoder.JSONDecodeError: error parsing JSON file (bad syntax)

        IOError: valid JSON file is empty
    Returns:
        dict: documents loaded from the JSON file
    """
    if isinstance(file_name, Path):
        file_name = str(file_name)

    try:
        with open(file_name, "r", encoding="utf-8") as input_file:
            json_data = json.load(input_file)

    except FileNotFoundError as err:
        print(f"Could not find JSON file: {file_name}")
        raise err

    except json.decoder.JSONDecodeError as err:
        print(f"Decoding error reading JSON file: {file_name}")
        raise err

    if not json_data:
        print(f"Empty JSON file: {file_name}")
        raise IOError
    return json_data


def saveJSONFile(file_name: str, a_dict: dict) -> str:
    """Save a Python dict as a JSON file.

    Args:
        file_name (str): Absolute path and file name (excluding ".json")
        jsonable (dict): A dictionary.

    Returns:
        str: The output of json.dumps.
    """
    json_object = json.dumps(a_dict)

    if file_name[-5:] != ".json":
        file_path_str = file_name + ".json"
    else:
        file_path_str = file_name

    with safe_open_w(file_path_str) as outfile:
        outfile.write(json_object)

    print(f"JSON saved to: {file_path_str}")

    return json_object


def safe_open_w(path):
    """Open "path" for writing, creating any parent directories as needed."""
    # From https://stackoverflow.com/questions/23793987/write-a-file-to-a-directory-that-doesnt-exist # noqa
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w")


def cumAvg(val: float, old_avg: float, num_entries: int) -> float:
    """Cumulative average.

    Args:
        val (float): ith data point
        old_avg (float): Cumulative average to i-1
        num_entries (int): i

    Returns:
        float: Cumulative average to i

    https://en.wikipedia.org/wiki/Moving_average
    """
    return old_avg + (val - old_avg) / (num_entries + 1)


def actionSpace2Array(
    actions: ndarray[int],
    num_sensors: int,
    num_targets: int,
) -> ndarray[int]:
    """Convert actions in MultiDiscrete (1d array) format to 2D array.

    M = number of sensors
    N = number of targets

    Args:
        actions (ndarray[int]): (M, ) Valued 0 to N, where N indicates inaction.
        num_sensors (int): Number of sensors.
        num_targets (int): Number of targets.

    Returns:
        ndarray[int]: (N+1, M) array of 0s and 1s. A 1 in the [m, n] position
            indicates the m-th sensor is tasked to the n-th target (or if the 1
            is in the last row, inaction by the sensor).
    """
    actions_array = zeros([num_targets + 1, num_sensors])
    for i, val in enumerate(actions):
        actions_array[int(val), i] = 1

    actions_array = actions_array.astype(int)
    return actions_array


class MaskConverter:
    """Handles conversions between visibility masks and action masks.

    Used when interfacing with flattened MultiDiscrete action spaces.

    Notation:
        N = num_targets
        M = num_sensors

    Definitions:
        2d visibility mask: [N, M] array
        2d action mask: [(N + 1), M] array, where bottom row is inaction and is
            always ones.
        1d action mask: A flattened 2d action mask, where the all entries in the
            i'th column are listed before all entries in the (i+1)'th column.
        There is no 1d visibility mask.
    """

    def __init__(self, num_targets: int, num_sensors: int):
        """Initializer MaskConverter."""
        self.vis_mask_space = Box(0, 1, shape=[num_targets, num_sensors], dtype=int)
        self.mask2d_space = Box(0, 1, shape=[num_targets + 1, num_sensors], dtype=int)
        self.mask2d_transpose_space = Box(
            0, 1, shape=[num_sensors, num_targets + 1], dtype=int
        )
        self.num_targets = num_targets
        self.num_sensors = num_sensors

        # save MD action space for reference
        self.action_space = MultiDiscrete([num_targets + 1] * num_sensors)

    def convertActionMaskFrom1dTo2d(self, mask1d: ndarray) -> ndarray:
        """Convert 1d action mask to 2d action mask.

        in = [0 0 0 1 0 0 0 1]
        out = [
            [0 0]
            [0 0]
            [0 0]
            [1 1]
            ]
        """
        mask2d = unflatten(self.mask2d_transpose_space, mask1d)
        mask2d = mask2d.transpose()

        return mask2d

    def convertActionMaskFrom2dTo1d(self, mask2d: ndarray) -> ndarray:
        """Convert a 2d action mask to a flat action mask (includes inaction).

        in = [
            [1 0]
            [0 1]
            [1 1]
            ]
        out = [1 0 1 0 1 1]
        """
        assert self.mask2d_space.contains(mask2d)
        flat_mask = flatten(self.mask2d_space, mask2d.transpose())
        return flat_mask

    def convert2dVisMaskTo1dActionMask(self, vis_mask: ndarray[int]) -> ndarray[int]:
        """Convert a 2d visibility mask to a flat action mask (includes inaction).

        in = [
            [1 0]
            [0 1]
            ]
        out = [1 0 1 0 1 1]

        Inaction is always assumed to be 1 (allowed).
        """
        assert self.vis_mask_space.contains(vis_mask)

        num_cols = vis_mask.shape[1]
        vis_mask = vstack((vis_mask, ones([1, num_cols])))
        # convert to int separately b/c vstack optional arg doesn't work
        vis_mask = vis_mask.astype(int)
        flat_mask = self.convertActionMaskFrom2dTo1d(vis_mask)
        return flat_mask

    def appendInactionRowToActionMask(self, mask: ndarray[int]) -> ndarray[int]:
        """Convert a 2d vis mask to a 2d action mask.

        Appends a row of 1s to bottom of vis mask, denoting inaction.

        Args:
            mask (ndarray[int]): (N, M) Binary array.

        Returns:
            ndarray[int]: (N+1, M) Binary array.
        """
        assert (
            mask.shape[0] == self.num_targets
        ), f"mask.shape = {mask.shape}; number of rows must == num_targets."
        action_mask2d = vstack(
            (
                mask,
                ones((1, self.num_sensors), dtype=int),
            )
        )
        return action_mask2d

    def removeInactionsFrom1dMask(self, mask1d: ndarray) -> ndarray:
        """Convert 1d mask with inactions to 1d mask without inactions.

        Removes every n-th element from a 1d array, except the 0th element.

        in = [1, 0, 1, 0, 0, 1]
        out = [1, 0, 0, 0]
        """
        null_action_idx = self.num_targets
        # Get all indices to remove; skip 0th entry because it is always 0
        indices_to_remove = arange(null_action_idx, mask1d.size, null_action_idx + 1)
        mask_no_nullaction = delete(mask1d, indices_to_remove)
        return mask_no_nullaction


# %% Helper print function
def printNestedDict(d, level=0):
    """Print all entries of a nested dict with spaces denoting nest level."""
    space = " "
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{space * level}{k} :")
            level_copy = level + 1
            printNestedDict(v, level=level_copy)
        else:
            print(f"{space * level}{k} : {v}")


# %% Check if multidiscrete action array violates NxM vis mask
def isActionValid(mask: ndarray, action: ndarray) -> bool:
    """Return True if all actions are valid.

    Args:
        mask (ndarray): 2d binary array.
        action (ndarray): 1d array where length == number of cols in mask, and
            values are 0 to number of rows in mask.
    """
    valid = [None] * len(action)
    for i, (act, col) in enumerate(zip(action, mask.T)):
        if col[act] == 1:
            valid[i] = True
        else:
            valid[i] = False

    all_actions_valid = False not in valid
    return all_actions_valid


# %% allEqual
def allEqual(iterable) -> bool:
    """Check if all items in an iterable are identical."""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# %% Convert string to numpy array
def fromStringArray(array_string: str) -> ndarray:
    """Convert an array string to a numpy array."""
    # Remove extra spaces after "[", then replace spaces with ","
    # This process sometimes inserts "[," (seems to only happen with long strs)
    # so remove those on a separate pass.
    # The same process sometimes adds ",,", so replace those with "," on a separate
    # pass.
    array_string = ",".join(array_string.replace("[ ", "[").split())
    array_string = array_string.replace("[,", "[")
    array_string = array_string.replace(",,", ",")
    return array(literal_eval(array_string))


# %% Convert ndarray to list
def array2List(obj):
    """Convert ndarrays to lists for json conversion."""
    if isinstance(obj, ndarray):
        return obj.tolist()
    raise TypeError("Not serializable")


# %% Get inequality func from str
def getInequalityFunc(inequality_str: str) -> Callable:
    """Get inequality from string representation."""
    inequality_map = {"<=": le, ">=": ge, "<": lt, ">": gt}

    assert (
        inequality_str in inequality_map.keys()
    ), "inequality_str must be one of ['<=', '>=', '<', '>']."

    inequality_func = inequality_map[inequality_str]

    return inequality_func


# %% get info from env
def getInfo(env: Env) -> dict:
    """Gets info from env in a safe way."""
    env_copy = deepcopy(env)
    _, info = env_copy.reset()
    return info


# %% Recusrively convert a dict of items to primitives
def recursivelyConvertDictToPrimitive(in_dict: dict) -> dict:
    """Recursively convert dict entries into primitives."""
    out = {}
    # Loop through key-value pairs of in_dict. If a value is a dict, then recurse.
    # Otherwise, convert value to a JSON-able type. Special handling if the
    # value is a list. Lists of dicts are recursed; lists of non-dicts and
    # empty lists are converted to JSON-able as normal.
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out[k] = recursivelyConvertDictToPrimitive(v)
        elif isinstance(v, list):
            if len(v) == 0:
                out[k] = [convertToPrimitive(a) for a in v]
            elif isinstance(v[0], dict):
                out[k] = [recursivelyConvertDictToPrimitive(a) for a in v]
            else:
                out[k] = [convertToPrimitive(a) for a in v]
        else:
            out[k] = convertToPrimitive(v)
    return out


def convertToPrimitive(entry: Any) -> list:
    """Convert a non-serializable object into a JSON-able type.

    Partner function to recursivelyConvertDictToPrimitive. Probably should not
    use this on its own.
    """
    if isinstance(entry, ndarray):
        # numpy arrays need their own tolist() method to convert properly.
        out = entry.tolist()
    elif isinstance(entry, set):
        out = list(entry)
    elif isinstance(entry, (float32, int64)):
        out = entry.item()
    else:
        out = entry

    return out


def findNearest(
    x: ndarray,
    val: float | int,
    round: str = None,
    return_index: bool = False,
):
    """Get the value of x that is closest to val.

    Args:
        x (ndarray): Array to search
        val (float | int): Value to search for nearest.
        round (str, optional): ["down" | "up" | None] Whether to search for the
            nearest entry of x that is less/greater than val. Defaults to None.
        return_index (bool, optional): If True, returns index as well as value
            of x. Defaults to False.

    Returns:
        float: Value of x closest to val.
        float, int: If return_index is True, return both value and index of x.
    """
    original = x
    # need to convert to float to use Inf
    x = deepcopy(asarray(x, dtype=float))
    if round == "down":
        x[x > val] = Inf
    elif round == "up":
        x[x < val] = -Inf

    idx = (abs(x - val)).argmin()

    if return_index is False:
        return original[idx]
    elif return_index is True:
        return original[idx], idx


# %% Recursively update a dict
def updateRecursive(d: dict, u: dict) -> dict:
    """Recursively update nested dict d with dict u."""
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = updateRecursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# %% Saturate Inf
def saturateInf(x: ndarray, dtype: type = float) -> ndarray:
    """Replace Inf with max float."""
    dtype_map = {
        float: [finfo(float).min, finfo(float).max],
        float32: [MIN_FLOAT32, MAX_FLOAT32],
        float64: [MIN_FLOAT64, MAX_FLOAT64],
        int: [iinfo(int).max, iinfo(int).max],
        int32: [MIN_INT32, MAX_INT32],
        int64: [MIN_INT64, MAX_INT64],
    }

    x_new = deepcopy(x)
    x_new[x_new == -Inf] = dtype_map[dtype][0]
    x_new[x_new == Inf] = dtype_map[dtype][1]

    return x_new


# %% Chained get
def chainedGet(dictionary: dict, *args, default: Any = None) -> Any:
    """Get a value nested in a dictionary by its nested path.

    Parameters:
        dictionary (dict): The dictionary to search in.
        *args: The nested path to the value.
        default (Any, optional): The default value to return if the nested value
            is not found. Defaults to None.

    Returns:
        Any: The nested value if found, otherwise the default value.
    """
    value_path = list(args)
    dict_chain = dictionary
    while value_path:
        try:
            dict_chain = dict_chain.get(value_path.pop(0))
        except AttributeError:
            return default

    return dict_chain


# %% chainedDelete
def chainedDelete(dictionary: dict, keys_path: list[str]) -> dict:
    """Delete a value nested in a dictionary by its nested path.

    If path keys_path is to a non-existent item, then the dictionary is returned
    unmodified.

    Args:
        dictionary (dict): The dictionary to delete the value from.
        keys_path (list[str]): The nested path of keys to traverse in the dictionary.

    Returns:
        dict: The modified dictionary after deleting the value.

    """
    if keys_path and dictionary:
        key = keys_path.pop(0)
        if key in dictionary:
            if keys_path:
                chainedDelete(dictionary[key], keys_path)
            else:
                del dictionary[key]
    return dictionary


# %% chainedAppend
def chainedAppend(dictionary: dict, keys: list[str], value: Any) -> dict:
    """Appends a value to a nested dictionary using a list of keys.

    Args:
        dictionary (dict): The dictionary to append the value to.
        keys (list[str]): The list of keys representing the nested structure.
        value (Any): The value to be appended.

    Returns:
        dict: The updated dictionary with the value appended.
    """
    if keys and dictionary is not None:
        key = keys.pop(0)
        if keys:
            if key not in dictionary:
                dictionary[key] = {}
            chainedAppend(dictionary[key], keys, value)
        else:
            dictionary[key] = value
    return dictionary


# %% chainedConvertDictSpaceToDict
def chainedConvertDictSpaceToDict(
    dict_space: gym.spaces.Dict,
) -> dict[str, gym.spaces.Space]:
    """Convert a gym.spaces.Dict to a dict.

    Args:
        dict_space (gym.spaces.Dict): The gym.spaces.Dict to be converted.

    Returns:
        dict: The converted dict.
    """
    out_dict = {}
    for key, space in dict_space.spaces.items():
        if isinstance(space, gym.spaces.Dict):
            out_dict[key] = chainedConvertDictSpaceToDict(space)
        else:
            out_dict[key] = space
    return out_dict


# %% chainedConvertDictToDictSpace
def chainedConvertDictToDictSpace(
    dictionary: dict[str, dict | Space]
) -> gym.spaces.Dict:
    """Convert a dict to a gym.spaces.Dict."""
    out_dict_space = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            out_dict_space[key] = chainedConvertDictToDictSpace(value)
        else:
            out_dict_space[key] = value
    return gym.spaces.Dict(out_dict_space)


# %% safeGetattr
def safeGetattr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely retrieves the value of an attribute from an object, even if the attribute is nested.

    Args:
        obj (object): The object from which to retrieve the attribute.
        attr (str): The name of the attribute to retrieve. It can be a nested attribute separated by dots.
        default (optional): The default value to return if the attribute is not found. Defaults to None.

    Returns:
        The value of the attribute if found, otherwise the default value.

    """
    try:
        parts = attr.split(".")
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part, default)
            else:
                obj = getattr(obj, part, default)
        return obj
    except AttributeError:
        return default


def fpe_equals(value: float, expected: float, resolution: float = None) -> bool:
    """Check if `value` and `expected` are approximately equal considering FPE.

    This function checks if the difference between `value` and `expected` is less
    than the resolution, which represents the floating point error (FPE). If the
    resolution is not provided, it defaults to the machine's resolution for float.

    Args:
        value (float): The value to be compared.
        expected (float): The expected value of `value` without considering FPE.
        resolution (float, optional): The maximum difference for which `value` and
            `expected` will be considered equal. Defaults to machine's resolution.

    Returns:
        bool: True if `value` and `expected` are approximately equal considering
        FPE, False otherwise.
    """
    if resolution is None:
        resolution = finfo(float).resolution
    check = fabs(value - expected) < resolution
    return check
