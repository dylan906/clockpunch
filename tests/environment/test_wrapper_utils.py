"""Tests for wrapper_utils.py."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.wrappers.filter_observation import FilterObservation
from numpy import Inf, array, ones, sum
from numpy.random import rand
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.misc_wrappers import IdentityWrapper
from punchclock.environment.wrapper_utils import (
    SelectiveDictObsWrapper,
    SelectiveDictProcessor,
    checkDictSpaceContains,
    convertBinaryBoxToMultiBinary,
    convertNumpyFuncStrToCallable,
    getIdentityWrapperEnv,
    getNumWrappers,
    getSpaceClosestCommonDtype,
    getSpacesRange,
    getWrapperList,
    getXLevelWrapper,
    remakeSpace,
)

# %% Tests
# %% Test SelectiveDictProcessor
print("\nTest SelectiveDictProcessor...")
in_dict = {
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [1, 2, 3],
}
sdp = SelectiveDictProcessor([sum], ["a", "b"])
out_dict = sdp.applyFunc(in_dict)
print(f"in_dict = {in_dict}")
print(f"out_dict = {out_dict}")

# %% Test SelectiveDictObsWrapper
print("\nTest SelectiveDictObsWrapper...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=[2, 3]),
                "b": MultiDiscrete([1, 1, 1]),
            }
        ),
    }
)


def testFunc(x):
    """Test function."""
    return sum(x).reshape((1,))


new_obs_space = deepcopy(rand_env.observation_space)
new_obs_space["a"] = Box(-Inf, Inf)
sdow = SelectiveDictObsWrapper(
    env=rand_env,
    funcs=[testFunc],
    keys=["a"],
    new_obs_space=new_obs_space,
)
print(f"unwrapped obs space = {rand_env.observation_space['a']}")
print(f"wrapped obs space = {sdow.observation_space['a']}")

# %% Test getNumWrappers
print("\nTest getNumWrappers()...")
env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Dict(
                    {
                        "aa": Box(0, 1, shape=[2, 3]),
                        "ab": MultiDiscrete([1, 1, 1]),
                    }
                ),
                "b": MultiDiscrete([2, 3, 2]),
            }
        ),
        "action_space": MultiDiscrete([3, 4, 3]),
    }
)
env_wrapped1 = FilterObservation(env, ["a", "b"])
env_wrapped2 = FilterObservation(env_wrapped1, ["a"])

base_env = deepcopy(env)
out = getNumWrappers(base_env)
print(f"Number of wrappers (base environment) = {out}")

out = getNumWrappers(env_wrapped2)
print(f"Number of wrappers (wrapped environment) = {out}")

# %% Test getWrapperList
print("\nTest getWrapperList()...")
wrappers = getWrapperList(env=env_wrapped2)
print(f"wrappers = {wrappers}")

# %% Test checkDictSpaceContains
print("\nTest checkDictSpaceContains()...")
report = checkDictSpaceContains(
    env.observation_space,
    {
        "a": rand(1, 2),
        "b": env.observation_space["b"].sample(),
    },
)
print(f"Check dict report: {report}")

# %% Test remakeSpace
print("\nTest remakeSpace()...")
space = Box(0, 1, shape=(2, 2), dtype=int)
new_space = remakeSpace(space, shape=(3, 3))
print(f"space = {space}")
print(f"new_space = {new_space}")

space = Box(0, 1, shape=(2, 2), dtype=int)
new_space = remakeSpace(space, lows=-1, highs=5, dtype=float)
print(f"space = {space}")
print(f"new_space = {new_space}")

# %% Test getSpaceClosestCommonDtype
print("\nTest getSpaceClosestCommonDtype...")

common_dtype = getSpaceClosestCommonDtype(
    Box(0, 1, shape=(2, 2), dtype=int),
    Box(0, 1, shape=(1, 3), dtype=float),
)
print(f"dtype = {common_dtype}")
common_dtype = getSpaceClosestCommonDtype(
    Box(0, 1, shape=(2, 2), dtype=int),
    MultiBinary(2),
)
print(f"dtype = {common_dtype}")

common_dtype = getSpaceClosestCommonDtype(
    MultiDiscrete([2, 3]),
    MultiBinary(2),
)
print(f"dtype = {common_dtype}")

# %% Test getSpacesRange
print("\nTest getSpacesRange...")
spacerange = getSpacesRange(
    Box(-3, 1, shape=(2, 2), dtype=int),
    Box(0, Inf, shape=(1, 3), dtype=float),
)
print(f"space range = {spacerange}")

spacerange = getSpacesRange(
    MultiBinary((2, 2)),
    MultiDiscrete((2, 3, 4)),
)
print(f"space range = {spacerange}")

spacerange = getSpacesRange(
    MultiBinary((2, 2)),
    Discrete(3),
)
print(f"space range = {spacerange}")

# %% Test convertBinaryBoxToMultiBinary
print("\nTest convertBinaryBoxToMultiBinary...")
space = Box(0, 1, shape=(2, 3), dtype=int)
new_space = convertBinaryBoxToMultiBinary(space)
print(f"new space = {new_space}")

space = Box(0, 1, shape=(2, 3))
new_space = convertBinaryBoxToMultiBinary(space)
print(f"new space = {new_space}")

# %% Test getIdentityWrapperEnv
print("\nTest getIdentityWrapperEnv...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=[2, 3]),
            }
        )
    }
)
wrapped_env = FilterObservation(
    IdentityWrapper(FilterObservation(rand_env, ["a"])),
    ["a"],
)

ienv = getIdentityWrapperEnv(wrapped_env)
print(ienv)

try:
    getIdentityWrapperEnv(rand_env)
except Exception as e:
    print(e)

# %% Test getXLevelWrapper
print("\nTest getXLevelWrapper...")
wrapped_env = IdentityWrapper(IdentityWrapper(RandomEnv()))
xlevel = getXLevelWrapper(wrapped_env, 2)
print(f"wrapped env = {wrapped_env}")
print(f"xlevel wrapper = {xlevel}")

# %% Test convertNumpyFuncStrToCallable
print("\nTest convertNumpyFuncStrToCallable...")

Np_partial = convertNumpyFuncStrToCallable("sum")
print(Np_partial([1, 2, 3]))

Np_partial = convertNumpyFuncStrToCallable("sum", axis=1)
print(Np_partial(ones((2, 2))))
# %% done
print("done")
