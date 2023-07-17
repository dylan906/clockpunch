"""Tests for wrapper_utils.py."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.wrappers.filter_observation import FilterObservation
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.wrapper_utils import (
    SelectiveDictProcessor,
    checkDictSpaceContains,
    getNumWrappers,
    getWrapperList,
)

# %% Tests
# %% Test SelectiveDictProcessor
in_dict = {
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [1, 2, 3],
}
sdp = SelectiveDictProcessor([sum], ["a", "b"])
out_dict = sdp.applyFunc(in_dict)
print(f"in_dict = {in_dict}")
print(f"out_dict = {out_dict}")

# %% Test getNumWrappers
print("\nTest getNumWrappers()...")
env = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a": gym.spaces.Dict(
                    {
                        "aa": gym.spaces.Box(0, 1, shape=[2, 3]),
                        "ab": gym.spaces.MultiDiscrete([1, 1, 1]),
                    }
                ),
                "b": gym.spaces.MultiDiscrete([2, 3, 2]),
            }
        ),
        "action_space": gym.spaces.MultiDiscrete([3, 4, 3]),
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
