"""Tests for policy_builder.py."""
# NOTE: This sccript generates files in tests/policies/data/
# %% Imports
# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import array

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile, saveJSONFile
from punchclock.policies.policy_builder import (
    BoxConfig,
    DictConfig,
    MultiBinaryConfig,
    MultiDiscreteConfig,
    buildCustomPolicy,
    buildSpace,
    buildSpaceConfig,
)

# %% Test Space Configs
print("\nTest space configs...")
box_config = BoxConfig(
    low=array([0, 0]),
    high=array([1, 1]),
    dtype="int",
)
print(f"box config = {box_config}")

md_config = MultiDiscreteConfig(nvec=array([3, 3, 3]))
print(f"md config = {md_config}")

mb_config = MultiBinaryConfig(n=[3, 2])
print(f"mb config = {mb_config}")

dict_config1 = DictConfig(spaces={"a": md_config})
print(f"Dict config = {dict_config1}")

dict_config2 = DictConfig(
    spaces={
        "a": md_config,
        "b": DictConfig({"b1": box_config, "b2": md_config}),
    }
)
print(f"Dict config = {dict_config2}")

# Try DictConfig with forbidden key
try:
    dict_config3 = DictConfig(spaces={"space": 1})
except Exception as err:
    print(err)

# Try DictConfig with incorrectly nested entry
try:
    dict_config3 = DictConfig(spaces={"a": {"a1": 1}})
except Exception as err:
    print(err)

# %% Convert space configs to dicts and save as jsons
print("\nConvert and save space configs...")
fpath = "tests/policies/data/"
fname_list = []
for i, config in enumerate(
    [
        box_config,
        md_config,
        mb_config,
        dict_config1,
        dict_config2,
    ]
):
    # assign a counter to end of fname to prevent duplicate names
    fname = fpath + config.space + str(i)
    converted_dict = config.toDict()
    saveJSONFile(fname, converted_dict)
    fname_list.append(fname + ".json")


# %% Test buildSpace
print("\nTest buildSpace...")
# load spaces from previous test
configs = []
for fname in fname_list:
    config = loadJSONFile(fname)
    print(f"loaded config = {config}")
    configs.append(config)

spaces = []
for config in configs:
    space = buildSpace(config)
    print(f"space = {space}")

# Test config with incorrect key
bad_config = deepcopy(configs[0])
bad_config["space"] = "der"
try:
    buildSpace(bad_config)
except Exception as err:
    print(err)

# %% Test buildSpaceConfig
gym_box = Box(low=array([0, 0]), high=array([1, 1]), dtype=int)
gym_md = MultiDiscrete([3, 3, 3])
gym_mb = MultiBinary([2, 3])
gym_dict = Dict(
    {
        "a": gym_box,
        "b": Dict(
            {"b1": gym_md},
        ),
    }
)

gym_spaces = [gym_box, gym_md, gym_mb, gym_dict]
for space in gym_spaces:
    config = buildSpaceConfig(space)
    print(f"config = {config}")

# %% Test buildCustomPolicy
print("\nTest buildCustomPolicy...")
obs_space = {
    "space": "Dict",
    "observations": {
        "space": "Dict",
        "a": {
            "space": "Box",
            "low": [0, 0],
            "high": [1, 1],
        },
        "b": {"space": "MultiBinary", "n": [2, 3]},
    },
    "action_mask": {
        "space": "Box",
        "low": [0, 0],
        "high": [1, 1],
        "dtype": "int",
    },
}
act_space = {
    "space": "MultiDiscrete",
    "nvec": [2, 2],
}

policy_config = {
    "policy": "RandomPolicy",
    "observation_space": obs_space,
    "action_space": act_space,
}
pol = buildCustomPolicy(policy_config)
print(f"policy = {pol}")

# %% Done
print("done")
