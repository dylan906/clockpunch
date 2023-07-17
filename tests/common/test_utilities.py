"""Test Utilities module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from gymnasium.spaces import MultiDiscrete, flatten
from numpy import array, zeros
from numpy.random import randint

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.common.utilities import (
    MaskConverter,
    calcVisMap,
    filterAgentsType,
    fromStringArray,
    isActionValid,
    printNestedDict,
    saveJSONFile,
)

# %% JSON loader
# dat_sat = loadJSONFile(
#     "/home/dylanrpenn/punchclock/tests/datafiles/" "leo_only_test_targets.json"
# )
# dat_ssn = loadJSONFile(
#     "/home/dylanrpenn/punchclock/tests/datafiles/" "dedicated_ssn_network.json"
# )
# print(f"dat_sat[0]={dat_sat[0]}")
# print(f"dat_ssn[0]={dat_ssn[0]}")

# for index in range(len(dat_sat)):
#     print(f"dat_sat[index][{index}]=")
#     print(dat_sat[index]["init_eci"])
# %% SaveJSON
print("\nsaveJSONFile()...")
fpath = "tests/common/data/test_file"
test_dict = {"a": 1}
saveJSONFile(fpath, test_dict)


# %% Test mask converter
print("\nConvert 1d mask to 2d...")
n = 3
m = 2
mc = MaskConverter(n, m)

md_action_space = MultiDiscrete([n + 1] * m)
act_md = array([3, 3])
print(f"act MD = {act_md}")
act_1d = flatten(md_action_space, act_md)
print(f"act_1d = {act_1d}")
act_2d = mc.convertActionMaskFrom1dTo2d(act_1d)
print(f"act_2d = \n{act_2d}")

act_mask_2d = array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=int)
print(f"act mask 2d = \n{act_mask_2d}")
act_mask_1d = mc.convertActionMaskFrom2dTo1d(act_mask_2d)
print(f"1d act mask = {act_mask_1d}")

vis_mask = array(
    [
        [1, 0],
        [0, 1],
        [0, 0],
    ],
    dtype=int,
)
print(f"vis mask = \n{vis_mask}")
act_mask_1d = mc.convert2dVisMaskTo1dActionMask(vis_mask)
print(f"act_mask_1d = {act_mask_1d}")

mask_no_inaction = mc.removeInactionsFrom1dMask(mask1d=act_mask_1d)
print(f"mask without inaction = {mask_no_inaction}")

# %% Test filterAgentType
list_of_agents = [
    Target("ground", 1, zeros([6]), target_filter=None),
    Sensor("ground", 1, zeros([6])),
]
der = filterAgentsType(list_of_agents, Sensor)
print(f"filtered agents = {der}")

der = filterAgentsType(list_of_agents, Target)
print(f"filtered agents = {der}")


# %% CalcVisMap
RE = 6378
sensor_states = array(
    [
        [RE + 100, 0, 0, 0, 0, 0],
        [-RE - 100, 0, 0, 0, 0, 0],
    ],
).T
target_states = array(
    [
        [RE + 200, 0, 0, 0, 0, 0],
        [-RE - 200, 0, 0, 0, 0, 0],
        [0, RE, 0, 0, 0, 0],
        [0, -RE, 0, 0, 0, 0],
    ],
).T

# test correct inputs
vis_map = calcVisMap(
    sensor_states=sensor_states,
    target_states=target_states,
    body_radius=RE,
)
print(f"vis_map = \n{vis_map}")

# test with incorrect dimensions
try:
    vis_map = calcVisMap(
        sensor_states=sensor_states.T,
        target_states=target_states,
        body_radius=RE,
    )
except Exception as err:
    print(err)
    pass

try:
    vis_map = calcVisMap(
        sensor_states=sensor_states,
        target_states=target_states.T,
        body_radius=RE,
    )
except Exception as err:
    print(err)
    pass


# %% Test printNestedDict
print("\nTest printNestedDict()...")
td = {
    "a": 1,
    "b": "thing",
    "c": {
        "c1": 1,
        "c2": 2,
    },
    "d": 4,
    "e": {
        "e1": 1,
        "e2": {
            "e21": 1,
            "e22": 2,
        },
        "e3": 3,
    },
}

printNestedDict(d=td)

# %% Test isActionValid
# Test 3 cases: no actions are valid, 1 of 2 actions are valid, both actions are
# valid.
vis_map = array([[0, 0], [1, 1], [0, 0]])
actions = [array([0, 0]), array([0, 1]), array([1, 1])]
print(f"vis_map = \n{vis_map}")
for a in actions:
    isvalid = isActionValid(vis_map, a)
    print(f"action = {a}")
    print(f"action valid? {isvalid}")

# %% fromStringArray
print("\nTest fromStringArray...")
# Test input with \n breaks
test_string = "[[1. 1. 1.]\n [1. 1. 1.]\n [1. 1. 1.]]"
out_array = fromStringArray(array_string=test_string)
print(f"input = {test_string}")
print(f"type(input) = {type(test_string)}")
print(f"output = {out_array}")
print(f"type(output) = {type(out_array)}")

# Test long string input
test_string = "[[1.     1.     1.    ]\n [0.     1.    1.  ]\n [1.52232     47.935154    40.039814  ]\n [  0.20038845   0.1535252    0.14399543]\n [  0.20006913   0.15339415   0.14389709]\n [  0.20006643   0.15338561   0.14389215]]"
out_array = fromStringArray(array_string=test_string)
print(f"input = {test_string}")
print(f"type(input) = {type(test_string)}")
print(f"output = {out_array}")
print(f"type(output) = {type(out_array)}")

# Test input with commas
test_string = "[0, 0, 0]"
out_array = fromStringArray(array_string=test_string)
print(f"input = {test_string}")
print(f"type(input) = {type(test_string)}")
print(f"output = {out_array}")
print(f"type(output) = {type(out_array)}")
# %%
print("done")
