"""Tests for binary.py reward function module."""
# %% Imports
# Third Party Imports
from numpy import array

# Punch Clock Imports
from scheduler_testbed.reward_funcs.binary import BinaryReward

# %% Test initialization
pol = BinaryReward(penalty=0.2)

# Test warnings
pol_warn = BinaryReward(penalty=0.2, penalties={"non_vis_assignment": 1})

# %% Test calc reward
# num targets = 1, num sensors = 3
obs = {"action_mask": array([1, 1, 0, 1, 0, 1])}
info = {
    "num_targets": 1,
    "num_sensors": 3,
    "vis_map_truth": array([[1, 0, 0]]),
}
actions = array([0, 0, 0])
reward = pol.calcReward(obs, info, actions)
print(f"reward = {reward}")
# %% done
print("done")
