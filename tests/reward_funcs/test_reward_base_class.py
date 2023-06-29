"""Test for reward_base_class.py."""
# %% Imports
# Third Party Imports
from numpy import array
from numpy.random import randint

# Punch Clock Imports
from scheduler_testbed.reward_funcs.reward_base_class import RewardFunc

# %% Test initialize
print("\nTest initialize...")
rf = RewardFunc(subsidies={"active_action": 1})
print(f"reward function = {rf}")
print(f"penalties = {rf.penalties}")

# %% Test calcPenalties
num_targs = 3
num_sens = 2
info = {
    "vis_map_truth": randint(0, 2, size=(num_targs, num_sens)),
    "num_sensors": num_sens,
    "num_targets": num_targs,
}
actions = randint(0, num_targs + 1, size=(num_sens))
[penalties, penalty_report] = rf.calcPenalties(info=info, actions=actions)
print(f"vis_map = \n{info['vis_map_truth']}")
print(f"actions = {actions}")
print(f"penalties = {penalties}")
print(f"penalty report = {penalty_report}")

# % Test calcSubsidies
actions = array([0, 3])
[subsidies, subsidy_report] = rf.calcSubsidies(info=info, actions=actions)
print(f"actions = {actions}")
print(f"subsidies = {subsidies}")
print(f"subsidy report = {subsidy_report}")
# %% done
print("done")
