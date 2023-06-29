"""Test script for policy_utils.py module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from matplotlib import pyplot as plt
from numpy import array
from numpy.random import randint

# Punch Clock Imports
from punchclock.policies.policy_utils import (
    appendSubsidyRow,
    multipleAssignmentCost,
    noAssignmentCost,
    nonVisibleAssignmentCost,
    switchingCost,
    upperConfidenceBounds,
)

# %% Build test inputs
sat_names = [1, 2, 3]
sens_names = ["A", "B"]
sat_states = array([[100, 0, 0], [0, 0, 100], [0, 100, 100]])
sens_states = array([[100, 0, 0], [0, 50, 0]])


# %% Test appendSubsidyRow
subsidy = 0.1
sub_rwd = array([[1, 2, 3], [1, 1, 1]])
subisized = appendSubsidyRow(sub_rwd, subsidy)
print(f"subsidized = {subisized}")


# %% Test UCB
ucb_Qt = 4.2
ucb_Nt = 2
ucb_C = 1
uxb_max = 100
# nominal case
ucb_value = upperConfidenceBounds(10, ucb_Qt, ucb_Nt, ucb_C, uxb_max)
print(f"ucb_vals = \n{ucb_value}")

# edge case Nt=0
ucb_value = upperConfidenceBounds(10, ucb_Qt, 0, ucb_C, uxb_max)
print(f"ucb_vals = \n{ucb_value}")

# edge case t<1
ucb_value = upperConfidenceBounds(0, ucb_Qt, ucb_Nt, ucb_C, uxb_max)
print(f"ucb_vals = \n{ucb_value}")

# %% Test switching cost
sc_last_act = array([[0, 1, 1], [1, 1, 1]])
sc_act = array([[1, 0, 1], [1, 0, 1]])
sc_out = switchingCost(sc_last_act, sc_act, 2.1)
print(f"switch cost = \n{sc_out}")

# %% Test multipleAssignmentCost()
print("\nTest multipleAssignmentCost()...")

test_actions = array([[1, 0], [1, 0]])
cost = 100
[ma_cost, report] = multipleAssignmentCost(test_actions, False, cost)
print(f"sensor assigned to multiple targets = {ma_cost}")
print(f"multiple assignment report = {report}")

test_actions = array([[1, 1], [0, 0]])
[ma_cost, report] = multipleAssignmentCost(test_actions, False, cost)
print(f"multiple sensors to one target = {ma_cost}")
print(f"multiple assignment report = {report}")

test_actions = array([[0, 0], [0, 0], [1, 1]])
[ma_cost, report] = multipleAssignmentCost(test_actions, True, cost)
print(f"multiple sensors assigned to inaction = {ma_cost}")
print(f"multiple assignment report = {report}")

test_actions = array([[1, 0], [0, 1], [1, 0]])
[ma_cost, report] = multipleAssignmentCost(test_actions, True, cost)
print(f"sensors assigned to targets and inaction = {ma_cost}")
print(f"multiple assignment report = {report}")

# %% Test nonVisibleAssignmentCost
print("\nTest nonVisibleAssignmentCost()...")
actions = array([[1, 1], [0, 0]])
print(f"actions = \n{actions}")
vis_map = randint(0, 2, size=(2, 2))
print(f"vis_map = \n{vis_map}")

[nv_cost, num_nv_penalties] = nonVisibleAssignmentCost(
    actions=actions,
    vis_map=vis_map,
    cost=7,
)
print(f"nv_cost = {nv_cost}")
print(f"num non-vis penalties = {num_nv_penalties}")

# %% Test noAssignmentCost()
print("\nTest noAssignmentCost()...")

test_actions = array([[0, 0], [0, 0], [0, 0]])
na_cost = noAssignmentCost(test_actions)
print(f"2 sensors with no tasking= {na_cost}")

test_actions = array([[1, 0], [0, 0], [0, 0]])
na_cost = noAssignmentCost(test_actions)
print(f"1 sensor with no tasking= {na_cost}")


# %%
plt.show()
print("done")
