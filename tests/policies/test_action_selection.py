"""Test script for action bank module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
import itertools

# Third Party Imports
from numpy import array, zeros
from numpy.random import rand, seed

# Punch Clock Imports
from punchclock.common.utilities import isActionValid
from punchclock.policies.action_selection import (
    argmaxTieBreak,
    epsGreedy,
    epsGreedyMask,
)

# %% Set RNG seed
seed(42)
# %% Test tie breaker
print("\nTest tie breaker...")
values = array([3, 3, 1])
sum_actions = zeros(len(values))
# Actions 0 and 1 should be chosen about evenly; action 2 should never be chosen.
for _ in range(100):
    ra_out = argmaxTieBreak(values)
    sum_actions[ra_out] += 1
print(f"values = {values}")
print(f"sum_actions (number of times picked) = {sum_actions}")

# %% Test e-Greedy
print("\nTest epsGreedy...")
# normal test
Q = array([[0, 0, 2, 1], [1, 1, 1, 1]])
epsilon = 0.01
egreedy_out = epsGreedy(Q, epsilon)
print(f"Q = \n{Q}")
print(f"actions selected = {egreedy_out}")

# Test with high epsilon
egreedy_out = epsGreedy(Q, epsilon=1)
print(f"actions selected (large eps) = {egreedy_out}")

# Test min and max options selected
print("\nMultiple iterations of action selection:")
for _ in range(10):
    egreedy_out = epsGreedy(Q, epsilon=1)
    print(f"  actions selected = {egreedy_out}")

# %% Test epsGreedyMask
print("\nTest epsGreedyMask...")
# Test with tied Q/random Q and fully random/no random actions
Qs = [zeros(shape=(4, 2), dtype=int), rand(4, 2)]
epsilons = [0, 1]
mask = array([[1, 1], [0, 0], [0, 1], [1, 0]])
# print("Test with mask:")
print(f"mask = \n{mask}")
combo_list = itertools.product(epsilons, Qs)
for e, Q in combo_list:
    actions = epsGreedyMask(Q=Q, epsilon=e, mask=mask)
    print(f"Q = {Q}")
    print(f"epsilon = {e}")
    print(f"actions = {actions}")
    print(f"valid? {isActionValid(mask, actions)}")

# Test with default vis_mask
actions = epsGreedyMask(Q=Q, epsilon=1)
print("Test with default (no mask):")
print(f"actions = {actions}")
