"""Test for gym_utils.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces.utils import flatten
from numpy import array

# Punch Clock Imports
from punchclock.environment.gym_utils import (
    boxActions2MultiDiscrete,
    chunker,
)

# %% Test utility functions
out = chunker([1, 2, 3, 4, 5, 6], 2)
print(f"chunker output = {out}")

out = boxActions2MultiDiscrete(array([22, 3.3, 6, 7, -1.1, 0.0]), 2)
print(f"boxActions2MultiDiscrete output = {out}")

test_space = MultiDiscrete([2, 2, 2])
sample = test_space.sample()
flat_sample = flatten(test_space, sample)
out = boxActions2MultiDiscrete(flat_sample, num_actions=2)
print(f"original =? output: {out ==sample}")
