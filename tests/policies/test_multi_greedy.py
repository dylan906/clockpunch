"""Tests for multi_greedy.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import Inf

# Punch Clock Imports
from punchclock.policies.multi_greedy import MultiGreedy

# %% Make generic observation and action spaces
print("\nMaking spaces...")
num_sensors = 2
num_targets = 3
obs_space = Dict(
    {
        "observations": Dict(
            {"a": Box(-Inf, Inf, shape=[num_targets, num_sensors], dtype=float)}
        ),
        "action_mask": MultiBinary((num_targets + 1, num_sensors)),
    }
)
action_space = MultiDiscrete([num_targets + 1] * num_sensors)

# %% Test instantiation
print("\nTest instantiation...")
policy = MultiGreedy(
    observation_space=obs_space,
    action_space=action_space,
    key="a",
)
print(f"policy = {policy}")

# %% Test computeAction
action = policy.computeAction(obs=policy.observation_space.sample())
