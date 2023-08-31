"""Test greedy_cov_v2.py module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import Inf, diagonal
from numpy.random import seed

# Punch Clock Imports
from punchclock.common.utilities import MaskConverter, isActionValid
from punchclock.policies.greedy_cov_v2 import GreedyCovariance

# %% Set random seed
seed(42)

# %% Test policy initialization
print("\nInitialize Policy...")
num_sensors = 2
num_targets = 4
mask_dim = (num_targets + 1, num_sensors)
obs_space = Dict(
    {
        "observations": Dict(
            {
                "est_cov": Box(
                    -Inf, Inf, shape=[num_targets, 6, 6], dtype=float
                ),
                "vis_mask": MultiBinary((num_targets, num_sensors)),
            }
        ),
        "action_mask": MultiBinary(mask_dim),
    }
)

action_space = MultiDiscrete([num_targets + 1] * num_sensors)

policy = GreedyCovariance(
    observation_space=obs_space,
    action_space=action_space,
    subsidy=0.1,
    epsilon=0.01,
)
print(f"policy.subsidy = {policy.subsidy}")
print(f"policy.action_space = {policy.action_space}")
print(f"policy.observation_space = {policy.observation_space}")

# %% Calc Q
print("\nCalculate Q...")
covariance = obs_space.sample()["observations"]["est_cov"]

cov_diags = diagonal(covariance, 1, 2)
Q = policy.calcQ(cov_diags)
print(f"Q = \n{Q}")


# %% Choose action
print("\nTest computeAction...")
# NOTE: Because obs was randomly generated, the action mask may prevent choosing
# inaction; this doesn't matter for these tests.
obs = policy.observation_space.sample()
action = policy.computeAction(obs)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")
# %% Choose action w/ high epsilon
print("\nChoose action with high epsilon...")

policy.epsilon = 1
action = policy.computeAction(obs)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")

# %% Make sure invalid actions aren't selected
policy = GreedyCovariance(
    observation_space=obs_space,
    action_space=action_space,
    subsidy=0.1,
    epsilon=0,
)
mask_converter = MaskConverter(num_targets=num_targets, num_sensors=num_sensors)
for _ in range(10):
    obs = policy.observation_space.sample()
    obs["action_mask"] = mask_converter.convert2dVisMaskTo2dActionMask(
        obs["observations"]["vis_mask"]
    )
    action = policy.computeAction(obs)
    is_val = isActionValid(
        mask=obs["action_mask"],
        action=action,
    )
    print(is_val)

# %%
print("done")
