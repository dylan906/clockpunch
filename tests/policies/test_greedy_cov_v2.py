"""Test greedy_cov_v2.py module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import Inf, array, diagonal, vstack
from numpy.random import seed

# Punch Clock Imports
from punchclock.common.utilities import isActionValid
from punchclock.policies.greedy_cov_v2 import GreedyCovariance

# %% Set random seed
seed(42)

# %% Test policy initialization
print("\nInitialize Policy...")
num_sensors = 2
num_targets = 4
mask_dim = num_sensors * (num_targets + 1)
obs_space = Dict(
    {
        "observations": Dict(
            {
                "est_cov": Box(
                    -Inf, Inf, shape=[num_targets, 6, 6], dtype=float
                ),
                "vis_map_est": MultiBinary((num_targets, num_sensors)),
            }
        ),
        "action_mask": Box(0, 1, shape=[mask_dim], dtype=int),
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
vis_map = obs_space.sample()["observations"]["vis_map_est"]

cov_diags = diagonal(covariance, 1, 2)
Q = policy.calcQ(cov_diags, vis_map)
print(f"Q = \n{Q}")

# %% getCovVisMask
print("\nTest getCovVisMask...")

obs = obs_space.sample()
[cov, vm, mask] = policy.getCovVisMask(obs)
print(f"cov = \n{cov}")
print(f"vis map = \n{vm}")
print(f"mask = {mask}")

# %% Choose action
print("\nTest computeAction...")
# NOTE: Because obs was randomly generated, the action mask may prevent choosing
# inaction; this doesn't matter for these tests.
action = policy.computeAction(obs)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")
# %% Choose action w/ high epsilon
print("\nChoose action with high epsilon...")

policy.epsilon = 1
action = policy.computeAction(obs)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")


# %% Test base class compute action
print("\nTest base class compute action...")

action = policy._computeSingleAction(obs)
print(f"obs['est_cov'] = \n{obs['observations']['est_cov']}")
print(f"action = {action}")

# %% Make sure invalid actions aren't selected
policy = GreedyCovariance(
    observation_space=obs_space,
    action_space=action_space,
    subsidy=0.1,
    epsilon=0,
)
for _ in range(10):
    obs = policy.observation_space.sample()
    action = policy.computeAction(obs)
    # print(f"vis_mask = \n{obs['observations']['vis_map_est']}")
    # print(f"action = {action}")
    action_mask = vstack(
        (
            obs["observations"]["vis_map_est"],
            array([1, 1]),
        )
    )
    is_val = isActionValid(
        mask=action_mask,
        action=action,
    )
    print(is_val)

# %%
print("done")
