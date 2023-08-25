"""Tests for reward_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.reward_wrappers import (
    NullActionReward,
    ThresholdReward,
    VismaskViolationReward,
)

# %% Test VismaskViolationReward
print("\nTest VismaskViolationReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((2, 4))}),
        "action_space": MultiDiscrete([3, 3, 3, 3]),
    }
)
binary_env = VismaskViolationReward(rand_env, "a", reward=0.1)
action = array([0, 0, 0, 2])

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"obs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with rewarding (penalizing) invalid actions
binary_env = VismaskViolationReward(
    rand_env, "a", reward=-0.1, reward_valid_actions=False
)

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# %% Test NullActionReward
print("\nTest NullActionReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((2, 2))}),
        "action_space": MultiDiscrete([3, 3]),
    }
)
nar_env = NullActionReward(rand_env, reward=-0.1)

action = array([0, 2])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"reward={reward}")


nar_env = NullActionReward(rand_env, reward_null_actions=False)

action = array([0, 0])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"reward = {reward}")

# %% Test ThresholdReward
print("\nTest ThresholdReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {"a": Box(low=-1, high=1), "b": MultiBinary((1,))}
        ),
        "action_space": MultiDiscrete([1]),
    }
)

thresh_env = ThresholdReward(rand_env, "a", -2)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"reward (Box space) = {reward}")

# Test with MultiBinary space
thresh_env = ThresholdReward(rand_env, "b", 1)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"reward (MultiBinary space) = {reward}")
# %% Done
print("done")
