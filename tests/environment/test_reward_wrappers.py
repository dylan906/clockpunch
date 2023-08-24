"""Tests for reward_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.reward_wrappers import BinaryReward

# %% Test BinaryReward
print("\nTest BinaryReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((2, 4))}),
        "action_space": MultiDiscrete([3, 3, 3, 3]),
    }
)
binary_env = BinaryReward(rand_env, "a", reward=0.1)
action = array([0, 0, 0, 2])

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"obs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with rewarding (penalizing) invalid actions
binary_env = BinaryReward(
    rand_env, "a", reward=-0.1, reward_valid_actions=False
)

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")
# %% Done
print("done")
