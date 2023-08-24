"""Tests for reward_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.reward_wrappers import BinaryReward

# %% Test BinaryReward
print("\nTest BinaryReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1),
            }
        )
    }
)
binary_env = BinaryReward(rand_env)
(obs, reward, term, trunc, info) = binary_env.step(
    binary_env.action_space.sample()
)
print(f"reward={reward}")
# %% Done
print("done")
