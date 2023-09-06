"""Tests for reward_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.reward_wrappers import (
    AssignObsToReward,
    LogisticTransformReward,
    MaskReward,
    NullActionReward,
    ThresholdReward,
    ZeroReward,
)

# %% Test ZeroReward
print("\nTest ZeroReward...")
rand_env = RandomEnv()
wrapped_env = ZeroReward(rand_env)

(_, reward, _, _, _) = wrapped_env.step(wrapped_env.action_space.sample())
print(f"reward = {reward}")

# %% Test AssignObsToReward
print("\nTest AssignObsToReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=-1, high=1),
                "b": Box(low=-1, high=1, shape=()),
            }
        ),
        "action_space": MultiDiscrete([1]),
        "reward_space": Box(0, 0),
    }
)

ass_env = AssignObsToReward(rand_env, "a")
(obs, reward, term, trunc, info) = ass_env.step(ass_env.action_space.sample())
print(f"obs = {obs}")
print(f"reward = {reward}")

ass_env = AssignObsToReward(rand_env, "b")
(obs, reward, term, trunc, info) = ass_env.step(ass_env.action_space.sample())
print(f"obs = {obs}")
print(f"reward = {reward}")
# %% Test MaskReward
print("\nTest MaskReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((3, 4))}),
        "action_space": MultiDiscrete([3, 3, 3, 3]),
        "reward_space": Box(0, 0),
    }
)
binary_env = MaskReward(rand_env, "a", reward=0.1)
action = array([0, 0, 0, 2])

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"obs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with rewarding (penalizing) invalid actions
binary_env = MaskReward(rand_env, "a", reward=-0.1, reward_valid_actions=False)

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with accounting for null actions
binary_env = MaskReward(rand_env, "a", ignore_null_actions=False)

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
rand_env = RandomEnv()

thresh_env = ThresholdReward(rand_env, -2)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"reward (Box space) = {reward}")

# Test with MultiBinary space
thresh_env = ThresholdReward(rand_env, 1)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"reward (MultiBinary space) = {reward}")

# %% Test LogisticTransformReward
print("\nTest LogisticTransformReward...")

rand_env = RandomEnv({"reward_space": Box(low=-2, high=-1)})  # %% Done
log_env = LogisticTransformReward(rand_env)

unwrapped_reward = 0
wrapped_reward = log_env.reward(unwrapped_reward)
print(f"unwrapped reward = {unwrapped_reward}")
print(f"wrapped reward = {wrapped_reward}")

(_, reward, _, _, _) = log_env.step(log_env.action_space.sample())
print(f"reward (via step) = {reward}")

# %% Done
print("done")
