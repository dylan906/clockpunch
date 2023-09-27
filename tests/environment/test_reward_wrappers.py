"""Tests for reward_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.misc_wrappers import RandomInfo
from punchclock.environment.reward_wrappers import (
    AssignInfoToReward,
    AssignObsToReward,
    LogisticTransformReward,
    ZeroReward,
)

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

# %% Assign InfoToReward
print("\nTest AssignInfoToReward...")
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict(
                {
                    "a": Box(low=-1, high=1),
                    "b": Box(low=-1, high=1, shape=()),
                }
            ),
            "action_space": MultiDiscrete([1]),
        }
    )
)
assinfo_env = AssignInfoToReward(rand_env, key=0)
(obs, reward, term, trunc, info) = assinfo_env.step(
    assinfo_env.action_space.sample()
)
print(f"info = {info}")
print(f"reward = {reward}")

# %% Test ZeroReward
print("\nTest ZeroReward...")
rand_env = RandomEnv()
wrapped_env = ZeroReward(rand_env)

(_, reward, _, _, _) = wrapped_env.step(wrapped_env.action_space.sample())
print(f"reward = {reward}")
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
