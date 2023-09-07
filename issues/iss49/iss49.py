"""Issue #49."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.obs_wrappers import CopyObsItem
from punchclock.environment.reward_wrappers import (
    AssignObsToReward,
    LogisticTransformReward,
    ZeroReward,
)
from punchclock.environment.wrapper_utils import getNumWrappers, getWrapperList
from punchclock.policies.random_policy import RandomPolicy
from punchclock.simulation.sim_runner import SimRunner


# %% Excerpt from SimRunner
# Defined as standalone function in this script for convenience
def transformObs1ataTime(wrapped_env):
    num_env_wrappers = getNumWrappers(wrapped_env)

    obs, _ = wrapped_env.unwrapped.reset()
    strcmd = "env."
    for i in range(num_env_wrappers):
        cmd = (num_env_wrappers - i - 1) * strcmd
        # cmd = (num_env_wrappers - i) * strcmd # as written in SimRunner
        cmd = "wrapped_env." + cmd + "observation(obs)"
        obs = eval(
            cmd,
            {
                "wrapped_env": wrapped_env,
                "obs": obs,
            },
        )

    return obs


# %% Build env

env = RandomEnv(
    {
        "observation_space": Dict({"a": Box(0, 1)}),
    }
)
wrapped_env = ZeroReward(CopyObsItem(env, "a", "aa"))

wrapped_env.reset()
obs, reward, _, _, _ = wrapped_env.step(wrapped_env.action_space.sample())
print(f"obs = \n{obs}")

# %% Tests
print("\nTest single reward wrapper...")
obs = transformObs1ataTime(wrapped_env)
print(f"obs = \n{obs}")

print("\nTest multiple obs wrappers...")
wrapped_env = CopyObsItem(CopyObsItem(env, "a", "aa"), "a", "aaa")
obs = transformObs1ataTime(wrapped_env)
print(f"obs = \n{obs}")

# Fails
print("\nTest obs(reward(env))...")
wrapped_env = CopyObsItem(ZeroReward(env), "a", "aaa")
try:
    obs = transformObs1ataTime(wrapped_env)
    print(f"obs = \n{obs}")
except Exception as e:
    print(e)

print("\nTest reward(obs(env))...")
wrapped_env = ZeroReward(CopyObsItem(env, "a", "aa"))
obs = transformObs1ataTime(wrapped_env)
print(f"obs = \n{obs}")

print("\nTest reward(reward(obs(env)))...")
wrapped_env = LogisticTransformReward(ZeroReward(CopyObsItem(env, "a", "aa")))
obs = transformObs1ataTime(wrapped_env)
print(f"obs = \n{obs}")

# %% Done
print("done")
