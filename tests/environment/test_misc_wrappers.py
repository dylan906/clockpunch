"""Tests for misc_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.environment.misc_wrappers import (
    AppendInfoItemToObs,
    IdentityWrapper,
    NumWindows,
    RandomInfo,
)

# %% Test IdentityWrapper
print("\nTest IdentityWrapper...")
rand_env = RandomEnv({"observation_space": Box(0, 1)})
identity_env = IdentityWrapper(rand_env)

print(f"identity env = {identity_env}")
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = identity_env.observation(unwrapped_obs)
print(f"unwrapped obs = {unwrapped_obs}")
print(f"wrapped obs = {wrapped_obs}")

obs, reward, term, trunc, info = identity_env.step(
    identity_env.action_space.sample()
)

identity_env = IdentityWrapper(rand_env, id="foo")
print(f"identity env = {identity_env}")
print(f"env.id = {identity_env.id}")

# %% Build env for NumWindows wrapper
print("\nBuild env for NumWindows test...")
rand_env = RandomEnv()
agents = [buildRandomAgent(agent_type="sensor") for ag in range(2)]
agents.extend([buildRandomAgent(agent_type="target") for ag in range(3)])
rand_env.agents = agents
rand_env.horizon = 10
rand_env.time_step = 100

# %% RandomInfo
print("\nTest RandomInfo...")
rand_env = RandomEnv()
_, info_unwrapped = rand_env.reset()
randinfo_env = RandomInfo(rand_env)
_, _, _, _, info_wrapped = randinfo_env.step(randinfo_env.action_space.sample())
print(f"unwrapped info = {info_unwrapped}")
print(f"wrapped info = {info_wrapped}")

randinfo_env = RandomInfo(rand_env, info_space=Dict({"a": Box(0, 1)}))
_, _, _, _, info_wrapped = randinfo_env.step(randinfo_env.action_space.sample())
print(f"wrapped info = {info_wrapped}")

# %% AppendInfoItemToObs
print("\nTest AppendInfoItemToObs...")
rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"a": Box(0, 1)})}),
    info_space=Dict({"b": MultiDiscrete(2)}),
)
appinfo_env = AppendInfoItemToObs(rand_env, info_key="b")
obs, info = appinfo_env.reset()
print(f"reset obs = {obs}")
print(f"reset info = {info}")

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"step obs = {obs}")
print(f"step info = {info}")
# %% Test NumWindows
print("\nTest NumWindows...")
nw_env = NumWindows(env=rand_env, use_estimates=False)
obs, _, _, _, info = nw_env.step(nw_env.action_space.sample())
print(f"info = {info}")

# %% Done
print("done")
