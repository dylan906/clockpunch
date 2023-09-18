"""Tests for misc_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict
from gymnasium.utils.env_checker import check_env
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.misc_wrappers import (
    AppendInfoItemToObs,
    IdentityWrapper,
    RandomInfo,
)
from punchclock.policies.policy_builder import buildSpace

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

try:
    check_env(identity_env)
except Exception as ex:
    print(ex)

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

try:
    check_env(randinfo_env)
except Exception as ex:
    print(ex)

# %% AppendInfoItemToObs
print("\nTest AppendInfoItemToObs...")
info_space_config = {"space": "Box", "low": 0, "high": 3}
rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"a": Box(0, 1)})}),
    info_space=Dict({"b": buildSpace(info_space_config)}),
)

appinfo_env = AppendInfoItemToObs(
    rand_env,
    info_key="b",
    info_space_config=info_space_config,
)
obs, info = appinfo_env.reset()
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space = {appinfo_env.observation_space}")
print(f"reset obs = {obs}")
print(f"reset info = {info}")
assert appinfo_env.observation_space.contains(obs)

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"step obs = {obs}")
print(f"step info = {info}")
assert appinfo_env.observation_space.contains(obs)

try:
    check_env(appinfo_env)
except Exception as ex:
    print(ex)

# %% Done
print("done")
