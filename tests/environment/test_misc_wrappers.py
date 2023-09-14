"""Tests for misc_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.environment.misc_wrappers import IdentityWrapper, NumWindows

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

# %% Test NumWindows
print("\nTest NumWindows...")
nw_env = NumWindows(env=rand_env, horizon=10, dt=100, use_estimates=False)
obs, _, _, _, info = nw_env.step(nw_env.action_space.sample())
print(f"info = {info}")

# %% Done
print("done")
