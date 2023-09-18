"""Tests for info_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.utils.env_checker import check_env
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.environment.info_wrappers import NumWindows

# %% Build env for NumWindows wrapper
print("\nBuild env for NumWindows test...")
rand_env = RandomEnv()
agents = [buildRandomAgent(agent_type="sensor") for ag in range(2)]
agents.extend([buildRandomAgent(agent_type="target") for ag in range(3)])
rand_env.agents = agents
rand_env.horizon = 10
rand_env.time_step = 100

# %% Test NumWindows
print("\nTest NumWindows...")
nw_env = NumWindows(env=rand_env, use_estimates=False)
obs, _, _, _, info = nw_env.step(nw_env.action_space.sample())
print(f"info = {info}")

# %% Use gym checker
# check_env(rand_env)

# %% done
print("done")
