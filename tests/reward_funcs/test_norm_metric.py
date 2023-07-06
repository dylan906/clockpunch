"""Tests for norm_metric.py."""
# %% Imports
# Punch Clock Imports
from punchclock.reward_funcs.norm_metric import GenericReward

# %% Test initialization
print("\nTest initialization...")
rf = GenericReward(
    obs_or_info="obs",
    metric="num_tasked",
    preprocessors=["min"],
)
# %% Test calcReward
print("\nTest calcReward...")
obs = {"num_tasked": [2, 5, 2, 1]}
reward = rf.calcReward(obs=obs, info=None)
print(f"reward = {reward}")
# %% Test configurable preprocessor
print("\nTest configurable preprocessor...")
rf = GenericReward(
    obs_or_info="obs",
    metric="num_tasked",
    preprocessors=["min", {"preprocessor": "divide", "config": {"x2": 2}}],
)
reward = rf.calcReward(obs=obs, info=None)
print(f"reward = {reward}")
# %% Test backward compatibility
print("\nTest backward compatibility...")

rf = GenericReward(
    obs_or_info="obs",
    metric="num_tasked",
    norm_denominator=3,
    preprocessors=["min"],
)
reward = rf.calcReward(obs=obs, info=None)
print(f"reward = {reward}")
