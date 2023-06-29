"""Tests for norm_metric.py."""
# %% Imports
# Punch Clock Imports
from punchclock.reward_funcs.norm_metric import NormalizedMetric

# %% Test initialization
print("\nTest initialization...")
rf = NormalizedMetric(
    obs_or_info="obs",
    metric="num_tasked",
    norm_denominator=2,
    preprocessors=["min"],
)

# %% Test calcReward
print("\nTest calcReward...")
obs = {"num_tasked": [2, 5, 2, 1]}
reward = rf.calcReward(obs=obs, info=None)
print(f"reward = {reward}")
