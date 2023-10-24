"""Tests for stoppers.py."""
# %% Imports
# Third Party Imports
from ray import train, tune
from ray.tune import Stopper

# Punch Clock Imports
from punchclock.ray.stoppers import StopOnTrend

# %% Test StopOnTrend
print("\nStopOnTrend...")
stopper = StopOnTrend()
tuner = tune.Tuner(
    my_trainable,
    run_config=train.RunConfig(stop=stopper),
    tune_config=tune.TuneConfig(num_samples=2),
)
result_grid = tuner.fit()
