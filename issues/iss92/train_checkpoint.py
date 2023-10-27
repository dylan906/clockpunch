"""Create a checkpoint."""
# Standard Library Imports
import os
import random
import string
from pathlib import Path

# Third Party Imports
from ray import air, tune
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from punchclock.ray.build_tuner import buildTuner

# %% Script
dir_path = Path(__file__).parent
storage_path = dir_path.joinpath("data")


param_space = {
    "framework": "torch",
    "env": "CartPole-v1",
    "model": {
        # "custom_model": "MaskedLSTM",
    },
}

rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
exp_name = "training_run_" + rand_str

# Train with normal method
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=air.RunConfig(
        stop={
            "training_iteration": 2,
        },
        name=exp_name,
        storage_path=storage_path,
    ),
)

results = tuner.fit()
