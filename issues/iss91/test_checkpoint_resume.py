"""Test to resume experiment level checkpoint."""
# https://discuss.ray.io/t/not-able-to-resume-experiment/8587
# Standard Library Imports
import os
from pathlib import Path

# Third Party Imports
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.nets.lstm_mask import MaskedLSTM
from punchclock.ray.build_env import buildEnv

# %% Script
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)
register_env("my_env", buildEnv)

dir_path = Path(__file__).parent

# change this line to the dir that the experiment checkpoint is stored
exp_name = "data/training_run_OYP"

# experiment_path = os.path.join(storage_path, exp_name)
experiment_path = os.path.join(dir_path, exp_name)
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(
    experiment_path,
    trainable="PPO",
    resume_unfinished=True,
    resume_errored=True,
)
restored_tuner.fit()
# %% Done
print("done")
