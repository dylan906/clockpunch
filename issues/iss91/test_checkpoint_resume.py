"""Test to resume experiment level checkpoint."""
# https://discuss.ray.io/t/not-able-to-resume-experiment/8587
# Standard Library Imports
import os

# Third Party Imports
from ray import tune
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from punchclock.nets.lstm_mask import MaskedLSTM

# %% Script
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

storage_path = "/home/dylanrpenn/ray_results/"
exp_name = "training_run_MMK"
experiment_path = os.path.join(storage_path, exp_name)
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
