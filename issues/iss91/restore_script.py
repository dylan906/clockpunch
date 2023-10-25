"""Restore and run tune script."""
# %% Imports
# Standard Library Imports
import argparse
import json
import os

# Third Party Imports
from ray import tune

# Punch Clock Imports
from punchclock.ray.build_tuner import buildTuner

# %% Print Working directory
current_dir = os.getcwd()
print(f"working directory: {current_dir}")

# %% Parse arguments
# instantiate parser (https://docs.python.org/3/library/argparse.html#nargs)
parser = argparse.ArgumentParser(description="this is a description")

# add argument to parser (this parser has only 1 argument)
parser.add_argument(
    "exp_path",  # key of argument
    type=str,
    nargs="+",  # gathers all command line arguments into a list
    help="path to experiment",
)
# get Namespace variable
exp_path = parser.parse_args()
# print(vars(exp_path))
# Extract path from Namespace
exp_path = vars(exp_path)["exp_path"][0]
print(f"experiment file path = {exp_path}")

# # %% Load config file
# # Opening JSON file
# # with open("bash_script_demo/test_config.json", "r", encoding="UTF-8") as f:
# with open(exp_path, "r", encoding="UTF-8") as f:
#     config = json.load(f)

# # Print config contents
# print(f"config file contents= {config}")

# %% Run tune job
# restore tuner
restored_tuner = tune.Tuner.restore(
    exp_path,
    trainable="PPO",
    resume_unfinished=True,
    resume_errored=True,
)

# run fit job
restored_tuner.fit()
