"""Argument parser script for use in bash_script_test.sh."""
# https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
# %% Imports
# Standard Library Imports
import argparse
import json
import os

# Punch Clock Imports
from scheduler_testbed.ray.build_tuner import buildTuner

# %% Print Working directory
current_dir = os.getcwd()
print(f"working directory: {current_dir}")

# %% Parse arguments
# instantiate parser (https://docs.python.org/3/library/argparse.html#nargs)
parser = argparse.ArgumentParser(description="this is a description")

# add argument to parser (this parser has only 1 argument)
parser.add_argument(
    "config_path",  # key of argument
    type=str,
    nargs="+",  # gathers all command line arguments into a list
    help="path to configuration file",
)
# get Namespace variable
config_path = parser.parse_args()
# print(vars(config_path))
# Extract path from Namespace
config_path = vars(config_path)["config_path"][0]
print(f"config file path = {config_path}")

# %% Load config file
# Opening JSON file
# with open("bash_script_demo/test_config.json", "r", encoding="UTF-8") as f:
with open(config_path, "r", encoding="UTF-8") as f:
    config = json.load(f)

# Print config contents
print(f"config file contents= {config}")

# %% Run tune job
# build tuner
tuner = buildTuner(config=config)
# run fit job
tuner.fit()
