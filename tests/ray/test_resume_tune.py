"""Tests for resume_tune.py."""
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
from ray import air
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import Policy
from ray.tune import Tuner

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.ray.resume_tune import restoreTuner

# %% Load json file
# path to the folder this script is contained
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"dir path: {dir_path}")

# config file name
config_name = "config_test.json"

file_path = dir_path + "/" + config_name
print(f"file path: {file_path}")

config = loadJSONFile(file_name=file_path)
# %% Test restoreTuner
# Run a quick tune fit on a dummy environment to test loading a checkpoint.
print("\nTest restoreTuner...")
algo_config = (
    ppo.PPOConfig()
    .training()
    .environment(
        env="CartPole-v1",
    )
    .framework(
        "torch",
    )
)
# Set low training iter so fit runs quickly.
tuner = Tuner(
    "PPO",
    param_space=algo_config.to_dict(),
    run_config=air.config.RunConfig(
        stop={"training_iteration": 1},
        name="test_restore_exp",
    ),
)
tuner.fit()

# # checkpoint_path = "tests/ray/exp_name"
# # checkpoint_path = tuner._local_tuner._experiment_checkpoint_dir
experiment_path = "~/ray_results/test_restore_exp"

# print("restore")
tuner = Tuner.restore(path=experiment_path)
# # tuner = restoreTuner(checkpoint_dir=checkpoint_path, config=config)
# print("Running fit...")
# tuner.fit()
# print("...fit complete.")
# %% Restore Policy
restored_policy = Policy.from_checkpoint(
    "tests/ray/exp_name/PPO_my_env_dac43_00000_0_2023-01-13_15-50-04/checkpoint_000001"
)

# %% done
print("done")
