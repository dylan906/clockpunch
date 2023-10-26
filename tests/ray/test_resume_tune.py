"""Tests for resume_tune.py."""
# %% Imports
# Standard Library Imports
import random
import string
from pathlib import Path

# Third Party Imports
from ray import air
from ray.tune import Tuner

# Punch Clock Imports
from punchclock.ray.resume_tune import resumeTune

# %% Test resumeTune
# Run a quick tune fit on a dummy environment to test loading a checkpoint.
print("\nTest resumeTune...")

param_space = {
    "framework": "torch",
    "env": "CartPole-v1",
}

# Store tune results in test dir
dir_path = Path(__file__).parent
storage_path = dir_path.joinpath("data")
rand_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
exp_name = "test_restore_exp_" + rand_str

# Set low training iter so fit runs quickly.
tuner = Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=air.config.RunConfig(
        stop={"training_iteration": 1},
        name=exp_name,
        storage_path=storage_path,
    ),
)
tuner.fit()

# %% Restore Tuner and resume fit
exp_path = storage_path.joinpath(exp_name)

resumeTune(
    checkpoint_dir=str(exp_path),
    trainable="PPO",
    num_cpus=None,
)

# %% done
print("done")
