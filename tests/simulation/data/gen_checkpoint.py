"""Generate Ray checkpoint to use in tests."""
# NOTE: This script requires "config_env.json" in the same directory to run
# NOTE: This script saves a set of files in a directory, and creates that dir if
# it doesn't already exist.
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
from ray import air, tune
from ray.air import CheckpointConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.stopper import MaximumIterationStopper

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_env import buildEnv

# %% Preliminaries
# Register environment constructor and model
register_env("ssa_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

# Load env config and build MC tuner param_space
fpath = os.path.dirname(os.path.realpath(__file__))
env_config_path = fpath + "/config_env.json"
env_config = loadJSONFile(env_config_path)

param_space = {
    "env_config": env_config,
    "model": {"custom_model": "action_mask_model"},
    "framework": "torch",
    "env": "ssa_env",
    "horizon": None,
    "sgd_minibatch_size": 10,
    "train_batch_size": 100,
}
# %% Build and run tuner
tuner = tune.Tuner(
    trainable="PPO",
    tune_config=tune.TuneConfig(),
    param_space=param_space,
    # Set iterations to 1 to finish trial quickly.
    run_config=air.RunConfig(
        stop=MaximumIterationStopper(max_iter=1),
        name="test_trial",
        storage_path=fpath + "/test_checkpoint2",
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
    ),
)
results = tuner.fit()

# %% done
print("done")
