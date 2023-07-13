"""Script to recreate issue #3."""
# %% Imports
# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_env import buildEnv

# %% Load config json
config = loadJSONFile("issues/iss3_config.json")

# %% Register custom environment and model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

# %% Assemble inputs for tune.Tuner()
param_space = deepcopy(config["param_space"])
param_space.update(
    {
        "framework": "torch",
        "env": "my_env",
        "horizon": None,  # environment has its own horizon
    }
)


run_config = air.RunConfig(
    stop={"timesteps_total": 10},
)
tune_config = tune.tune_config.TuneConfig()

# %% Make tuner
ray.init(num_cpus=16)
# tuner = buildTuner(config)
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=run_config,
    tune_config=tune_config,
)
tuner.fit()
