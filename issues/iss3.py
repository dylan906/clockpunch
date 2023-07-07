"""Script to recreate issue #3."""
# %% Imports
# Third Party Imports
import gymnasium as gym
import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.ray.build_env import buildEnv
from punchclock.ray.build_tuner import buildTuner

# %% Load config json
config = loadJSONFile("issues/iss3_config.json")

# %% Register custom environment and model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

# %% Assemble inputs for tune.Tuner()

# run_config = air.RunConfig(
#     **config["run_config"],
# )
run_config = air.RunConfig(
    name="exp_name",
    local_dir="tests/ray",
    stop={"timesteps_total": 10},
)
tune_config = tune.tune_config.TuneConfig(**config["tune_config"])

print(config["param_space"])
param_space = config["param_space"].update(
    {
        "framework": "torch",
        "env": "SSAScheduler",
        "horizon": None,  # environment has its own horizon
        "ignore_worker_failures": True,
        "log_level": "DEBUG",
        "num_gpus": 0,
    }
)
print(param_space)

ray.init(num_cpus=16)

# tuner = buildTuner(config)
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=run_config,
    tune_config=tune_config,
)
tuner.fit()
