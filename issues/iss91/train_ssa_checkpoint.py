"""Create a checkpoint using the SSA environment."""
# %% Imports
# Standard Library Imports
import os
import random
import string

# Third Party Imports
import ray
from numpy import array, diag, pi
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env as ray_check_env
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.common.utilities import recursivelyConvertDictToPrimitive
from punchclock.nets.lstm_mask import MaskedLSTM
from punchclock.ray.build_env import buildEnv
from punchclock.ray.build_tuner import buildTuner

# %% Environment params
# Build environment with action mask
RE = getConstants()["earth_radius"]

horizon = 10
time_step = 10

x_sensors_lla = [
    array([0, 0, 0]),
    array([0, pi / 4, 0]),
    array([pi / 4, 0, 0]),
]

num_sensors = len(x_sensors_lla)
num_targets = 5

x_sensors_ecef = [lla2ecef(x_lla=x) for x in x_sensors_lla]
x_sensors_eci = [ecef2eci(x_ecef=x, JD=0) for x in x_sensors_ecef]

# agent params
agent_params = {
    "num_sensors": num_sensors,
    "num_targets": num_targets,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": None,
    "target_dist": "uniform",
    "sensor_dist_frame": None,
    "target_dist_frame": "COE",
    "sensor_dist_params": None,
    "target_dist_params": [
        [RE + 300, RE + 800],
        [0, 0],
        [0, pi / 2],
        [0, 2 * pi],
        [0, 2 * pi],
        [0, 2 * pi],
    ],
    "fixed_sensors": x_sensors_eci,
    "fixed_targets": None,
}

# Set the UKF parameters. We are using the abbreviated interface for simplicity,
# see ezUKF for details.
temp_list = array([1, 1, 1, 0.01, 0.01, 0.01])
temp_matrix = diag(temp_list)
# p_init = {
#     "dist": "normal",
#     "params": [10 * temp_list, temp_list],
# }
filter_params = {
    "Q": 0.001 * temp_matrix,
    "R": 1 * 0.1 * temp_matrix,
    "p_init": 10 * temp_matrix,
}

# Set environment constructor params
constructor_params = {
    "wrappers": [
        {
            "wrapper": "FilterObservation",
            "wrapper_config": {"filter_keys": ["vis_map_est", "est_cov"]},
        },
        {
            "wrapper": "CopyObsInfoItem",
            "wrapper_config": {
                "copy_from": "obs",
                "copy_to": "info",
                "from_key": "vis_map_est",
                "to_key": "vis_map_est",
            },
        },
        {
            "wrapper": "VisMap2ActionMask",
            "wrapper_config": {
                "obs_info": "info",
                "vis_map_key": "vis_map_est",
                "new_key": "action_mask",
            },
        },
        {
            "wrapper": "CopyObsInfoItem",
            "wrapper_config": {
                "copy_from": "info",
                "copy_to": "obs",
                "from_key": "action_mask",
                "to_key": "action_mask",
                "info_space_config": {
                    "space": "Box",
                    "low": 0,
                    "high": 1,
                    "shape": [num_targets + 1, num_sensors],
                    "dtype": "int",
                },
            },
        },
        {
            "wrapper": "NestObsItems",
            "wrapper_config": {
                "new_key": "observations",
                "keys_to_nest": ["vis_map_est", "est_cov"],
            },
        },
        {"wrapper": "FlatDict"},
    ]
}

env_config = {
    "horizon": horizon,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "time_step": time_step,
    "constructor_params": constructor_params,
}
env_config = recursivelyConvertDictToPrimitive(env_config)

env = buildEnv(env_config)
ray_check_env(env)

# %% Train model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

dir_path = os.path.dirname(os.path.realpath(__file__))
storage_path = dir_path + "/data"

param_space = {
    "framework": "torch",
    "env": "my_env",
    "env_config": env_config,
    "model": {
        "custom_model": "MaskedLSTM",
        "custom_model_config": {
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
            "lstm_state_size": tune.grid_search([2, 4, 6, 8]),
        },
    },
    "lr": 1e-5,
    "gamma": 0.999,
    "num_workers": 19,
}

# Train with normal method
rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
exp_name = "training_run_" + rand_str
# tuner = tune.Tuner(
#     trainable="PPO",
#     param_space=param_space,
#     run_config=air.RunConfig(
#         stop={
#             "training_iteration": 1,
#         },
#         name=exp_name,
#         storage_path=storage_path,
#     ),
# )

# use buildTuner() instead of manually making Tuner
ray.init(num_cpus=20)
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(19)

tuner = buildTuner(
    {
        "trainable": "PPO",
        "param_space": param_space,
        "run_config": {
            "stop": {"training_iteration": 10},
            "name": exp_name,
            "storage_path": storage_path,
        },
    },
    override_date=True,
)
results = tuner.fit()
