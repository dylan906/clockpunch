"""Build a tuner with LSTM Mask model."""
# %5 Imports
# Standard Library Imports
from typing import Any

# Third Party Imports
import gymnasium as gym
import ray
import ray.rllib.algorithms.ppo as ppo
from lstm_mask import MaskedLSTM  # NOTE: Relative import is brittle
from numpy import array, diag, float32, int64, ndarray, pi
from ray import air, tune
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.stopper import MaximumIterationStopper

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.common.utilities import loadJSONFile, printNestedDict
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_env import buildEnv, genConfigFile
from punchclock.ray.build_tuner import (
    _getDefaults,
    _getExperimentName,
    buildTuner,
)


# %% util function
def recursivelyConvertDictToPrimitive(in_dict: dict) -> dict:
    """Recursively convert dict entries into primitives."""
    out = {}
    # Loop through key-value pairs of in_dict. If a value is a dict, then recurse.
    # Otherwise, convert value to a JSON-able type. Special handling if the
    # value is a `list`. Lists of dicts are recursed; lists of non-dicts and
    # empty lists are converted to JSON-able as normal.
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out[k] = recursivelyConvertDictToPrimitive(v)
        elif isinstance(v, list):
            if len(v) == 0:
                out[k] = [convertToPrimitive(a) for a in v]
            elif isinstance(v[0], dict):
                out[k] = [recursivelyConvertDictToPrimitive(a) for a in v]
            else:
                out[k] = [convertToPrimitive(a) for a in v]
        else:
            out[k] = convertToPrimitive(v)
    return out


def convertToPrimitive(entry: Any) -> list:
    """Convert a non-serializable object into a JSON-able type."""
    if isinstance(entry, ndarray):
        # numpy arrays need their own tolist() method to convert properly.
        out = entry.tolist()
    elif isinstance(entry, set):
        out = list(entry)
    elif isinstance(entry, (float32, int64)):
        out = entry.item()
    else:
        out = entry

    return out


# %% Env params
print("\nSet env params...")
RE = getConstants()["earth_radius"]
sensor_init_ecef = array([[RE, 0, 0, 0, 0, 0], [0, RE, 0, 0, 0, 0]])
sensor_init_eci = ecef2eci(sensor_init_ecef.transpose(), 0).transpose()

# Set parameters for agents. In this example, we are using 2 fixed sensors and
# 4 targets with stochastic initial conditions.
agent_params = {
    "num_sensors": 2,
    "num_targets": 4,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": None,
    "target_dist": "normal",
    "sensor_dist_frame": None,
    "target_dist_frame": "COE",
    "sensor_dist_params": None,
    "target_dist_params": [
        [5000, 200],
        [0, 0],
        [0, pi],
        [0, 2 * pi],
        [0, 2 * pi],
        [0, 2 * pi],
    ],
    "fixed_sensors": sensor_init_eci,
    "fixed_targets": None,
}

# Set the UKF parameters. We are using the abbreviated interface for simplicity,
# see ezUKF for details.
temp_matrix = diag([1, 1, 1, 0.01, 0.01, 0.01])
filter_params = {
    "Q": 0.001 * temp_matrix,
    "R": 0.1 * temp_matrix,
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
                "copy_to": "obs",
                "from_key": "vis_map_est",
                "to_key": "vm_copy",
            },
        },
        {
            "wrapper": "VisMap2ActionMask",
            "wrapper_config": {
                "obs_info": "obs",
                "vis_map_key": "vm_copy",
                "new_key": "action_mask",
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

# Set the environment parameters. Parameters must be in dict to interface with
# RLlib env register.
env_config = {
    "horizon": 10,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "time_step": 100,
    "constructor_params": constructor_params,
}

env_config = recursivelyConvertDictToPrimitive(env_config)
# %% Make test Env
rand_env_config = {
    "observation_space": gym.spaces.Dict(
        {
            "observations": gym.spaces.Box(0, 1, shape=[4]),
            "action_mask": gym.spaces.Box(0, 1, shape=[2], dtype=int),
        }
    ),
    "action_space": gym.spaces.MultiDiscrete([2]),
}

rand_env = RandomEnv(env_config)

# %% Register model and env
print("\nRegister model and env...")
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)
register_env("my_env", buildEnv)
register_env("rand_env", RandomEnv)
# %% Set tuner config
print("\nSet tuner config...")
param_space = {
    "framework": "torch",
    "env": "my_env",
    "horizon": None,  # environment has its own horizon
    "env_config": env_config,
    "model": {
        # Specify our custom model from above.
        "custom_model": "MaskedLSTM",
        # Extra kwargs to be passed to your model's c'tor.
        # "custom_model_config": {},
        "custom_model_config": {
            # "num_outputs": 2,
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
            "fc_size": 5,
            "lstm_state_size": 20,
        },
    },
}

run_config = air.config.RunConfig(stop=MaximumIterationStopper(max_iter=1))
# %% Build tuner
print("\nBuild tuner...")
ray.init()

tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=run_config,
    tune_config={},
)
print("run fit")
tuner.fit()


# %% done
print("done")
