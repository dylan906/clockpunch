"""Tests for env_builder.py."""
# NOTE: This test generates a new file (config_test.json) in the same directory
# as the test script.
# %% Imports
# Standard Library Imports
import warnings
from copy import deepcopy

# Third Party Imports
from numpy import array, diag, pi
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_env import buildEnv, genConfigFile

# %% Set test environment params
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
                "to_key": "action_mask",
            },
        },
        {
            "wrapper": "VisMap2ActionMask",
            "wrapper_config": {
                "obs_info": "obs",
                "vis_map_key": "action_mask",
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

# Set model (neural net) config to use custom MyActionMask model. This line isn't
# actually used in this test, but is here as an example.
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

model_config = {
    "custom_model": "action_mask_model",
    "fcnet_activation": "relu",
    "fcnet_hiddens": [30, 20],
}

# %% Test environment builder
print("\nTest buildEnv...")

# deepcopy so buildEnv doesn't modify params
print("Test basic functionality...")
env = buildEnv(deepcopy(env_config))
check_env(env)

print(f"env = {env}")
print(f"env observation space = {env.observation_space.spaces}")

# Test default argument insertion
print("Test defaults...")

# If constructor_params is not designated, then the env builder should use default
# wrappers.
env_config2 = deepcopy(env_config)
env_config2.pop("constructor_params")
env2 = buildEnv(deepcopy(env_config2))
print("constructor_params not set:")
print(f"env = {env2}")
print(f"env2 observation space = {env2.observation_space.spaces}")

# Use base environment if wrappers == []
env_config3 = deepcopy(env_config)
env_config3["constructor_params"]["wrappers"] = []
env3 = buildEnv(deepcopy(env_config3))
print("Wrappers set to []:")
print(f"env3 = {env3}")

# If an observation_space_filter is designated, but vis_map_est is not included
# in the list of states to target_filter, then the env builder should add it and warn
# the user.
constructor_params4 = {
    "wrappers": [
        {
            "wrapper": "FilterObservation",
            "wrapper_config": {"filter_keys": ["num_tasked"]},
        },
    ]
}
env_config4 = deepcopy(env_config)
env_config4["constructor_params"] = constructor_params4

# Treat warnings as errors to make sure warning happens. Reset afterwards to normal
# behavior.
warnings.filterwarnings("error")
try:
    env4 = buildEnv(deepcopy(env_config4))
except Warning as w:
    print(w)
warnings.resetwarnings()

env4 = buildEnv(deepcopy(env_config4))
print("vis_map_est not included in obs target_filter:")
print(f"    env4 = {env4}")
print(f"    env4 observation space = {env4.observation_space.spaces}")
# %% Test repeatability of buildEnv
# Make sure that when building environment with randomly-distributed initial conditions,
# that ICs are different from one function call to the next.
env1 = buildEnv(deepcopy(env_config))
env2 = buildEnv(deepcopy(env_config))
env1.reset()
env2.reset()

info1 = env1.unwrapped._getInfo()
info2 = env2.unwrapped._getInfo()
# Differenced array should be 0s cols 0-1 (fixed sensors), and non-zero for cols
# 2-5 (targets).
print(info1["true_states"] - info2["true_states"])

# %% Test genConfigFile
print("\nTest genConfigFile...")
# NOTE: This test generates a new file in the same directory as the test script.
config_dir = "tests/ray"
experiment_name = "exp_name"
local_dir = "tests/ray"
file_name = "config_test"

param_space = {
    "env_config": env_config,
    "model": model_config,
    "lr": 5e-5,
}
trainable = "PPO"
tune_config = {}
run_config = {
    "stop": {
        "timesteps_total": 10,
        # "episodes_total": 1,
        # "training_iteration": 1,
    },
    "name": experiment_name,
    "local_dir": local_dir,
}

dat = genConfigFile(
    config_dir=config_dir,
    config_file_name=file_name,
    num_cpus=None,
    trainable=trainable,
    param_space=param_space,
    tune_config=tune_config,
    run_config=run_config,
)
# %%
print("done")
