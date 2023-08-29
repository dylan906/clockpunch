"""Generate config files for testing run_tune_script."""
# %% Imports
# Third Party Imports
from numpy import diag, pi

# Punch Clock Imports
from punchclock.ray.build_env import genConfigFile

# from ray import tune


# %% Environment params
agent_params = {
    "num_sensors": 3,
    "num_targets": 10,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": "uniform",
    "target_dist": "uniform",
    "sensor_dist_frame": "LLA",
    "target_dist_frame": "COE",
    "sensor_dist_params": [
        [0, pi / 2],
        [0, 2 * pi],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ],
    "target_dist_params": [
        [42000, 43000],
        [0, 0],
        [0, pi / 8],
        [0, 0],
        [0, 0],
        [0, 2 * pi],
    ],
    "fixed_sensors": None,
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
            "wrapper": "CopyObsItem",
            "wrapper_config": {
                "key": "vis_map_est",
                "new_key": "vis_map_copy",
            },
        },
        {
            "wrapper": "VisMap2ActionMask",
            "wrapper_config": {
                "vis_map_key": "vis_map_copy",
                "rename_key": "action_mask",
            },
        },
        {
            "wrapper": "NestObsItems",
            "wrapper_config": {
                "new_key": "observations",
                "keys_to_nest": [
                    "vis_map_est",
                    "est_cov",
                ],
            },
        },
        {"wrapper": "FlatDict"},
    ]
}

# Set the environment parameters. Parameters must be in dict to interface with RLlib env
# register.
builder_params = {
    "horizon": 10,
    "agent_params": agent_params,
    "filter_params": filter_params,
    # "policy_params": policy_params,
    "reward_params": {},
    "time_step": 100,
    "constructor_params": constructor_params,
}
# %% Hidden layers
# Use a grid search for hidden layers. For 2 layers, search combinations of layer 1 size
# = 100 and 50, and layer 2 size = 50 and 25. This gives a 2x2 grid to search.

# hidden_layers = [tune.grid_search([100, 50]), 25]
model_config = {
    "custom_model": "action_mask_model",
    "fcnet_activation": "relu",
    "fcnet_hiddens": [30, 20],
}
# %% Other Params
config_dir = "tests/training_scripts"
config_file_name = "config_training_test"
num_cpus = 10
num_workers = None
num_episodes = 1
training_method = "PPO"
lr = 1e-5

param_space = {
    "env_config": builder_params,
    "model": model_config,
    "lr": lr,
    "num_workers": num_workers,
}

run_config = {
    "stop": {
        "episodes_total": num_episodes,
    },
    "name": "training_results",
    "local_dir": "tests/training_scripts",
}
# %% Gen Config
genConfigFile(
    config_dir=config_dir,
    config_file_name=config_file_name,
    num_cpus=num_cpus,
    trainable=training_method,
    param_space=param_space,
    run_config=run_config,
)

print("done")
