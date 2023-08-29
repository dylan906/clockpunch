"""Generate config files for testing run_mc_script.py."""
# NOTE: This script generates a file in the script's dir.
# %% Imports
# Standard Library Imports
import json
import os

# Third Party Imports
from numpy import diag, pi

# Punch Clock Imports
from punchclock.common.utilities import array2List, loadJSONFile, saveJSONFile
from punchclock.policies.policy_builder import buildSpaceConfig
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.mc_config import MonteCarloConfig

# %% Environment Config
# agent config
agent_params = {
    "num_sensors": 3,
    "num_targets": 4,
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

# Set the environment config
env_config = {
    "horizon": 10,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "reward_params": {},
    "time_step": 100,
    "constructor_params": constructor_params,
}


# %% Policy Config
env = buildEnv(env_config)

obs_space_config = buildSpaceConfig(env.env.observation_space).toDict()
act_space_config = buildSpaceConfig(env.env.action_space).toDict()

policy_config = {
    "policy": "RandomPolicy",
    "observation_space": obs_space_config,
    "action_space": act_space_config,
}

# %% MC Results path
fpath = os.path.dirname(os.path.realpath(__file__))
results_dir = fpath + "/data"
# %% Handling env config

# convert to json-able for saving
env_config_json = json.dumps(env_config, default=array2List)
with open(fpath + "/env_config.json", "w") as outfile:
    outfile.write(env_config_json)

# env_config only works in MonteCarloConfig if it is loaded from a saved jon. Not
# sure why.
# TODO: Fix.
env_config_loaded = loadJSONFile(fpath + "/env_config.json")

# %% Gen MC config
mc_config = MonteCarloConfig(
    num_episodes=1,
    policy_configs=policy_config,
    env_config=env_config_loaded,
    results_dir=results_dir,
    print_status=True,
)

mc_config_path = fpath + "/config_mc_test.json"
mc_config.save(mc_config_path, append_timestamp=False)

print("done")
