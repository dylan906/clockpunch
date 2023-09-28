"""Gen config for simulation tests."""
# Generate a config file, "config_env.json" to to used in other tests in the simulation
# directory.
# NOTE: This script saves a file, "config_env.json" in the same dir as this script.
# %% Imports

# Standard Library Imports
import json
import os

# Third Party Imports
from numpy import array, diag, pi

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.common.utilities import array2List

# %% Config env
RE = getConstants()["earth_radius"]
sensor_init_ecef = array([[RE, 0, 0, 0, 0, 0], [0, RE, 0, 0, 0, 0]])
sensor_init_eci = ecef2eci(sensor_init_ecef.transpose(), 0).transpose()

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
                "vis_map_key": "vm_copy",
                "rename_key": "action_mask",
            },
        },
        {
            "wrapper": "NestObsItems",
            "wrapper_config": {
                "new_key": "observations",
                "keys_to_nest": ["vis_map_est", "est_cov"],
            },
        },
        {"wrapper": "IdentityWrapper"},
        {"wrapper": "FlatDict"},
    ]
}

env_config = {
    "horizon": 10,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "time_step": 100,
    "constructor_params": constructor_params,
}

# %% Save file
json_obj = json.dumps(env_config, default=array2List)

fdir = os.path.dirname(os.path.realpath(__file__))
fpath = fdir + "/config_env.json"
with open(fpath, "w") as outfile:
    outfile.write(json_obj)
