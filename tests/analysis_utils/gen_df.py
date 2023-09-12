"""Generate a data file for analysis_utils tests."""
# This script generates a simulation data file for use in analysis_utils tests.
# %% Imports

# Standard Library Imports
import os

# Third Party Imports
from numpy import array, diag, pi

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.policies.random_policy import RandomPolicy
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.sim_runner import SimRunner

# %% Build environment
print("\nBuild test environment...")

# %% Environment params
RE = getConstants()["earth_radius"]

horizon = 10
time_step = 100
# sensor locations (fixed)
x_sensors_lla = [
    array([0, 0, 0]),
    array([0, pi / 4, 0]),
    array([pi / 4, 0, 0]),
]

num_sensors = len(x_sensors_lla)
num_targets = 20

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
temp_matrix = diag([1, 1, 1, 0.01, 0.01, 0.01])
filter_params = {
    "Q": 0.001 * temp_matrix,
    "R": 1 * 0.1 * temp_matrix,
    "p_init": 10 * temp_matrix,
}

reward_params = None

# assign build params as a dict for easier env recreation/changing
# ActionMask wrapper required for SimRunner
constructor_params = {
    "wrappers": [
        {
            "wrapper": "FilterObservation",
            "wrapper_config": {
                "filter_keys": [
                    "eci_state",
                    "est_cov",
                    "num_tasked",
                    "num_windows_left",
                    "obs_staleness",
                    "vis_map_est",
                ]
            },
        },
        {
            "wrapper": "CopyObsItem",
            "wrapper_config": {
                "key": "vis_map_est",
                "new_key": "action_mask",
            },
        },
        {
            "wrapper": "VisMap2ActionMask",
            "wrapper_config": {"vis_map_key": "action_mask"},
        },
        {
            "wrapper": "NestObsItems",
            "wrapper_config": {
                "new_key": "observations",
                "keys_to_nest": [
                    "eci_state",
                    "est_cov",
                    "num_tasked",
                    "num_windows_left",
                    "obs_staleness",
                    "vis_map_est",
                ],
            },
        },
        {"wrapper": "IdentityWrapper"},
    ],
}

env_config = {
    "horizon": horizon,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "reward_params": reward_params,
    "time_step": time_step,
    "constructor_params": constructor_params,
}


# %% Build policy
env = buildEnv(env_config)

policy = RandomPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    use_mask=True,
)

# %% Build and run SimRunner
simrunner = SimRunner(env=env, policy=policy, max_steps=horizon)
results = simrunner.runSim()
df = results.toDataFrame()

# %% Save dataframe as pickle
fpath = os.path.dirname(os.path.realpath(__file__))
savepath = fpath + "/simresults_df.pkl"
df.to_pickle(savepath)

# %% done
print("done")
