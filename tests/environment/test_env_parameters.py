"""Test env_parameters.py."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from numpy import asarray, eye, ones

# Punch Clock Imports
from scheduler_testbed.common.agents import Sensor, Target
from scheduler_testbed.environment.env_parameters import SSASchedulerParams

# %% Test buildRewardFunc
print("\nTest buildRewardFunc...")
reward_params = {
    "reward_func": "Threshold",
    "obs_or_info": "obs",
    "metric": "num_tasked",
    "metric_value": 3,
    "penalty": 1,
    "inequality": ">",
    "preprocessors": ["min"],
}
reward_func = SSASchedulerParams.buildRewardFunc(
    None,
    params=deepcopy(reward_params),
)


# %% Test genFilters
print("\nTest genFilters...")
initial_conditions = [ones([6, 1]) for i in range(3)]
dynamics_type = "satellite"

Q = (0.5 * eye(6)).tolist()
R = (0.9 * eye(6)).tolist()
p_init = (0.1 * eye(6)).tolist()
p_init[0][0] = 0.5
mini_filter_params = {
    "Q": Q,
    "R": R,
    "p_init": p_init,
}

list_of_filters = SSASchedulerParams.genFilters(
    self=None,
    initial_conditions=initial_conditions,
    dynamics_type=dynamics_type,
    filter_params=mini_filter_params,
)
init_estimates = [filt.est_x for filt in list_of_filters]
print("Initial estimates - Initial Conditions:")
print(asarray(init_estimates) - asarray(initial_conditions))

# Test genFilters with floats instead of nested lists
print("Test with float inputs...")
mini_filter_params = {
    "Q": 0.1,
    "R": 1,
    "p_init": 10,
}
list_of_filters = SSASchedulerParams.genFilters(
    self=None,
    initial_conditions=initial_conditions,
    dynamics_type=dynamics_type,
    filter_params=mini_filter_params,
)
# %% Build dicts of parameters
agent_params = {
    "num_sensors": 2,
    "num_targets": 4,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist_frame": "ECI",
    "target_dist_frame": "ECI",
    "sensor_dist": "uniform",
    "target_dist": "normal",
    "sensor_dist_params": [
        [7000, 7500],
        [0, 0],
        [0, 0],
        [0, 0],
        [7.5, 8.5],
        [0, 0],
    ],
    "target_dist_params": [
        [10000, 100],
        [0, 0],
        [0, 0],
        [0, 0],
        [8, 0.1],
        [0, 0],
    ],
    "sensor_starting_num": 1000,
    "target_starting_num": 9000,
    "fixed_sensors": None,
    "fixed_targets": None,
    "init_num_tasked": None,
    "init_last_time_tasked": None,
}

filter_params = {
    "dynamics_type": "satellite",
    "Q": (1 * eye(6)).tolist(),
    "R": (1 * eye(6)).tolist(),
    "p_init": (1 * eye(6)).tolist(),
}


# %% Test SSAScheduleParams class
print("\nTest SSAScheduleParams class...")
env_params = SSASchedulerParams(
    time_step=1.1,
    horizon=10,
    reward_params=deepcopy(reward_params),
    agent_params=agent_params,
    filter_params=filter_params,
)

print("agent initial conditions:\n")
[print(a.eci_state) for a in env_params.agents]


# %% Test class with fixed agents
print("\n Test SSAScheduleParams with fixed agents...")

fixed_sensors = [
    [6500, 0, 0, 0, 0, 0],
    [0, 6500, 0, 0, 0, 0],
]

fixed_targets = [
    [7500, 0, 0, 0, 8, 0],
    [0, 8000, 0, -8, 0, 0],
    [0, 8000, 0, 8, 0, 0],
    [42000, 0, 0, 0, 3, 0],
]


agent_params2 = {
    "num_sensors": 2,
    "num_targets": 4,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist_frame": None,
    "target_dist_frame": None,
    "sensor_dist": None,
    "target_dist": None,
    "sensor_dist_params": None,
    "target_dist_params": None,
    "sensor_starting_num": 1000,
    "target_starting_num": 9000,
    "fixed_sensors": fixed_sensors,
    "fixed_targets": fixed_targets,
    "init_num_tasked": None,
    "init_last_time_tasked": None,
}
env_params2 = SSASchedulerParams(
    time_step=100,
    horizon=4,
    reward_params=deepcopy(reward_params),
    agent_params=agent_params2,
    filter_params=filter_params,
)
print("agent initial conditions:\n")
[print(a.eci_state) for a in env_params2.agents]
# %% Test _calcAccessWindows
print("\nTest _calcAccessWindows...")
list_of_sensors = [
    agent for agent in env_params2.agents if type(agent) is Sensor
]
list_of_targets = [
    agent for agent in env_params2.agents if type(agent) is Target
]

# Number of possible windows is number of steps in sim (env.horizon)
# All starting num_windows for targets should be <=horizon
num_windows = env_params2._calcAccessWindows(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
)
print(f"num_windows = {num_windows}")

# Include sensor-overlap in window count.
# Starting num_windows can be >horizon. In this example, one satellite (#4) is at GEO, so
# can be accessed by both sensors through all time steps.
num_windows = env_params2._calcAccessWindows(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
    merge_windows=False,
)
print(f"num_windows = {num_windows}")


# %% Test argument checks
print("\nTest argument checker...")

# fixed agents size doesn't match num_sensors
agent_params3 = deepcopy(agent_params2)
agent_params3["num_targets"] = 1
try:
    env_params3 = SSASchedulerParams(
        time_step=1.1,
        horizon=10,
        reward_params=deepcopy(reward_params),
        agent_params=agent_params3,
        filter_params=filter_params,
    )
except ValueError as err:
    print(err)

# inputs for both fixed agents and non-fixed distribution specified
agent_params4 = deepcopy(agent_params2)
agent_params4["sensor_dist"] = "normal"
try:
    env_params4 = SSASchedulerParams(
        time_step=1.1,
        horizon=10,
        reward_params=deepcopy(reward_params),
        agent_params=agent_params4,
        filter_params=filter_params,
    )
except ValueError as err:
    print(err)


agent_params5 = deepcopy(agent_params2)
agent_params5["target_dist_frame"] = "ECI"
try:
    env_params5 = SSASchedulerParams(
        time_step=1.1,
        horizon=10,
        reward_params=deepcopy(reward_params),
        agent_params=agent_params5,
        filter_params=filter_params,
    )
except ValueError as err:
    print(err)

# %% Test Defaults
# Check defaults for a few arguments
agent_params6 = {
    "num_sensors": 2,
    "num_targets": 4,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist_frame": None,
    "target_dist_frame": None,
    # "sensor_dist": None,
    "target_dist": None,
    # "sensor_dist_params": None,
    "target_dist_params": None,
    # "sensor_starting_num": 1000,
    # "target_starting_num": 9000,
    "fixed_sensors": fixed_sensors,
    "fixed_targets": fixed_targets,
    # "init_num_tasked": None,
    # "init_last_time_tasked": None,
}

env_params6 = SSASchedulerParams(
    horizon=10,
    reward_params=reward_params,
    agent_params=agent_params6,
    filter_params=filter_params,
)
print(f"env_params6.time_step = {env_params6.time_step}")
print(f"env_params6.reward_func = {env_params6.reward_func}")
print(f"sensor_dist = {env_params6.agent_params['sensor_dist']}")
print(f"sensor_dist_params = {env_params6.agent_params['sensor_dist_params']}")
print(
    f"starting sensor number = {env_params6.agent_params['sensor_starting_num']}"
)
print(f"last_time_tasked = {env_params6.agent_params['init_last_time_tasked']}")
print(f"num_tasked = {env_params6.agent_params['init_num_tasked']}")

# %%
print("done")
