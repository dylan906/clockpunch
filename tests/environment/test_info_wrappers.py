"""Tests for info_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete

# from gymnasium.utils.env_checker import check_env
from numpy import array, diag, pi
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.environment.info_wrappers import (
    ActionTypeCounter,
    MaskReward,
    NumWindows,
)
from punchclock.ray.build_env import buildEnv

# %% Build env for NumWindows wrapper
print("\nBuild env for NumWindows test...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": Box(low=0, high=1)}),
        "action_space": MultiDiscrete([1]),
    }
)
agents = [buildRandomAgent(agent_type="sensor") for ag in range(2)]
agents.extend([buildRandomAgent(agent_type="target") for ag in range(3)])
rand_env.agents = agents
rand_env.horizon = 10
rand_env.time_step = 100

# %% Test NumWindows
print("\nTest NumWindows...")
nw_env = NumWindows(env=rand_env, use_estimates=False)
obs, _, _, _, info = nw_env.step(nw_env.action_space.sample())
print(f"info = {info}")

obs, info = nw_env.reset()
print(f"info = {info}")

# %% Test with SSAScheduler
# Environment params
RE = getConstants()["earth_radius"]

horizon = 50
time_step = 100
# sensor locations (fixed)
x_sensors_lla = [
    array([0, 0, 0]),
    array([0, pi / 4, 0]),
]

num_sensors = len(x_sensors_lla)
num_targets = 3

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

constructor_params = {
    "wrappers": [
        {
            "wrapper": "NumWindows",
            "wrapper_config": {
                "use_estimates": False,
                "new_keys": ["num_windows_alt", "vis_forecast"],
            },
        }
    ]
}

env_config = {
    "horizon": horizon,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "time_step": time_step,
    "constructor_params": constructor_params,
}
ssa_env = buildEnv(env_config)
ssa_env.reset()
# test in loop
for _ in range(horizon - 1):
    obs, _, _, _, info = ssa_env.step(ssa_env.action_space.sample())
    print(f"num windows left = {info['num_windows_alt']}")
    print(f"vis vorecast shape = {info['vis_forecast'].shape}")
# %% Use gym checker
# check_env(rand_env)
# %% Test ActionTypeCounter
print("\nTest ActionTypeCounter...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((2, 2))}),
        "action_space": MultiDiscrete([3, 3]),
    }
)
nar_env = ActionTypeCounter(rand_env, new_key="count")

action = array([0, 2])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"info={info}")


nar_env = ActionTypeCounter(rand_env, new_key="count", count_null_actions=False)

action = array([0, 0])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"info={info}")

# %% Test MaskReward
print("\nTest MaskReward...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((3, 4))}),
        "action_space": MultiDiscrete([3, 3, 3, 3]),
        "reward_space": Box(0, 0),
    }
)
binary_env = MaskReward(rand_env, "a", reward=0.1)
action = array([0, 0, 0, 2])

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"obs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with rewarding (penalizing) invalid actions
binary_env = MaskReward(rand_env, "a", reward=-0.1, reward_valid_actions=False)

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# Test with accounting for null actions
binary_env = MaskReward(rand_env, "a", ignore_null_actions=False)

(obs, reward, term, trunc, info) = binary_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"reward={reward}")

# %% done
print("done")
