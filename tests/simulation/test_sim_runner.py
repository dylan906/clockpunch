"""Test sim_runner module."""
# NOTE: This script writes a file (see toDataFrame test).
# %% Imports
from __future__ import annotations

# Standard Library Imports
import json
from collections import OrderedDict
from copy import deepcopy
from os import path

# Third Party Imports
import matplotlib.pyplot as plt
from gymnasium.wrappers import FilterObservation
from numpy import array, cumsum, linspace, pi, zeros
from numpy.random import default_rng
from pandas import read_csv, read_pickle
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.environment.wrapper_utils import getWrapperList
from punchclock.environment.wrappers import ActionMask, FlatDict
from punchclock.policies.greedy_cov_v2 import GreedyCovariance
from punchclock.policies.random_policy import RandomPolicy
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.sim_runner import (
    SimResults,
    SimRunner,
    concatenateStates,
    formatSimResultsDF,
)

# %% RNG
rng = default_rng()


# %% Define shortcut functions for plotting and printing sim results
def plotSimResults(results: SimResults, env, time):
    """Shortcut function for plotting standard sim results."""
    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0, 0].plot(time, results.done)
    axs[0, 0].set(ylabel="Done")
    axs[0, 0].set(xlabel="Time")
    axs[1, 0].plot(time, results.actions)
    axs[1, 0].set(ylabel="actions")
    axs[1, 0].set(xlabel="Time")
    axs[0, 1].plot(time, results.reward)
    axs[0, 1].set(ylabel="reward")
    axs[0, 1].set(xlabel="Time")
    axs[1, 1].plot(time, cumsum(results.reward))
    axs[1, 1].set(ylabel="cum_reward")
    axs[1, 1].set(xlabel="Time")
    plt.tight_layout()

    # Get true states from info. Assemble as [T x (N+M) x 6] array
    states = zeros([len(time), env.num_agents, 6])
    for i, time_frame in enumerate(results.info):
        # for j, agent in enumerate(time_frame["true_states"]):
        states[i, :, :] = time_frame["true_states"].T
    # Plot state histories
    fig, axs = plt.subplots(3, sharex=True)
    for i in range(env.num_agents):
        axs[0].plot(time, states[:, i, 0])
        axs[1].plot(time, states[:, i, 1])
        axs[2].plot(time, states[:, i, 2])
    axs[0].set(ylabel="I")
    axs[1].set(ylabel="J")
    axs[2].set(ylabel="K")
    axs[2].set(xlabel="Time")


def printSimResults(sim_results: SimResults):
    """Shortcut function for printing selected sim results."""
    # Make sure variable types are consistent between initial values and non-initial
    # values.
    print(" Variable Types:")
    dict_of_vars = {
        "actions": sim_results.actions,
        "done": sim_results.done,
        "obs": sim_results.obs,
        "reward": sim_results.reward,
    }
    for key, var in dict_of_vars.items():
        print(f"  {key} = {var[0]=}, type = {type(var[0])}")
        print(f"  {key} = {var[1]=}, type = {type(var[1])}")


# %% time vector
# number of steps to take
horizon = 10
# Make sure to use enough points for the pre-calculated visibility history to not
# miss any visibility windows (0-crossings).
time = linspace(0, 90 * 3, num=horizon).tolist()
# %% Build environment
print("\nBuild test environment...")
filter_params = {
    "Q": 0.1,
    "R": 0.1,
    "p_init": 1,
}
# 2 sensors / 3 targets at fixed positions (not randomly generated)
sensor_ICs = array(
    [
        [6500, 0, 0, 0, 0, 0],
        [0, 6500, 0, 0, 0, 0],
    ]
)
target_ICs = array(
    [
        [7000, 0, 0, 0, 8, 0],
        [7000, 1000, 0, 0, 8, 0],
        [-7500, 1000, 0, 0, 8, 0],
    ]
)
agent_params = {
    "num_sensors": 2,
    "num_targets": 3,
    "sensor_starting_num": 101,
    "target_starting_num": 1,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist_frame": None,
    "target_dist_frame": None,
    "sensor_dist": None,
    "target_dist": None,
    "sensor_dist_params": None,
    "target_dist_params": None,
    "fixed_sensors": sensor_ICs,
    "fixed_targets": target_ICs,
    "init_num_tasked": None,
    "init_last_time_tasked": None,
}

reward_params = {
    "reward_func": "Threshold",
    "obs_or_info": "obs",
    "metric": "num_tasked",
    "metric_value": 3,
    "penalty": 1,
    "inequality": ">=",
    "preprocessors": ["min"],
}

# assign build params as a dict for easier env recreation/changing
# ActionMask wrapper required for SimRunner
config = OrderedDict(
    {
        "time_step": time[1] - time[0],
        "horizon": horizon,
        "agent_params": agent_params,
        "filter_params": filter_params,
        "reward_params": reward_params,
        "constructor_params": {
            "wrappers": [
                {
                    "wrapper": "filter_observation",
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
                    "wrapper": "copy_obs_item",
                    "wrapper_config": {
                        "key": "vis_map_est",
                        "new_key": "action_mask",
                    },
                },
                {
                    "wrapper": "vis_map_action_mask",
                    "wrapper_config": {"vis_map_key": "action_mask"},
                },
                {
                    "wrapper": "nest_obs_items",
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
                {"wrapper": "flat_dict"},
            ],
        },
    }
)

env = buildEnv(env_config=config)
print(f"environment = {env}")
# %% Build dummy policy for simRunner
print("\nBuild Torch-based test policy...")
register_env("my_env", buildEnv)
algo_config = (
    ppo.PPOConfig()
    .training()
    .environment(
        env="my_env",
        env_config=config,
    )
    .framework("torch")
)
algo = algo_config.build()
policy = algo.get_policy()
print(f"policy = {policy}")

# %% Initialize SimRunner
print("\nInitialize SimRunner...")
sim_runner = SimRunner(env=env, policy=policy, max_steps=horizon)
print(f"sim_runner = {sim_runner}")
print(f"sim_runner.env.info['time_now'] = {sim_runner.env.info['time_now']}")

# %% Test ConcatenateStates()
print("\nTest concatenateStates()...")
list_of_agents = deepcopy(env.agents)
states = concatenateStates(list_of_agents)
print(f"states = \n{states}")

# %% Test step()
print("\nTest step()...")
act = sim_runner.env.action_space.sample()
[obs, reward, done, info, next_action] = sim_runner.step(action=act)
print(f"obs = {obs}")
print(f"reward = {reward}")
print(f"done = {done}")
print(f"info = {info}")
print(f"next_action = {next_action}")

# %% Test runSim()
print("\nTest runSim()...")
results1 = sim_runner.runSim()

# %% Plot results
print("\nPlot results...")
plotSimResults(results=results1, env=env, time=time)
printSimResults(sim_results=results1)

# %% Test with wrapped environment/Ray policy
print("\nTest with wrapped env/Ray policy...")
# Copy original environment config and add wrappers.
config_wrap = deepcopy(config)
config_wrap["constructor_params"] = {
    "wrappers": [
        {
            "wrapper": "filter_observation",
            "wrapper_config": {"filter_keys": ["est_cov", "vis_map_est"]},
        },
        {
            "wrapper": "copy_obs_item",
            "wrapper_config": {"key": "vis_map_est", "new_key": "action_mask"},
        },
        {
            "wrapper": "vis_map_action_mask",
            "wrapper_config": {"vis_map_key": "action_mask"},
        },
        {
            "wrapper": "nest_obs_items",
            "wrapper_config": {
                "new_key": "observations",
                "keys_to_nest": ["vis_map_est", "est_cov"],
            },
        },
        {"wrapper": "flat_dict"},
    ]
}

algo_wrap_config = (
    ppo.PPOConfig()
    .training()
    .environment(
        env="my_env",
        env_config=config_wrap,
    )
    .framework("torch")
)
algo_wrap = algo_wrap_config.build()
policy_wrap = algo_wrap.get_policy()
wrapped_env = buildEnv(env_config=config_wrap)
sim_runner3 = SimRunner(env=wrapped_env, policy=policy_wrap, max_steps=horizon)
results_wrap = sim_runner3.runSim()
print("Wrapped env sim results:")
printSimResults(sim_results=results_wrap)
# Check that observations are same for 0th and other values
print(
    "Keys same for all obs?",
    f"{results_wrap.obs[0].keys()== results_wrap.obs[1].keys()}",
)
# %% Test with custom policy
print("\nTest runSim() with CustomPolicy...")
# Rebuild environment, but without flattening (don't use last wrapper)
config_noflat = deepcopy(config)
del config_noflat["constructor_params"]["wrappers"][-1]
env2 = buildEnv(config_noflat)
env2.reset()
# Build new policy
obs_space = env2.observation_space
act_space = env2.action_space
policy2 = GreedyCovariance(
    observation_space=obs_space,
    action_space=act_space,
    epsilon=0,
)

sim_runner2 = SimRunner(env=env2, policy=policy2, max_steps=horizon)
results2 = sim_runner2.runSim()
plotSimResults(results=results2, env=env2, time=time)
print("CustomPolicy sim results:")
printSimResults(sim_results=results2)

# %% Test different wrapper configs
print("\nTest different wrapper configs")
# Test 4 configs:
#   1. ActionMask as top and only wrapper
#   2. ActionMask as bottom wrapper
#   3. ActionMask as neither top nor bottom wrapper
#   4. Bare environment

# Create list of environments. Will manually wrap envs instead of using relying
# on constructor, so remove ActionMask wrapper from config.
config2 = deepcopy(config)
config2["constructor_params"] = {}
envs = [
    ActionMask(buildEnv(config2)),
    FlatDict(ActionMask(buildEnv(config2))),
    FlatDict(ActionMask(FilterObservation(buildEnv(config2)))),
    buildEnv(config2),
]
# To automate building policies, need to get the index of the layer that ActionMask
# is in the stack of wrappers. First, create a list of wrappers (for each env).
# Then find index of ActionMask in list of wrappers.
lists_of_wrappers = [getWrapperList(e) for e in envs]
obs_space_indices = []
for wrappers in lists_of_wrappers:
    if ActionMask in wrappers:
        obs_space_indices.append(wrappers.index(ActionMask))
    else:
        obs_space_indices.append(0)

# Loop through environments and observation space indices. Use action mask location
# (act_mask_loc) to programmatically get the observation space from the environment,
# then build the corresponding policy. With the policy built, use this and the
# env to build a sim_runner and then run a sim. Use try/except to test assertions
# in sim_runner creation.
for env, act_mask_loc in zip(envs, obs_space_indices):
    print(f"\nobs space = {env.observation_space}")
    # The number of "env." is variable, so use eval() to get obs space.
    strcmd = "env." + (act_mask_loc * "env.") + "observation_space"
    obs_space = eval(strcmd, {"env": env})
    try:
        policy = GreedyCovariance(
            observation_space=obs_space,
            action_space=env.action_space,
        )
        sim_runner = SimRunner(
            env=env,
            policy=policy,
            max_steps=horizon,
        )
        results = sim_runner.runSim()
    except Exception as err:
        print("error caught")
        print(err)

# %% Test random initial conditions
print("\nTest with random initial conditions and many targets...")
config_rand = deepcopy(config_noflat)
config_rand["seed"] = rng.integers(99999999)
print(f"seed = {config_rand['seed']}")

# Make target params for randomly distributed, then replace selected keys.
rand_agent_params = {
    "num_targets": 100,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "target_dist_frame": "COE",
    "target_dist": "uniform",
    "target_dist_params": [
        [1000, 2000],
        [0, 0],
        [0, pi],
        [0, 2 * pi],
        [0, 2 * pi],
        [0, 2 * pi],
    ],
    "fixed_targets": None,
}
for k, v in rand_agent_params.items():
    config_rand["agent_params"][k] = v

env_rand = buildEnv(config_rand)
policy_rand = RandomPolicy(
    observation_space=env_rand.observation_space,
    action_space=env_rand.action_space,
)
sim_runner4 = SimRunner(env=env_rand, policy=policy_rand, max_steps=2)
results_rand = sim_runner4.runSim()

# %% Test SimResults.toPrimitiveSimResults()
print("\nTest SimResults.toPrimitiveSimResults...")
primitive_results = results1.toPrimitiveSimResults()
# check if primitive results dict is json-able
try:
    json.dumps(primitive_results.__dict__)
    print("JSON dump successful")
except Exception as err:
    print(err)

# %% Set toDataFrame()
print("\nTest SimResults.toDataFrame...")
results_df = results1.toDataFrame()
print(f"DataFrame columns = {results_df.columns}")

fpath = path.dirname(path.realpath(__file__)) + "/data/test_df.pkl"
results_df.to_pickle(fpath)

df_loaded = read_pickle(fpath)

# %% Test PrimitiveSimResults.toDataFrame()
print("\nTest PrimitiveSimResults.toDataFrame...")
primitive_df = primitive_results.toDataFrame()

fpath = path.dirname(path.realpath(__file__)) + "/data/test_df.csv"
primitive_df.to_csv(fpath)

# Check csv loading to DF
primitive_loaded = read_csv(fpath)
print(primitive_loaded.columns)
# %% Test formatSimResultsDF
print("\nTest formatSimResultsDF...")
df = read_csv("tests/simulation/data/test_df.csv")
df_new = formatSimResultsDF(df, print_output=True)
print(df_new.columns)

# %% Done
plt.show()
print("done")
