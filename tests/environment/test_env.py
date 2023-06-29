"""Test for env module."""

# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
import matplotlib.pyplot as plt
from numpy import array, diag, linspace, zeros
from numpy.linalg import norm
from ray.rllib.utils import check_env

# Punch Clock Imports
from scheduler_testbed.common.constants import getConstants
from scheduler_testbed.common.math import getCircOrbitVel
from scheduler_testbed.common.transforms import ecef2eci
from scheduler_testbed.common.utilities import actionSpace2Array
from scheduler_testbed.environment.env import SSAScheduler
from scheduler_testbed.environment.env_parameters import SSASchedulerParams

# %% Supporting test Parameters
RE = getConstants()["earth_radius"]
time = linspace(0, 10, num=10).tolist()


# %% Instantiate environment class
# Build initial state vectors for 3 sensors, 4 targets.
#   Sensors 0, 1, 2 can see Targets 0, 2, 3
#   No sensors can see Target 1
# Sensors are terrestrial, so set in ECEF frame, then convert to ECI.
fixed_sensors_ecef = zeros([3, 6])
fixed_sensors_ecef[:, 0] = [RE, RE + 1, RE + 2]
fixed_sensors_eci = ecef2eci(fixed_sensors_ecef.transpose(), JD=0).transpose()

# targets are satellites, so can set in ECI with no need to convert
fixed_targets = zeros([4, 6])
fixed_targets[:, 0] = [RE + 400, -RE - 500, RE + 600, RE + 700]
for i in range(fixed_targets.shape[0]):
    r = norm(fixed_targets[i, :])
    v = getCircOrbitVel(r)
    fixed_targets[i, 4] = v

# fixed_targets[:, 4] = [getCircOrbitVel(r) for r in fixed_targets[:, 0]]

agent_params = {
    "num_sensors": 3,
    "num_targets": 4,
    "sensor_starting_num": 1000,
    "target_starting_num": 9000,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": None,
    "target_dist": None,
    "sensor_dist_frame": None,
    "target_dist_frame": None,
    "sensor_dist_params": None,
    "target_dist_params": None,
    "fixed_sensors": fixed_sensors_eci,
    "fixed_targets": fixed_targets,
    "init_num_tasked": None,
    "init_last_time_tasked": None,
}
diag_matrix = diag([1, 1, 1, 0.01, 0.01, 0.01])
ukf_params = {
    "Q": 0.001 * diag_matrix,
    "R": 0.01 * diag_matrix,
    "p_init": 1 * diag_matrix,
}

reward_params = {
    "reward_func": "Threshold",
    "obs_or_info": "info",
    "metric": "num_tasked",
    "metric_value": 3,
    "inequality": ">",
    "preprocessors": min,
    "reward": 0,
    "penalty": 0,
    "penalties": {
        "multi_assignment": 1.1,
        "non_vis_assignment": 0.3,
    },
    "subsidies": {
        "inaction": 0.1,
        "active_action": 0.2,
    },
}

env_params = SSASchedulerParams(
    time_step=1.2,
    horizon=10,
    agent_params=agent_params,
    filter_params=ukf_params,
    reward_params=deepcopy(reward_params),
)

env = SSAScheduler(env_params)
# %% Test reset() (functionality)
# More thorough test a bit later...
print("\nTest .reset()...")
reset_return = env.reset()
print(f".reset() return=\n{reset_return}")


# %% Test sample action space
print("\nTest .action_space.sample()...")
print(f"action space sample:\n{env.action_space.sample()}")


# %% Test _getObs()
print("\nTest ._getObs()...")

obs = env._getObs()
print(f"env._getObs()=\n{obs}")
print(f"observation space sample:\n{env.observation_space.sample()}")

# %% Test _getInfo()
print("\nTest ._getInfo()...")

info = env._getInfo()
print(f"env._getInfo()=\n{info}")

# %% Test _earnReward()
print("\nTest ._earnReward()...")
print("Remember that Target1 can't see any sensors")
action_array_blank = zeros([env.num_sensors], dtype=int)

print(f"vis map truth = \n{env.info['vis_map_truth']}")

# Task 1 non-visible sensor-target pair
action_test1 = deepcopy(action_array_blank)
action_test1[0] = 1
rewards1 = env._earnReward(action_test1)
print(f"rewards (2 visible, 1 not) = {rewards1}")

# Task a sensor to inaction, other 2 sensors to non-conflicting targets
action_test2 = deepcopy(action_array_blank)
action_test2[0] = env.num_targets
action_test2[1] = 1
rewards2 = env._earnReward(action_test2)
print(f"rewards (2 visible, 1 inaction (subsidy)) = {rewards2}")

# Task multiple sensors to same target
action_test4 = deepcopy(action_array_blank)
rewards4 = env._earnReward(action_test4)
print(f"rewards (multi-sensors to single target) = {rewards4}")

# %% Test updateInfoPreTasking()
# time and num_unique_targets_tasked should both update.
print("\nTest .updateInfoPreTasking()...")
print(f"  time now (pre-update) ={env._getInfo()['time_now']}")
print(
    f"  num unique targets tasked (pre-update) =\
    {env._getInfo()['num_unique_targets_tasked']}"
)
dummy_action = array([1, 3, 4])
env.updateInfoPreTasking(dummy_action)
print(f"  time now (post-update) ={env._getInfo()['time_now']}")
print(
    f"  num unique targets tasked (post-update) =\
        {env._getInfo()['num_unique_targets_tasked']}"
)

# %% Test step() and _taskAgents()
print("\nTest .step() and ._taskAgents()...")

action_test = env.action_space.sample()
[obs, reward, done, truncated, info] = env.step(action_test)
print("step output:")
print(f"    obs keys = {obs.keys()}")
print(f"    reward = {reward}")
print(f"    done = {done}")
print(f"    info keys = {info.keys()}")

# %% Test updateInfoPostTasking
print("\nTest .updateInfoPostTasking()...")
# Need to do an updateInfoPreTasking and taskAgents first.
action = env.action_space.sample()
action_array = actionSpace2Array(
    action,
    env.num_sensors,
    env.num_targets,
)
env.updateInfoPreTasking(action)

# State estimates should change, but sim time should not
print(f"  time now (pre-update) = {env._getInfo()['time_now']}")
print(f"  cov[0][0] (pre-update) = {env._getInfo()['est_cov'][0][0]}")
env._propagateAgents(env.info["time_now"])
env._taskAgents(action_array[:-1, :])
env.updateInfoPostTasking()
print(f"  time now (pre-update) = {env._getInfo()['time_now']}")
print(f"  cov[0][0] (post-update) = {env._getInfo()['est_cov'][0][0]}")

# %% Test non-vis counter
# Build an env with a very long time step to guarantee visibility status changes
# from step 0 to 1.
print("\n Test non-vis tracker...")
env_params_long_ts = SSASchedulerParams(
    time_step=4000,
    horizon=10,
    agent_params=agent_params,
    filter_params=ukf_params,
    reward_params=deepcopy(reward_params),
)
env_long_ts = SSAScheduler(env_params_long_ts)
env_long_ts.reset()

print(
    f"num non-vis taskings = {env_long_ts._getInfo()['num_non_vis_taskings_est']}"
)
print(f"vis map est (pre tasking) = \n{env_long_ts._getInfo()['vis_map_est']}")
# Task only visible targets
action = array([0, 1, 2])
env_long_ts.step(action)
print(f"action = {action}")
print(
    f"num non-vis taskings = {env_long_ts._getInfo()['num_non_vis_taskings_est']}"
)
print(f"vis map est (post tasking) = \n{env_long_ts._getInfo()['vis_map_est']}")
print(
    f"non-vis (est) by sensor (post tasking) = \
    \n{env_long_ts._getInfo()['non_vis_by_sensor_est']}"
)


# %% Test step and reset in loop (short)
# ensure agents and all attributes get reset
print("\nTest step/reset in short loop...")
# number of steps before reset
max_steps = 4
env_params = SSASchedulerParams(
    time_step=300,
    horizon=max_steps,
    agent_params=agent_params,
    filter_params=ukf_params,
    reward_params=deepcopy(reward_params),
)

# build and reset environment
env = SSAScheduler(env_params)
env.reset()

# step through environment through multiple resets
for _ in range(max_steps * 3 + 1):
    # num_windows_left should all decrement every step (until 0)
    print(f"num_windows_left = {env._getObs()['num_windows_left']}")

    # pick arbitrary action
    action_step = env.action_space.sample()

    [obs, rwd, done, truncated, info] = env.step(action_step)

    # print post-reset states
    if done is True:
        print(
            f"AFTER RESET: \n"
            f"  env._getInfo() = {env._getInfo()} \n"
            f"  eci_state[:3] = {env.agents[3].eci_state[:3]}\n"
            f"  num_windows_left = {env._getObs()['num_windows_left']}"
        )
# %% More thorough reset() test.
# Make sure that all properties of agent and agent.filter get reset to initial conditions
print("\nMore thorough reset test...")
print(f"backup agent[0] state = {env.reset_params.agents[0].eci_state}")
print(f"current agent[0] state = {env.agents[0].eci_state}")
print(
    f"backup agent[3] filter initial est_x = {env.reset_params.agents[3].filter.est_x}"
)
print(f"current agent[3] filter est_x = {env.agents[3].filter.est_x}")
print(f"tracker = {env.tracker.unique_tasks}")
print("RESET env")
env.reset()
print(f"post-reset agent[0] state = {env.agents[0].eci_state}")
print(f"post-reset agent[3] filter est_x = {env.agents[3].filter.est_x}")
print(f"tracker = {env.tracker.unique_tasks}")

# %% Test step/dynamics in long loop
# make sure dynamics make sense
print("\nTest step/dynamics in long loop (takes a minute)...")
# Note that this test is mostly to see dynamics and info are working correctly with
# different step sizes. Actions are random.

# build 2 environments that are the same except for step size
params_short_step = SSASchedulerParams(
    time_step=1.0,
    horizon=50,
    agent_params=agent_params,
    filter_params=ukf_params,
    reward_params=deepcopy(reward_params),
)
params_long_step = SSASchedulerParams(
    time_step=200,
    horizon=50,
    agent_params=agent_params,
    filter_params=ukf_params,
    reward_params=deepcopy(reward_params),
)

env_short_step = SSAScheduler(params_short_step)
env_long_step = SSAScheduler(params_long_step)
env_short_step.reset()
env_long_step.reset()

# horizon is same for both environments
max_steps = env_short_step.horizon

# initialize logging variables
state_hist_short = zeros([max_steps, 6, len(env_short_step.agents)])
state_hist_long = zeros([max_steps, 6, len(env_short_step.agents)])
info_hist_short = []
info_hist_long = []
obs_hist_short = []
obs_hist_long = []

# initialize counter and reset variable
done = False
i = 0
while not done:
    # print(i)
    # action doesn't matter for this test-- random action.
    action_step = env_short_step.action_space.sample()
    # save state and info
    info = env_short_step._getInfo()
    info_hist_short.append(deepcopy(info))
    info = env_long_step._getInfo()
    info_hist_long.append(deepcopy(info))
    obs = env_short_step._getObs()
    obs_hist_short.append(deepcopy(obs))
    obs = env_long_step._getObs()
    obs_hist_long.append(deepcopy(obs))
    state_hist_short[i, :, :] = info["true_states"]
    state_hist_long[i, :, :] = info["true_states"]

    # step envs forward
    [obs, rwd, done, truncated, info] = env_short_step.step(action_step)
    [obs, rwd, done, truncated, info] = env_long_step.step(action_step)

    i += 1


# plot state history, look for abnormalities
fig, axs = plt.subplots(2, 2)
fig.suptitle("Short vs Long Step Size States")
axs[0, 0].set_title("position, short step, sat")
axs[0, 0].plot(state_hist_short[:, :3, 0])
axs[1, 0].set_title("position, long step, sat")
axs[1, 0].plot(state_hist_long[:, :3, 0])
axs[0, 1].set_title("position, short step, terrestrial")
axs[0, 1].plot(state_hist_short[:, :3, 2])
axs[1, 1].set_title("position, long step, terrestrial")
axs[1, 1].plot(state_hist_long[:, :3, 2])
plt.tight_layout()

# plot estimate covariances for sample target
# diagonal elements should be positive; off-diagonals may have negative values
sample_targ = 0
fig, axs = plt.subplots(6, 6)
fig.suptitle("Covariance History of All Targets")
for row in range(6):
    for col in range(6):
        cov_hist = [i["est_cov"] for i in info_hist_long]
        time_hist_element = [frame[:, row, col] for frame in cov_hist]
        axs[row, col].plot(time_hist_element)

# plot primary metrics
fig, axs = plt.subplots(2, 2)
fig.suptitle("Catalog Metrics History")
axs[0, 0].set_title("log mean pos cov, short step")
# plt.yscale("log")
axs[0, 0].plot([i["mean_pos_var"] for i in info_hist_short])
axs[0, 0].set_yscale("log")
axs[0, 1].set_title("unique taskings, short step")
axs[0, 1].plot([i["num_unique_targets_tasked"] for i in info_hist_short])
axs[1, 0].set_title("log mean pos cov, long step")
axs[1, 0].plot([i["mean_pos_var"] for i in info_hist_long])
axs[1, 0].set_yscale("log")
axs[1, 1].set_title("unique taskings, long step")
axs[1, 1].plot([i["num_unique_targets_tasked"] for i in info_hist_long])
plt.tight_layout()

# plot secondary metrics
# not plotting both long and short step for this one
fig, axs = plt.subplots(2, 1)
fig.suptitle("Secondary Metrics, Long step")
axs[0].set_title("num_non_vis_taskings_truth")
axs[0].plot([i["num_non_vis_taskings_truth"] for i in info_hist_long])
axs[1].set_title("num_multiple_taskings")
axs[1].plot([i["num_multiple_taskings"] for i in info_hist_long])
plt.tight_layout()

# plot number of windows left
fig, axs = plt.subplots(2)
fig.suptitle("Number of Windows Left (short vs long step)")
axs[0].set_title("num_windows_left, short step")
axs[0].plot([i["num_windows_left"].squeeze() for i in obs_hist_short])
axs[1].set_title("num_windows_left, long step")
axs[1].plot([i["num_windows_left"].squeeze() for i in obs_hist_long])
plt.tight_layout()

# %% Check environment passes Ray checker
print("\nRay environment checker...")
check_env(env)
# %% Test registration and gym.make()
print("\nTest gym environment creation stuff...")
# NOTE: See __init__.py in "scheduler_testbed" for registration code.
env3 = gym.make("SSAScheduler-v0", scenario_params=env_params)
env3.reset()
[obs, rwd, done, truncated, info] = env3.step(action_test)
print(f"post-step: obs={obs}, reward={rwd}, done={done}, info={info}")

# %% Done
plt.show()
print("done")
