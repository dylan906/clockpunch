"""Test policy.py module."""

# %% Imports

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from numpy import array, zeros
from numpy.random import rand, randint, seed

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.dynamics.dynamics_classes import SatDynamicsModel
from punchclock.estimation.ez_ukf import ezUKF
from punchclock.policies.need import NeedParameter

# %% Dummy Parameters
print("\nSetting parameters...")
# Set random seed
rand_seed = randint(0, 9999)
print(f"rand_seed = {rand_seed}")
seed(rand_seed)
# scenario stuff
num_targets = 5
num_sensors = 2

initial_p = rand(num_targets)
nvec = [num_targets + 1 for a in range(num_sensors)]
action_space = gym.spaces.MultiDiscrete(nvec)
print(f"action space sample = {action_space.sample()}")
print(f"action space nvec = {action_space.nvec}")
subsidy = 0.1
Q_init = zeros([num_targets + 1, num_sensors])
Q_init[-1, :] = subsidy
act_init = action_space.sample()
# %% Test policy initialization defaults
print("\nPolicy defaults...")
policy = NeedParameter(
    initial_p=initial_p,
    num_sensors=num_sensors,
    num_targets=num_targets,
)
print(f"policy.subsidy = {policy.subsidy}")
print(f"policy.Q = {policy.Q}")
print(f"policy.action = {policy.action}")

# %% Initialize policy
print("\nInitialize Policy...")
policy = NeedParameter(
    initial_p=initial_p,
    num_sensors=num_sensors,
    num_targets=num_targets,
    subsidy=subsidy,
    Q_initial=Q_init,
    action_initial=act_init,
    epsilon=0.01,
)

# %% Calc Q
print("\nCalculate Q...")

# visibility map (low inclusive, high exclusive)
vis_map = randint(0, 2, size=[num_targets, num_sensors])
print(f"vis_map = \n{vis_map}")

obs = {
    "vis_map_est": vis_map,
}

Q = policy.calcQ(vis_map)
print(f"p = {policy.p}")
print(f"mu = {policy.mu}")
print(f"Q = \n{Q}")

# %% Choose action
print("\nChoose action...")

action = policy.chooseAction(Q)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")
# %% Choose action w/ high epsilon
print("\nChoose action with high epsilon...")

policy.epsilon = 1
action = policy.chooseAction(Q)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")
print("Check again for good measure")
action = policy.chooseAction(Q)
print(f"action = {action}")
print(f"actions contained in action space? {action_space.contains(action)}")

# %% Generate p
print("\nGenerate p...")
p = policy.p
print(f"p = {p}")
p = policy._regenP(ones_flag=False)
print(f"_regenP(ones_flag=False) = {p}")

p = policy._regenP(ones_flag=True)
print(f"_regenP(ones_flag=True) = {p}")

# %% Evolve mu
print("\nRandomize mu...")
mu = policy.mu
print(f"mu = {mu}")
mu = policy._randomizeMu(mu, num_sensors=1)
print(f"mu after randomizer = {mu}")

print(f"policy.mu = {policy.mu}")
for i in range(num_targets):
    policy.update(obs, reward=1, debug=True)
    print(f"policy.mu after {i+1}x randomizes = {policy.mu}")

# %% Test calcPotentialReward
print("\nTest calcPotentialReward...")
# build targets and sensors
ukf = ezUKF(
    params={
        "Q": 0.01,
        "R": 0.1,
        "p_init": 1,
        "x_init": array([7000, 0, 0, 0, 4, 0]),
        "dynamics_type": "satellite",
    }
)
targ = Target(
    SatDynamicsModel(),
    agent_id=2,
    init_eci_state=array([7000, 0, 0, 0, 4, 0]),
    filter=deepcopy(ukf),
    time=0,
)
list_of_targets = [deepcopy(targ) for a in range(num_targets)]

sens = Sensor(
    SatDynamicsModel(),
    "A",
    array([1, 2, 3, 4, 5, 6]),
    5,
)
list_of_sensors = [deepcopy(sens) for a in range(num_sensors)]

# reset mu to be all ones
policy.mu = [1, 1, 0, 1, 0]
policy.calcPotentialReward(
    sensors=list_of_sensors, targets=list_of_targets, vis_map=vis_map
)

# %% Test getNetReward
print("\nTest getNetReward...")
actions = action_space.sample()
# actions = [5, 2]
print(f"actions = {actions}")
print(f"mu before = {policy.mu}")
reward = policy.getNetReward(
    sensors=list_of_sensors,
    targets=list_of_targets,
    actions=actions,
    vis_map_truth=vis_map,
    vis_map_est=vis_map,
)
print(f"mu after = {policy.mu}")
print(f"reward = {reward}")

# %%
print("done")
