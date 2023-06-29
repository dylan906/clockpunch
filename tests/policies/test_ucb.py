"""Test ucb.py."""

# %% Imports
# Third Party Imports
from numpy import ones
from numpy.random import randint

# Punch Clock Imports
from scheduler_testbed.policies.ucb import UpperConfidenceBounds

# %% Initialize policy
num_targets = 4
num_sensors = 2

ucb = UpperConfidenceBounds(
    exploration_param=0.2,
    max_reward=100,
    num_sensors=num_sensors,
    num_targets=num_targets,
)

vis_map = randint(0, 2, size=(num_targets, num_sensors))
print(f"vis_map = \n{vis_map}")
# %% Calculate Q
Q = ucb.calcQ(vis_map)
print(f"Q = \n{Q}")

# %% Update
obs = {"vis_map_est": vis_map}
[Q, cum_reward, action] = ucb.update(observation=obs, reward=1)

print(f"Q = \n{Q}")
print(f"action = {action}")

# %% Reinitialize policy with non-default Nt
num_targets = 4
num_sensors = 2

Nt = randint(0, 4, size=(num_targets))
print(f"Nt = {Nt}")

ucb2 = UpperConfidenceBounds(
    exploration_param=0.2,
    max_reward=100,
    num_sensors=num_sensors,
    num_targets=num_targets,
    num_previous_actions=Nt,
)
vis_map = ones((num_targets, num_sensors))
obs = {"vis_map_est": vis_map}

# %% Test in loop
for _ in range(5):
    [Q, cum_reward, action] = ucb2.update(observation=obs, reward=1)
    # print(f"Q = \n{Q}")
    print(f"action = {action}")
    print(f"Nt = {ucb2.Nt}")

# %% done
print("done")
