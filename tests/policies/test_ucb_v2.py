"""Test ucb_v2.py."""

# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from numpy import ones
from numpy.random import randint

# Punch Clock Imports
from scheduler_testbed.policies.ucb_v2 import UpperConfidenceBounds

# %% Initialize policy
print("\nTest initialization...")
num_targets = 4
num_sensors = 2

obs_space = Dict(
    {
        "observations": Dict(
            {
                "vis_map_est": Box(
                    0, 1, shape=[num_targets, num_sensors], dtype=int
                )
            }
        ),
        "action_mask": Box(
            0, 1, shape=[(num_targets + 1) * num_sensors], dtype=int
        ),
    }
)
act_space = MultiDiscrete([num_targets] * num_sensors)

ucb = UpperConfidenceBounds(
    observation_space=obs_space,
    action_space=act_space,
    exploration_param=0.2,
    max_reward=100,
)

# %% Calculate Q
print("\nTest calcQ...")

vis_map = ucb.observation_space.sample()["observations"]["vis_map_est"]
Q = ucb.calcQ(vis_map)
print(f"Q = \n{Q}")

# %% computeAction
print("\nTest computeAction...")
# TODO: Bookmark: Pickup rebuild here
obs = ucb.observation_space.sample()
action = ucb.computeAction(obs=obs)

print(f"vis_map = \n{obs['observations']['vis_map_est']}")
print(f"action = {action}")

# %% Reinitialize policy with non-default Nt
num_targets = 4
num_sensors = 2

Nt = randint(0, 4, size=(num_targets))
print(f"Nt = {Nt}")

ucb2 = UpperConfidenceBounds(
    observation_space=obs_space,
    action_space=act_space,
    exploration_param=0.2,
    max_reward=100,
    num_previous_actions=Nt,
)
obs = ucb2.observation_space.sample()
action = ucb2.computeAction(obs)

print(f"vis_map = \n{obs['observations']['vis_map_est']}")
print(f"action = {action}")
# %% Test in loop
# Check that Nt is incrementing and that policy is balancing actions among available
# targets
obs = ucb2.observation_space.sample()
obs["observations"]["vis_map_est"] = ones(
    (num_targets, num_sensors),
    dtype=int,
)
for _ in range(10):
    action = ucb2.computeAction(obs=obs)
    # print(f"vis_map = \n{obs['observations']['vis_map_est']}")
    print(f"action = {action}")
    print(f"Nt = {ucb2.Nt}")

# %% done
print("done")
