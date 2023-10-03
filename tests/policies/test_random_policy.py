"""Test for random_policy.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from numpy import Inf

# Punch Clock Imports
from punchclock.common.utilities import MaskConverter
from punchclock.policies.random_policy import RandomPolicy

# %% Build action/obs spaces
num_sensors = 2
num_targets = 4
mask_dim = (num_targets + 1, num_sensors)
obs_space = Dict(
    {
        "observations": Dict(
            {
                "est_cov": Box(-Inf, Inf, shape=[6, num_targets], dtype=float),
                "vis_map_est": MultiBinary((num_targets, num_sensors)),
            }
        ),
        "action_mask": MultiBinary(mask_dim),
    }
)

action_space = MultiDiscrete([num_targets + 1] * num_sensors)
# %% Initialize policy
print("\nInitialize policy...")
pol = RandomPolicy(observation_space=obs_space, action_space=action_space)
print(f"policy = {pol}")
print(f"policy.use_mask = {pol.use_mask}")

# %% test computeAction
print("\ncomputeAction...")
# Test child method, then test base method (which includes space checking)
obs = pol.observation_space.sample()
action = pol.computeAction(obs=obs)
print(f"action = {action}")

try:
    action = pol._computeSingleAction(obs=obs)
    print("Checks passed")
except Exception as er:
    print(er)

# %% Test with action mask
print("\ncomputeAction with action mask enabled...")

# Try to instantiate policy without "action_mask" as a key (should throw error).
obs_space = Dict(
    {
        "observations": Dict(
            {
                "est_cov": Box(-Inf, Inf, shape=[6, num_targets], dtype=float),
                "vis_map_est": MultiBinary((num_targets, num_sensors)),
            }
        )
    }
)

try:
    pol = RandomPolicy(
        observation_space=obs_space,
        action_space=action_space,
        use_mask=True,
    )
except Exception as err:
    print(err)

# Build policy correctly
obs_space = Dict(
    {
        "observations": Dict(
            {
                "est_cov": Box(-Inf, Inf, shape=[6, num_targets], dtype=float),
                "vis_map_est": MultiBinary((num_targets, num_sensors)),
            }
        ),
        "action_mask": MultiBinary(mask_dim),
    }
)
pol = RandomPolicy(
    observation_space=obs_space,
    action_space=action_space,
    use_mask=True,
)
obs = obs_space.sample()
print(f"action mask = {obs['action_mask']}")

action = pol.computeAction(obs=obs)
print(f"action = {action}")

# Test random actions in loop to make sure they don't violate mask. There is a
# mask violation checker built into computeAction(); any violations will be tripped
# automatically. Randomly sampling observation space doesn't guarantee that vis_map
# is consistent with action mask, so manually modify with a mask_converter.
mask_converter = MaskConverter(num_targets, num_sensors)
for _ in range(10):
    obs = obs_space.sample()
    obs["action_mask"] = mask_converter.appendInactionRowToActionMask(
        obs["observations"]["vis_map_est"]
    )
    vis_map = obs["observations"]["vis_map_est"]
    action = pol.computeAction(obs=obs)
    print(f"vis_map = {vis_map}")
    print(f"action = {action}")


# %% done
print("done")
