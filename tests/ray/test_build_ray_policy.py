"""Tests for build_ray_policy.py."""
# Punch Clock Imports
from punchclock.ray.build_ray_policy import buildCustomRayPolicy

# %% Tests
path = "tests/ray/exp_name/PPO_my_env_3c871_00000_0_2023-01-17_10-25-56/checkpoint_000001/policies/default_policy"

ray_policy = buildCustomRayPolicy(checkpoint_path=path)
print(ray_policy)
# %% Done
print("done")
