"""Tests for sim_utils.py."""
# NOTE: This script requires a Ray checkpoint in "data/" to run.
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from numpy import pi
from numpy.linalg import norm

# Punch Clock Imports
from punchclock.common.transforms import eci2ecef
from punchclock.simulation.sim_utils import (
    buildCustomOrRayPolicy,
    genInitStates,
    saturateCOEs,
)

# %% Test saturateCOEs
print("\nTest saturateCOEs...")
coe = {
    "sma": [0, 7000],
    "ecc": [-0.1, 1.1],
    "inc": [-1, 2 * pi],
    "raan": [-pi, 2 * pi + 1],
    "argp": [-pi, 2 * pi + 1],
    "ta": [-pi, 2 * pi + 1],
}

treated_coe = saturateCOEs(**coe)
print(f"coe = {coe}")
print(f"treated coe = {treated_coe}")

# Test return as array
treated_coe_array = saturateCOEs(**coe, return_array=True)
print(f"array return = {treated_coe_array}")


# %% Test genInitStates
print("\nTest genInitStates...")
# Test with distributions in ECI frame
init_uniform = genInitStates(
    2,
    "uniform",
    [[7000, 7500], [8000, 10000], [0, 0], [0, 0], [7.5, 8.5], [0, 0]],
    frame="ECI",
)
print(f"ECI init_uniform = \n{init_uniform}")

init_normal = genInitStates(
    3,
    "normal",
    [[7000, 100], [8000, 500], [0, 0], [0, 0], [7.5, 0.1], [0, 0]],
    frame="ECI",
)
print(f"ECI init_normal = \n{init_normal}")

# Test with distributions in LLA frame
init_lla = genInitStates(
    2,
    "normal",
    [[pi / 2, 0.1], [0, 1], [0, 0], [0, 0], [3, 3], [0, 0]],
    frame="LLA",
)
print(f"LLA init (in ECI frame) = \n{init_lla}")
init_lla_ecef = [eci2ecef(vec, 0) for vec in init_lla]
# velocities in ECEF should be 0
print(f"LLA init (in ECEF frame) = \n{init_lla_ecef}")
# check magnitude of position vector
print(f"norm(position vectors) = {norm(init_lla_ecef[0])}")

# Test with distribution in COEs
init_coe = genInitStates(
    2,
    "normal",
    [[6000, 8000], [0, 1], [0, pi], [-pi, pi], [-pi, pi], [-pi, pi]],
    frame="COE",
)
print(f"COE init (in ECI frame) = \n{init_coe}")

# %% Test buildCustomOrRayPolicy
print("\nTest buildCustomOrRayPolicy...")
# ray_path = "tests/simulation/data/test_checkpoint/checkpoint_000200/policies/default_policy"
fpath = os.path.dirname(os.path.realpath(__file__))
ray_path = (
    fpath
    + "/data/test_checkpoint2/test_trial/PPO_ssa_env_f26ba_00000_0_2023-08-22_11-52-50/checkpoint_000001/policies/default_policy"
)

ray_policy = buildCustomOrRayPolicy(config_or_path=ray_path)
print(f"Ray policy = {ray_policy}")

custom_policy_config = {
    "policy": "RandomPolicy",
    "observation_space": Dict(
        {
            "observations": Dict(
                {
                    "a": Box(0, 1, shape=(2, 2)),
                }
            ),
            "action_mask": Box(0, 1, shape=(2, 2)),
        }
    ),
    "action_space": MultiDiscrete([2, 2, 2]),
}
custom_policy = buildCustomOrRayPolicy(config_or_path=custom_policy_config)
print(f"Custom policy = {custom_policy}")

# %% Done
print("done")
