"""Tests for visibility.py."""
# Third Party Imports
from numpy import array

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.common.visibility import calcVisMap, calcVisMapAndDerivative

RE = getConstants()["earth_radius"]
# %% Make test data
sensor_states = ecef2eci(
    array(
        [
            [RE + 100, 0, 0, 0, 0, 0],
            [-RE - 100, 0, 0, 0, 0, 0],
        ],
    ).T
)
target_states = array(
    [
        [RE + 200, 0, 0, 0, 1, 0],
        [-RE - 200, 0, 0, 0, 0, 1],
        [0, RE + 0.1, 0, 1, 0, 0],
        [0, -RE - 0.1, 0, 1, 0, 0],
    ],
).T

# %% CalcVisMap
print("\nTest calcVisMap()...")
# test correct inputs
vis_map = calcVisMap(
    sensor_states=sensor_states,
    target_states=target_states,
    body_radius=RE,
)
print(f"vis_map = \n{vis_map}")

vis_map = calcVisMap(
    sensor_states=sensor_states,
    target_states=target_states,
    body_radius=RE,
    binary=False,
)
print(f"vis_map = \n{vis_map}")

# test with incorrect dimensions
try:
    vis_map = calcVisMap(
        sensor_states=sensor_states.T,
        target_states=target_states,
        body_radius=RE,
    )
except Exception as err:
    print(err)
    pass

try:
    vis_map = calcVisMap(
        sensor_states=sensor_states,
        target_states=target_states.T,
        body_radius=RE,
    )
except Exception as err:
    print(err)
    pass

# %% Test calcVisMapAndDerivative
print("\nTest calcVisMapAndDerivative()...")
vis_map, vis_map_der = calcVisMapAndDerivative(
    sensor_states=sensor_states,
    target_states=target_states,
    body_radius=RE,
    binary=False,
)
print(f"{vis_map=}")
print(f"{vis_map_der=}")
