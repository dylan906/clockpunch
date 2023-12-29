"""Tests for orbits.py."""
# %% Imports
# Third Party Imports
from numpy import array
from numpy.linalg import norm

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.orbits import (
    getCircOrbitVel,
    getRadialRate,
    getTrueAnomaly,
)

RE = getConstants()["earth_radius"]
MU = getConstants()["mu"]

# %% Test getTrueAnomaly
print("\nTest getTrueAnomaly...")
# Test with arbitrary input
r_vec = array([RE + 400, 0, 0])
v_vec = array([1, 8, 0])
e_unit_vec = array([1, 0, 0])
ta = getTrueAnomaly(r_vec, v_vec, e_unit_vec)
print(f"{ta=}")

# test with terrestrial "orbit" edge case. This case will results in
# dot(e_unit_vec, r_vec) / norm(r_vec) > 1, which will cause an error in arccos.
# safeArccos should handle it given a tolerance. Expect ta=0.0.
v_vec = array([-0.32788384, 0.32788384, 0])
r_vec = array([4510.094, 4510.094, 0])
e_unit_vec = array([0.71, 0.704214, 0])
ta = getTrueAnomaly(r_vec, v_vec, e_unit_vec)
print(f"{ta=}")

# %% Test getRadialRate
print("\nTest getRadialRate...")
r_vec = array([RE + 400, 0, 0])
v_vec = array([1, 8, 0])

rdot = getRadialRate(r_vec, v_vec)
print(f"{rdot=}")

# test with circular orbit (rdot=0)
v = getCircOrbitVel(r=norm(r_vec))
v_vec = array([0, v, 0])
rdot = getRadialRate(r_vec, v_vec)
print(f"{rdot=}")

# %% Done
print("done")
