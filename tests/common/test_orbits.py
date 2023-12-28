"""Tests for orbits.py."""
# %% Imports
# Third Party Imports
from numpy import array
from numpy.linalg import norm

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.orbits import getCircOrbitVel, getRadialRate

RE = getConstants()["earth_radius"]
MU = getConstants()["mu"]

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
