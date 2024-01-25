"""Test for transforms module."""
# %% Imports
# Third Party Imports
import matplotlib.pyplot as plt
from numpy import array, pi
from numpy.random import rand

# Punch Clock Imports
from punchclock.common.transforms import (
    coe2eci,
    ecef2eci,
    ecef2eci_test,
    eci2ecef,
    lla2ecef,
    rot3,
)

# %% lat-lon-alt 2 ECEF converter
lla = array([3 / 4, 0.9, 10])
r_ecef = lla2ecef(lla)
print(f"lla={lla}")
print(f"r_ecef={r_ecef}")

# keys = ["lat", "lon", "alt"]
# vals = list(map(dat_ssn[0].get, keys))
# print(f"extracted lla={vals}")
# print("alternative method: lla=")
# print(dat_ssn[0].get("lat"), dat_ssn[0].get("lon"), dat_ssn[0].get("alt"))

# print(f"lla2ecef(extracted)= {lla2ecef(vals)}")

# %% ECEF 2 ECI Converter
print("\nTest ecef2eci()...")
print("  Test for correct values")
# Test for correct values
x_ecef = array([6000, 0, 0, 0, 0, 0])
# expected answer should be [6000, 0, 0, 0, +num, 0]
x_eci = ecef2eci(x_ecef, 0)
print(f"{x_ecef=}")
print(f"{x_eci=}")

for i in [1, 2, 3, 5]:
    assert x_eci[i] == 0, f"x_eci[{i}] should be zero {x_eci[i]=}"
assert x_eci[0] == 6000, f"x_eci[0] should be 6000 {x_eci[0]=}"
assert x_eci[4] > 0, f"x_eci[4] should be positive {x_eci[4]=}"

# expected answer should be [+num, +num, 0, -num, +num, 0]
x_eci = ecef2eci_test(x_ecef, 1)
print(f"{x_ecef=}")
print(f"{x_eci=}")

assert x_eci[0] > 0, f"x_eci[0] should be positive {x_eci[0]=}"
assert x_eci[1] > 0, f"x_eci[1] should be positive {x_eci[1]=}"
assert x_eci[2] == 0, f"x_eci[2] should be zero {x_eci[2]=}"
assert x_eci[3] < 0, f"x_eci[3] should be negative {x_eci[3]=}"
assert x_eci[4] > 0, f"x_eci[4] should be positive {x_eci[4]=}"
assert x_eci[5] == 0, f"x_eci[5] should be zero {x_eci[5]=}"

x_ecef = array([6000, 0, 1, 0, 0, 0])
# expceted answer should be [+num, +num, 1, -num, +num, 0]
x_eci = ecef2eci(x_ecef, 1)
print(f"{x_ecef=}")
print(f"{x_eci=}")

assert x_eci[0] > 0, f"x_eci[0] should be positive {x_eci[0]=}"
assert x_eci[1] > 0, f"x_eci[1] should be positive {x_eci[1]=}"
assert x_eci[2] == 1, f"x_eci[2] should be 1 {x_eci[2]=}"
assert x_eci[3] < 0, f"x_eci[3] should be negative {x_eci[3]=}"
assert x_eci[4] > 0, f"x_eci[4] should be positive {x_eci[4]=}"
assert x_eci[5] == 0, f"x_eci[5] should be zero {x_eci[5]=}"

x_ecef = array([-6000, 0, 0, 0, 0, 0])
# expceted answer should be [-num, -num, 0, +num, -num, 0]
x_eci = ecef2eci(x_ecef, 1)
print(f"{x_ecef=}")
print(f"{x_eci=}")

assert x_eci[0] < 0, f"x_eci[0] should be negative {x_eci[0]=}"
assert x_eci[1] < 0, f"x_eci[1] should be negative {x_eci[1]=}"
assert x_eci[2] == 0, f"x_eci[2] should be 0 {x_eci[2]=}"
assert x_eci[3] > 0, f"x_eci[3] should be positive {x_eci[3]=}"
assert x_eci[4] < 0, f"x_eci[4] should be negative {x_eci[4]=}"
assert x_eci[5] == 0, f"x_eci[5] should be zero {x_eci[5]=}"

# test with (6,1) input
print("  Test (6,1) input")
x_ecef = array([[6000, 2000, 1000, 1, 2, 3]]).transpose()
x_eci = ecef2eci(x_ecef, 10)
print(f"x_ecef={x_ecef}")
print(f"x_eci={x_eci}")
assert x_eci.shape == (6,)

# test with (1,6) input (will error)
print("  Test (1,6) input")
x_ecef = array([[6000, 2000, 1000, 1, 2, 3]])
print(f"x_ecef={x_ecef}")
try:
    x_eci = ecef2eci(x_ecef, 10)
    print(f"x_eci={x_eci}")
    print("Test failed")
except Exception as e:
    print(e)
    print("Test passed")

# test with 2D input
print("  2D input")
x_ecef = rand(6, 3)
x_eci = ecef2eci(x_ecef, 10)
print(f"x_ecef=\n{x_ecef}")
print(f"x_eci=\n{x_eci}")
assert x_eci.shape == (6, 3)

# test with 2D input, swapped dimensions (will error)
print("  2D input, swapped dimensions")
x_ecef = rand(3, 6)
print(f"x_ecef=\n{x_ecef}")
try:
    x_eci = ecef2eci(x_ecef, 10)
    print(f"x_eci=\n{x_eci}")
    print("Test failed")
except Exception as e:
    print(e)
    print("Test passed")

# %% eci2ecef
print("\nTest eci2ecef()...")
# test with 1D input
# Two-way test
print("  Convert ECI->ECEF->ECI")
x_eci = array([6000, 6000, 6000, 1, 2, 3])
print(f"x_eci={x_eci}")
print(f"x_ecef={eci2ecef(x_eci, 100)}")
print(f"x_eci reconvert={ecef2eci(eci2ecef(x_eci, 100), 100)}")

# Two-way test
print("  Convert ECEF->ECI->ECEF")
x_ecef = array([6000, 6000, 6000, 1, 2, 3])
print(f"x_ecef={x_ecef}")
print(f"x_eci={ecef2eci(x_ecef, 200)}")
print(f"x_ecef reconvert={eci2ecef(ecef2eci(x_ecef, 200), 200)}")

# test with (1,6) input
print("  (1,6) input")
x_eci = array([[6000, 2000, 1000, 1, 2, 3]])
print(f"x_eci={x_eci}")
try:
    x_ecef = eci2ecef(x_eci, 5000)
    print(f"x_ecef={x_ecef}")
except Exception as e:
    print(e)


# test with 2D input
print("  2D input")
x_eci = rand(6, 3)
x_ecef = eci2ecef(x_eci, 5000)
print(f"x_eci=\n{x_eci}")
print(f"x_ecef=\n{x_ecef}")

# test with 2D input, swapped dimensions
print("  2D input, swapped dimensions")
x_eci = rand(3, 6)
print(f"x_eci=\n{x_eci}")
try:
    x_ecef = eci2ecef(x_eci, 5000)
    print(f"x_ecef=\n{x_ecef}")
except Exception as e:
    print(e)

# test with (6,1) input
print("  (6,1) input")
x_eci = array([[6000, 2000, 1000, 1, 2, 3]]).transpose()
print(f"x_eci={x_eci}")
x_ecef = eci2ecef(x_eci, 5000)
print(f"x_ecef={x_ecef}")

# %% Test coe2eci
print("\nTest coe2eci()...")
x_eci = coe2eci(
    sma=7000,
    ecc=0.1,
    inc=pi / 4,
    raan=0,
    argp=0,
    true_anom=pi,
)
print(f"x_eci = {x_eci}")

# Try with out of range inputs
try:
    x_eci = coe2eci(
        sma=0,
        ecc=-0.1,
        inc=-pi / 4,
        raan=3 * pi,
        argp=-1,
        true_anom=3 * pi,
    )
except ValueError as e:
    print(e)

# %% Done
plt.show()
print("done")
