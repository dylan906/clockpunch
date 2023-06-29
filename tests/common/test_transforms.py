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
# test with 1D input
print("  1D input")
x_ecef = array([6000, 2000, 1000, 1, 2, 3])
x_eci = ecef2eci(x_ecef, 5000)
print(f"x_ecef={x_ecef}")
print(f"x_eci={x_eci}")

# test with (1,6) input
print("  (1,6) input")
x_ecef = array([[6000, 2000, 1000, 1, 2, 3]])
print(f"x_ecef={x_ecef}")
try:
    x_eci = ecef2eci(x_ecef, 5000)
    print(f"x_eci={x_eci}")
except Exception as e:
    print(e)


# test with 2D input
print("  2D input")
x_ecef = rand(6, 3)
x_eci = ecef2eci(x_ecef, 5000)
print(f"x_ecef=\n{x_ecef}")
print(f"x_eci=\n{x_eci}")

# test with 2D input, swapped dimensions
print("  2D input, swapped dimensions")
x_ecef = rand(3, 6)
print(f"x_ecef=\n{x_ecef}")
try:
    x_eci = ecef2eci(x_ecef, 5000)
    print(f"x_eci=\n{x_eci}")
except Exception as e:
    print(e)

# test with (6,1) input
print("  (6,1) input")
x_ecef = array([[6000, 2000, 1000, 1, 2, 3]]).transpose()
print(f"x_ecef={x_ecef}")
x_eci = ecef2eci(x_ecef, 5000)
print(f"x_eci={x_eci}")


plt.style.use("default")
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x_eci[0], x_eci[1], marker=".", linestyle="none")
axs[0, 0].plot(x_ecef[0], x_ecef[1], marker="x", linestyle="none")

axs[0, 1].plot(x_eci[0], x_eci[2], marker=".", linestyle="none")
axs[0, 1].plot(x_ecef[0], x_ecef[2], marker="x", linestyle="none")

axs[1, 0].plot(x_eci[3], x_eci[4], marker=".", linestyle="none")
axs[1, 0].plot(x_ecef[3], x_ecef[4], marker="x", linestyle="none")

axs[1, 1].plot(x_eci[3], x_eci[5], marker=".", linestyle="none")
axs[1, 1].plot(x_ecef[3], x_ecef[5], marker="x", linestyle="none")

axs[0, 0].legend(["ECI", "ECEF"])
axs[0, 0].set_title("position")
axs[0, 0].set_xlabel("I")
axs[0, 0].set_ylabel("J")
axs[0, 1].set_title("position")
axs[0, 1].set_ylabel("K")
axs[0, 1].set_xlabel("I")
axs[1, 0].set_title("velocity")
axs[1, 0].set_xlabel("I")
axs[1, 0].set_ylabel("J")
axs[1, 1].set_title("velocity")
axs[1, 1].set_ylabel("K")
axs[1, 1].set_xlabel("I")
plt.tight_layout()

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
print("done")
