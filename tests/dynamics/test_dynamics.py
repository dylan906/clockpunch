"""Tests for dynamics module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import copy

# Third Party Imports
from matplotlib import pyplot as plt
from numpy import append, array, linspace
from numpy.linalg import norm

# Punch Clock Imports
from scheduler_testbed.common.constants import getConstants
from scheduler_testbed.common.transforms import ecef2eci, eci2ecef
from scheduler_testbed.dynamics.dynamics import (
    a2body,
    satelliteDynamics,
    terrestrialDynamics,
)

# %% Get parameters
consts = getConstants()
RE = consts["earth_radius"]
mu = consts["mu"]

# %% Test a2body()
r0_eci = array([RE + 400, 0, 0])
print(f"acceleration (2-body) = {a2body(r0_eci, mu)}")
# %% Test satelliteDynamics()
t0 = 0
tf = 300
t = linspace(t0, tf, 10)

x0_eci = array([RE + 400, 0, 0, 0, 4, 0])
xdot = satelliteDynamics(t, x0_eci)
print(f"xdot = \n{xdot}\n")

# %% Test terrestrialDynamics()
print("\nTest terrestrial dynamics...")
print("\nPropagate at 0, 100, and 1000 sec")
t_terrestrial = [0, 100, 1000]
x_terrestrial = terrestrialDynamics(t_terrestrial, x0_eci, 0)
print(f"x_init (eci) = \n{x0_eci}")
print(f"x_final (eci) = \n{x_terrestrial}")

print("Propagate from 0-velocity (eci) initial condition")
x0_eci = array([RE, 0, 0, 0, 0, 0])
x1_eci = terrestrialDynamics(20, x0_eci, 0)
print(f"x_init (eci) = \n{x0_eci}")
print(f"x_final (eci) = \n{x1_eci}")

print("Propagate from surface of Earth, 0-vel ECEF")
x0_ecef = array([RE, 0, 0, 0, 0, 0])
x0_eci = ecef2eci(x0_ecef, 0)
x1_eci = terrestrialDynamics(2000, x0_eci, 0)
x1_ecef = eci2ecef(x1_eci, 0)
print(f"x0_ecef = \n{x0_ecef}")
print(f"x0_eci = \n{x0_eci}")
print(f"x1_eci = \n{x1_eci}")
print(f"x1_ecef = \n{x1_ecef}")


# Test in loop (ensure position magnitude is constant)
print("Test in loop")
x_now = x0_eci
x_hist = copy(x0_eci)
x_hist.shape = [6, 1]
for t in range(0, 10000, 100):
    x_now = terrestrialDynamics(t, x_now, 0)
    x_hist = append(x_hist, x_now, axis=1)
    # print(norm(x_now[:3, 0]))

fig, ax = plt.subplots()
ax.plot(norm(x_hist[:3, :], axis=0))
ax.set_ylabel("position magnitude")
# %%
plt.show()
print("done")
