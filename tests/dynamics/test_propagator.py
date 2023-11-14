"""Test for propagator module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import array

# Punch Clock Imports
from punchclock.dynamics.propagator import simplePropagate

# %% Test simplePropagate
print("\nTest simple functionality...")


def dummyFunc(t, x0):  # noqa
    xdot = array([1, 1, -0.1])
    return xdot


x1 = simplePropagate(dummyFunc, array([0, 0, 0]), 0, 5)
print(f"propagated state=\n{x1}")

# %% Test with t0=tf
print("\nTest t0=tf...")
try:
    x1 = simplePropagate(dummyFunc, array([0, 0, 0]), 0, 0)
except Exception:
    print("An exception occurred")

# %% Test with IVP failure
print("\nTest if IVP fails...")


def dummyFunc2(t, x0):  # noqa
    xdot = [0, 0]
    if x0[0] < 0:
        xdot[0] = 1e15
    else:
        xdot[0] = -1e15

    xdot[1] = 1
    return xdot


try:
    x1 = simplePropagate(dummyFunc2, array([1e15, 1e15]), 0, 5)
    print(f"propagated state=\n{x1}")
except Exception:
    print("An exception occurred")

# %% Test with 2D initial conditions
print("\nTest with 2D initial conditions...")

x0 = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
x1 = simplePropagate(dummyFunc, x0, 0, 5)
print(f"propagated state=\n{x1}")


# %% Test with 2D but singleton dimension ICs
print("\nTest with 2D initial conditions where one is a singleton...")

x0 = array([[1], [1], [1]])
x1 = simplePropagate(dummyFunc, x0, 0, 5)
print(f"propagated state=\n{x1}")


# %%
print("done")
