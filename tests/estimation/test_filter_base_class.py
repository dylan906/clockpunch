"""Test for filter base class module."""

# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import array, eye, ndarray, random

# Punch Clock Imports
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.filter_base_class import Filter
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter


# %% Define Test Parameters
def dummyCallable(a):
    return random.rand(2, 1)


# Define 2 classes, one of which is missing a required base class method
class dummyClass(Filter):
    def __init__(self, time, x_est, x_cov):
        super().__init__(time, x_est, x_cov)

    def reset():
        pass


class dummyClassError(Filter):
    def __init__(self, time, x_est, x_cov):
        super().__init__(time, x_est, x_cov)

    # def reset():
    #     pass


# %% Simple tests
try:
    a = dummyClass(4, 2, 3)
    print("Class instantiated correctly")
except Exception:
    print("Class instantiation error")
    pass

try:
    b = dummyClassError(4, 2, 3)
    print("Class instantiated correctly")
except Exception:
    print("Class instantiation error")
    pass

print(f"a.time = {a.time}")


# %% Test with real dynamics and UKF
ground_dynamics = StaticTerrestrial()


def passThruMeasurement(state: ndarray):
    """Fully observable measurement function."""
    meas = state
    return meas


q = eye(6)
r = eye(6)

my_filter = UnscentedKalmanFilter(
    time=0,
    est_x=array([6800, 0, 0, 0, 0, 0]),
    est_p=eye(6),
    dynamics_model=ground_dynamics,
    measurement_model=passThruMeasurement,
    q_matrix=q,
    r_matrix=r,
)
print(f"my_filter.est_x = \n{my_filter.est_x}")

print("done")
