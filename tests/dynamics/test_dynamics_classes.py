"""Tests for dynamics_classes module."""
# %% Imports
# Standard Library Imports
from typing import Callable

# Third Party Imports
from matplotlib import pyplot as plt
from numpy import array, linspace, ndarray, ones

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.dynamics.dynamics_classes import (
    DynamicsModel,
    SatDynamicsModel,
    StaticTerrestrial,
)

# %% Constants
RE = getConstants()["earth_radius"]


# %% Dummy Functions
def testFunc(
    x0: ndarray,
    t0: float,
    tf: float,
    params: list,
) -> ndarray:
    x1 = x0 + (tf - t0 + params[0]) * ones([x0.size, 1])
    return x1


def testFuncNoParams(
    x0: ndarray,
    t0: float,
    tf: float,
) -> ndarray:
    x1 = x0 + (tf - t0) * ones([x0.size, 1])
    return x1


def testFuncAlt(
    x0: ndarray,
    t0: float,
    tf: float,
    a: float,
    b: float,
) -> ndarray:
    x1 = x0 + (tf - t0 + a * b) * ones([x0.size, 1])
    return x1


# %% Dummy Classes
class TestDynamics(DynamicsModel):
    def __init__(
        self,
        dynamics_func: Callable[[ndarray, float, float], ndarray],
        params: list,
    ):
        super().__init__(dynamics_func)
        self.a = params[0]
        self.b = params[1]

    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> ndarray:
        return super().propagate(
            start_state,
            start_time,
            end_time,
            params=[self.a, self.b],
        )


class TestDynamicsNoParams(DynamicsModel):
    def __init__(
        self,
        dynamics_func: Callable[[ndarray, float, float], ndarray],
    ):
        super().__init__(dynamics_func)

    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
    ) -> ndarray:
        return super().propagate(
            start_state,
            start_time,
            end_time,
        )


class TestDynamicsMoreParams(DynamicsModel):
    def __init__(
        self,
        dynamics_func: Callable[[ndarray, float, float], ndarray],
        a: float,
        b: float,
    ):
        super().__init__(dynamics_func)
        self.a = a
        self.b = b

    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> ndarray:
        return super().propagate(start_state, start_time, end_time, [self.a, self.b])


# %% Test base class
a = TestDynamics(testFunc, [1, 2])
print(vars(a))
x0 = ones([6, 1])
x1 = a.propagate(x0, 0, 4)
print(x1)

b = TestDynamicsNoParams(testFuncNoParams)
print(vars(b))
x1 = b.propagate(x0, 0, 3)
print(x1)

# b = TestDynamicsMoreParams(testFuncAlt, 1, 2)
# print(vars(b))
# x1 = b.propagate(x0, 0, 4)
# print(x1)

# %% Test SatDynamicsModel
a = SatDynamicsModel()
print(f"vars(SatDynamics) = \n{vars(a)}")
x1 = a.propagate(array([8000, 0, 0, 0, 6, 0]), 0, 5)
print(f"propagated states (Satellite) = \n{x1}\n")

time = linspace(0, 10000)
state = array([7000, 0, 0, 0, 7.5, 0])
state_hist = []
for i, t in enumerate(time[:-1]):
    state = a.propagate(start_state=state, start_time=t, end_time=time[i + 1])
    state_hist.append(state.copy())

fig, axs = plt.subplots(2)
axs[0].set_title("Satellite Dynamics")
axs[0].set_ylabel("position")
axs[0].plot([x[:3] for x in state_hist])
axs[1].set_ylabel("velocity")
axs[1].plot([x[3:] for x in state_hist])


# %% Test StaticTerrestrial model
print("\nTest StaticTerrestrial...")
b = StaticTerrestrial()

# basic test
print("  basic test")
print(f"vars(StaticTerrestrial) = \n{vars(b)}")
x1 = b.propagate(array([7000, 0, 0, 0, 0, 0]), 500, 7000)
print(f"propagated states (StaticTerrestrial) = \n{x1}\n")

# 2D initial conditions
print("  Propagate w/ 2D initial conditions")
x_init = array(
    [
        [7000, 0, 0, 0, 0, 0],
        [0, 8000, 0, 0, 0, 0],
    ]
).transpose()
print(f"x_init = \n{x_init}")
x1 = b.propagate(x_init, 0, 400)
print(f"propagated states (StaticTerrestrial) = \n{x1}\n")

x_ecef = array([RE, 0, 0, 0, 0, 0])
x_eci = ecef2eci(x_ecef, 0)
state = x_eci

time = linspace(0, 1e4)
state_hist = []
for i, t in enumerate(time[:-1]):
    state = b.propagate(start_state=state, start_time=t, end_time=time[i + 1])
    state_hist.append(state.copy())

state_hist_arr = array(state_hist)
fig, axs = plt.subplots(2)
axs[0].set_title("Terrestrial Dynamics")
axs[0].set_ylabel("position")
axs[0].plot(state_hist_arr[:, :3], label=["I", "J", "K"])
axs[1].set_ylabel("velocity")
axs[1].plot(state_hist_arr[:, 3:], label=["dI", "dJ", "dK"])
axs[0].legend()
axs[1].legend()

state_hist_arr = array(state_hist)
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(state_hist_arr[:, 0], state_hist_arr[:, 1])
axs[1].plot(state_hist_arr[:, 0], state_hist_arr[:, 2])
axs[0].set_xlabel("I")
axs[0].set_ylabel("J")
axs[1].set_ylabel("K")
axs[0].set_xlim(-1.1 * RE, 1.1 * RE)
axs[0].set_ylim(-1.1 * RE, 1.1 * RE)
axs[1].set_xlim(-1.1 * RE, 1.1 * RE)
axs[1].set_ylim(-1.1 * RE, 1.1 * RE)
axs[0].plot(state_hist_arr[0, 0], state_hist_arr[0, 1], "*")
axs[1].plot(state_hist_arr[0, 0], state_hist_arr[0, 2], "*")

# %%
plt.show()
print("done")
