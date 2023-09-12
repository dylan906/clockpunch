"""Test for agents module."""
# %% Imports

# Standard Library Imports
import json
from copy import deepcopy

# Third Party Imports
import numpy as np
from matplotlib import pyplot as plt
from numpy import array, asarray, eye, linspace, sqrt, zeros

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter

# %% Build Filter
print("Build target_filter...")
dummy_dynamics = SatDynamicsModel()
sat_dynamics = SatDynamicsModel()
ground_dynamics = StaticTerrestrial()


def dummyMeasurementDynamics(a):
    """Dummy dynamics."""
    return a.reshape([6, 1])


est_x_inital = array([[7000, 0, 0, 0, 2, 0]]).transpose()
est_p_initial = 1 * eye(6)
r = 0.1 * eye(6)
q = 0.01 * eye(6)
my_filter = UnscentedKalmanFilter(
    time=0,
    est_x=est_x_inital,
    est_p=est_p_initial,
    dynamics_model=dummy_dynamics,
    measurement_model=dummyMeasurementDynamics,
    q_matrix=q,
    r_matrix=r,
)


# %% Basic Tests
print("Instantiation tests...")

a = Agent(
    dynamics_model=sat_dynamics,
    init_eci_state=array([1, 2, 3, 4, 5, 6]),
    agent_id="A",
)
print(f"Satellite agent: vars(agent) = \n{vars(a)}\n")

c = Sensor(
    dynamics_model=ground_dynamics,
    init_eci_state=array([1, 2, 3, 4, 5, 6]),
    agent_id="A",
    time=5,
)
print(f"Ground agent: vars(ground sensor) = \n{vars(c)}\n")

c.propagate(10)
print(f"vars(ground sensor) after updating dynamics= \n{vars(c)}\n")

mu = getConstants()["mu"]
d = Agent(
    dynamics_model=sat_dynamics,
    init_eci_state=array([7000, 0, 0, 0, sqrt(mu / 7000), 0]),
    agent_id="derSat",
)
print(f"New satellite agent: vars(sat agent) = \n{vars(d)}\n")

d.propagate(10)
print(
    f"New satellite agent: vars(sat agent) after updating dynamics = \n{vars(d)}\n"
)

# %% Test get measurement
print("Test getMeasurement...")
e = Target(
    dynamics_model=sat_dynamics,
    init_eci_state=array([1, 2, 3, 4, 5, 6]),
    agent_id=2,
    target_filter=deepcopy(my_filter),
)
print("Test with noise")
print(f"    true state = {e.eci_state}")
print(f"    measured state = \n{e.getMeasurement()}")

# Test with no noise
no_noise_filter = UnscentedKalmanFilter(
    time=0,
    est_x=est_x_inital,
    est_p=est_p_initial,
    dynamics_model=dummy_dynamics,
    measurement_model=dummyMeasurementDynamics,
    q_matrix=q,
    r_matrix=zeros([6, 6]),
)
e1 = Target(
    dynamics_model=sat_dynamics,
    init_eci_state=array([1, 2, 3, 4, 5, 6]),
    agent_id=2,
    target_filter=no_noise_filter,
)
print("Test without noise")
print(f"    true state = {e1.eci_state}")
print(f"    measured state = \n{e1.getMeasurement()}")
print(
    f"    truth - measured = {asarray(e1.eci_state).squeeze() - e1.getMeasurement()}"
)
# %% Test Propagation over multiple iterations
print("\nTest Propagation over multiple iterations...")

x0_ecef = array([[7000, 0, 0, 0, 0, 0]]).transpose()
x0_eci = ecef2eci(x0_ecef, 0)
ag_ground = Sensor(
    dynamics_model=StaticTerrestrial(),
    agent_id="A",
    init_eci_state=x0_eci,
)
ag_space = Sensor(
    dynamics_model=SatDynamicsModel(),
    agent_id="B",
    init_eci_state=array([[8000, 1000, 0, 8, 0, 0]]).transpose(),
)

time_vec = linspace(5, 5000)
x_hist_ground = zeros([6, len(time_vec)])
x_hist_space = zeros([6, len(time_vec)])

for i, t in enumerate(time_vec):
    ag_ground.propagate(t)
    ag_space.propagate(t)
    x_hist_ground[:, i] = ag_ground.eci_state.squeeze()
    x_hist_space[:, i] = ag_space.eci_state.squeeze()


fig, ax = plt.subplots(2)
ax[0].plot(time_vec, x_hist_ground[:3, :].transpose())
ax[0].set_title("ground")
ax[1].plot(time_vec, x_hist_space[:3, :].transpose())
ax[1].set_title("space")
plt.tight_layout()
# %% Filter Tests
print("\nFilter tests...")
# Functionality test
print("  functionality test")
b = Target(
    dynamics_model=sat_dynamics,
    init_eci_state=array([7000, 0, 0, 0, 4, 0]),
    agent_id=2,
    target_filter=deepcopy(my_filter),
    time=0,
)
print(f"Target: vars(sat target) = \n{vars(b)}\n")
print(f"Target state: b.eci_state = \n{b.eci_state}")

# targets must be tasked after state propagation
print("  test propagation")
b.propagate(10)
print(f"Target state after propagating: b.eci_state = \n{b.eci_state}")

# test update nonphysical with tasking
print("  test updateNonPhyiscal with tasking")
b.updateNonPhysical(task=True)

print("\nrelevant vars(sat target) after nonphysical update with tasking:")
print(f"target.time = {b.time}")
print(f"target.num_tasked = {b.num_tasked}")
print(f"target.last_time_tasked = {b.last_time_tasked}")
print(f"target.target_filter.time = {b.target_filter.time}")
print(f"target.target_filter.est_x = \n{b.target_filter.est_x}")

# test update nonphysical without tasking
print("  test updateNonPhyiscal w/o tasking")
b.propagate(20)
b.updateNonPhysical(task=False)

print("\nrelevant vars(sat target) after  nonphysical update without tasking:")
print(f"target.time = {b.time}")
print(f"target.num_tasked = {b.num_tasked}")
print(f"target.last_time_tasked = {b.last_time_tasked}")
print(f"target.target_filter.time = {b.target_filter.time}")
print(f"target.target_filter.est_x = \n{b.target_filter.est_x}")

# test error catcher
print("  Test updateNonPhyiscal with bad input")
try:
    b.updateNonPhysical()
except TypeError as err:
    print(err)

# test second tasking
print("  Test 2nd tasking")
b.propagate(60)
b.updateNonPhysical(task=True)
print("\nrelevant vars(sat target) after updating dynamics and tasking again:")
print(f"target.time = {b.time}")
print(f"target.num_tasked = {b.num_tasked}")
print(f"target.last_time_tasked = {b.last_time_tasked}")
print(f"target.target_filter.time = {b.target_filter.time}")
print(f"target.eci_state = \n{b.eci_state}")
print(f"target.target_filter.est_x = \n{b.target_filter.est_x}")

# %% Test num_windows_left
print("\nTest num_window_left decrement...")

c = Target(
    dynamics_model=sat_dynamics,
    init_eci_state=array([7000, 0, 0, 0, 4, 0]),
    agent_id=2,
    target_filter=deepcopy(my_filter),
    num_windows_left=2,
)
print(f"num_windows_left = {c.num_windows_left}")

# num_windows_left should decrement until 0, then stay at 0
for t in [10, 20, 30]:
    c.propagate(t)
    c.updateNonPhysical(task=False)
    print(f"num_windows_left = {c.num_windows_left}")

# %% Test toJson
print("\ntoJson...")
# assign a attributes to be numpy dtypes to test conversion to python types
c.last_time_tasked = np.float32(2.2)
c.num_tasked = np.int64(10)
json_dict = c.toDict()

# check to make sure json can be dumped
json_object = json.dumps(json_dict)
# %%
plt.show()
print("done")
