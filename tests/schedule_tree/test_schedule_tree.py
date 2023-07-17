"""Test for schedule_tree module."""
# %% Imports

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import matplotlib.pyplot as plt
from numpy import arange, array, eye, ndarray

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter
from punchclock.schedule_tree.schedule_tree import ScheduleTree

# %% Build test data for calcIntervalTree
print("\nTest calcIntervalTree()...")

print(" Build test parameters...")


# measurement model
def dummy_measurement_model(state: ndarray):
    """Dummy function."""
    return state


# target_filter params
est_x_init = array([[7000, 0, 0, 0, 1, 0]])
est_p_init = eye(6)
q = 0.1 * eye(6)
r = 0.01 * eye(6)
# Dummy Filter
dummy_filter = UnscentedKalmanFilter(
    time=0,
    est_x=est_x_init,
    est_p=est_p_init,
    dynamics_model=SatDynamicsModel(),
    measurement_model=dummy_measurement_model,
    q_matrix=q,
    r_matrix=r,
)


# agents
list_of_agents = [
    Target(
        SatDynamicsModel(),
        agent_id=1,
        init_eci_state=array([[7000, 0, 0, 0, 8, 0]]).transpose(),
        target_filter=deepcopy(dummy_filter),
        time=1,
    ),
    Sensor(
        SatDynamicsModel(),
        agent_id="A",
        init_eci_state=array([[8000, 0, 0, 0, 8, 0]]).transpose(),
        time=1,
    ),
    Sensor(
        StaticTerrestrial(),
        agent_id="B",
        init_eci_state=array([[7000, 0, 0, 0, 0, 0]]).transpose(),
        time=1,
    ),
    Target(
        SatDynamicsModel(),
        agent_id=2,
        init_eci_state=array([[7000, 1000, 0, 0, -8, 0]]).transpose(),
        target_filter=deepcopy(dummy_filter),
        time=1,
    ),
    Target(
        SatDynamicsModel(),
        agent_id=3,
        init_eci_state=array([[7500, 0, 0, 0, 8, 0]]).transpose(),
        target_filter=deepcopy(dummy_filter),
        time=1,
    ),
]


# %% Test calcIntervalTree
print(" Test function...")

# initial time (s)
t0 = 0
# final time (s)
tf = 40 * 60 * 3
# number of steps in simulation
numStep = 100
# time step
dt = tf / numStep

# time vector
t = arange(t0, tf, dt)

sched_tree = ScheduleTree(list_of_agents, t)

print("\nsched_tree attributes:")
print(f"len(sched_tree.sched_tree) = {len(sched_tree.sched_tree)}")
print(f"sched_tree.num_sats = {sched_tree.num_targets}")
print(f"sched_tree.num_sensors = {sched_tree.num_sensors}")
print(f"sched_tree.sat_list = {sched_tree.target_list}")
print(f"sched_tree.sensor_list = {sched_tree.sensor_list}")

# Plot visibility and state history
plt.style.use("default")
fig, ax = plt.subplots(3)
for i_sens in range(sched_tree.num_sensors):
    for i_targ in range(sched_tree.num_targets):
        text = str(sched_tree.sensor_list[i_sens]) + str(
            sched_tree.target_list[i_targ]
        )
        ax[0].plot(
            t, sched_tree.vis_array[:, i_targ, i_sens], marker=".", label=text
        )

ax[0].set_title("target-sensor pairs")
ax[0].set_ylabel("V")
ax[0].set_xlabel("t")
ax[0].legend()

for i in range(sched_tree.num_targets):
    text = str(sched_tree.target_list[i])
    ax[1].plot(
        sched_tree.x_targets[:, 0, i],
        sched_tree.x_targets[:, 1, i],
        marker=".",
        label=text,
    )
ax[1].set_title("targets")
ax[1].set_xlabel("I")
ax[1].set_ylabel("J")
ax[1].set_xlim(-12000, 12000)
ax[1].set_ylim(-12000, 12000)
ax[1].legend(["1", "2"])

for i in range(sched_tree.num_sensors):
    text = str(sched_tree.sensor_list[i])
    ax[2].plot(
        sched_tree.x_sensors[:, 0, i],
        sched_tree.x_sensors[:, 1, i],
        marker=".",
        label=text,
    )
ax[2].set_title("sensors")
ax[2].set_xlabel("I")
ax[2].set_ylabel("J")
ax[2].set_xlim(-12000, 12000)
ax[2].set_ylim(-12000, 12000)
ax[2].legend(["A", "B"])

plt.tight_layout()

# plot state history in components
fig2, ax2 = plt.subplots()
ax2.plot(t, sched_tree.x_targets[:, :3, 0])
ax2.set_title("position, agent 0")

# %% Test Schedule tree Class
print("\nTest ScheduleTree class...")

print("Print all Intervals")
for i in range(len(sched_tree.sched_tree)):
    print(f"  Interval[{i}] = {list(sched_tree.sched_tree)[i]}")
    print(f"  Sat Name[{i}] = {sched_tree.getTargAtInt(i)}")
    print(f"  Sens Name[{i}] = {sched_tree.getSensAtInt(i)}")

print("Query targets/sensors at given interval index")
print(f"  sched_tree.getTargAtInt(0) = {sched_tree.getTargAtInt(0)}")
print(f"  sched_tree.getTargAtInt(1) = {sched_tree.getTargAtInt(1)}")
print(f"  sched_tree.getSensAtInt(0) = {sched_tree.getSensAtInt(0)}")

print("Query all targets/sensors at given time")
print(f"  sched_tree.getTargs(100) = {sched_tree.getTargs(100)}")
print(f"  sched_tree.getSens(100) = {sched_tree.getSens(100)}")

print("Check visibility")
print(
    f"  isVis: {sched_tree.checkVisibility(time=0, sensor_id='A', target_id=1)}"
)
print(
    f"  isVis: {sched_tree.checkVisibility(time=2000, sensor_id='B', target_id=2)}"
)

# %%
plt.show()
print("done")
