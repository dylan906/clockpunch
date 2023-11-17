"""Test access_windows.py."""
# %% Imports
# Third Party Imports
from numpy import array, asarray
from numpy.random import randint

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.schedule_tree.access_windows import AccessWindowCalculator

# %% Build dummy agents to get realistic states
print("\nBuilding dummy agents...")
num_sensors = 2
num_targets = 3

list_of_sensors = [buildRandomAgent(agent_type="sensor") for i in range(num_sensors)]
list_of_targets = [buildRandomAgent(agent_type="target") for i in range(num_targets)]

x_sensors = asarray([a.eci_state for a in list_of_sensors]).squeeze().T
x_targets = asarray([a.eci_state for a in list_of_targets]).squeeze().T

# %% Test _buildAgents
print("\nTest _buildAgents...")
agents = AccessWindowCalculator._buildAgents(
    None,
    x_sensors,
    time=0,
    dynamics="terrestrial",
)

print(f"agents = {agents}")

# %% Test AccessWindowCalculator instantiation
print("\nTest AccessWindowCalculator instantiation...")

print("  Test with default args")
# Number of possible windows is number of steps in sim (env.horizon)
awc = AccessWindowCalculator(
    num_sensors=num_sensors,
    num_targets=num_targets,
    dynamics_sensors="terrestrial",
    dynamics_targets="satellite",
)

# %% Test setup()
awc._setup(x_sensors=x_sensors, x_targets=x_targets, t=0)
print(f"t_now = {awc.t_now}")
print(f"time vec = {awc.time_vec}")

# %% Test _getStates
print("\nTest _getStates...")
x = awc._getStates()
print(f"x = {x}")

# %% Test _getVis
print("\nTest _getVis...")
vis = awc._getVis(x_sensors=x_sensors, x_targets=x_targets)
print(f"vis status = \n{vis}")

# %% Test calcVisHist
print("\nTest calcVisHist...")
vis_hist = awc.calcVisHist(x_sensors=x_sensors, x_targets=x_targets, t=0)
print(f"vis hist = \n{vis_hist}")

# %% Test _mergeWindows
print("\nTest _mergeWindows...")
awc = AccessWindowCalculator(
    num_sensors=num_sensors,
    num_targets=num_targets,
    dynamics_sensors="terrestrial",
    dynamics_targets="satellite",
    horizon=2,
)

dummy_vis_hist = randint(0, 2, size=(awc.horizon, num_targets, num_sensors))
# Test a full row and a full col of 0s
dummy_vis_hist[0, 0, :] = 0
dummy_vis_hist[1, :, 0] = 0
merged = awc._mergeWindows(vis_hist=dummy_vis_hist)
print(f"raw vis hist = \n{dummy_vis_hist}")
print(f"merged vis hist = \n{merged}")

# %% Test calcStepsToWindow
print("\nTest calcStepsToWindow...")

steps_to_window = awc.calcStepsToWindow(A=array([1, 0, 0, 1, 0, 1, 0, 0, 0]))
print(f"{steps_to_window=}")

# %% Test calcTimeToWindowHist
print("\nTest calcTimeToWindowHist...")
vis_hist_targets = randint(0, 2, size=(4, 2))
time_to_window_hist = awc._calcTimeToWindowHist(
    t_now=0, vis_hist_targets=vis_hist_targets
)
print(f"{time_to_window_hist=}")

# %% Test calcNumWindows
print("\nTest calcNumWindows...")
num_windows, vis_hist, vis_hist_sensors, time, time_to_next_window = awc.calcNumWindows(
    x_sensors=x_sensors,
    x_targets=x_targets,
    t=0,
    return_vis_hist=True,
)
print(f"{num_windows=}")
print(f"{vis_hist=}")
print(f"{vis_hist_sensors=}")
print(f"{time=}")
print(f"{time_to_next_window=}")

# %% Test with non-default args and merge_windows=False
print("\nTest with merge_windows=False and non-default args")
awc = AccessWindowCalculator(
    num_sensors=num_sensors,
    num_targets=num_targets,
    dynamics_sensors="terrestrial",
    dynamics_targets="satellite",
    horizon=50,
    dt=13,
    merge_windows=False,
)

(
    num_windows,
    vis_hist,
    vis_hist_targets,
    time_hist,
    time_to_next_window,
) = awc.calcNumWindows(
    x_sensors=x_sensors,
    x_targets=x_targets,
    t=321.2,
    return_vis_hist=True,
)
print(f"num_windows = {num_windows}")

# %% Test with non-default args, merge_windows=True
print("\nTest with merge_windows=True and non-default args")

awc.merge_windows = True
(
    num_windows,
    vis_hist,
    vis_hist_targets,
    time_hist,
    time_to_next_window,
) = awc.calcNumWindows(
    x_sensors=x_sensors,
    x_targets=x_targets,
    t=321.2,
    return_vis_hist=True,
)
print(f"num_windows = {num_windows}")

# %% Test with DynamicsModel input
print("\nTest with DynamicsModel inputs")

awc = AccessWindowCalculator(
    num_sensors=num_sensors,
    num_targets=num_targets,
    dynamics_sensors=StaticTerrestrial(),
    dynamics_targets=SatDynamicsModel(),
)
num_windows = awc.calcNumWindows(
    x_sensors=x_sensors,
    x_targets=x_targets,
    t=2.1,
)
print(f"num_windows = {num_windows}")

# %% Done
print("done")
