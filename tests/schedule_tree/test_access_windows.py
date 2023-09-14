"""Test access_windows.py."""
# %% Imports
# Third Party Imports
from numpy import asarray
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

list_of_sensors = [
    buildRandomAgent(agent_type="sensor") for i in range(num_sensors)
]
list_of_targets = [
    buildRandomAgent(agent_type="target") for i in range(num_targets)
]

x_sensors = asarray([a.eci_state for a in list_of_sensors]).squeeze().T
x_targets = asarray([a.eci_state for a in list_of_targets]).squeeze().T

# %% Test _buildAgents
print("\nTest _buildAgents...")
agents = AccessWindowCalculator._buildAgents(
    None, x_sensors, time=0, dynamics="terrestrial"
)

print(f"agents = {agents}")

# %% Test AccessWindowCalculator instantiation
print("\nTest AccessWindowCalculator instantiation...")

print("  Test with default args")
# Number of possible windows is number of steps in sim (env.horizon)
awc = AccessWindowCalculator(
    x_sensors=x_sensors,
    x_targets=x_targets,
    dynamics_sensors="terrestrial",
    dynamics_targets="satellite",
)

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
vis_hist = awc.calcVisHist()
print(f"vis hist = \n{vis_hist}")

# %% Test _mergeWindows
print("\nTest _mergeWindows...")
dummy_vis_hist = randint(
    0, 2, size=(len(awc.time_vec), num_targets, num_sensors)
)
# Test a full row and a full col of 0s
dummy_vis_hist[0, 0, :] = 0
dummy_vis_hist[1, :, 0] = 0
merged = awc._mergeWindows(vis_hist=dummy_vis_hist)
print(f"raw vis hist = \n{dummy_vis_hist}")
print(f"merged vis hist = \n{merged}")

# %% Test calcNumWindows
print("\nTest calcNumWindows...")
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"vis hist = \n{vis_hist}")
print(f"num_windows = {num_windows}")

# %% Test with non-default args and merge_windows=False
print("\nTest with merge_windows=False and non-default args")
awc = AccessWindowCalculator(
    x_sensors=x_sensors,
    x_targets=x_targets,
    dynamics_sensors="terrestrial",
    dynamics_targets="satellite",
    t_start=321,
    horizon=50,
    dt=13,
    merge_windows=False,
)

vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"num_windows = {num_windows}")

# %% Test with non-default args, merge_windows=True
print("\nTest with merge_windows=True and non-default args")

awc.merge_windows = True
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"num_windows = {num_windows}")

# %% Test with DynamicsModel input
print("\nTest with DynamicsModel inputs")

awc = AccessWindowCalculator(
    x_sensors=x_sensors,
    x_targets=x_targets,
    dynamics_sensors=StaticTerrestrial(),
    dynamics_targets=SatDynamicsModel(),
)
num_windows = awc.calcNumWindows()
print(f"num_windows = {num_windows}")

# %% Done
print("done")
