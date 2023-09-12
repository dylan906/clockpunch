"""Test access_windows.py."""
# Punch Clock Imports
from punchclock.common.agents import Sensor, Target, buildRandomAgent
from punchclock.schedule_tree.access_windows import calcAccessWindows

# %% Test calcAccessWindows
print("\nTest calcAccessWindows...")
num_sensors = 2
num_targets = 2

list_of_sensors = [
    buildRandomAgent(target_sensor="sensor") for i in range(num_sensors)
]
list_of_targets = [
    buildRandomAgent(target_sensor="target") for i in range(num_targets)
]

# Number of possible windows is number of steps in sim (env.horizon)
# All starting num_windows for targets should be <=horizon
num_windows = calcAccessWindows(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
)
print(f"num_windows = {num_windows}")

# Include sensor-overlap in window count.
# Starting num_windows can be >horizon. In this example, one satellite (#4) is at GEO, so
# can be accessed by both sensors through all time steps.
num_windows = calcAccessWindows(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
    horizon=10,
    merge_windows=False,
)
print(f"num_windows = {num_windows}")
