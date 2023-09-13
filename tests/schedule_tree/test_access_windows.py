"""Test access_windows.py."""
# Third Party Imports
from numpy import asarray

# Punch Clock Imports
from punchclock.common.agents import Agent, buildRandomAgent
from punchclock.schedule_tree.access_windows import AccessWindowCalculator


# %% Util func
def resetAgents(list_of_agents: list[Agent]):
    """Reset agents."""
    for ag in list_of_agents:
        ag.time = 0
    return list_of_agents


# %% Build dummy agents to get realistic states
print("\nBuilding dummy agents...")
num_sensors = 2
num_targets = 3

list_of_sensors = [
    buildRandomAgent(target_sensor="sensor") for i in range(num_sensors)
]
list_of_targets = [
    buildRandomAgent(target_sensor="target") for i in range(num_targets)
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
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
)
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"vis hist = \n{vis_hist}")
print(f"num_windows = {num_windows}")

# Include sensor-overlap in window count.
print("  Test with merge_windows=False and non-default args")
# Starting num_windows can be >horizon. In this example, one satellite (#4) is at GEO, so
# can be accessed by both sensors through all time steps.
list_of_sensors = resetAgents(list_of_sensors)
list_of_targets = resetAgents(list_of_targets)

awc = AccessWindowCalculator(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
    horizon=100,
    dt=200,
    merge_windows=False,
    truth_or_estimated="estimated",
)
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"num_windows = {num_windows}")

print("  Test with merge_windows=True and non-default args")
awc.merge_windows = True
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"num_windows = {num_windows}")

# Test with non-zero agent start times
list_of_sensors = [
    buildRandomAgent(target_sensor="sensor", time=100)
    for i in range(num_sensors)
]
list_of_targets = [
    buildRandomAgent(target_sensor="target", time=100)
    for i in range(num_targets)
]
awc = AccessWindowCalculator(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
)
vis_hist = awc.calcVisHist()
num_windows = awc.calcNumWindows()
print(f"vis hist = \n{vis_hist}")
print(f"num_windows = {num_windows}")
# %% Done
print("done")
