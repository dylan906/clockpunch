"""Test access_windows.py."""
# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target, buildRandomAgent
from punchclock.schedule_tree.access_windows import AccessWindowCalculator


# %% Util func
def resetAgents(list_of_agents: list[Agent]):
    for ag in list_of_agents:
        ag.time = 0
    return list_of_agents


# %% Test AccessWindowCalculator
print("\nTest AccessWindowCalculator...")
num_sensors = 2
num_targets = 2

list_of_sensors = [
    buildRandomAgent(target_sensor="sensor") for i in range(num_sensors)
]
list_of_targets = [
    buildRandomAgent(target_sensor="target") for i in range(num_targets)
]

list_of_sensors = resetAgents(list_of_sensors)
list_of_targets = resetAgents(list_of_targets)
# Number of possible windows is number of steps in sim (env.horizon)
# All starting num_windows for targets should be <=horizon
awc = AccessWindowCalculator(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
    horizon=1,
)
num_windows = awc.calcAccessWindows()
print(f"num_windows = {num_windows}")

# Include sensor-overlap in window count.
# Starting num_windows can be >horizon. In this example, one satellite (#4) is at GEO, so
# can be accessed by both sensors through all time steps.
list_of_sensors = resetAgents(list_of_sensors)
list_of_targets = resetAgents(list_of_targets)

num_windows = AccessWindowCalculator(
    list_of_sensors=list_of_sensors,
    list_of_targets=list_of_targets,
    horizon=10,
    dt_eval=200,
    dt_propagate=200,
    merge_windows=False,
)
num_windows = awc.calcAccessWindows()
print(f"num_windows = {num_windows}")

# %% Done
print("done")
