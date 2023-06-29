"""Environment utilities."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from numpy import array, asarray, ndarray

# Punch Clock Imports
from scheduler_testbed.common.agents import Agent, Sensor, Target
from scheduler_testbed.common.constants import getConstants
from scheduler_testbed.common.utilities import allEqual, calcVisMap

# %% Constants
RE = getConstants()["earth_radius"]


# %% GetVisMapEstOrTruth
def getVisMapEstOrTruth(
    list_of_agents: list[Agent],
    truth_flag: bool,
) -> ndarray[int]:
    """Get a visibility map from either estimated or truth states.

    Args:
        list_of_agents (`list[Agent]`): List of agents.
        estimate_flag (`bool`): If True, calculate truth visibility map. If False,
            calculate estimated visibility map.

    Returns:
        `ndarray[int]`: (N, M) True visibility map.
    """
    # Make sure states being used for vis map calculation are all on same
    # time step.
    sensor_times = [a.time for a in list_of_agents if isinstance(a, Sensor)]
    if truth_flag is True:
        target_times = [a.time for a in list_of_agents if isinstance(a, Target)]
    else:
        target_times = [
            a.filter.time for a in list_of_agents if isinstance(a, Target)
        ]
    assert allEqual(
        sensor_times + target_times
    ), "Sensor and target times are not equal. If truth_flag == False, target \
        filter times are not equal to sensor times."

    RE = getConstants()["earth_radius"]

    # arrange states into (6, X) arrays
    # Use true states of sensors
    sensor_states = (
        asarray(
            [
                agent.eci_state
                for agent in list_of_agents
                if type(agent) is Sensor
            ]
        )
        .squeeze()
        .T
    )
    # Use true or estimated target states depending on truth_flag
    if truth_flag is True:
        target_states = (
            asarray(
                [
                    agent.eci_state
                    for agent in list_of_agents
                    if type(agent) is Target
                ]
            )
            .squeeze()
            .T
        )
    elif truth_flag is False:
        target_states = (
            asarray(
                [
                    agent.filter.est_x
                    for agent in list_of_agents
                    if type(agent) is Target
                ]
            )
            .squeeze()
            .T
        )

    # calculate visibility map
    vis_map = calcVisMap(
        sensor_states=sensor_states,
        target_states=target_states,
        body_radius=RE,
    )

    return vis_map


# %% forecastVisMap
def forecastVisMap(
    agents: list[Agent],
    time_step: float,
    estimate: bool = False,
) -> ndarray:
    """Forecast a visibility map at a future time.

    Args:
        agents (`list[Agent]`): List of agents.
        time_step (`float`): The time difference forward at which to forecast
            (from current agent time).
        estimate (`bool`, optional): If True, calculates vis_map based on targets'
            estimated states rather than true states. Defaults to False.

    Returns:
        `ndarray`: Binary vis map. See calcVisMap for details.
    """
    # Check that times are identical for all agents.
    times = [ag.time for ag in agents]
    assert allEqual(times)

    # Get 1-step ahead states for agents (true for sensors, est for targets)
    agents = deepcopy(agents)
    x_sensors = []
    x_targets = []
    for ag in agents:
        time_now = ag.time
        if isinstance(ag, Target) and (estimate is True):
            # Get estimates states for targets (if estimate is True)
            ag.filter.predict(time_now + time_step)
            x_targets.append(ag.filter.pred_x)
        elif isinstance(ag, Target) and (estimate is False):
            # Get true states for targets (if estimate is False)
            ag.propagate(time_now + time_step)
            x_targets.append(ag.eci_state)
        elif isinstance(ag, Sensor):
            # Get true states for sensors
            ag.propagate(time_now + time_step)
            x_sensors.append(ag.eci_state)

    # Convert to arrays for interface with calcVisMap
    x_sensors = array(x_sensors).T
    x_targets = array(x_targets).T

    vis_map = calcVisMap(
        sensor_states=x_sensors,
        target_states=x_targets,
        body_radius=RE,
    )

    return vis_map
