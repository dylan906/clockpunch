"""Environment utilities."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from numpy import array, asarray, ndarray

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.common.constants import getConstants
from punchclock.common.utilities import allEqual
from punchclock.common.visibility import calcVisMap

# %% Constants
RE = getConstants()["earth_radius"]


# %% GetVisMapEstOrTruth
def getVisMapEstOrTruth(
    list_of_agents: list[Agent],
    truth_flag: bool,
    binary: bool = True,
) -> ndarray[int]:
    """Generate a visibility map using either estimated or true states.

    This function calculates a visibility map based on the states of a list of
    agents. The states can be either the true states or the estimated states,
    depending on the `truth_flag`. The true states of the sensors are always
    used. The visibility map is a binary or continuous matrix where each entry
    indicates whether a sensor can see a target.

    Args:
        list_of_agents (list[Agent]): A list of agents, which can be either
            sensors or targets.
        truth_flag (bool): A flag to determine whether to use target true or
            estimated states for the visibility map calculation. If True, the
            true states are used. If False, the estimated states are used.
        binary (bool, optional): A flag to determine whether the visibility map
            should be binary. If True, the visibility map is binary, where a
            value of 1 indicates the sensor-target pair can see each other. If
            False, the visibility is a float where a value >0 indicates the
            sensor-target pair can see each other. Defaults to True.

    Returns:
        ndarray[int]: A visibility map as a 2D array. Each entry in the array
            indicates whether a sensor can see a target or not. If binary is
            True, (1) indicates the sensor can see the target, and (0) indicates
            the sensor cannot see the target. If binary is False, the value of
            the entry is the a float, where a value >0 indicates the sensor can
            see the target, and a value <=0 indicates the sensor cannot see the
            target.

    Raises:
        AssertionError: If the times of sensors and targets (or target filters
            if `truth_flag` is False) are not equal.
    """
    # Make sure states being used for vis map calculation are all on same
    # time step.
    sensor_times = [a.time for a in list_of_agents if isinstance(a, Sensor)]
    if truth_flag is True:
        target_times = [a.time for a in list_of_agents if isinstance(a, Target)]
    else:
        target_times = [
            a.target_filter.time for a in list_of_agents if isinstance(a, Target)
        ]
    assert allEqual(
        sensor_times + target_times
    ), "Sensor and target times are not equal. If truth_flag == False, target \
        target_filter times are not equal to sensor times."

    RE = getConstants()["earth_radius"]

    # arrange states into (6, X) arrays
    # Use true states of sensors
    sensor_states = _getTrueStates([a for a in list_of_agents if isinstance(a, Sensor)])

    # Use true or estimated target states depending on truth_flag
    if truth_flag is True:
        target_states = _getTrueStates(
            [a for a in list_of_agents if isinstance(a, Target)]
        )
    elif truth_flag is False:
        target_states = _getEstimatedStates(
            [a for a in list_of_agents if isinstance(a, Target)]
        )

    # calculate visibility map
    vis_map = calcVisMap(
        sensor_states=sensor_states,
        target_states=target_states,
        body_radius=RE,
        binary=binary,
    )

    return vis_map


def _getTrueStates(list_of_agents: list[Agent]) -> ndarray:
    """Get the true states of the target agents.

    Args:
        list_of_agents (`list[Agent]`): List of agents.

    Returns:
        `ndarray`: True states of the target agents.
    """
    target_states = asarray([agent.eci_state for agent in list_of_agents]).squeeze().T
    return target_states


def _getEstimatedStates(list_of_agents: list[Agent]) -> ndarray:
    """Get the estimated states of the target agents.

    Args:
        list_of_agents (`list[Agent]`): List of agents.

    Returns:
        `ndarray`: Estimated states of the target agents.
    """
    target_states = (
        asarray([agent.target_filter.est_x for agent in list_of_agents]).squeeze().T
    )
    return target_states


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
            ag.target_filter.predict(time_now + time_step)
            x_targets.append(ag.target_filter.pred_x)
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


def buildAgentInfoMap(agents: list[Agent]) -> list[dict]:
    """This function takes a list of Agent objects and returns a list of dictionaries.

    This function takes a list of Agent objects and returns a list of dictionaries.
    Each dictionary represents an agent and is structured like:
        {
            'agent_type': value of 'sensor' or 'target',
            'id': value of agent id
        }.

    Args:
        agents (list[Agent]): A list of Agent objects.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents an
            agent.
    """
    agent_map = []
    for i, agent in enumerate(agents):
        if isinstance(agent, Sensor):
            agent_map.append({"agent_type": "Sensor"})
        elif isinstance(agent, Target):
            agent_map.append({"agent_type": "Target"})
        agent_map[i]["id"] = agent.agent_id

    return agent_map
