"""Schedule tree: module for defining ScheduleTree class and creating instances."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Any

# Third Party Imports
from intervaltree import IntervalTree
from numpy import asarray, moveaxis, ndarray, zeros
from satvis.vis_history import getVisHist

# Punch Clock Imports
from scheduler_testbed.common.agents import Agent, Sensor, Target
from scheduler_testbed.common.constants import getConstants


# %% Class Definition
class ScheduleTree:
    """Relates access windows between sensors and targets.

    Contains an `IntervalTree` of visibility windows between sensor-target pairs
    (`self.sched_tree`). Also contains some handy properties and useful functions
    for easily getting information from `self.sched_tree`.

    Notation:
        N = number of targets
        M = number of sensors
        T = length of time vector

    Attributes:
        num_sensors (`int`): Number of sensors.
        num_targets (`int`): Number of targets.
        sched_tree (`IntervalTree`): Structure that shows all time periods when
            sensors can see targets.
        sensor_list (`list`): List of sensor IDs
        target_list (`list`): List of target IDs
        vis_array (`ndarray`): [T x N x M] array of visibility function values
            where the n- and m-indices indicate the corresponding target-sensor
            pair.
        x_sensors (`ndarray`): [T x 6 x M] State history of sensors.
        x_targets (`ndarray`):[T x 6 x N] State history of targets.
    """

    def __init__(self, list_of_agents: list, time: list):
        """Initialize ScheduleTree() instance.

        Args:
            list_of_agents (`list[Agent]`): List of Agent instances.
            time (`list`): List of times (s) at which motion propagation will be
                evaluated. No need to use fine resolution, just fine enough so
                that a pass between two objects won't be skipped over.
        """
        [
            interval_tree,
            visibility_array,
            x_sensors,
            x_targets,
        ] = self.calcIntervalTree(list_of_agents, time)

        self.sched_tree = interval_tree
        self.vis_array = visibility_array
        self.x_sensors = x_sensors
        self.x_targets = x_targets

        self.target_list = [
            agent.id for agent in list_of_agents if type(agent) is Target
        ]
        self.sensor_list = [
            agent.id for agent in list_of_agents if type(agent) is Sensor
        ]

        self.num_targets = len(self.target_list)
        self.num_sensors = len(self.sensor_list)

    # %% Rise-Set Function
    def calcIntervalTree(
        self,
        list_of_agents: list[Agent],
        time: list,
    ) -> tuple[IntervalTree, ndarray, ndarray, ndarray]:
        """Generates target-sensor pair access windows.

        Notation:
            N = number of targets
            M = number of sensors
            T = length of time vector

        Args:
            list_of_agents (`list[Agent]`): List of Agent instances.
            time (`list`): List of times (s) at which motion propagation will be
                evaluated. No need to use fine resolution, just fine enough so
                that a pass between two objects won't be skipped over.

        Returns:
            rise_set_tree (`IntervalTree`): Intervals where sensors/targets can
                see each other.
            v (`ndarray`): [T x N x M] Values of v are the values of the visibility
                function evaluated at input `time`. Positive values mean the n-m'th
                target-sensor pair can see each other.
            x_sensors (`ndarray`): [T x 6 x M] State history of sensors.
            x_targets (`ndarray`): [T x 6 x N] State history of targets.
        """
        # Earth radius
        RE = getConstants()["earth_radius"]

        # get initial conditions of agents and arrange as [6 x (N+M)] matrix
        x0 = [agent.eci_state for agent in list_of_agents]
        x0 = asarray(x0).squeeze().transpose()

        # initialize state history
        x_hist = zeros([len(time), 6, x0.shape[1]])

        # set initial values of x_hist
        x_hist[0, :, :] = x0

        # loop through agents and propagate motion
        for agent_indx, agent in enumerate(list_of_agents):
            for i, t in enumerate(time[1:], start=1):
                agent.propagate(t)
                x_hist[i, :, agent_indx] = agent.eci_state.squeeze()

        # get indices of sensors and targets
        indx_targets = [
            i
            for i in range(len(list_of_agents))
            if type(list_of_agents[i]) is Target
        ]
        indx_sensors = [
            i
            for i in range(len(list_of_agents))
            if type(list_of_agents[i]) is Sensor
        ]

        # get state histories of sensors and targets
        x_targets = x_hist[:, :, indx_targets]
        x_sensors = x_hist[:, :, indx_sensors]

        # calculate visibility over state history
        # convert list_of_agents to 2 lists of dicts for getVisHist compatibility
        targets_dict = [a.__dict__ for a in list_of_agents if type(a) is Target]
        sensors_dict = [a.__dict__ for a in list_of_agents if type(a) is Sensor]

        rise_set_tree, v = getVisHist(
            targets_dict,
            sensors_dict,
            x_targets,
            x_sensors,
            time,
            RE,
        )

        # move axis for consistency w/ state history and maintain interface w/
        # getVisHist
        v = moveaxis(v, 2, 0)
        v = moveaxis(v, 2, 1)

        return rise_set_tree, v, x_sensors, x_targets

    def getTargAtInt(self, interval_num: int) -> Any:
        """Get target id at given visibility interval.

        Args:
            interval_num (`int`): Visibility interval index (starts at 0).

        Returns:
            `Any`: Typically a `str`.
        """
        tree_list = list(self.sched_tree)
        target_id = tree_list[interval_num][2]["target_id"]
        return target_id

    def getSensAtInt(self, interval_num: int) -> Any:
        """Get sensor id at given visibility interval.

        Args:
            interval_num (`int`): Visibility interval index (starts at 0).

        Returns:
            `Any`: Typically a `str`.
        """
        sensor_id = list(self.sched_tree)[interval_num][2]["sensor_id"]
        return sensor_id

    def getStartAtInt(self, interval_num: int) -> float:
        """Get start time of given interval.

        Args:
            interval_num (`int`): Visibility interval index (starts at 0).

        Returns:
            `float`: Time at beginning of interval `interval_num`.
        """
        start_time = list(self.sched_tree)[interval_num][0]
        return start_time

    def getFinishAtInt(self, interval_num: int) -> float:
        """Get finish time of given interval.

        Args:
            interval_num (`int`): Visibility interval index (starts at 0).

        Returns:
            `float`: Time at end of interval `interval_num`.
        """
        finish_time = list(self.sched_tree)[interval_num][1]
        return finish_time

    def getTargs(self, time: float) -> set:
        """Get list of targets visible at given time.

        Args:
            time (`float`): Time (must be within bounds of `ScheduleTree`).

        Returns:
            `set`: Sorted set of targets visible at `time`.
        """
        list_tree = list(self.sched_tree[time])
        targets = [x[2]["target_id"] for x in list_tree]
        return sorted(set(targets))

    def getSens(self, time: float) -> set:
        """Get list of sensors that can see targets at given time.

        Args:
            time (`float`): Time (must be within bounds of `ScheduleTree`).

        Returns:
            `set`: Sorted set of sensors that can see targets at `time`.
        """
        list_tree = list(self.sched_tree[time])
        sensors = [x[2]["sensor_id"] for x in list_tree]
        return sorted(set(sensors))

    def checkVisibility(
        self, time: float, sensor_id: Any, target_id: Any
    ) -> bool:
        """Checks if given target and sensor are visible at given time.

        Args:
            time (`float`): Time (must be within bounds of `ScheduleTree`).
            sensor_id (`Any`): Typically a `str`.
            target_id (`Any`): Typically a `str`.

        Returns:
            `bool`: True if sensor can see target. False otherwise.
        """
        list_tree = list(self.sched_tree[time])
        if len(list_tree) == 0:
            return False

        for window in list_tree:
            if (window[2]["sensor_id"] == sensor_id) and (
                window[2]["target_id"] == target_id
            ):
                return True
            else:
                vis_status = False

        return vis_status
