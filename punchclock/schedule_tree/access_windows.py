"""Calculate access windows."""
# %% Imports
# Standard Library Imports
from collections import Counter
from copy import deepcopy

# Third Party Imports
from intervaltree import Interval, IntervalTree
from numpy import arange

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.schedule_tree.schedule_tree import ScheduleTree

# %% calcAccessWindows


class AccessWindowCalculator:
    def __init__(
        self,
        list_of_sensors: list[Sensor],
        list_of_targets: list[Target],
        horizon: int = 1,
        dt_eval: int | float = 100,
        dt_propagate: int | float = 100,
        merge_windows: bool = True,
    ):
        """Calculate access windows for all targets.

        Args:
            list_of_sensors (list[Sensor]): List of sensors at initial dynamic states.
            list_of_targets (list[Target]): List of targets at initial dynamic states.
            horizon (int, optional): Number of time steps to calculate forward to.
                Defaults to 1.
            dt_eval (int | float, optional): Time step (sec) at which to evaluate
                access windows. Defaults to 100.
            dt_propagate (int | float, optional): Time step (sec) at which to propagate
                dynamics. Must be <= dt_eval. Defaults to 100.
            merge_windows (bool, optional): Whether of not to count an interval where
                a target can be seen by multiple sensors as 1 or multiple windows.
                True means that such situations will be counted as 1 window. Defaults
                to True.

        Returns:
            list[int]: Number of access windows per target from current time to horizon.
                The order of the list corresponds to order in list_of_targets.

        Notes:
            - Access windows are defined as discrete events (no duration) set to
                occur at the beginning of a time interval (whose duration is specified
                by dt_eval). If a sensor-target pair are visible to each other at
                t = i * dt_eval (the beginning of the interval),
                then this is counted as an access window. The time duration before
                or after the instant in time the sensor-target pair can see each
                other has no bearing on window count.
            - merge_windows should be set to True (the default) when you do not
                want to account for multiple sensors tasked to a single target at
                the same time (the typical case).

        Access window examples:
            - An access period of dt_eval duration is counted as one window.
            - An access period of eps << dt_eval encompassing t = i * dt_eval
                is counted as one window.
            - An access period of eps << dt_eval that occurs in the interval
                i * dt_eval < t < (i+1) * dt_eval is not counted.
            - An access period of dt_eval + eps starting at t = dt_eval
                is counted as two windows.
            - An access period of dt_eval + eps starting at
                t = dt_eval + eps is counted as one window.

        """
        assert (
            dt_propagate <= dt_eval
        ), f"""dt_propagate is not <= dt_eval
        (dt_propagate = {dt_propagate}, dt_eval = {dt_eval})"""

        self.merge_windows = merge_windows

        self.list_of_targets = self.checkAgentIDs(list_of_targets)
        self.list_of_sensors = self.checkAgentIDs(list_of_sensors)

        # If dt_propagate is close in magnitude to the horizon, overwrite with a smaller
        # value. This guarantees at least a reasonable (albeit small) number of sim
        # steps.
        self.dt_propagate = min(dt_propagate, (horizon * dt_eval) / 5)
        self.time_propagate = arange(
            start=0,
            stop=(horizon + 1) * dt_eval,
            step=dt_propagate,
        )

        # time_eval will be same as time_propagate if dt_propagate == dt_eval (assuming
        # large horizon)
        self.dt_eval = dt_eval
        self.time_eval = arange(
            start=0,
            stop=(horizon + 1) * dt_eval,
            step=dt_eval,
        )

        return

    def calcAccessWindows(self) -> list[int]:
        # Get access windows
        avail = ScheduleTree(
            self.list_of_sensors + self.list_of_targets, self.time_propagate
        )
        avail_tree = avail.sched_tree

        # slice availability tree by dt_eval (not dt_propagate)
        sliced_tree = deepcopy(avail_tree)
        for int_slice in self.time_eval:
            sliced_tree.slice(int_slice)

        # convert to list
        main_ival_list = list(sliced_tree.items())

        # initialize debugging variables
        num_windows_dicts = []
        list_trees = []
        # initialize outputs
        num_windows = [None for i in range(len(self.list_of_targets))]
        for i, targ in enumerate(
            targ.agent_id for targ in self.list_of_targets
        ):
            # get intervals involving targ; strip data from intervals
            intervals = [
                ival
                for ival in main_ival_list
                if (ival.data["target_id"] == targ)
            ]

            if self.merge_windows is True:
                intervals_no_data = [
                    Interval(ival.begin, ival.end) for ival in intervals
                ]

                # Build tree from intervals, which automatically merges identical
                # intervals.
                target_tree = IntervalTree(intervals_no_data)
                # merge overlapping (but not identical) intervals
                target_tree.merge_overlaps()

                # record for debugging and output
                list_trees.append(target_tree)
                num_windows[i] = len(target_tree)
                num_windows_dicts.append(
                    {"targ_id": targ, "num_windows": num_windows[i]}
                )
            else:
                num_windows[i] = len(intervals)

        return num_windows

    def checkAgentIDs(self, list_of_agents: list[Agent]) -> list[Agent]:
        """Check that agents have unique IDs, assign IDs if none provided."""
        # Check for duplicate IDs. If found, check if all IDs are None. If so,
        # assign sequential ints as IDs. Otherwise, throw a flag. If no duplicate
        # IDs found, return is same as input.
        agent_id = [ag.agent_id for ag in list_of_agents]
        repeat_ids = [
            item for item, count in Counter(agent_id).items() if count > 1
        ]

        if repeat_ids == [None]:
            # Assign sequential IDs if agents were input w/o IDs
            list_of_agents = self.assignIDs(list_of_agents)
        else:
            # If duplicate IDs were input, stop method.
            assert (
                len(repeat_ids) == 0
            ), f"""Duplicate ids in list_of_agents not allowed. Duplicate ids:
            {repeat_ids}."""

        return list_of_agents

    def assignIDs(self, list_of_agents: list[Agent]) -> list[Agent]:
        """Assign sequential IDs to a list of ID-less Agents."""
        for i, t in enumerate(list_of_agents):
            t.agent_id = i

        return list_of_agents
