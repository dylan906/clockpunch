"""Calculate access windows."""
# %% Imports
# Standard Library Imports
from collections import Counter
from copy import deepcopy

# Third Party Imports
from intervaltree import Interval, IntervalTree
from numpy import arange

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.schedule_tree.schedule_tree import ScheduleTree

# %% calcAccessWindows


def calcAccessWindows(
    list_of_sensors: list[Sensor],
    list_of_targets: list[Target],
    horizon: int = 1,
    dt_eval: int | float = 100,
    dt_propagate: int | float = 100,
    merge_windows: bool = True,
) -> list[int]:
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

    # get list of target ids
    target_ids = [targ.agent_id for targ in list_of_targets]
    repeat_ids = [
        item for item, count in Counter(target_ids).items() if count > 1
    ]
    if repeat_ids == [None]:
        # Assign sequential IDs if targets were input w/o IDs
        list_of_targets = assignIDs(list_of_targets)
    else:
        # If duplicate IDs were input, stop method.
        assert (
            len(repeat_ids) == 0
        ), f"Duplicate ids in list_of_targets not allowed. Duplicate ids: {repeat_ids}."

    # Ff dt_propagate is close in magnitude to the horizon, overwrite with a smaller
    # value. This gaurantees at least a reasonable (albiet small) number of sim
    # steps.
    dt_propagate = min(dt_propagate, (horizon * dt_eval) / 5)
    time_propagate = arange(
        start=0,
        stop=horizon * dt_eval,
        step=dt_propagate,
    )

    # Get access windows
    avail = ScheduleTree(list_of_sensors + list_of_targets, time_propagate)
    avail_tree = avail.sched_tree

    # slice availability tree by dt_eval (not dt_propagate)
    # time_sim will be same as dt_propagate == dt_eval (assuming large horizon)
    time_sim = arange(
        start=0,
        stop=horizon * dt_eval,
        step=dt_eval,
    )
    sliced_tree = deepcopy(avail_tree)
    for int_slice in time_sim:
        sliced_tree.slice(int_slice)

    # convert to list
    main_ival_list = list(sliced_tree.items())

    # initialize debugging variables
    num_windows_dicts = []
    list_trees = []
    # initialize outputs
    num_windows = [None for i in range(len(list_of_targets))]
    for i, targ in enumerate(target_ids):
        # get intervals involving targ; strip data from intervals
        intervals = [
            ival for ival in main_ival_list if (ival.data["target_id"] == targ)
        ]

        if merge_windows is True:
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


def assignIDs(list_of_targets: list[Target]) -> list[Target]:
    """Assign sequential IDs to a list of ID-less Targets."""
    for i, t in enumerate(list_of_targets):
        t.agent_id = i

    return list_of_targets
