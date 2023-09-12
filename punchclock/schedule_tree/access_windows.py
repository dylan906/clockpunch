"""Calculate access windows."""
# %% Imports
# Standard Library Imports
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
    time_step: int | float = 100,
    merge_windows: bool = True,
) -> list[int]:
    """Calculate access windows for all targets.

    Args:
        list_of_sensors (list[Sensor]): List of sensors at initial dynamic states.
        list_of_targets (list[Target]): List of targets at initial dynamic states.
        horizon (int, optional): Number of time steps to calculate forward to.
            Defaults to 1.
        time_step (int | float, optional): Duration of time step (sec). Defaults
            to 100.
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
            by time_step). If a sensor-target pair are visible to each other at
            t = i * time_step (the beginning of the interval),
            then this is counted as an access window. The time duration before
            or after the instant in time the sensor-target pair can see each
            other has no bearing on window count.
        - merge_windows should be set to True (the default) when you do not
            want to account for multiple sensors tasked to a single target at
            the same time (the typical case).

    Access window examples:
        - An access period of time_step duration is counted as one window.
        - An access period of eps << time_step encompassing t = i * time_step
            is counted as one window.
        - An access period of eps << time_step that occurs in the interval
            i * time_step < t < (i+1) * time_step is not counted.
        - An access period of time_step + eps starting at t = time_step
            is counted as two windows.
        - An access period of time_step + eps starting at
            t = time_step + eps is counted as one window.

    """
    # get list of target ids
    target_ids = [targ.agent_id for targ in list_of_targets]

    # propagate motion in ScheduleTree at 100sec steps, unless horizon time is
    # shorter, in which case pick some fraction of horizon.
    step = min(100, (horizon * time_step) / 5)
    time_propagate = arange(
        start=0,
        stop=horizon * time_step,
        step=step,
    )

    # Get access windows
    avail = ScheduleTree(list_of_sensors + list_of_targets, time_propagate)
    avail_tree = avail.sched_tree

    # slice availability tree by simulation time (not time_propagate)
    # time_sim will be same as time_propagate if time_step>100
    time_sim = arange(
        start=0,
        stop=horizon * time_step,
        step=time_step,
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
