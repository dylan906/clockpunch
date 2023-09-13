"""Calculate access windows."""
# %% Imports
# Standard Library Imports
from collections import Counter
from copy import deepcopy

# Third Party Imports
from intervaltree import Interval, IntervalTree
from numpy import arange, asarray, ndarray, squeeze, sum, zeros
from satvis.visibility_func import isVis

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.common.constants import getConstants
from punchclock.schedule_tree.schedule_tree import ScheduleTree

# %% calcAccessWindows


class AccessWindowCalculator:
    def __init__(
        self,
        list_of_sensors: list[Sensor],
        list_of_targets: list[Target],
        horizon: int = 1,
        dt: int | float = 100,
        truth_or_estimated: str = "truth",
        merge_windows: bool = True,
    ):
        """Calculate access windows for all targets.

        Args:
            list_of_sensors (list[Sensor]): List of sensors at initial dynamic states.
            list_of_targets (list[Target]): List of targets at initial dynamic states.
            horizon (int, optional): Number of time steps to evaluate access windows
                forward from start time. Defaults to 1.
            dt (int | float, optional): Time step (sec) at which to propagate
                dynamics at evaluate visibility. Defaults to 100.
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
        assert horizon >= 1

        start_times = [ag.time for ag in list_of_sensors]
        start_times.extend([ag.time for ag in list_of_targets])
        assert all(
            [start_times[0] == st for st in start_times]
        ), "All agents must have same time stamp."
        self.start_time = start_times[0]

        self.backup_list_of_targets = deepcopy(list_of_targets)
        self.backup_list_of_sensors = deepcopy(list_of_sensors)
        self.reset()

        self.merge_windows = merge_windows
        if truth_or_estimated == "truth":
            self.use_true_states = True
        elif truth_or_estimated == "estimated":
            self.use_true_states = False

        self.RE = getConstants()["earth_radius"]
        self.num_sensors = len(self.list_of_sensors)
        self.num_targets = len(self.list_of_targets)
        self.num_agents = self.num_targets + self.num_sensors

        # If dt is close in magnitude to the horizon, overwrite with a smaller
        # value. This guarantees at least a reasonable (albeit small) number of sim
        # steps.
        self.dt = min(dt, (horizon * dt) / 5)
        self.time_propagate = arange(
            start=self.start_time,
            stop=self.start_time + (horizon + 1) * dt,
            step=dt,
        )

        return

    def reset(self):
        self.list_of_sensors = deepcopy(self.backup_list_of_sensors)
        self.list_of_targets = deepcopy(self.backup_list_of_targets)

    def calcVisHist(self) -> ndarray[int]:
        """Calculate visibility history array.

        Returns:
            ndarray[int]: (T, N, M) binary array.
        """
        self.reset()
        # Setup state history arrays
        x0 = self.getStates()

        vis_hist = zeros(
            (
                len(self.time_propagate),
                len(self.list_of_targets),
                len(self.list_of_sensors),
            ),
            dtype=int,
        )
        vis_hist[0, :, :] = self.getVis(
            x_sensors=x0[:, : self.num_sensors],
            x_targets=x0[:, self.num_sensors :],
        )

        # loop through agents and propagate motion
        for i, t in enumerate(self.time_propagate[1:], start=1):
            for agent in self.list_of_sensors + self.list_of_targets:
                agent.propagate(t)

            xi = self.getStates()
            vis_hist[i, :, :] = self.getVis(
                x_sensors=xi[:, : self.num_sensors],
                x_targets=xi[:, self.num_sensors :],
            )

        return vis_hist

    def calcNumWindows(self) -> ndarray[int]:
        vis_hist = self.calcVisHist()
        if self.merge_windows is False:
            num_windows = sum(sum(vis_hist, axis=2), axis=0)
        else:
            # merge multi-sensor windows and sum
            vis_hist_merge = self.mergeWindows(vis_hist)  # returns (T, N)
            num_windows = sum(vis_hist_merge, axis=0)

        return num_windows

    def mergeWindows(self, vis_hist: ndarray) -> ndarray:
        """Merge sensor elements of a (T, N, M) visibility history array.

        For every (N, M) frame in vis_hist, a N-long binary vector is created.
        If there are any 1s in the i'th row of the t'th frame, the i'th value
        of the binary vector is set to 1. Otherwise, the value is 0. The binary
        vectors are output as a (T, N) array.

        Args:
            vis_hist (ndarray): (T, N, M)

        Returns:
            ndarray: (T, N)
        """
        vis_hist_merge = zeros(
            (vis_hist.shape[0], vis_hist.shape[1]),
            dtype=int,
        )
        for t in range(vis_hist.shape[0]):
            va = vis_hist[t, :, :]
            for n in range(va.shape[0]):
                row = va[n, :]
                if 1 in row:
                    vis_hist_merge[t, n] = 1

        return vis_hist_merge

    def getStates(self) -> ndarray:
        """Get current state (truth or estimated) from all agents.

        If self.use_true_states == True, then truth states are fetched. Otherwise,
        truth states are fetched for sensors and estimated states are fetched for
        targets.

        Returns:
            ndarray: (6, M + N) ECI states. Sensor states are in columns 0:M-1,
                target states are in columns M:N-1.
        """
        if self.use_true_states:
            x = [
                agent.eci_state
                for agent in self.list_of_sensors + self.list_of_targets
            ]
        else:
            # Get truth states for sensors but estimated states for targets
            x = [agent.eci_state.squeeze() for agent in self.list_of_sensors]
            x.extend(
                [agent.target_filter.est_x for agent in self.list_of_targets]
            )

        # return (6, M+N) array
        x = asarray(x).squeeze().transpose()

        return x

    def getVis(self, x_sensors: ndarray, x_targets: ndarray) -> ndarray:
        """Get visibility status array (N, M).

        Args:
            x_sensors (ndarray): (6, M) ECI state.
            x_targets (ndarray): (6, N) ECI state.

        Returns:
            ndarray: (N, M) binary array, where 1 indicates sensor-target pair
                are visible to each other.
        """
        # Loop through sensor/target pairs and check visibility via isVis(), which
        # just requires two position vectors.
        vis_status = zeros((self.num_targets, self.num_sensors), dtype=int)
        for sens in range(self.num_sensors):
            r_sens = x_sensors[:3, sens]
            for targ in range(self.num_targets):
                r_targ = x_targets[:3, targ]
                vis_status[targ, sens] = isVis(r_sens, r_targ, self.RE)

        return vis_status
