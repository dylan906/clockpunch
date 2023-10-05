"""Calculate access windows."""
# %% Imports
# Standard Library Imports
from copy import deepcopy
from typing import Tuple

# Third Party Imports
from numpy import arange, asarray, ndarray, sum, zeros
from satvis.visibility_func import isVis

# Punch Clock Imports
from punchclock.common.agents import Agent
from punchclock.common.constants import getConstants
from punchclock.dynamics.dynamics_classes import (
    DynamicsModel,
    SatDynamicsModel,
    StaticTerrestrial,
)


# %% AccessWindowCalculator
class AccessWindowCalculator:
    """Calculates access windows between Sensors and Targets.

    AccessWindowCalculator creates surrogate agents corresponding to the input
    states and dynamic models. It then propagates the surrogate agents' state
    history to predict visibility status.

    Works with either a fixed or receding horizon.

    Notation:
        M: number of sensors
        N: number of targets
        T: number of time steps
        t: time
        dt: time step

    Notes:
        - Access window forecast range is (now:horizon]. Forecast does not include
            the current time step, and includes the horizon (final) time step.
        - If using a receding horizon, the number of time steps forecast T is constant.
            If using fixed horizon, T changes as a function of current time.
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

    def __init__(
        self,
        num_sensors: int,
        num_targets: int,
        dynamics_sensors: DynamicsModel | list[DynamicsModel] | str | list[str],
        dynamics_targets: DynamicsModel | list[DynamicsModel] | str | list[str],
        horizon: int = 1,
        dt: float = 100.0,
        fixed_horizon: bool = True,
        merge_windows: bool = True,
    ):
        """Calculate access windows for all targets.

        Args:
            num_sensors (int): Number of sensors.
            num_targets (int): Number of targets.
            dynamics_sensors (DynamicsModel | list[DynamicsModel] | str | list[str]):
                Dynamic model/model tag for sensors. If str or list input, entry(ies)
                must be one of ["terrestrial" | "satellite"]. If str input, all
                sensors are assigned same dynamic model.
            dynamics_targets (DynamicsModel | list[DynamicsModel] | str | list[str]):
                Dynamic model/model tag for targets. If str or list input, entry(ies)
                must be one of ["terrestrial" | "satellite"]. If str input, all
                targets are assigned same dynamic model.
            horizon (int, optional): Number of time steps to evaluate access windows
                forward in time. Defaults to 1.
            dt (float, optional): Time step (sec) at which to propagate
                dynamics at evaluate visibility. Defaults to 100.
            fixed_horizon (bool, optional): If True, input for horizon will be
                fixed, and subsequent calls to forecast visibility windows will
                use a window from current time to the fixed value. If False, the
                same number of steps are forecast on every call.
            merge_windows (bool, optional): Whether of not to count an interval where
                a target can be seen by multiple sensors as 1 or multiple windows.
                True means that such situations will be counted as 1 window. Defaults
                to True.
        """
        assert isinstance(dynamics_sensors, (list, str, DynamicsModel))
        assert isinstance(dynamics_targets, (list, str, DynamicsModel))
        if isinstance(dynamics_sensors, list):
            assert len(dynamics_sensors) == num_sensors
        if isinstance(dynamics_targets, list):
            assert len(dynamics_targets) == num_targets
        if isinstance(dynamics_sensors, str):
            assert dynamics_sensors in ["terrestrial", "satellite"]
        if isinstance(dynamics_targets, str):
            assert dynamics_targets in ["terrestrial", "satellite"]

        assert horizon >= 1, "horizon not >= 1."

        self.dynamics_sensors = dynamics_sensors
        self.dynamics_targets = dynamics_targets

        self.merge_windows = merge_windows
        self.fixed_horizon = fixed_horizon

        self.RE = getConstants()["earth_radius"]
        self.num_sensors = num_sensors
        self.num_targets = num_targets

        # # If dt is close in magnitude to the horizon, overwrite with a smaller
        # # value. This guarantees at least a reasonable (albeit small) number of sim
        # # steps.
        self.dt = min(dt, (horizon * dt) / 5)
        self.horizon = horizon

        if fixed_horizon is True:
            self.fixed_horizon_time = (horizon + 1) * self.dt
        else:
            self.fixed_horizon_time = None

        return

    def calcVisHist(
        self,
        x_sensors: ndarray,
        x_targets: ndarray,
        t: float,
    ) -> ndarray[int]:
        """Calculate visibility history array.

        Args:
            x_sensors (ndarray): (6, M) ECI state vectors in columns.
            x_targets (ndarray): (6, N) ECI state vectors in columns.
            t (float): Current time (sec).

        Returns:
            ndarray[int]: (T, N, M) binary array. A 1 indicates that the n-m
                target-sensor can see each other at time t.
        """
        # build agents and time vector
        self._setup(x_sensors=x_sensors, x_targets=x_targets, t=t)

        # Setup state history arrays
        x0 = self._getStates()

        vis_hist = zeros(
            (
                len(self.time_vec),
                self.num_targets,
                self.num_sensors,
            ),
            dtype=int,
        )
        vis_hist[0, :, :] = self._getVis(
            x_sensors=x0[:, : self.num_sensors],
            x_targets=x0[:, self.num_sensors :],
        )

        # loop through agents and propagate motion
        for i, t in enumerate(self.time_vec[1:], start=1):
            for agent in self.sensors + self.targets:
                agent.propagate(t)

            xi = self._getStates()
            vis_hist[i, :, :] = self._getVis(
                x_sensors=xi[:, : self.num_sensors],
                x_targets=xi[:, self.num_sensors :],
            )

        return vis_hist

    def calcNumWindows(
        self,
        x_sensors: ndarray,
        x_targets: ndarray,
        t: float,
        return_vis_hist: bool = False,
    ) -> (
        ndarray[int]
        | Tuple[ndarray[int], ndarray[int], ndarray[int], ndarray[float]]
    ):
        """Get number of visibility windows per target from current time to horizon.

        Optionally return detailed data (time history of access windows).

        Args:
            x_sensors (ndarray): (6, M) ECI state vectors in columns.
            x_targets (ndarray): (6, N) ECI state vectors in columns.
            t (float): Current time (sec).
            return_vis_hist (bool, optional): Set to True to return vis_hist in
                addition to num_windows. Defaults to False.

        Returns:
            num_windows (ndarray[int]): (N,) Number of windows per target.
            vis_hist (ndarray[int], optional): (T, N, M) Same as return from
                self.calcVisHist(). Outputs if return_vis_hist == True on input.
            vis_hist_targets (ndarray[int], optional): (T, N) Each row has the
                number of windows left in time period for the n'th target. Outputs
                if return_vis_hist == True on input.
            time (ndarray[float], optional): (T, ) Time history (sec) corresponding
                to 0th dimensions of vis_hist and vis_hist_targets. Outputs if
                return_vis_hist == True on input.
        """
        vis_hist = self.calcVisHist(
            x_sensors=x_sensors,
            x_targets=x_targets,
            t=t,
        )
        # merge multi-sensor windows and sum
        vis_hist_targets = self._mergeWindows(vis_hist)  # returns (T, N)

        num_windows = self.sumWindows(
            vis_hist=vis_hist,
            vis_hist_targets=vis_hist_targets,
            merge_windows=self.merge_windows,
        )

        if return_vis_hist is False:
            return num_windows
        else:
            time_hist = deepcopy(self.time_vec)
            return num_windows, vis_hist, vis_hist_targets, time_hist

    def sumWindows(
        self,
        vis_hist: ndarray[int],
        vis_hist_targets: ndarray[int],
        merge_windows: bool,
    ) -> ndarray[int]:
        """Get total number of windows available per target.

        Args:
            vis_hist (ndarray[int]): (T, N, M) Binary array of sensor-target
                accessibility, where T is the final time index.
            vis_hist_targets (ndarray[int]): (T, N) Number of windows per n'th
                target at t'th time step, regardless of how many sensors (beyond 1)
                have access to the target.
            merge_windows (bool): Whether or not to count multiple simultaneous
                sensor accesses as multiple or one window.

        Returns:
            ndarray[int]: (N, ) Number of windows per target.
        """
        if merge_windows is False:
            # sum unmerged windows
            num_windows = sum(sum(vis_hist, axis=2), axis=0)
        elif merge_windows is True:
            # sum merged windows
            num_windows = sum(vis_hist_targets, axis=0)

        return num_windows

    def _setup(
        self,
        x_sensors: ndarray,
        x_targets: ndarray,
        t: float,
    ):
        """Build surrogate agents and generate time vector.

        Args:
            x_sensors (ndarray): (6, M) ECI state vectors in columns.
            x_targets (ndarray): (6, N) ECI state vectors in columns.
            t (float): Current time (sec).
        """
        # update time
        self.t_now = t

        # Generate time vector
        self.time_vec = self._genTime()

        # Create surrogate agents
        self.sensors = self._buildAgents(
            x=x_sensors,
            time=self.t_now,
            dynamics=self.dynamics_sensors,
        )
        self.targets = self._buildAgents(
            x=x_targets,
            time=self.t_now,
            dynamics=self.dynamics_targets,
        )

    def _getStates(self) -> ndarray:
        """Get current state from all agents.

        Returns:
            ndarray: (6, M + N) ECI states. Sensor states are in columns 0:M-1,
                target states are in columns M:N-1.
        """
        x = [agent.eci_state for agent in self.sensors + self.targets]

        # return (6, M+N) array
        x = asarray(x).squeeze().transpose()

        return x

    def _getVis(self, x_sensors: ndarray, x_targets: ndarray) -> ndarray:
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

    def _mergeWindows(self, vis_hist: ndarray) -> ndarray:
        """Merge sensor elements of a (T, N, M) visibility history array.

        For every (N, M) frame in vis_hist, a N-long binary vector is created.
        If there are any 1s in the n'th row of the t'th frame, the n'th value
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

    def _buildAgents(
        self,
        x: ndarray,
        time: float,
        dynamics: str | list[str] | DynamicsModel | list[DynamicsModel],
    ) -> list[Agent]:
        """Build generic agents.

        Args:
            x (ndarray): (6, A) ECI state vectors in the columns.
            time (float): JD time.
            dynamics (str | list[str] | DynamicsModel | list[DynamicsModel]):
                Dynamic model/model tag for agents. If str or list input, entry(ies)
                must be one of ["terrestrial" | "satellite"]. If str input, all
                agents are assigned same dynamic model.

        Returns:
            list[Agent]: A list of Targets or Sensors.
        """
        # Map if dynamics tag is used vice DynamicsModel
        dynamics_model_map = {
            "terrestrial": StaticTerrestrial(),
            "satellite": SatDynamicsModel(),
        }

        if isinstance(dynamics, (str, DynamicsModel)):
            # Copy single instance to list of identical items
            dynamics = [dynamics for a in range(x.shape[1])]

        if isinstance(dynamics[0], str):
            # Map from str->DynamicsModel if str inputs are used
            dynamics = [dynamics_model_map[d] for d in dynamics]

        agents = []
        for i, dyn in zip(range(x.shape[1]), dynamics):
            xi = x[:, i]
            agents.extend(
                [
                    Agent(
                        dynamics_model=dyn,
                        init_eci_state=xi,
                        time=time,
                    )
                ]
            )

        return agents

    def _genTime(self) -> ndarray:
        """Generate a time vector.

        If using receding horizon, T = self.horizon is constant. If using fixed
        horizon, T is the difference between current time and the fixed horizon
        (in integer steps of dt size).

        Returns:
            ndarray: (T, ) Time vector (sec).
        """
        if self.fixed_horizon is True:
            assert (
                self.t_now < self.fixed_horizon_time
            ), "Current time exceeds fixed horizon."

            fixed_horizon_delta = self.fixed_horizon_time - self.t_now
            time_vec = arange(
                start=self.t_now + self.dt,
                stop=self.t_now + fixed_horizon_delta,
                step=self.dt,
            )
        else:
            time_vec = arange(
                start=self.t_now + self.dt,
                stop=self.t_now + (self.horizon + 1) * self.dt,
                step=self.dt,
            )

        return time_vec
