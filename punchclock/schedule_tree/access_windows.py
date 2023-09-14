"""Calculate access windows."""
# %% Imports
# Standard Library Imports
from copy import deepcopy

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

# %% calcAccessWindows


class AccessWindowCalculator:
    """Calculates access windows between Sensors and Targets.

    AccessWindowCalculator creates surrogate agents corresponding to the input
    states and dynamic models. It then propagates the surrogate agents' state
    history to predict visibility status.
    """

    def __init__(
        self,
        x_sensors: ndarray,
        x_targets: ndarray,
        dynamics_sensors: DynamicsModel | list[DynamicsModel] | str | list[str],
        dynamics_targets: DynamicsModel | list[DynamicsModel] | str | list[str],
        t_start: float = 0.0,
        horizon: int = 1,
        dt: float = 100.0,
        merge_windows: bool = True,
    ):
        """Calculate access windows for all targets.

        Args:
            x_sensors (ndarray): (6, M) ECI state vectors.
            x_targets (ndarray): (6, N) ECI state vectors.
            dynamics_sensors (DynamicsModel | list[DynamicsModel] | str | list[str]):
                Dynamic model/model tag for sensors. If str or list input, entry(ies)
                must be one of ["terrestrial" | "satellite"]. If str input, all
                sensors are assigned same dynamic model.
            dynamics_targets (DynamicsModel | list[DynamicsModel] | str | list[str]):
                Dynamic model/model tag for targets. If str or list input, entry(ies)
                must be one of ["terrestrial" | "satellite"]. If str input, all
                targets are assigned same dynamic model.
            t_start (float, optional): JD initialization time. Defaults to 0.
            horizon (int, optional): Number of time steps to evaluate access windows
                forward from start time. Defaults to 1.
            dt (float, optional): Time step (sec) at which to propagate
                dynamics at evaluate visibility. Defaults to 100.
            merge_windows (bool, optional): Whether of not to count an interval where
                a target can be seen by multiple sensors as 1 or multiple windows.
                True means that such situations will be counted as 1 window. Defaults
                to True.

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
        assert x_sensors.shape[0] == 6
        assert x_targets.shape[0] == 6
        assert isinstance(dynamics_sensors, (list, str, DynamicsModel))
        assert isinstance(dynamics_targets, (list, str, DynamicsModel))
        if isinstance(dynamics_sensors, list):
            assert len(dynamics_sensors) == x_sensors.shape[1]
        if isinstance(dynamics_targets, list):
            assert len(dynamics_targets) == x_targets.shape[1]
        if isinstance(dynamics_sensors, str):
            assert dynamics_sensors in ["terrestrial", "satellite"]
        if isinstance(dynamics_targets, str):
            assert dynamics_targets in ["terrestrial", "satellite"]

        assert horizon >= 1

        self.start_time = t_start

        # Create surrogate agents
        self.sensors = self._buildAgents(
            x=x_sensors,
            time=self.start_time,
            dynamics=dynamics_sensors,
        )
        self.targets = self._buildAgents(
            x=x_targets,
            time=self.start_time,
            dynamics=dynamics_targets,
        )

        self.backup_sensors = deepcopy(self.sensors)
        self.backup_targets = deepcopy(self.targets)

        self.merge_windows = merge_windows

        self.RE = getConstants()["earth_radius"]
        self.num_sensors = len(self.sensors)
        self.num_targets = len(self.targets)
        self.num_agents = self.num_targets + self.num_sensors

        # If dt is close in magnitude to the horizon, overwrite with a smaller
        # value. This guarantees at least a reasonable (albeit small) number of sim
        # steps.
        self.dt = min(dt, (horizon * dt) / 5)
        self.time_vec = arange(
            start=self.start_time,
            stop=self.start_time + (horizon + 1) * dt,
            step=dt,
        )

        return

    def reset(self):
        """Reset agents.

        Because propagating the agents to calculate visibility windows changes their
        states, need to be able to reset agents to original state.

        Agents are only modified within scope of AccessWindowCalculator (no leakage).
        """
        self.sensors = deepcopy(self.backup_sensors)
        self.targets = deepcopy(self.backup_targets)

    def calcVisHist(self) -> ndarray[int]:
        """Calculate visibility history array.

        Returns:
            ndarray[int]: (T, N, M) binary array. A 1 indicates that the n-m
                target-sensor can see each other at time t.
        """
        self.reset()
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

    def calcNumWindows(self) -> ndarray[int]:
        """Get number of visibility windows per target from current time to horizon.

        Returns:
            ndarray[int]: (N,) Number of windows per target.
        """
        vis_hist = self.calcVisHist()
        if self.merge_windows is False:
            num_windows = sum(sum(vis_hist, axis=2), axis=0)
        else:
            # merge multi-sensor windows and sum
            vis_hist_merge = self._mergeWindows(vis_hist)  # returns (T, N)
            num_windows = sum(vis_hist_merge, axis=0)

        return num_windows

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

    def _buildAgents(
        self,
        x: ndarray,
        time: float,
        dynamics: str | list[str] | DynamicsModel | list[DynamicsModel],
    ) -> list[Agent]:
        """Build generic agents."""
        dynamics_model_map = {
            "terrestrial": StaticTerrestrial(),
            "satellite": SatDynamicsModel(),
        }

        if isinstance(dynamics, (str, DynamicsModel)):
            dynamics = [dynamics for a in range(x.shape[1])]

        if isinstance(dynamics[0], str):
            dynamics = [dynamics_model_map[d] for d in dynamics]

        agents = []
        for i, dyn in zip(range(x.shape[1]), dynamics):
            xi = x[:, i]
            # dyn_model = dynamics_model_map[dyn]
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
