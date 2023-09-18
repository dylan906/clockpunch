"""Info wrappers."""
# %% Import
# Standard Library Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Tuple, final
from warnings import warn

# Third Party Imports
from gymnasium import Env, Wrapper
from numpy import asarray, ndarray

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.dynamics.dynamics_classes import DynamicsModel
from punchclock.schedule_tree.access_windows import AccessWindowCalculator


# %% Info Wrapper
class InfoWrapper(ABC, Wrapper):
    """Base class for custom info wrappers."""

    def __init__(self, env: Env):
        """Wrap env with InfoWrapper."""
        super().__init__(env)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    @final
    def step(self, action):
        """Step environment forward. Do not modify."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        new_info = self.updateInfo(
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
        infos.update(new_info)

        return (observations, rewards, terminations, truncations, infos)

    @abstractmethod
    def updateInfo(
        self,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ) -> dict:
        """Create a new info dict."""
        new_info = {}
        return new_info


# %% NumWindows wrapper
class NumWindows(InfoWrapper):
    """Calculate number of target access windows over a time period.

    Wraps `info` returned from env.step(). Appends 2 items to info:
        "num_windows_left": ndarray[int] (N, ) Each entry is the number of access
            windows to the n'th target from now to the horizon.
        "vis_forecast" : ndarray[int] (T, N, M) Binary array where a 1 in the
            (t, n, m) position indicates that sensor m has access to target n at
            time step t.

    The names of the new info items are overridable via the new_keys arg.

    For details on access window definitions, see AccessWindowCalculator.
    """

    def __init__(
        self,
        env: Env,
        horizon: int = None,
        dt: float = None,
        merge_windows: bool = True,
        fixed_horizon: bool = True,
        use_estimates: bool = True,
        new_keys: list[str] = None,
    ):
        """Wrap environment with NumWindows InfoWrapper.

        Args:
            env (Env): A Gymnasium environment.
            horizon (int, optional): Number of time steps forward to forecast access
                windows. Defaults to env horizon.
            dt (float, optional): Time step (sec). Defaults to env time step.
            merge_windows (bool, optional): If True, steps when a target is accessible
                by J sensors is counted as 1 window instead of J windows. Defaults
                to True.
            fixed_horizon (bool, optional): If True, wrapper forecasts access windows
                to fixed time horizon, set by `horizon` at instantiation. If False,
                forecasts forward by `horizon` steps every call. Defaults to True.
            use_estimates (bool, optional): If True, use estimated states of targets
                to forecast access windows. Otherwise, use true states. True states
                are always used for sensors. Defaults to True.
            new_keys (list[str], optional): Override default names to be appended
                to info. The 0th value will override "num_windows_left"; the 1st
                value will override "vis_forecast". Defaults to None, meaning
                "num_windows_left" and "vis_forecast" are used.
        """
        super().__init__(env)
        # Type checking
        assert hasattr(env, "agents"), "env.agents does not exist."
        assert isinstance(
            env.agents, list
        ), "env.agents must be a list of Targets/Sensors."
        assert all(
            [isinstance(ag, (Target, Sensor)) for ag in env.agents]
        ), "env.agents must be a list of Targets/Sensors."

        assert hasattr(env, "horizon"), "env.horizon does not exist."
        if horizon is None:
            horizon = env.horizon

        assert hasattr(env, "time_step"), "env.time_step does not exist."
        if dt is None:
            dt = env.time_step

        if new_keys is None:
            new_keys = ["num_windows_left", "vis_forecast"]
        else:
            assert len(new_keys) == 2, "len(new_keys) != 2."
            print(
                f"""Default keys for NumWindows wrapper overridden. Using following
                map for new info key names: {{
                    'num_windows_left': {new_keys[0]},
                    'vis_forecast': {new_keys[1]},
                }}
                """
            )

        # check items in unwrapped info
        env_copy = deepcopy(env)
        [_, info] = env_copy.reset()
        if "num_windows_left" or "vis_forecast" in info:
            warn(
                """info already has 'num_windows_left' or 'vis_forecast'. These
            items will be overwritten with this wrapper. Consider renaming the
            unwrapped items."""
            )

        # names of new keys to append to info
        self.new_keys_map = {
            "num_windows_left": new_keys[0],
            "vis_forecast": new_keys[1],
        }
        self.use_estimates = use_estimates

        # Separate sensors from targets and dynamics
        sensors, targets = self._getAgents()
        self.num_sensors = len(sensors)
        self.num_targets = len(targets)
        dyn_sensors = self._getDynamics(sensors)
        dyn_targets = self._getDynamics(targets)

        self.awc = AccessWindowCalculator(
            num_sensors=self.num_sensors,
            num_targets=self.num_targets,
            dynamics_sensors=dyn_sensors,
            dynamics_targets=dyn_targets,
            horizon=horizon,
            dt=dt,
            fixed_horizon=fixed_horizon,
            merge_windows=merge_windows,
        )

    def _getStates(
        self,
        sensors: list[Sensor],
        targets: list[Target],
        use_estimates: bool,
    ) -> ndarray:
        """Get current state (truth or estimated) from all agents.

        If self.use_estimates == False, then truth states are fetched. Otherwise,
        truth states are fetched for sensors and estimated states are fetched for
        targets.

        Returns:
            x_sensors (ndarray): (6, M) ECI states.
            x_targets (ndarray): (6, N) ECI states.
        """
        if use_estimates is False:
            x_sensors = [agent.eci_state for agent in sensors]
            x_targets = [agent.eci_state for agent in targets]
        else:
            # Get truth states for sensors but estimated states for targets
            x_sensors = [agent.eci_state for agent in sensors]
            x_targets = [agent.target_filter.est_x for agent in targets]

        # return (6, M) and (6, N) arrays
        x_sensors = asarray(x_sensors).squeeze().transpose()
        x_targets = asarray(x_targets).squeeze().transpose()

        return x_sensors, x_targets

    def _getTime(self, agents: list[Agent]) -> float:
        """Gets current simulation time (sec)."""
        start_times = [ag.time for ag in agents]
        assert all(
            [start_times[0] == st for st in start_times]
        ), "All agents must have same time stamp."

        t0 = start_times[0]

        return t0

    def updateInfo(
        self,
        observations: Any,
        rewards: Any,
        terminations: Any,
        truncations: Any,
        infos: dict,
    ) -> dict:
        """Append items to info returned from env.step().

        Args:
            observations, rewards, terminations, truncations (Any): Unused.
            infos (dict): Unwrapped info dict.

        Returns:
            dict: Same as input info, but with two new items, "num_windows_left"
                and "vis_forecast".
                {
                    ...
                    "num_windows_left": ndarray[int] (N, ) Each entry is the number
                        of access windows to the n'th target from now to the horizon.
                    "vis_forecast" : ndarray[int] (T, N, M) Binary array where
                        a 1 in the (t, n, m) position indicates that sensor m has
                        access to target n at time step t.
                }
        """
        out = self._getCalcWindowInputs()

        self.num_windows_left, self.vis_forecast = self.awc.calcNumWindows(
            x_sensors=out["x_sensors"],
            x_targets=out["x_targets"],
            t=out["t0"],
            return_vis_hist=True,
        )

        new_info = {
            self.new_keys_map["num_windows_left"]: self.num_windows_left,
            self.new_keys_map["vis_forecast"]: self.vis_forecast,
        }

        return new_info

    def _getCalcWindowInputs(self) -> dict:
        """Get sensor/target states and current time.

        Returns:
            dict: {
                "x_sensors": ndarray (6, M), ECI state vectors in columns,
                "x_targets": ndarray (6, N), ECI state vectors in columns,
                "t0": float, Current simulation time,
            }
        """
        # Separate sensors from targets and get relevant attrs
        sensors, targets = self._getAgents()

        [x_sensors, x_targets] = self._getStates(
            sensors=sensors,
            targets=targets,
            use_estimates=self.use_estimates,
        )

        t0 = self._getTime(sensors + targets)

        return {
            "x_sensors": x_sensors,
            "x_targets": x_targets,
            "t0": t0,
        }

    def _getAgents(self) -> Tuple[list[Sensor], list[Target]]:
        """Get agents from environment, divided into Sensor and Target lists."""
        agents = deepcopy(self.agents)
        sensors = [ag for ag in agents if isinstance(ag, Sensor)]
        targets = [ag for ag in agents if isinstance(ag, Target)]
        return sensors, targets

    def _getDynamics(self, agents: list[Agent]) -> list[DynamicsModel]:
        """Get dynamics from a list of Agents."""
        # This is its own separate method because later I may want to add more
        # dynamics models that may make fetching them more complicated. So just
        # making this method separated in prep for that.
        dynamics = [ag.dynamics for ag in agents]

        return dynamics
