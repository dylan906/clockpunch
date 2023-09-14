"""Misc wrappers."""
# %% Imports
# Standard Library Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Tuple, final

# Third Party Imports
from gymnasium import Env, Wrapper
from numpy import asarray, ndarray

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.dynamics.dynamics_classes import DynamicsModel
from punchclock.schedule_tree.access_windows import AccessWindowCalculator


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    # NOTE: SimRunner is hugely dependent on this class. Be careful about modifying
    # it.

    def __init__(self, env: Env, id: Any = None):  # noqa
        """Wrap environment with IdentityWrapper.

        Args:
            env (Env): A Gymnasium environment.
            id (Any, optional): Mainly used to distinguish between multiple instances
                of IdentityWrapper. Defaults to None.
        """
        super().__init__(env)

        self.id = id
        return

    def observation(self, obs):
        """Pass-through observation."""
        return obs


# %% Info Wrapper
class InfoWrapper(ABC, Wrapper):
    """Base class for custom info wrappers."""

    def __init__(self, env: Env):
        """Wrap env with InfoWrapper."""
        super().__init__(env)

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

        return (observations, rewards, terminations, truncations, new_info)

    @abstractmethod
    def updateInfo(
        self,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        return infos


# %% NumWindows wrapper
class NumWindows(InfoWrapper):
    def __init__(
        self,
        env: Env,
        horizon: int = None,
        dt: float = None,
        merge_windows: bool = True,
        fixed_horizon: bool = True,
        use_estimates: bool = True,
    ):
        super().__init__(env)
        # Type checking
        assert hasattr(env, "agents")
        assert isinstance(env.agents, list)
        assert all([isinstance(ag, (Target, Sensor)) for ag in env.agents])

        self.use_estimates = use_estimates

        # Separate sensors from targets and dynamics
        sensors, targets = self._getAgents()
        dyn_sensors = self._getDynamics(sensors)
        dyn_targets = self._getDynamics(targets)
        # t0 = self._getStartTime(agents)
        # out = self_getAwcInputs()

        self.num_sensors = len(sensors)
        self.num_targets = len(targets)

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

        # self.num_windows_left, self.vis_forecast = self.awc.calcNumWindows(
        #     x_sensors=x_sensors,
        #     x_targets=x_targets,
        #     t=t0,
        #     return_vis_hist=True,
        # )

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

    def _getStartTime(self, agents: list[Agent]) -> float:
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
        out = self._getCalcWindowInputs()

        self.num_windows_left, self.vis_forecast = self.awc.calcNumWindows(
            x_sensors=out["x_sensors"],
            x_targets=out["x_targets"],
            t=out["t0"],
            return_vis_hist=True,
        )

        new_info = {
            "num_windows_left": self.num_windows_left,
            "vis_forecast": self.vis_forecast,
        }

        return new_info

    def _getCalcWindowInputs(self) -> dict:
        # Separate sensors from targets and get relevant attrs
        sensors, targets = self._getAgents()

        [x_sensors, x_targets] = self._getStates(
            sensors=sensors,
            targets=targets,
            use_estimates=self.use_estimates,
        )

        t0 = self._getStartTime(sensors + targets)

        return {
            "x_sensors": x_sensors,
            "x_targets": x_targets,
            "t0": t0,
        }

    def _getAgents(self) -> Tuple[list[Sensor], list[Target]]:
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
