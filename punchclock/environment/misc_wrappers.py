"""Misc wrappers."""
# %% Imports
# Standard Library Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, final

# Third Party Imports
from gymnasium import Env, Wrapper
from numpy import asarray, ndarray

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
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
    def __init__(self, env: Env):
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
        use_estimates: bool = True,
    ):
        super().__init__(env)
        assert hasattr(env, "agents")
        assert isinstance(env.agents, list)
        assert all([isinstance(ag, (Target, Sensor)) for ag in env.agents])

        self.use_estimates = use_estimates

        agents = deepcopy(env.agents)
        sensors = [ag for ag in agents if isinstance(ag, Sensor)]
        targets = [ag for ag in agents if isinstance(ag, Target)]
        [x_sensors, x_targets] = self._getStates(
            sensors=sensors, targets=targets
        )

        dyn_sensors = self._getDynamics(sensors)
        dyn_targets = self._getDynamics(targets)

        t0 = self._getStartTime(agents)

        self.window_calculator = AccessWindowCalculator(
            x_sensors=x_sensors,
            x_targets=x_targets,
            dynamics_sensors=dyn_sensors,
            dynamics_targets=dyn_targets,
            t_start=t0,
            horizon=horizon,
            dr=dt,
            merge_windows=merge_windows,
        )

    def _getStates(
        self, sensors: list[Sensor], targets: list[Target]
    ) -> ndarray:
        """Get current state (truth or estimated) from all agents.

        If self.use_estimates == False, then truth states are fetched. Otherwise,
        truth states are fetched for sensors and estimated states are fetched for
        targets.

        Returns:
            x_sensors (ndarray): (6, M) ECI states.
            x_targets (ndarray): (6, N) ECI states.
        """
        if self.use_estimates is False:
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
    ):
        new_info = infos.update({})
        return new_info
