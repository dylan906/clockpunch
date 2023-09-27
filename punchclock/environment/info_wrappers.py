"""Info wrappers."""
# %% Import
# Standard Library Imports
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, Tuple, final
from warnings import warn

# Third Party Imports
from gymnasium import Env, Wrapper
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete
from numpy import asarray, ndarray

# Punch Clock Imports
from punchclock.common.agents import Agent, Sensor, Target
from punchclock.common.utilities import actionSpace2Array, getInequalityFunc
from punchclock.dynamics.dynamics_classes import DynamicsModel
from punchclock.environment.wrapper_utils import (
    countMaskViolations,
    countNullActiveActions,
    getInfo,
)
from punchclock.schedule_tree.access_windows import AccessWindowCalculator


# %% Info Wrapper
class InfoWrapper(ABC, Wrapper):
    """Base class for custom info wrappers."""

    def __init__(self, env: Env):
        """Wrap env with InfoWrapper."""
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a Dict."
        super().__init__(env)
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a MultiDiscrete."
        assert all(
            env.action_space.nvec == env.action_space.nvec[0]
        ), "All values in action_space.nvec must be same."
        super().__init__(env)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        new_info = self.updateInfo(obs, 0, False, False, info, None)
        info.update(new_info)
        self.info = deepcopy(info)

        return obs, info

    @final
    def _getInfo(self):
        return deepcopy(self.info)

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
            action,
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
        action,
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
        info = getInfo(env)
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
        action: Any,
    ) -> dict:
        """Append items to info returned from env.step().

        Args:
            observations, rewards, terminations, truncations, action (Any): Unused.
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


# %% ActionTypeCounter
class ActionTypeCounter(InfoWrapper):
    """Counts null or active actions.

    The null action is the max value allowed in a MultiDiscrete action space. All
        values in action space must be identical.

    Example:
        action_space = MultiDiscrete([3, 3, 3]) # 3 is the null action
        wrapped_env = ActionTypeCounter(env)
        action = array([0, 1, 3])
        count = 0 + 0 + 1 = 1

    Example:
        action_space = MultiDiscrete([3, 3, 3]) # 3 is the null action
        wrapped_env = ActionTypeCounter(env, count_null_actions=False)
        action = array([0, 1, 3])
        count = 1 + 1 + 0 = 2

    """

    def __init__(
        self,
        env: Env,
        new_key: str,
        count_null_actions: bool = True,
    ):
        """Wrap environment.

        Args:
            env (Env): See InfoWrapper for requirements.
            new_key (str): New key in info to assign action count value to.
            count_null_actions (bool, optional): If True, null actions are counted.
                If False, active actions are counted. Defaults to True.
        """
        super().__init__(env)
        info = getInfo(env)
        if new_key in info:
            warn(
                f"""{new_key} already in info returned by env. Will be overwritten
                by wrapper. Consider using different value for new_key={new_key}."""
            )

        self.new_key = new_key
        self.count_null_actions = count_null_actions
        self.null_action_index = env.action_space.nvec[0] - 1

    def updateInfo(
        self,
        obs: Any,
        reward: Any,
        termination: Any,
        truncationAny: Any,
        info: Any,
        action: ndarray[int],
    ) -> dict:
        """Count actions.

        Args:
            obs, reward, termination, truncation, info: Unused.
            action (ndarray[int]): A (N,) array of ints where the i-th value is
                the i-th sensor and the value denotes the target number (0 to N-1);
                the value N denotes number of null/active actions.

        Returns:
            info (dict[str[int]]): {
                self.new_key: Sum of null/active actions this step.
                }
        """
        act_count = countNullActiveActions(
            action=action,
            null_action_index=self.null_action_index,
            count_null=self.count_null_actions,
        )
        info = {self.new_key: act_count}

        return info


# %% MaskViolationCounter
class MaskViolationCounter(InfoWrapper):
    """Count sensors assigned to valid (or invalid) action.

    Nomenclature:
        M: Number of sensors.
        N: Number of targets.

    Example:
        # for 3 sensors, 2 targets, null actions included, count valid actions
        wrapped_env = MaskViolationCounter(env, "action_mask", ignore_null_actions=False)
        # action_mask = array([[1, 1, 1],
                               [0, 0, 1]
                               [1, 1, 1]])  # last row is null action
        action = array([0, 1, 2])
        # count = 1 + 0 + 1 = 2

        Sensor 0 counts 1 because it tasked a valid (1) action.
        Sensor 1 counts 0 because it tasked an invalid (0) action.
        Sensor 2 counts 1 because it tasked a valid (1) action.

    Example:
        # for 3 sensors, 2 targets, null actions ignored, count invalid actions
        wrapped_env = MaskViolationCounter(env, "action_mask",
            count_valid_actions=False, ignore_null_actions=True)
        # action_mask = array([[1, 1, 1],
                               [0, 0, 1],
                               [1, 1, 1]])
        action = array([0, 1, 2])
        # count = 0 + 1 + 0 = 1

        Sensor 0 counts 0 because it tasked a valid (1) action.
        Sensor 1 counts 1 because it tasked an invalid (0) action.
        Sensor 2 counts 0 because null-actions (2) are ignored.
    """

    def __init__(
        self,
        env: Env,
        new_key: str,
        action_mask_key: str,
        count_valid_actions: bool = False,
        ignore_null_actions: bool = True,
    ):
        """Wrap environment.

        Args:
            env (Env): See InfoWrapper for requirements.
            new_key (str): New key in info to assign mask violation count value to.
            action_mask_key (str): Key corresponding to action mask in observation
                space. Value associated with action_mask_key must be (N+1, M) binary
                array where a 1 indicates the sensor-action the pairing is a valid
                action). The bottom row denotes null action.
            count_valid_actions (bool, optional): If True, valid actions are counted.
                If False, invalid actions are counted. Defaults to False.
            ignore_null_actions (bool, optional): If True, the bottom row of the
                action mask is ignored; action values of N are ignored. If False,
                Null actions are counted according to mask value and count_valid_actions
                value. Defaults to True.
        """
        super().__init__(env)
        assert (
            action_mask_key in env.observation_space.spaces
        ), f"'{action_mask_key}' not in env.observation_space."
        assert isinstance(
            env.observation_space.spaces[action_mask_key], MultiBinary
        ), f"env.observation_space['{action_mask_key}'] must be MultiBinary."

        self.num_sensors = len(env.action_space)
        self.num_targets = env.action_space.nvec[0] - 1

        assert env.observation_space.spaces[action_mask_key].shape == (
            self.num_targets + 1,
            self.num_sensors,
        ), f"""env.observation_space['{action_mask_key}'] must have shape (N+1, M),
        which in this case is ({self.num_targets+1}, {self.num_sensors})."""

        self.new_key = new_key
        self.action_mask_key = action_mask_key
        self.count_valid_actions = count_valid_actions
        self.ignore_null_actions = ignore_null_actions
        self.action_converter = partial(
            actionSpace2Array,
            num_sensors=self.num_sensors,
            num_targets=self.num_targets,
        )

    def updateInfo(
        self,
        obs: OrderedDict,
        reward: Any,
        termination: Any,
        truncation: Any,
        info: Any,
        action: ndarray[int],
    ) -> float:
        """Count invalid/valid actions.

        Args:
            obs (OrderedDict): Must have action_mask_key in it.
            reward, termination, truncation, info: Unused.
            action (ndarray[int]): A (N,) array of ints where the i-th value is
                the i-th sensor and the value denotes the target number (0 to N-1);
                a value of N denotes null action.

        Returns:
            info[str[int]]: {
                self.new_key: valid/invalid action count (int)
            }
        """
        action_2d = self.action_converter(action)
        action_mask = obs[self.action_mask_key]

        tot = countMaskViolations(
            action=action_2d,
            mask=action_mask,
            count_valid_actions=self.count_valid_actions,
            ignore_null_actions=self.ignore_null_actions,
        )

        info = {self.new_key: tot}

        return info


# %% ThresholdReward
class ThresholdReward(InfoWrapper):
    """Outputs a binary if value in info meets an inequality operation.

    If specified value is <= (by default) the threshold, then True is output.
    Otherwise, False is returned. The inequality is set on instantiation (can be
    <=, >=, <, or >) and does not change.

    If threshold_reward is set, then updateInfo returns a value instead of a bool.
    """

    def __init__(
        self,
        env: Env,
        info_key: str,
        new_key: str,
        threshold: float | int,
        threshold_reward: float | None = None,
        inequality: str = "<=",
    ):
        """Wrap environment with ThresholdReward.

        Args:
            env (Env): A Gymnasium environment.
            info_key (str): Key to item in info to check against threshold.

            threshold (float | int): Threshold to evaluate info[info_key] against.
            threshold_reward (float | None, optional): Reward generated per step
                that threshold evaluates to True. If not set (or set to None),
                output of updateInfo is a bool. Defaults to None.
            inequality (str, optional): String representation of inequality operator
                to use in threshold operation. Must be one of ['<=', '>=', '<', '>'].
                Defaults to "<=".
        """
        super().__init__(env)
        info = getInfo(env)
        assert info[info_key].shape in [
            (),
            (1,),
        ], f"info[{info_key}] must be a singleton dimension."

        self.info_key = info_key
        self.new_key = new_key
        self.threshold_reward = threshold_reward
        # getInequalityFunc checks arg type
        self.inequalityFunc = getInequalityFunc(inequality)
        self.threshold = threshold

    def updateInfo(
        self,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
        action,
    ) -> dict:
        """Append threshold item to info.

        Args:
            observations (_type_): _description_
            rewards (_type_): _description_
            terminations (_type_): _description_
            truncations (_type_): _description_
            infos (_type_): _description_
            action (_type_): _description_

        Returns:
            dict: _description_
        """
        if infos[self.info_key].ndim > 0:
            assert len(infos[self.info_key]) == 1
            val_noarray = infos[self.info_key][0]
        else:
            val_noarray = infos[self.info_key]

        inbounds = self.inequalityFunc(val_noarray, self.threshold)

        if self.threshold_reward is not None:
            # inequalityFunc returns numpy bool, which needs to be compared with "=="
            # instead of "is"
            if inbounds == True:  # noqa
                inbounds = self.threshold_reward
            elif inbounds == False:  # noqa
                inbounds = 0
            else:
                TypeError("inbounds is neither True nor False")

        info = {self.new_key: inbounds}

        return info
