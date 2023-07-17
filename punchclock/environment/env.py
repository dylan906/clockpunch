"""Gym environment for scenario."""

# %% Imports
from __future__ import annotations

# Standard Library Imports
from cmath import inf
from collections import OrderedDict
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
from numpy import array, asarray, diagonal, float32, int64, ndarray, ones, zeros

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.common.metrics import TaskingMetricTracker, meanVarUncertainty
from punchclock.common.utilities import actionSpace2Array
from punchclock.environment.env_parameters import SSASchedulerParams
from punchclock.environment.env_utils import getVisMapEstOrTruth


# %% Environment
class SSAScheduler(gym.Env):
    """Class for SSA scheduling environment.

    Description:
        Observation Space: An `OrderedDict` with the following keys:
            {
                "eci_state": `ndarray[float]` (6, N+M),  # Dynamic state of agents
                    at time t. The 0-dim elements are position (km) and velocity
                    (km/s) in ECI frame. For sensors, this is the true state; for
                    targets this is the estimated state.
                "est_cov": `ndarray[float]` (6, N), # Diagonals of target_filter's
                    estimate covariance matrix at time t.
                "vis_map_est": `ndarray[int]` (N, M) # Visibility map at time t.
                    Values are binary (0/1). 1 indicates the sensor-target pair
                    can see each other.
                "obs_staleness": `ndarray[float]` (1, N),  # The difference between
                    the simulation time and the last time the given target was tasked
                    (sec).
                "num_windows_left": `ndarray[int]`(1, N),  # Number of observation # noqa
                    windows left in scenario for given target.
                "num_tasked": `ndarray[int]` (1, N),  # Number of times target
                    has been tasked during simulation.
            }
        Action Space: (M, ) `gym.spaces.MultiDiscrete`. Each entry is valued 0-N.
            All action space instantiations are converted within `SSAScheduler`
            to binary arrays representing sensor-target tasking pairs, a process
            which is opaque to the user, but not necessary to see.
        Info: Use `._getInfo()` to get information which is not available in the
            observation. Returns a dict with the following keys:
            {
                "num_steps": (`int`) Number of steps taken in the simulation so
                    far.
                "time_now": (`float`) Current simulation time (sec).
                "mean_pos_var": (`float`) Mean of estimate position covariance across
                    whole catalog.
                "mean_vel_var": (`float`) Mean of estimate velocity covariance across
                    whole catalog.
                "est_x": (`ndarray[float]`) with shape (6, N+M) State estimates
                    of all agents. Sensor states are truth values. ECI frame (km,
                    km/s)
                "est_cov": (`ndarray[float]`) with shape (N, 6, 6), Estimate covariance
                    matrices for all targets.
                "num_unique_targets_tasked": (`int`) Number of unique targets that
                    have been tasked from the beginning of the simulation until
                    the current step.
                "num_non_vis_taskings_est": (`int`) Number of instances in which
                    a sensor was tasked to an estimated non-visible target.
                "num_non_vis_taskings_truth": (`int`) Number of instances in which
                    a sensor was tasked to a truly non-visible target.
                "num_multiple_taskings": (`int`) Number of instances in which multiple
                    sensors were assigned to a single target.
                "targets_tasked": (`list`) Non-repeating IDs of all targets tasked
                    since simulation started,.
                "true_states": (`ndarray[float]`) (6, N + M) True states of all agents.
                "vis_map_est": (`ndarray[int]`) (N, M) Estimated visibility map.
                "vis_map_truth": (`ndarray[int]`) (N, M) True visibility map.
                "num_windows_left": `list[int]` N-long,  Number of observation
                    windows left in scenario for given target.
                "obs_staleness": `list[float]` N-long,  The difference between
                    the simulation time and the last time the given target was tasked
                    (sec).
                "num_tasked": (`list[int]`) N-long, Number of times each target has
                    been tasked.
            }

    Attributes (not all-inclusive):
        action_space (`gym.spaces.MultiDiscrete`): See Action Space.
        agents (`list[Agent]`): List of `Agent`s (both `Target` and `Sensor` types).
        horizon (`int`): Number of steps in simulation.
        info (`dict`): See Info, above.
        num_agents (`int`): Number of agents in environment (targets and sensors).
        num_sensors (`int`): Number of sensors in environment.
        num_targets (`int`): Number of targets in environment.
        observation_space (`gym.spaces.Box`): See Observation Space.
        reward_func (`RewardFunc`): Reward function.
        reset_params (`SSASchedulerParams`): Parameters used to reset environment.
        sensor_ids (`list`): List of sensor IDs.
        target_ids (`list`): List of target IDs.
        time_step (`float`): Simulation time step (sec).
        tracker (`TaskingMetricTracker`): Used to track metrics in simulation.

    Notation:
        N = number of targets
        M = number of sensors
    """

    # line seems to be necessary for gym interface
    metadata = {"render_modes": [None]}

    def __init__(self, scenario_params: SSASchedulerParams):
        """Initialize SSA scheduling environment.

        Args:
            scenario_params (`SSASchedulerParams`): See `env_parameters.py` for
            documentation.
        """
        # %% Direct copy attributes (simple stuff)
        self.time_step = scenario_params.time_step
        self.horizon = scenario_params.horizon

        # deepcopy list of agents needed for reset() to work properly
        self.agents = deepcopy(scenario_params.agents)
        self.num_targets = scenario_params.agent_params["num_targets"]
        self.num_sensors = scenario_params.agent_params["num_sensors"]
        self.num_agents = len(self.agents)

        # policy parameters
        # self.policy = scenario_params.policy
        self.reward_func = scenario_params.reward_func

        # initialize `info` (filled out in reset())
        self.info = {}
        self.info["num_targets"] = self.num_targets
        self.info["num_sensors"] = self.num_sensors
        self.info["num_agents"] = self.num_agents

        # save backups for reset() (deepcopy needed for reset() to work properly)
        self.reset_params = deepcopy(scenario_params)

        # %% Derived attributes (more complex)
        # lists of sensor and target names
        self.sensor_ids = [
            agent.agent_id for agent in self.agents if type(agent) is Sensor
        ]
        self.target_ids = [
            agent.agent_id for agent in self.agents if type(agent) is Target
        ]

        # Initialize metrics tracker (tracks unique targets tracked, non-visible
        # taskings, and instances of multi-tasking).
        self.tracker = TaskingMetricTracker(self.sensor_ids, self.target_ids)

        # observation space as a Dict-- for human readability
        self.observation_space = Dict(
            {
                "eci_state": Box(
                    low=-inf, high=inf, shape=(6, self.num_agents)
                ),
                "est_cov": Box(low=-inf, high=inf, shape=(6, self.num_targets)),
                "vis_map_est": Box(
                    low=0,
                    high=1,
                    shape=(self.num_targets, self.num_sensors),
                    dtype=int,
                ),
                "obs_staleness": Box(
                    low=0, high=inf, shape=(1, self.num_targets)
                ),
                "num_windows_left": Box(
                    low=0, high=inf, shape=(1, self.num_targets), dtype=int
                ),
                "num_tasked": Box(
                    low=0, high=inf, shape=(1, self.num_targets), dtype=int
                ),
            }
        )
        # observation space as a flat array-- for running with a neural net
        # self.observation_space = self._dict_obs_space

        # action space is a (M,) MultiDiscrete array valued 0-N, where the index
        # of m is the sensor, and the value of m is the action taken. Values 0
        # to (N-1) are all targets; N is inaction.
        self.action_space = MultiDiscrete(
            (self.num_targets + 1) * ones([self.num_sensors])
        )

    def reset(self, *, seed=None, options=None) -> tuple[ndarray, dict]:
        """Reset environment to original state.

        Returns:
            observation (`ndarray`): See `_getObs`.
            info (`dict`): See `_getInfo`.
        """
        # reset tasking metrics tracker
        self.tracker.reset()
        # Reset reward function
        self.reward_func.reset()
        # reset parameters that always start at 0
        self.info["num_steps"] = 0
        self.info["time_now"] = 0.0
        self.info["num_tasked"] = zeros(self.num_targets, dtype=int).tolist()
        # reset parameters associated with tracker
        self.info["targets_tasked"] = self.tracker.targets_tasked
        self.info["num_unique_targets_tasked"] = self.tracker.unique_tasks
        self.info["num_non_vis_taskings_est"] = self.tracker.non_vis_tasked_est
        self.info[
            "num_non_vis_taskings_truth"
        ] = self.tracker.non_vis_tasked_truth
        self.info["non_vis_by_sensor_est"] = self.tracker.non_vis_by_sensor_est
        self.info[
            "non_vis_by_sensor_truth"
        ] = self.tracker.non_vis_by_sensor_truth
        self.info["num_multiple_taskings"] = self.tracker.multiple_taskings

        # deepcopy needed for reset() to work properly
        self.agents = deepcopy(self.reset_params.agents)

        # Update info AFTER resetting agents. This step creates entries in
        # self.info.
        self.updateInfoPostTasking()

        observation = self._getObs()

        # new gymnasium API requires reset() to return an info dict
        return observation, {}

    def _getObs(self) -> OrderedDict:
        """Get observation as dict.

        Returns:
            `OrderedDict`: {
                "eci_state": `ndarray[float]` (6, self.num_agents),  # Dynamic
                    state of agents. The 0-dim elements are position (km) and velocity
                    (km/s) in ECI frame.
                "est_cov": `ndarray[float]` (6, self.num_targets), # Diagonals
                    of target_filter's estimate covariance matrix.
                "vis_map_est": `ndarray[int]` (self.num_targets, self.num_sensors)
                    # Values are binary (0/1). 1 indicates the sensor-target pair
                    # can see each other.
                "obs_staleness": `ndarray[float]` (1, self.num_targets),  # The
                    difference between the simulation time and the last time the
                    given target was tasked (sec).
                "num_windows_left": `ndarray[int]`(1, self.num_targets),  # Number
                    of observation windows left in scenario for given target.
                "num_tasked": `ndarray[int]` (1, self.num_targets),  # Number of
                    times target has been tasked during simulation.
                } # noqa
        """
        # Calculate visibility matrix. Use ._getInfo() to ensure type checks are done.
        info_local = deepcopy(self._getInfo())
        # Get state estimates (and true states of sensors)
        eci_state = deepcopy(info_local["est_x"])

        # Copy from info so we don't need to recalculate vis map
        vis_map_est = deepcopy(info_local["vis_map_est"])

        # Get the diagonal of each covariance matrix and arrange into (6, N) array
        cov_copy = deepcopy(info_local["est_cov"])
        cov_diags = diagonal(cov_copy, axis1=1, axis2=2).T

        obs_staleness_list = info_local["obs_staleness"]

        num_windows_left_list = info_local["num_windows_left"]

        num_tasked_list = deepcopy(info_local["num_tasked"])

        est_cov = cov_diags
        obs_staleness = asarray(obs_staleness_list, dtype=float32)
        obs_staleness = obs_staleness.reshape((1, -1))
        num_windows_left = asarray(num_windows_left_list)
        num_windows_left = num_windows_left.reshape((1, -1))
        num_tasked = asarray(num_tasked_list)
        num_tasked = num_tasked.reshape((1, -1))

        ob_dict = OrderedDict(
            {
                "eci_state": eci_state,
                "est_cov": est_cov,
                "vis_map_est": vis_map_est,
                "obs_staleness": obs_staleness,
                "num_windows_left": num_windows_left,
                "num_tasked": num_tasked,
            }
        )
        return ob_dict

    def _getInfo(self) -> dict:
        """Returns info from environment.

        Returns:
            `dict`: See SSAScheduler class description.
        """
        # Check output types for more complicated outputs
        assert self.info["est_x"].shape == (
            6,
            self.num_agents,
        ), "est_x must have shape (6, N+M)"
        assert self.info["est_cov"].shape == (
            self.num_targets,
            6,
            6,
        ), "est_cov must have shape (N, 6, 6)"
        assert isinstance(
            self.info["est_cov"][0, 0, 0], float32
        ), "Entries of est_cov must be floats"
        assert self.info["true_states"].shape == (
            6,
            self.num_agents,
        ), "true_states shape must be (6, N+M)"
        assert isinstance(
            self.info["true_states"][0, 0], float32
        ), "true_states dtype must be float32"
        assert self.info["vis_map_est"].shape == (
            self.num_targets,
            self.num_sensors,
        ), "vis_map_est shape must be (N, M)"
        assert isinstance(
            self.info["vis_map_est"][0, 0], int64
        ), "vis_map_est entries must be ints"
        assert self.info["vis_map_truth"].shape == (
            self.num_targets,
            self.num_sensors,
        ), "vis_map_truth shape must be (N, M)"
        assert isinstance(
            self.info["vis_map_truth"][0, 0], int64
        ), "vis_map_truth entries must be ints"
        assert (
            len(self.info["num_windows_left"]) == self.num_targets
        ), "num_windows_left must be N-long list"
        assert isinstance(
            self.info["num_windows_left"][0], int
        ), "Entries of num_windows_left must be ints"
        assert (
            len(self.info["num_tasked"]) == self.num_targets
        ), "num_tasked must be N-long list"
        assert isinstance(
            self.info["num_tasked"][0], int64
        ), "Entries of num_tasked must be ints"
        assert (
            len(self.info["obs_staleness"]) == self.num_targets
        ), "obs_staleness must be N-long list"
        assert isinstance(
            self.info["obs_staleness"][0], float
        ), "Entries of obs_staleness must be floats"
        # Deep copy info before returning it to prevent weird pointing issues.
        return deepcopy(self.info)

    def _earnReward(self, actions: ndarray[int]) -> float:
        """Calculates true reward received from environment.

        Args:
            actions (`ndarray[int]`): (M, )

        Returns:
            `float`: Reward earned from environment at single time step.
        """
        reward_earned = self.reward_func.calcNetReward(
            obs=self._getObs(),
            info=self._getInfo(),
            actions=actions,
        )
        return reward_earned

    def updateInfoPreTasking(self, action: ndarray[int]):
        """Update information prior to tasking.

        Updates the following entries in self.info:
            "time_now": (`float`) Current simulation time (sec).
            "num_steps": (`int`) Number of steps taken in the simulation so far.
            "num_unique_targets_tasked": (`int`) Number of unique targets that have
                been tasked from the beginning of the simulation until the current
                step.
            "targets_tasked": (`list`) The target IDs that have been tasked since
                the last reset.
            "num_non_vis_taskings_est": (`int`) The number of times estimated
                non-visible targets have been tasked since the last reset.
            "num_non_vis_taskings_truth": (`int`) The number of times truly non-visible
                targets have been tasked since the last reset.
            "non_vis_by_sensor_est": (`ndarray[int]`) The number of times each
                sensor has been tasked to an estimated non-visible target.
            "non_vis_by_sensor_truth": (`ndarray[int]`) The number of times each
                sensor has been tasked to a truly non-visible target.
            "num_multiple_taskings": (`int`) The number of times multiple tasking
                events (multiple sensors tasked to a single target) have occurred
                since the last reset.

        Args:
            action (`ndarray[int]`): (M,) In format of MultiDiscrete action space.
        """
        # Update environment time; must be done before _taskAgents() so filters update
        # with correct time.
        self.info["time_now"] += self.time_step
        # print(self.info["time_now"])

        # NOTE: Time (info["time_now"]) is updated outside of updateInfo.
        # update number of steps taken
        self.info["num_steps"] += 1

        # Update tasking metrics tracker using current (old) vis_map_truth.
        # print(f"env: vis map est = {self.info['vis_map_est']}")
        # print(f"env: action={action}")
        [
            self.info["num_unique_targets_tasked"],
            self.info["targets_tasked"],
            self.info["num_non_vis_taskings_est"],
            self.info["num_non_vis_taskings_truth"],
            self.info["num_multiple_taskings"],
            self.info["non_vis_by_sensor_est"],
            self.info["non_vis_by_sensor_truth"],
        ] = self.tracker.update(
            actions=action,
            vis_mask_est=self.info["vis_map_est"],
            vis_mask_truth=self.info["vis_map_truth"],
        )

    def updateInfoPostTasking(self):
        """Update information after tasking has been complete.

        Updates the following entries in self.info:
            "est_x": (`ndarray[float]`) with shape (6, N+M) State estimates
                    of all agents. Sensor states are truth values. ECI frame (km,
                    km/s)
            "est_cov": (`ndarray[float]`) with shape (N, 6, 6), Estimate covariance
                matrices for all targets.
            "mean_pos_var": (`float`) Mean of estimate position covariance across
                whole catalog.
            "mean_vel_var": (`float`) Mean of estimate velocity covariance across
                whole catalog.
            "vis_map_est": (`ndarray[int]`) (N, M) Estimated visibility map.
            "vis_map_truth": (`ndarray[int]`) (N, M) True visibility map.
            "num_tasked": (`list[int]`) (N, ) Number of times each target has been
                tasked.
            "true_states": (`ndarray[float]`) (6, N+M) Column i is true ECI state
                of i-th agent.
        """
        self.info["est_x"] = self.getEstStates()

        # Update covariance matrices of targets.
        # NOTE: The diagonals are already recorded in `observation`, but storing
        # in `info` to save off-diagonals.
        covariance_list = [
            agent.target_filter.est_p
            for agent in self.agents
            if type(agent) is Target
        ]
        # "est_cov" is (N, 6, 6) array. Convert to float32 to prevent float64 types
        # when mixed ints/float32s are returned from agents.
        self.info["est_cov"] = asarray(covariance_list, dtype=float32)

        # Update mean covariance.
        # make list of covariances; need to swap Target-axis to interface with
        # meanVarUncertainty.
        # covariance_list = list(self._getObs()["est_cov"].transpose())
        self.info["mean_pos_var"] = meanVarUncertainty(covariance_list)
        self.info["mean_vel_var"] = meanVarUncertainty(
            covariance_list, pos_vel="velocity"
        )

        # Get true states
        true_states_list = [agent.eci_state.squeeze() for agent in self.agents]
        self.info["true_states"] = array(true_states_list, dtype=float32).T

        # Get number of times each target has been tasked.
        self.info["num_tasked"] = [
            ag.num_tasked for ag in self.agents if isinstance(ag, Target)
        ]

        # Get number of visibility windows left
        self.info["num_windows_left"] = [
            agent.num_windows_left
            for agent in self.agents
            if type(agent) is Target
        ]

        # Calc observation staleness. Force convert to float to avoid np.float64
        # types.
        self.info["obs_staleness"] = [
            float(self.info["time_now"] - agent.last_time_tasked)
            for agent in self.agents
            if type(agent) is Target
        ]

        # Update visibility maps for next step.
        self.info["vis_map_truth"] = getVisMapEstOrTruth(
            list_of_agents=self.agents,
            truth_flag=True,
        )
        self.info["vis_map_est"] = getVisMapEstOrTruth(
            list_of_agents=self.agents,
            truth_flag=False,
        )

    def _taskAgents(
        self,
        actions_array: ndarray[int],
    ) -> None:
        """Tasks agents according to their rows in `actions_array`.

        Args:
            actions_array (`ndarray[int]`): (N, M) array where N=number of targets and
                M=number of sensors. Values are 0 or 1, where 1 indicates a sensor/target
                pair action.

        Target will be tasked the same if there are multiple 1s in a row or a single
            1 in a row. Meaning, the non-physical states of the agent (ie: target_filter
            estimates) will evolve the same whether one or multiple sensors are
            tasked to the target.
        """
        # pull out targets from list_of_agents
        list_of_targets = [x for x in self.agents if type(x) is Target]

        # Loop through actions rows and task target if there is a 1 in its row.
        # Note: Logic doesn't change if there are multiple 1s in a row (multiple
        # sensors looking at one target).
        # Nonphysical states will update as if a single 1 were in the row.
        for i, row in enumerate(actions_array):
            # print(row)
            if 1 in row:
                # print(f"tasking target {list_of_targets[i].agent_id}")
                list_of_targets[i].updateNonPhysical(task=True)
            else:
                # print(f"Non-tasking target {list_of_targets[i].agent_id}")
                # if not tasked, still update non-physical states
                list_of_targets[i].updateNonPhysical(task=False)

    def _propagateAgents(self, time):
        """Propagate all agents forward to time."""
        for agent in self.agents:
            agent.propagate(time)

    def getEstStates(self) -> ndarray[float]:
        """Get agent estimated states.

        For sensors, estimates are true.

        Returns:
            `ndarray[float]`: (6, N+M) Each column is a state (estimate) in ECI
                frame (km, km/s).
        """
        # %% Get all states as list
        eci_state_list = []
        # for targets get estimated state; for sensors get true state
        for agent in self.agents:
            if type(agent) is Target:
                state = agent.target_filter.est_x
            elif type(agent) is Sensor:
                state = agent.eci_state
            eci_state_list.append(state)

        # convert to array for compatibility with observation space
        eci_state = asarray(eci_state_list, dtype=float32).squeeze().transpose()

        return eci_state

    # %% Step
    def step(
        self,
        action: ndarray[int],
    ) -> tuple[OrderedDict, float, bool, dict]:
        """Step environment forward given actions.

        Args:
            action (`ndarray[int]`): See `__init__` for format details.

        Returns:
            observation (`OrderedDict`): See `_getObs`
            reward (`float`): See `_earnReward`
            done (`bool`): True if `num_steps => max_steps`, False otherwise.
            info (`dict`): See `_getInfo`

        Notation:
            N = number of targets
            M = number of sensors
        """
        # Info is updated in 2 steps: before and after tasking. Pre-tasking updates
        # the attributes that require the current state and action. Post-tasking
        # updates the attributes that are required to output to the agent to make
        # a next step decision.
        self.updateInfoPreTasking(action)

        # Propagate dynamic states; Do this AFTER updateInfoPreTasking b/c time_now
        # needs to be current.
        # NOTE: doesn't modify filters
        self._propagateAgents(self.info["time_now"])

        # convert actions from action space format to to ndarray
        action_array = actionSpace2Array(
            action,
            self.num_sensors,
            self.num_targets,
        )

        # Don't pass in bottom row of action array (inaction row)
        self._taskAgents(action_array[:-1, :])

        # earn reward; deepcopy so that values don't change from reset on final iteration
        reward = deepcopy(self._earnReward(action))

        # Update info dependent on forecasted states
        self.updateInfoPostTasking()

        # Deepcopy so that values don't change from reset on final iteration.
        # NOTE: Observation and info should be fetched together sot that the estimated
        # state in obs is synched with the true state in info.
        observation = deepcopy(self._getObs())
        info = deepcopy(self._getInfo())

        # ---for Gym 0.22---
        # if using >0.22 Gym, need to specify `truncated` and `terminated` variables
        # (instead of just `done`)
        if self.info["num_steps"] == self.horizon:
            done = True
            print("Horizon reached, resetting...")
            self.reset()
        else:
            done = False

        truncated = False

        return observation, reward, done, truncated, info
