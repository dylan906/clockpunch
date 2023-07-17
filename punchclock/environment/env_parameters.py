"""Environment parameters module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from intervaltree import Interval, IntervalTree
from numpy import arange, asarray, diagonal, eye, ndarray, sqrt, zeros
from numpy.random import normal

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ez_ukf import ezUKF
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter
from punchclock.reward_funcs.generic_reward import GenericReward
from punchclock.reward_funcs.reward_base_class import RewardFunc

# from punchclock.policies.threshold import Threshold
from punchclock.reward_funcs.threshold_v2 import Threshold
from punchclock.schedule_tree.schedule_tree import ScheduleTree
from punchclock.simulation.sim_utils import genInitStates


# %% Class definition
class SSASchedulerParams:
    """Class for parameters to input to SSAScheduler environment.

    Description:
        - Allows user to build an `SSAScheduler` from primitive inputs.
        - An instance of `SSASchedulerParams` can be passed into `SSAScheduler`
            without any other arguments.
        - The dict `agent_params` defines if agent initial conditions are generated
            randomly or fixed. If generated randomly, `agent_params` sets what
            statistical distribution and in which reference frame the initial
            conditions are generated.
        - All initial conditions in a population (sensors or targets) are generated
            in the same way. All sensor ICs must be either probabilistic or fixed,
            and the same for target ICs. Sensor ICs can be generated in a different
            method than target ICs. For example, sensors can have fixed ICs and
            targets can have probabilistically ICs. Another example, sensors can
            have ICs generated in a uniform distribution in the ECI frame, while
            target ICs can be generated with a normal distribution in the COE frame.
        - The dict `filter_params` defines UKFs for all targets. All targets are
            build with an identical filter structure.

    Notation:
        M = number of sensors.
        N = number of targets.

    Attributes:
        - agent_params (`dict`): See __init__ for details.
        - agents (`list`): (N + M)-long list of `Sensor` and `Target` objects. Sensors
            listed first.
        - filter_params (`dict`): See __init__ for details.
        - reward_func (`RewardFunc`): A child of `RewardFunc` base class.
        - horizon (`int`): Number of steps at which simulation resets.
        - time_step (`float`): Time in seconds.
    """

    def __init__(
        self,
        horizon: int,
        agent_params: dict,
        filter_params: dict,
        reward_params: dict,
        time_step: float = 100,
        seed: int = None,
    ):
        """Initialize SSASchedulerParams.

        Args:
            horizon (`int`): Number of steps at which simulation resets.
            agent_params (`dict`): Spatial parameters of agents.
                {
                    "num_sensors": `int`,
                    "num_targets": `int`,
                    "sensor_starting_num": `int`, Defaults to 1000.
                    "target_starting_num": `int`, Defaults to 5000.
                    "sensor_dynamics": 'terrestrial' | 'satellite',
                    "target_dynamics": 'terrestrial' | 'satellite',
                    "sensor_dist": None | 'normal' | 'uniform',  # Enter None
                        if using fixed sensors. Defaults to None.
                    "target_dist": None | 'normal' | 'uniform',  # Enter None
                        if using fixed targets. Defaults to None.
                    "sensor_dist_frame": None | 'ECI' | 'COE' | 'LLA',  # Enter
                        None if using fixed sensors. Defaults to None.
                    "target_dist_frame": None | 'ECI' | 'COE' | 'LLA',  # Enter
                        None if using fixed targets. Defaults to None.
                    "sensor_dist_params": None | `list[list[low, high] *6]`,
                        # Enter None if using fixed sensors. Defaults to None.
                    "target_dist_params": None | `list[list[low, high] *6]`,
                        # Enter None if using fixed targets. Defaults to None.
                    "fixed_sensors": `list[list[float *6] *M]` | None,  # ECI
                        states of fixed sensors. Enter None if using stochastically-
                        generated sensors. Defaults to None.
                    "fixed_targets": `list[list[float *6] *M]` | None,  # ECI
                        states of fixed targets.  Enter None if using stochastically-
                        generated sensors. Defaults to None.
                    "init_num_tasked": `list[int]` (N) | None,  # Number of times
                        targets have been tasked prior to simulation. Defaults
                        to 0 if not entered or if None used.
                    "init_last_time_tasked": `list[float]` (N) | None,  # Absolute
                        time at which targets were previously tasked (sec). Inputs
                        should be negative or 0. Defaults to 0 if not entered or
                        if None used.
                }
            filter_params (`dict`): Parameters of UKF for sensors to use.
                {
                    "Q": `float` | `list[list[float *6] *6]`,  # Process noise
                    "R": `float` | `list[list[float *6] *6]`,  # Measurement noise
                    "p_init": `float` | `list[list[float *6] *6]`,  # recommend
                        setting an order of magnitude greater than R
                }
            reward_params (`dict`): Parameters of a reward function.
                {
                    "reward_func": ("Threshold", "NormalizedMetric") Name of reward
                        function. Must be in predefined list.
                    [other_params]: ([type_varies]),  # Keyword arguments for
                        reward function
                }
            time_step (`float`, optional): Time in seconds. Defaults to 100.
            seed (`int`, optional): Initial conditions RNG seed. Defaults to None.

        Notes:
            - All sensors must have the same dynamics (all sensors are either
                terrestrial or satellite).
            - All targets must have the same dynamics (all sensors are either
                terrestrial or satellite).
            - If "*_dist" is 'normal', each row in "*_dist_params" is [mean, std]
            - If "*_dist" is 'uniform', each row is "*_dist_params" is [low, high]
        """
        # %% Assign agent_params defaults if keys not specified
        if "sensor_starting_num" not in agent_params:
            agent_params["sensor_starting_num"] = 1000
        if "target_starting_num" not in agent_params:
            agent_params["target_starting_num"] = 5000
        if "sensor_dist" not in agent_params:
            agent_params["sensor_dist"] = None
        if "target_dist" not in agent_params:
            agent_params["target_dist"] = None
        if "sensor_dist_frame" not in agent_params:
            agent_params["sensor_dist_frame"] = None
        if "target_dist_frame" not in agent_params:
            agent_params["target_dist_frame"] = None
        if "sensor_dist_params" not in agent_params:
            agent_params["sensor_dist_params"] = None
        if "target_dist_params" not in agent_params:
            agent_params["target_dist_params"] = None
        if "fixed_sensors" not in agent_params:
            agent_params["fixed_sensors"] = None
        if "fixed_targets" not in agent_params:
            agent_params["fixed_targets"] = None
        if "init_num_tasked" not in agent_params:
            agent_params["init_num_tasked"] = None
        if "init_last_time_tasked" not in agent_params:
            agent_params["init_last_time_tasked"] = None

        # %% Argument checks
        # For fixed agents, make sure that the number of agents specified matches
        # the shape of initial conditions arrays.
        if type(agent_params["fixed_sensors"]) is list:
            if (
                len(agent_params["fixed_sensors"])
                != agent_params["num_sensors"]
            ):
                print(
                    "Error: len(agent_params['fixed_sensors']) != ",
                    "agent_params['num_sensors']",
                )
                raise ValueError(
                    "Input dimensions of 'fixed_sensors'!='num_sensors'. If you",
                    "are using fixed agents, ensure they match.",
                )

        if type(agent_params["fixed_targets"]) is list:
            if (
                len(agent_params["fixed_targets"])
                != agent_params["num_targets"]
            ):
                print(
                    "Error: len(agent_params['fixed_targets']) != ",
                    "agent_params['num_targets']",
                )
                raise ValueError(
                    "Input dimensions of 'fixed_targets'!='num_targets'. If you",
                    "are using fixed agents, ensure they match.",
                )

        # Check if distributions were input for stochastic agents but also fixed
        # agents were input (conflicting arguments).
        if (
            (agent_params["sensor_dist"] is not None)
            or (agent_params["sensor_dist_frame"] is not None)
            or (agent_params["sensor_dist_params"] is not None)
        ) and (type(agent_params["fixed_sensors"]) is list):
            print(
                "Error: Conflicting values set for 'sensor_dist*' and 'fixed_sensors'."
            )
            raise ValueError(
                """Argument(s) was/were provided for stochastically-generated sensors,
                but values were also provided for agent_params["fixed_sensors"].
                Resolve by setting agent_params["sensor_dist"],
                agent_params["sensor_dist_frame"], and agent_params["sensor_dist_params"]
                to None or setting agent_params["fixed_sensors"] to None."""
            )

        if (
            (agent_params["target_dist"] is not None)
            or (agent_params["target_dist_frame"] is not None)
            or (agent_params["target_dist_params"] is not None)
        ) and (type(agent_params["fixed_targets"]) is list):
            print(
                "Error: Conflicting values set for 'target_dist*' and 'fixed_targets'."
            )
            raise ValueError(
                """Argument(s) was/were provided for stochastically-generated targets,
                but values were also provided for agent_params['fixed_targets'].
                Resolve by setting agent_params["target_dist"],
                agent_params["target_dist_frame"], and agent_params["target_dist_params"]
                to None or setting agent_params["fixed_targets"] to None."""
            )

        # %% Assign pass-through attributes (uncomplicated stuff)
        self.time_step = time_step
        self.horizon = horizon
        self.agent_params = agent_params
        self.filter_params = filter_params

        # %% Reward function lookup
        # Lookup and build reward function.
        self.reward_func = self.buildRewardFunc(reward_params)
        # %% Build agents
        num_agents = agent_params["num_sensors"] + agent_params["num_targets"]
        # ---Generate initial states---
        # Check for fixed vs stochastic IC generation -- sensors
        if self.agent_params["sensor_dist"] is None:
            # skip initial condition generation if using fixed ICs
            sensor_ICs_array = asarray(self.agent_params["fixed_sensors"])
            # format into list for Agent interface
            sensor_ICs = [a.reshape((6, 1)) for a in sensor_ICs_array]
        else:
            sensor_ICs = genInitStates(
                num_initial_conditions=self.agent_params["num_sensors"],
                dist=self.agent_params["sensor_dist"],
                dist_params=self.agent_params["sensor_dist_params"],
                frame=self.agent_params["sensor_dist_frame"],
                seed=seed,
            )

        # Check for fixed vs stochastic IC generation -- targets
        if self.agent_params["target_dist"] is None:
            # skip initial condition generation if using fixed ICs
            target_ICs_array = asarray(self.agent_params["fixed_targets"])
            # format into list for Agent interface
            target_ICs = [a.reshape((6, 1)) for a in target_ICs_array]
        else:
            target_ICs = genInitStates(
                num_initial_conditions=self.agent_params["num_targets"],
                dist=self.agent_params["target_dist"],
                dist_params=self.agent_params["target_dist_params"],
                frame=self.agent_params["target_dist_frame"],
                seed=seed,
            )

        # ---Set defaults for optional agent parameters---
        # randomly generate num_tasked (if needed)
        if agent_params["init_num_tasked"] is None:
            agent_params["init_num_tasked"] = zeros(
                shape=[agent_params["num_targets"]],
                dtype=int,
            )

        # randomly generate last_time_checked (if needed)
        if agent_params["init_last_time_tasked"] is None:
            agent_params["init_last_time_tasked"] = zeros(
                shape=[agent_params["num_targets"]],
                dtype=float,
            )

        # ---Generate filters---
        # each filter has same basic structure, but with initial estimates close to
        # initial state
        target_filters = self.genFilters(
            target_ICs,
            self.agent_params["target_dynamics"],
            self.filter_params,
        )

        # ---Build lists of agents---
        # get list of sensor parameters
        sensor_params = self.getAgentParams(
            sensor_ICs,
            self.agent_params["sensor_dynamics"],
            self.agent_params["sensor_starting_num"],
            "sensor",
        )
        # build list of sensors
        list_of_sensors = [
            Sensor(**sensor_params[i])
            for i in range(self.agent_params["num_sensors"])
        ]

        # get list of target parameters
        target_params = self.getAgentParams(
            agent_initial_conditions=target_ICs,
            dynamics_type=self.agent_params["target_dynamics"],
            id_start=self.agent_params["target_starting_num"],
            sensor_target="target",
            list_of_filters=target_filters,
            num_tasked=self.agent_params["init_num_tasked"],
            last_time_tasked=self.agent_params["init_last_time_tasked"],
        )
        # build list of targets
        list_of_targets = [
            Target(**target_params[i])
            for i in range(self.agent_params["num_targets"])
        ]

        # ---Calc number of access windows and update agent attributes---
        # deep copy so that access windows propagation doesn't propagate real
        # sensors/targets
        num_windows = self._calcAccessWindows(
            list_of_sensors=deepcopy(list_of_sensors),
            list_of_targets=deepcopy(list_of_targets),
        )
        for targ, window_iter in zip(list_of_targets, num_windows):
            targ.num_windows_left = window_iter

        # combine sensors/targets into one list
        self.agents = list_of_sensors + list_of_targets

    def genFilters(
        self,
        initial_conditions: list[ndarray],
        dynamics_type: str,
        filter_params: dict,
    ) -> list[UnscentedKalmanFilter]:
        """Generate list of UKFs.

        Args:
            initial_conditions (`list[ndarray]`): List of (6,1) arrays of initial
                conditions in ECI frame.
            dynamics_type (`str`): "terrestrial" | "satellite"
            filter_params (`dict`): Values are nested lists for Q, R, and p_init
                matrices. Each outer list is a row, each value within an inner
                list corresponds to a column.
                {
                    "Q": `list[list[float *6] *6]`,
                    "R": `list[list[float *6] *6]`,
                    "p_init": `list[list[float *6] *6]`,
                }

        Returns:
            `list[UnscentedKalmanFilter]`: List of UKFs

        Notes:
            - All filters have identical noise parameters (Q, R, initial P) and
                dynamics. Initial state estimates are centered on true initial
                states, with Gaussian random noise added.
        """
        # Convert filter params to arrays if lists are provided; skip if params
        # are floats.
        if type(filter_params["Q"]) is list:
            filter_params["Q"] = asarray(filter_params["Q"])
        if type(filter_params["R"]) is list:
            filter_params["R"] = asarray(filter_params["R"])
        if type(filter_params["p_init"]) is list:
            filter_params["p_init"] = asarray(filter_params["p_init"])

        # Need initial covariance to generate noisy initial estimates. Convert to array
        # if float was input.
        if type(filter_params["p_init"]) is float or int:
            filter_params["p_init"] = filter_params["p_init"] * eye(6)

        # convert covariance to standard deviation
        std = sqrt(diagonal(filter_params["p_init"]))
        # generate noise centered on initial conditions
        noisy_initial_conditions = [
            (normal(ic.transpose(), std)).transpose()
            for ic in initial_conditions
        ]

        # list comprehension of filters using identical parameters EXCEPT for
        # initial conditions.
        filters = [
            ezUKF(
                {
                    "x_init": x_init,
                    "dynamics_type": dynamics_type,
                    "Q": filter_params["Q"],
                    "R": filter_params["R"],
                    "p_init": filter_params["p_init"],
                }
            )
            for x_init in noisy_initial_conditions
        ]

        return filters

    def getAgentParams(
        self,
        agent_initial_conditions: list[ndarray],
        dynamics_type: str,
        id_start: int,
        sensor_target: str,
        list_of_filters: list[UnscentedKalmanFilter] = None,
        num_tasked: list[int] = None,
        last_time_tasked: list[float] = None,
    ) -> list[dict]:
        """Get parameters to build list of agents.

        Args:
            agent_initial_conditions (`list[ndarray]`): List of (6,1) arrays.
                Length of list is number of agents for which parameters are
                generated.
            dynamics_type (`str`): "terrestrial" | "satellite"
            id_start (`int`): Number at which IDs will start being assigned as
                agent names (goes into Agent.agent_id)
            sensor_target (`str`): "sensor" | "target"
            list_of_filters (`list[UKF]`): Only used if sensor_target =="target".
                Defaults to None.
            num_tasked (`list[int]`): Initial number of times targets have been
                tasked. Not used if sensor_target is "sensor". Defaults to random
                values.
            last_time_tasked (`list[float]`): Initial values of last time targets
                have been tasked. Values should be negative. Not used if sensor_target
                is "sensor". Defaults to random values.

        Returns:
            `list[dict]`: List of parameters for later use in instantiating Agents.
                Dict keys are different for Targets and Sensors. For details
                on contents, see Agent, Sensor, and Target documentation.
                Return length is len(agent_initial_conditions).

        All returned agent parameters are either for Targets or Sensors; mix-
            and-match is not allowed.
        """
        num_agents = len(agent_initial_conditions)

        # generate list of dynamics models (assumes all agents have same dynamics)
        if dynamics_type == "terrestrial":
            agent_dynamics = [StaticTerrestrial() for i in range(num_agents)]
        elif dynamics_type == "satellite":
            agent_dynamics = [SatDynamicsModel() for i in range(num_agents)]

        # generate ID numbers as list for easy zipping
        # convert ints to str for consistency between targets/sensors
        ids = list(map(str, arange(id_start, id_start + num_agents, 1)))

        # Build some parameters differently depending on if target vs sensor
        if sensor_target == "sensor":
            # keys must match argument names of `Sensor` class.
            keys = [
                "dynamics_model",
                "agent_id",
                "init_eci_state",
            ]

            # Prepend sensors with "S" to distinguish from targets
            ids = ["S" + item for item in ids]

            # build list of parameter tuples
            params_tuples = list(
                zip(agent_dynamics, ids, agent_initial_conditions)
            )
        elif sensor_target == "target":
            # keys must match argument names of `Target` class.
            keys = [
                "dynamics_model",
                "agent_id",
                "init_eci_state",
                "filter",
                "init_num_tasked",
                "init_last_time_tasked",
            ]
            # build list of parameter tuples
            params_tuples = list(
                zip(
                    agent_dynamics,
                    ids,
                    agent_initial_conditions,
                    list_of_filters,
                    num_tasked,
                    last_time_tasked,
                )
            )

        return [dict(zip(keys, values)) for values in params_tuples]

    def _calcAccessWindows(
        self,
        list_of_sensors: list[Sensor],
        list_of_targets: list[Target],
        merge_windows: bool = True,
    ) -> list[int]:
        """Calculate access windows for all targets.

        Args:
            list_of_sensors (`list[Sensor]`): List of sensors at initial dynamic
                states.
            list_of_targets (`list[Target]`): List of targets at initial dynamic
                states.
            merge_windows (`bool`, optional): Whether of not to count an interval
                where a target can be seen by multiple sensors as 1 or multiple
                windows. True means that such situations will be counted as 1 window.
                Defaults to True.

        Returns:
            `list[int]`: Number of access windows per target at simulation
                initialization. The order of the list corresponds to order of targets
                in `list_of_targets`.

        Notes:
            - Access windows are defined as discrete events (no duration) set to
                occur at the beginning of intervals specified by
                `SSASchedulerParams.time_step`. If a sensor-target pair are visible
                to each other at `t = i * time_step` (the beginning of the interval),
                then this is counted as an access window. The time duration before
                or after the instant in time the sensor-target pair can see each
                other has no bearing on window count. Examples:
                - An access period of `time_step` duration is counted as one window.
                - An access period of `eps << time_step` spanning `t = i * time_step`
                    is counted as one window.
                - An access period of `eps << time_step` that occurs between
                    `i * time_step < t < (i+1) * time_step` is not counted.
                - An access period of `time_step + eps` starting at `t = time_step`
                    is counted as two windows.
                - An access period of `time_step + eps` starting at
                    `t = time_step + eps` is counted as one window.
            - `merge_windows` should be set to True (the default) when you do not
                want to account for multiple sensors tasked to a single target at
                the same time (the typical case).
        """
        # get list of target ids
        target_ids = [targ.agent_id for targ in list_of_targets]

        # propagate motion in ScheduleTree at 100sec steps, unless horizon time is
        # shorter, in which case pick some fraction of horizon.
        step = min(100, (self.horizon * self.time_step) / 5)
        time_propagate = arange(
            start=0,
            stop=self.horizon * self.time_step,
            step=step,
        )

        # Get access windows
        avail = ScheduleTree(list_of_sensors + list_of_targets, time_propagate)
        avail_tree = avail.sched_tree

        # slice availability tree by simulation time (not time_propagate)
        # time_sim will be same as time_propagate if self.time_step>100
        time_sim = arange(
            start=0,
            stop=self.horizon * self.time_step,
            step=self.time_step,
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
                ival
                for ival in main_ival_list
                if (ival.data["target_id"] == targ)
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

    def buildRewardFunc(
        self,
        params: dict,
    ) -> RewardFunc:
        """Build reward function given parameters.

        Input keys must match keyword arguments of chosen reward function.

        Args:
            params (`dict`): Keys vary depending on the reward function, but at
                minimum must have
                {
                    "reward_func" (`str`): Name of reward function,
                }

        Returns:
            `RewardFunc`: Instance of specified reward function.
        """
        # Add to list of recognized reward functions as more are built.
        reward_func_map = {
            "Threshold": Threshold,
            "GenericReward": GenericReward,
            # NormalizedMetric is for backward compatibility (pre v0.6.0)
            "NormalizedMetric": GenericReward,
        }

        # get reward function name
        name = params["reward_func"]
        # Deepcopy params. Needed for multiple env instantiation.
        params_copy = deepcopy(params)
        # remove "reward_func" from keys so the rest of params can be input via **kwargs
        params_copy.pop("reward_func")

        # Build Reward function.
        reward_func_uncalled = reward_func_map[name]

        reward_func = reward_func_uncalled(**params_copy)
        return reward_func
