"""Environment parameters module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from numpy import arange, asarray, diag, diagonal, eye, ndarray, pi, sqrt, zeros
from numpy.random import default_rng, normal

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ez_ukf import ezUKF, getRandomParams
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter

# from punchclock.policies.threshold import Threshold
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
            build with an identical target_filter structure.

    Notation:
        M = number of sensors.
        N = number of targets.

    Attributes:
        - agent_params (`dict`): See __init__ for details.
        - agents (`list`): (N + M)-long list of `Sensor` and `Target` objects. Sensors
            listed first.
        - filter_params (`dict`): See __init__ for details.
        - horizon (`int`): Number of steps at which simulation resets.
        - time_step (`float`): Time in seconds.
    """

    def __init__(
        self,
        horizon: int,
        agent_params: dict,
        filter_params: dict, = None,
        time_step: float = 100,
        seed: int = None,
    ):
        """Initialize SSASchedulerParams.

        Args:
            horizon (`int`): Number of steps at which simulation resets.
            agent_params (`dict`): Spatial parameters of agents.
                {
                    "num_sensors": `int`, Defaults to 1.
                    "num_targets": `int`, Defaults to 1.
                    "sensor_starting_num": `int`, Defaults to 1000.
                    "target_starting_num": `int`, Defaults to 5000.
                    "sensor_dynamics": 'terrestrial' | 'satellite', Defaults to 'terrestrial'.
                    "target_dynamics": 'terrestrial' | 'satellite', Defaults to 'satellite'.
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
            filter_params (`dict`, optional): Parameters of UKF for sensors to use.
                {
                    "Q": `float` | `list[list[float *6] *6]`,  # Process noise.
                        Defaults to 0.1 * diag([1, 1, 1, 1e-2, 1e-2, 1e-2]).
                    "R": `float` | `list[list[float *6] *6]`,  # Measurement noise.
                        Defaults to diag([1, 1, 1, 1e-2, 1e-2, 1e-2]).
                    "p_init": `float` | `list[list[float *6] *6]`,  # recommend
                        setting an order of magnitude greater than R
                        Defaults to 10 * diag([1, 1, 1, 1e-2, 1e-2, 1e-2]).
                }
            time_step (`float`, optional): Time in seconds. Defaults to 100.
            seed (`int`, optional): Initial conditions RNG seed. Defaults to None.

        Notes:
            - All sensors must have the same dynamics (all sensors are either
                terrestrial or satellite).
            - All targets must have the same dynamics (all sensors are either
                terrestrial or satellite).
            - If "*_dist" is 'normal', each row in "*_dist_params" is [mean, std]  # noqa
            - If "*_dist" is 'uniform', each row is "*_dist_params" is [low, high]
        """
        # %% Assign agent_params defaults if keys not specified
        agent_params["num_sensors"] = agent_params.get("num_sensors", 1)
        agent_params["num_targets"] = agent_params.get("num_targets", 1)
        agent_params["sensor_starting_num"] = agent_params.get(
            "sensor_starting_num", 1000
        )
        agent_params["target_starting_num"] = agent_params.get(
            "target_starting_num", 5000
        )
        agent_params["sensor_dist"] = agent_params.get("sensor_dist", None)
        agent_params["target_dist"] = agent_params.get("target_dist", None)
        agent_params["sensor_dist_frame"] = agent_params.get(
            "sensor_dist_frame", None
        )
        agent_params["target_dist_frame"] = agent_params.get(
            "target_dist_frame", None
        )
        agent_params["sensor_dist_params"] = agent_params.get(
            "sensor_dist_params", None
        )
        agent_params["target_dist_params"] = agent_params.get(
            "target_dist_params", None
        )
        agent_params["fixed_sensors"] = agent_params.get("fixed_sensors", None)
        agent_params["fixed_targets"] = agent_params.get("fixed_targets", None)
        agent_params["init_num_tasked"] = agent_params.get(
            "init_num_tasked", None
        )
        agent_params["init_last_time_tasked"] = agent_params.get(
            "init_last_time_tasked", None
        )
        agent_params["sensor_dynamics"] = agent_params.get(
            "sensor_dynamics", "terrestrial"
        )
        agent_params["target_dynamics"] = agent_params.get(
            "target_dynamics", "satellite"
        )

        # Default to randomly generated agents if distribution and fixed params are None
        agent_params.update(self.getDefaultAgentDists(agent_params))

        # Get default filter config
        filter_params.update(self.getDefaultFilterConfig(filter_params))

        # %% Argument checks
        # For fixed agents, make sure that the number of agents specified matches
        # the shape of initial conditions arrays.
        if isinstance(agent_params["fixed_sensors"], list):
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

        if isinstance(agent_params["fixed_targets"], list):
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
        ) and (isinstance(agent_params["fixed_sensors"], list)):
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
        ) and (isinstance(agent_params["fixed_targets"], list)):
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

        # %% Build agents
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
        # each target_filter has same basic structure, but with initial estimates
        # close to initial state
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
            filter_params (`dict`): Structure of value is shown below. See ezUKF
                for details.
                {
                    "Q": `list[list[float *6] *6]` | dict,
                    "R": `list[list[float *6] *6]` | dict,
                    "p_init": `list[list[float *6] *6]` | dict,
                }

        Returns:
            `list[UnscentedKalmanFilter]`: List of UKFs

        Notes:
            - All filters have identical noise parameters (Q, R, initial P) and
                dynamics. Initial state estimates are centered on true initial
                states, with Gaussian random noise added.
        """
        # Convert target_filter params to arrays if lists are provided; skip if params
        # are floats.
        if isinstance(filter_params["Q"], list):
            filter_params["Q"] = asarray(filter_params["Q"])
        if isinstance(filter_params["R"], list):
            filter_params["R"] = asarray(filter_params["R"])
        if isinstance(filter_params["p_init"], list):
            filter_params["p_init"] = asarray(filter_params["p_init"])

        # Need initial covariance to generate noisy initial estimates. Convert to array
        # if float was input.
        if isinstance(filter_params["p_init"], (float, int)):
            p_init_eye = filter_params["p_init"] * eye(6)
        elif isinstance(filter_params["p_init"], dict):
            p_init_eye = diag(getRandomParams(**filter_params["p_init"]))
        elif isinstance(filter_params["p_init"], ndarray):
            p_init_eye = filter_params["p_init"]

        # convert covariance to standard deviation
        std = sqrt(diagonal(p_init_eye))
        # generate noise centered on initial conditions
        noisy_initial_conditions = [
            (normal(ic.transpose(), std)).transpose()
            for ic in initial_conditions
        ]

        # list comprehension of filters with noise-injected initial state estimated
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
                "target_filter",
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

    def getDefaultAgentDists(self, config: dict) -> dict:
        """Set default distribution params for agents.

        Targets default to LEO, sensors default to Earth's surface.

        Targets and sensors default to uniform distributions.

        Does not modify sensor/target config if either "x_dist" or "fixed_x" are set.

        Args:
            config (dict): Agent config.

        Returns:
            dict: Potentially modified agent config.
        """
        new_config = deepcopy(config)
        if (config["sensor_dist"] is None) and (
            config["fixed_sensors"] is None
        ):
            # default sensors are uniformly distributed on globe with 0 altitude
            new_config["sensor_dist"] = "uniform"
            new_config["sensor_dist_frame"] = "LLA"
            new_config["sensor_dist_params"] = [
                [-pi / 2, pi / 2],
                [-pi, pi],
                [0, 0],
                [0, 0],  # last 3 entries not used in LLA
                [0, 0],  # ''
                [0, 0],  # ''
            ]
        if (config["target_dist"] is None) and (
            config["fixed_targets"] is None
        ):
            # default targets are uniformly distributed in LEO with 0 eccentricity
            new_config["target_dist"] = "uniform"
            new_config["target_dist_frame"] = "COE"
            new_config["target_dist_params"] = [
                [400, 1000],
                [0, 0],
                [-pi, pi],
                [0, 2 * pi],
                [0, 2 * pi],
                [0, 2 * pi],
            ]

        return new_config

    def getDefaultFilterConfig(self, config: dict) -> dict:
        """Get default Q, R, P for filters if any are not provided.

        Does not modify entries of filter config that are already populated.

        Args:
            config (dict): Can have any combination or none of "Q", "R", "p_init".

        Returns:
            dict: Potentially modified filter config.
        """
        new_config = deepcopy(config)
        noise_vec = [1, 1, 1, 1e-2, 1e-2, 1e-2]
        noise_matrix = diag(noise_vec)
        new_config["Q"] = new_config.get("Q", 0.1 * noise_matrix)
        new_config["R"] = new_config.get("R", 1 * noise_matrix)
        new_config["p_init"] = new_config.get("p_init", 10 * noise_matrix)

        return new_config
