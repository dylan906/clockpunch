"""Agents class module."""
# %% Imports

# Standard Library Imports
from copy import copy
from typing import Any

# Third Party Imports
from numpy import concatenate, eye
from numpy import float32 as npfloat32
from numpy import int64 as npint64
from numpy import ndarray, pi
from numpy.linalg import norm
from numpy.random import default_rng, multivariate_normal

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.math import getCircOrbitVel, normalVec
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.dynamics.dynamics_classes import (
    DynamicsModel,
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ez_ukf import ezUKF
from punchclock.estimation.filter_base_class import Filter


# %% Base Class
class Agent:
    """Parent class for a generic Agent object."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        init_eci_state: ndarray,
        agent_id: Any = None,
        time: float | int = 0,
    ):
        """Initialize Agent superclass.

        Args:
            dynamics_model (DynamicsModel): Determines how the agent propagates
                states. See DynamicsModel for details.
            init_eci_state (ndarray): (6,) ECI state array (km, km/sec)
                [I, J, K, dI, dJ, dK]
            agent_id (any, optional): Unique identifier. Defaults to None.
            time (float | int, optional): Agent time at start of simulation.
                Defaults to 0.
        """
        assert isinstance(
            init_eci_state, ndarray
        ), "init_eci_state must be a ndarray"

        self.dynamics = dynamics_model
        self.agent_id = agent_id
        self.time = time
        self.eci_state = init_eci_state.reshape([init_eci_state.size, 1])

    def propagate(self, time_next: float):
        """Update dynamic state (ECI position/velocity and time) of agent.

        Args:
            time_next (float): Next time step of simulation (s).
        """
        self.eci_state = self.dynamics.propagate(
            start_state=self.eci_state,
            start_time=self.time,
            end_time=time_next,
        )

        self.time = time_next

    def toDict(self) -> dict:
        """Convert Agent to json-able dict.

        Returns:
            dict: A dict.
        """
        json_dict = {}
        for attr, attr_val in self.__dict__.items():
            if attr in ["dynamics", "target_filter"]:
                # Save dynamics/target_filter as a string (avoid having to json-ify
                # Dynamics/Filter classes).
                json_dict[attr] = str(attr_val.__class__)
            elif isinstance(attr_val, ndarray):
                json_dict[attr] = attr_val.tolist()
            elif isinstance(attr_val, npfloat32):
                json_dict[attr] = float(attr_val)
            elif isinstance(attr_val, npint64):
                json_dict[attr] = int(attr_val)
            else:
                json_dict[attr] = attr_val

        return json_dict


# %% Target Class
class Target(Agent):
    """A subclass of Agent."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        init_eci_state: ndarray,
        target_filter: Filter,
        agent_id: Any = None,
        time: float | int = 0,
        init_num_tasked: int = 0,
        init_last_time_tasked: float = 0,
        num_windows_left: int = 0,
    ):
        """Initialize target subclass of Agent superclass.

        Args:
            dynamics_model (DynamicsModel): Determines how the agent propagates
                states. See DynamicsModel for details.
            init_eci_state (ndarray): (6,) ECI state array (km, km/sec)
                [I, J, K, dI, dJ, dK]
            target_filter (Filter): Filter used for state estimation.
            agent_id (Any, optional): Unique identifier. Defaults to None.
            time (float | int, optional): Agent time at start of simulation.
                Defaults to 0.
            init_num_tasked (int, optional): Initial number of times target has
                been tasked. Defaults to 0.
            init_last_time_tasked (float, optional): Initial time stamp of last
                time target was tasked. Defaults to 0.
            num_windows_left (int, optional): Number of viewing windows of target
                left in simulation. Defaults to 0.

        Derived attribute:
            meas_cov (ndarray): Measurement covariance matrix. Not to be confused
                with observation covariance, which is an attribute of the target_filter.
                Measurement covariance is constant.
        """
        super().__init__(
            dynamics_model=dynamics_model,
            init_eci_state=init_eci_state,
            agent_id=agent_id,
            time=time,
        )
        self.target_filter = target_filter
        self.num_tasked = init_num_tasked
        self.last_time_tasked = npfloat32(init_last_time_tasked)
        self.num_windows_left = num_windows_left

    def updateNonPhysical(
        self,
        task: bool,
    ):
        """Updates non-physical states.

        Arguments:
            task (bool): Set to True if target is tasked, otherwise  set to False

        Notes:
            - If self.num_windows_left was not set (default=None), then it is not
                updated with this function call.
            - If self.num_windows_left is 0, then it will remain as 0.
        """
        # If target is tasked,
        #   - increment num_tasked
        #   - generate a measurement
        if task is True:
            self.num_tasked = self.num_tasked + 1
            # self.last_time_tasked = self.time
            measurement = self.getMeasurement()
        elif task is False:
            measurement = None

        self.target_filter.predictAndUpdate(
            final_time=self.time, measurement=measurement
        )

        # Target.last_time_tasked can have initial value < 0, whereas
        #   Filter.last_measurement_time always initializes as None. After the first
        #   measurement is taken, both values stay equal.
        if self.target_filter.last_measurement_time is not None:
            self.last_time_tasked = npfloat32(
                self.target_filter.last_measurement_time
            )

        if self.num_windows_left is not None:
            if self.num_windows_left == 0:
                pass
            else:
                self.num_windows_left = self.num_windows_left - 1

    def getMeasurement(self) -> ndarray:
        """Measure dynamic state of target.

        Returns:
            ndarray: (6,) state array. Units are km, km/s.
        """
        # multivariate_norm needs to have singleton dimension arguments
        true_state = self.eci_state.squeeze()
        noisy_state = multivariate_normal(
            true_state, self.target_filter.r_matrix
        )

        return noisy_state


# %% Sensor Class


class Sensor(Agent):
    """A subclass of Agent."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        init_eci_state: ndarray,
        agent_id: Any = None,
        time: float | int = 0,
    ):
        """Initialize sensor subclass of Agent superclass.

        Args:
            dynamics_model (DynamicsModel):Determines how the agent propagates
                states. See DynamicsModel for details.
            init_eci_state (ndarray): (6,) ECI state array (km, km/sec)
                [I, J, K, dI, dJ, dK]
            agent_id (Any, optional): Defaults to None.
            time (float | int, optional): Agent time at start of simulation.
                Defaults to 0.
        """
        super().__init__(
            dynamics_model=dynamics_model,
            init_eci_state=init_eci_state,
            agent_id=agent_id,
            time=time,
        )


# %% Random Agent
def buildRandomAgent(
    target_sensor: str = "target",
    dynamics_model: str = "satellite",
    init_eci_state: ndarray[float] = None,
    id: str = None,  # noqa
    time: float | int = 0,
):
    """Make a random Agent.

    This function builds a quick-use agent to use in testing, when you don't care
    about the realism or accuracy of the agent's state, you just need the same
    interfaces.

    Args:
        target_sensor (str, optional): ["target" | "sensor"]. Defaults to
            "target".
        dynamics_model (str, optional): ["satellite" | "terrestrial"]. Defaults
            to "satellite".
        init_eci_state (ndarray[float], optional): ECI state vector. Defaults
            to random.
        id (str, optional): Agent id. Defaults to None.
        time (float | int, optional): Time at initial state. Defaults to 0.
    """
    rng = default_rng()
    dynamics_model_tag = copy(dynamics_model)

    if dynamics_model == "satellite":
        dynamics_model = SatDynamicsModel()
    elif dynamics_model == "terrestrial":
        dynamics_model = StaticTerrestrial()

    if init_eci_state is None:
        init_eci_state = getRandomIC(dynamics_model_tag, time)

    if target_sensor == "sensor":
        agent = Sensor(
            dynamics_model=dynamics_model,
            init_eci_state=init_eci_state,
            agent_id=id,
            time=time,
        )
    elif target_sensor == "target":
        ukf = ezUKF(
            {
                "x_init": init_eci_state,
                "p_init": rng.normal() * eye(6),
                "dynamics_type": dynamics_model_tag,
                "Q": rng.normal() * eye(6),
                "R": rng.normal() * eye(6),
            }
        )
        agent = Target(
            dynamics_model=dynamics_model,
            init_eci_state=init_eci_state,
            target_filter=ukf,
            agent_id=id,
            time=time,
        )

    return agent


def getRandomIC(
    satellite_terrestrial: str = "satellite", time: float = 0
) -> ndarray[float]:
    """Generate a random initial condition.

    Circular satellite orbits or static terrestrial ICs are generated.

    Args:
        satellite_terrestrial (str, optional): ["satellite" | "terrestrial"].
            Defaults to "satellite".
        time (float, optional): JD time (sec).

    Returns:
        ndarray[float]: ECI state vector.
    """
    rng = default_rng()
    RE = getConstants()["earth_radius"]
    if satellite_terrestrial == "satellite":
        lla = rng.uniform([-pi / 2, -pi, RE + 400], [pi / 2, pi, RE + 36000])
        x_ecef = lla2ecef(lla)
        r = x_ecef[:3]
        # Get randomly-pointed velocity vector that is normal to r and magnitude
        # for circular orbit.
        v_mag = getCircOrbitVel(norm(r))
        v_hat = normalVec(r)
        v = v_mag * v_hat

        x = concatenate([r, v])
    elif satellite_terrestrial == "terrestrial":
        # [lat, lon, alt]
        lla = rng.uniform([-pi / 2, -pi, 0], [pi / 2, pi, 0])
        x = ecef2eci(lla2ecef(lla), time)

    return x
