"""Tests for env_utils.py."""
# %% Imports
# Third Party Imports
from numpy import arange, array, sqrt

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.environment.env_utils import (
    forecastVisMap,
    getVisMapEstOrTruth,
)
from punchclock.estimation.ez_ukf import ezUKF

# %% Build agents for testing
sat_dynamics = SatDynamicsModel()
ground_dynamics = StaticTerrestrial()
mu = getConstants()["mu"]
RE = getConstants()["earth_radius"]

dynamics = [
    ground_dynamics,
    ground_dynamics,
    sat_dynamics,
    sat_dynamics,
]

x_inits = [
    ecef2eci(array([RE, 0, 0, 0, 0, 0]), 0),
    ecef2eci(array([0, -RE, 0, 0, 0, 0]), 0),
    array([7000, 0, 0, 0, sqrt(mu / 7000), 0]),
    array([0, 7000, 0, sqrt(mu / 7000), 0, 0]),
]

agents = []
for i, (dyn, x) in enumerate(zip(dynamics, x_inits)):
    if isinstance(dyn, StaticTerrestrial):
        agents.append(Sensor(dyn, i, x))
    else:
        filt = ezUKF(
            {
                "x_init": x,
                "p_init": 100,
                "dynamics_type": "satellite",
                "Q": 1e-3,
                "R": 10,
            }
        )
        agents.append(Target(dyn, i, x, filt))

# %% Test getVisMapEstOrTruth
# Make Target2 state estimate very wrong so that the estimated visibility is different
# from the true visibility.
agents[3].filter.est_x = array([7000, 0, 0, 0, sqrt(mu / 7000), 0])
vis_map = getVisMapEstOrTruth(list_of_agents=agents, truth_flag=True)
print(f"vis_map (true) = \n{vis_map}")
vis_map = getVisMapEstOrTruth(list_of_agents=agents, truth_flag=False)
print(f"vis_map (est) = \n{vis_map}")

# %% Test forecastVisMap
print("\nTest forecastVisMap...")
# Check over range of time steps to see truth vs estimate forecast differentiate
for dt in arange(367, 371):
    print(dt)
    vis_map = forecastVisMap(agents, time_step=dt, estimate=False)
    print(f"    vis_map, truth = \n{vis_map}")
    vis_map = forecastVisMap(agents, time_step=dt, estimate=True)
    print(f"    vis_map, estimate = \n{vis_map}")
