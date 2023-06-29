"""Dynamics module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import concatenate, ndarray, zeros
from numpy.linalg import norm

# Punch Clock Imports
from scheduler_testbed.common.constants import getConstants
from scheduler_testbed.common.transforms import ecef2eci, eci2ecef

# %% Functions


def satelliteDynamics(t: list, x0: ndarray) -> ndarray:
    """1st order ODE satellite dynamics for use in IVP solver.

    Args:
        t (``list``): List of times to propagate motion (seconds). Used if
            satelliteDynamics() is input to an IVP solver, but not
            standalone.
        x0 (``ndarray``): Initial conditions (ECI).

    Returns:
        ``ndarray``: State derivative vector (Xdot).

    Assumes Earth orbiting satellite, 2-body motion.
    """
    const = getConstants()

    r1_ECI = x0[:3]
    v1_ECI = x0[3:6]

    a1_ECI = a2body(r1_ECI, const["mu"])

    # assemble state vector derivative
    xDot = concatenate((v1_ECI, a1_ECI))

    return xDot


def terrestrialDynamicsAlt(x0: ndarray, t0: float, tf: float) -> ndarray:
    """Aligns argument order between `StaticTerrestrial` and `terrestiralDynamics`."""
    return terrestrialDynamics(t=tf, x0=x0, JD=t0)


def terrestrialDynamics(t: list, x0: ndarray, JD: float) -> ndarray:
    """Generates state history for Earth's surface-fixed point.

    Args:
        t (``list``): Times at which to propagate motion (seconds).
        x0 (``ndarray``): [6 x 1] ECI position vector (km and km/s).
        JD (``float``): Julian Date at starting time.

    Returns:
        ``ndarray``: [6 x T] state history, where T is len(t).
    """
    # check for format of t
    if type(t) is not list:
        t = [t]

    x_hist = zeros([6, len(t)])
    x0_ecef = eci2ecef(x0, JD)
    for i, t_step in enumerate(t):
        x_hist[:, i] = ecef2eci(x0_ecef, t_step)

    return x_hist


def a2body(r_eci: ndarray, mu: float) -> ndarray:
    """2-body dynamics.

    Args:
        r_eci (``ndarray``): [3 x 1] ECI position of particle (km).
        mu (``float``): gravitational parameter (km^3/s^2).

    Returns:
        ``ndarray``: [3 x 1] ECI acceleration of particle (km/s^2).
    """
    rMag = norm(r_eci)

    #  acceleration, ECI (km/s^2)
    a2b = -mu * r_eci / (rMag**3)

    return a2b
