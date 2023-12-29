"""Functions for orbit calculations.

[1] D. A. Vallado and W. D. Mcclain, Fundamentals of astrodynamics and applications.
Hawthorne: Microcosm Press, 2013.
"""
# %% Imports
# Standard Library Imports
from math import isnan

# Third Party Imports
from numpy import cos, dot, ndarray, pi, sin, sqrt, vdot
from numpy.linalg import norm

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.math import safeArccos
from punchclock.common.utilities import fpe_equals

# %% Constants
MU = getConstants()["mu"]
RE = getConstants()["earth_radius"]


# %% Functions
def getCircOrbitVel(r: float, mu: float = MU) -> float:
    """Calculates circular Earth-orbit velocity given radius.

    Args:
        r (float): Circular orbit radius (km)
        mu (float, optional): Gravitational parameter (km^3/s^2). Defaults to MU.

    Returns:
        float: Circular orbit velocity (km/s)
    """
    return sqrt(mu / r)


def getRadialRate(r_vec: ndarray, v_vec: ndarray, mu: float = MU) -> float:
    """Calculate the rate of change of the magnitude of the position vector.

    This function computes the rate at which the magnitude of the position vector
    changes, given the position and velocity vectors and the gravitational parameter.

    Args:
        r_vec (ndarray): Position vector in ECI coordinates (km).
        v_vec (ndarray): Velocity vector in ECI coordinates (km/s).
        mu (float, optional): Gravitational parameter (km^3/s^2). Defaults to
            Earth's mu.

    Returns:
        float: The rate of change of the magnitude of the position vector (km/s).
    """
    r = norm(r_vec)
    v = norm(v_vec)
    e, e_vec = getEccentricity(r_vec, v_vec, mu=mu)
    ta = getTrueAnomaly(r_vec, v_vec, e_unit_vec=e_vec)
    sma = getSemiMajorAxis(r, v, mu=mu)
    n = getMeanMotion(mu, sma)
    ta_dot = getTrueAnomalyRate(n, sma, e, r)

    r_dot = r * ta_dot * e * sin(ta) / (1 + e * cos(ta))
    if isnan(r_dot):
        raise ValueError(
            f"r_dot is NaN! This is probably caused by a 0 velocity vector. {v_vec=}"
        )

    return r_dot


def getTrueAnomaly(r_vec: ndarray, v_vec: ndarray, e_unit_vec: ndarray) -> float:
    """Calculate the true anomaly of an orbit.

    This function calculates the true anomaly of an orbit given the position, velocity,
    and eccentricity vectors.

    Args:
        r_vec (ndarray): The position vector in ECI coordinates (km).
        v_vec (ndarray): The velocity vector in ECI coordinates (km/s).
        e_unit_vec (ndarray): The eccentricity unit vector.

    Returns:
        float: The true anomaly in radians, ranging from 0 to 2*pi.
    """
    # pylint: disable=invalid-name
    anomaly = safeArccos(dot(e_unit_vec, r_vec) / norm(r_vec))
    if isnan(anomaly):
        raise ValueError(f"anomaly is NaN!, {e_unit_vec=}, {r_vec=}")
    ta = fixAngleQuadrant(anomaly, dot(r_vec, v_vec))
    return ta


def getTrueAnomalyRate(n: float, a: float, e: float, r: float) -> float:
    """Calculate true anomaly rate.

    From front cover of [1].

    Args:
        n (float): Mean motion (rad/s).
        a (float): Semi-major axis (km).
        e (float): Eccentricity.
        r (float): Position magnitude (km).

    Returns:
        float: True anomaly rate (rad/s).
    """
    return (n * (a**2) / (r**2)) * sqrt(1 - e**2)


def getMeanMotion(mu: float, sma: float) -> float:
    """Calculate mean motion of an elliptical or circular orbit.

    Not valid for parabolic or hyperbolic orbits.

    From front cover of [1].

    Args:
        mu (float): Gravitational parameter (km^3/s^2).
        sma (float): Semi-major axis (km).

    Returns:
        float: Mean motion (rad/s).
    """
    return sqrt(mu / sma**3)


def getEccentricity(
    r_vec: ndarray, v_vec: ndarray, mu: float = MU
) -> tuple[float, ndarray]:
    """Get the eccentricity magnitude & unit vector from position & velocity vectors.

    References:
        [1] Eqn 2-78

    Args:
        r_vec (ndarray): 3x1 ECI position vector (km).
        v_vec (ndarray): 3x1 ECI velocity vector (km/s).
        mu (float, optional): gravitational parameter of central body (km^3/sec^2).
            Defaults to MU.

    Returns:
        tuple[float, ndarray]: A tuple containing two elements:
            - float: The eccentricity, e.
            - ndarray: The 3x1 eccentricity unit vector.
    """
    r, v = norm(r_vec), norm(v_vec)
    ecc_vector = ((v**2 - mu / r) * r_vec - vdot(r_vec, v_vec) * v_vec) / mu
    ecc = norm(ecc_vector)

    # Normalize the eccentricity vector if orbit is eccentric
    if not fpe_equals(ecc, 0.0):
        return ecc, ecc_vector / ecc

    # else
    return ecc, ecc_vector


def getOrbitalEnergy(r: float, v: float, mu: float = MU) -> float:
    """Get the orbital specific energy from orbital radius & speed.

    Args:
        r (float): Orbital radius (km).
        v (float): Orbital speed (km/s).
        mu (float, optional): Gravitational parameter of central body (km^3/sec^2).
            Defaults to MU.

    Returns:
        float: Orbital specific energy (km^2/s^2).
    """
    return 0.5 * v**2 - mu / r


def getSemiMajorAxis(r: float, v: float, mu: float = MU) -> float:
    """Get the semi-major axis from orbital radius & speed.

    References:
        [1] Eqn 1-21

    Args:
        r (float): Orbital radius (km).
        v (float): Orbital speed (km/s).
        mu (float, optional): Gravitational parameter of central body (km^3/sec^2).
            Defaults to MU.

    Returns:
        float: Semi-major axis (km).
    """
    energy = getOrbitalEnergy(r, v, mu=mu)
    return -0.5 * mu / energy


def fixAngleQuadrant(angle: float, check: float) -> float:
    """Adjusts angle based on quadrant check.

    This function is used when a quadrant check is required on the output
    angular value. If `check` is less than zero, `theta = 2*pi - theta` is
    returned, otherwise `theta` is directly returned. This is due to the
    range of arccos(x) being [0, pi].

    References: [1]

    Args:
        angle (float): Angular value (in radians) to be adjusted.
        check (float): Value used in the check.

    Returns:
        float: Angular value adjusted for the proper quadrant.
    """
    if check < 0.0:
        angle = (2 * pi) - angle

    return angle
