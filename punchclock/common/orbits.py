"""Functions for orbit calculations.

[1] D. A. Vallado and W. D. Mcclain, Fundamentals of astrodynamics and applications.
Hawthorne: Microcosm Press, 2013.
"""
# %% Imports
# Third Party Imports
from numpy import arccos, cos, dot, ndarray, pi, sin, sqrt, vdot
from numpy.linalg import norm

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.utilities import fpe_equals

# %% Constants
MU = getConstants()["mu"]
RE = getConstants()["earth_radius"]


# def getRadialRate(r: float, ta: float, ta_dot: float, e: float, sma: float) -> float:
def getRadialRate(r_vec: ndarray, v_vec: ndarray, mu: float = MU) -> float:
    """Calculate rate of change of position magnitude.

    From front cover of [1].

    Args:
        r (float): Position magnitude (km).
        ta (float): True anomaly (rad).
        ta_dot (float): True anomaly rate (rad/s).
        e (float): Eccentricity.
        sma (float): Semi-major axis (km).

    Returns:
        float: Rate of change of position magnitude (km/s).
    """
    r = norm(r_vec)
    v = norm(v_vec)
    e, e_vec = getEccentricity(r_vec, v_vec, mu=mu)
    ta = getTrueAnomaly(r_vec, v_vec, e_unit_vec=e_vec)
    sma = getSemiMajorAxis(r, v, mu=mu)
    n = getMeanMotion(mu, sma)
    ta_dot = getTrueAnomalyRate(n, sma, e, r)

    return r * ta_dot * e * sin(ta) / (1 + e * cos(ta))


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
    anomaly = arccos(dot(e_unit_vec, r_vec) / norm(r_vec))
    return fixAngleQuadrant(anomaly, dot(r_vec, v_vec))


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
    """Called from within functions that require a quadrant check on the output angular value.

    If check is less than zero, theta = 2*pi - theta is returned, otherwise
    theta is directly returned.

    This occurs due to directly using arccos(x) because its range is [0, pi].

    References:
        [1] Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).
            Hawthorne, CA: Microcosm Press.

    Args:
        angle (float): The angular value (in radians) that is to be (potentially) adjusted.
        check (float): The value used in the check.

    Returns:
        float: Angular value adjusted for the proper quadrant.
    """
    if check < 0.0:
        angle = (2 * pi) - angle

    return angle
