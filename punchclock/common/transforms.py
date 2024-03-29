"""Coordinate transforms."""
# Implementations of eci2coe originally done by Dylan Thomas via RESONAATE
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Optional

# Third Party Imports
from numpy import (
    array,
    concatenate,
    cos,
    cross,
    matmul,
    ndarray,
    pi,
    reshape,
    sin,
    sqrt,
    zeros,
)

# Punch Clock Imports
from punchclock.common.constants import getConstants

# %% Constants
mu = getConstants()["mu"]
RE = getConstants()["earth_radius"]
OMEGA_EARTH = getConstants()["earth_rotation_rate"]


# %% Lat-Lon-Alt -> ECEF
def lla2ecef(x_lla: ndarray) -> ndarray:
    """Converts lat, lon, altitude to state vector in ECEF.

    Args:
        x_lla (ndarray): (3,) Latitude (rad), longitude (rad), altitude (km).

    Returns:
        ndarray: [I, J, K, dI, dJ, dK] state vector (km, km/s)

    Notes:
        - If x_lla has more than 3 elements, the elements after index 2 are ignored.
    """
    lat, lon, alt = x_lla[0], x_lla[1], x_lla[2]
    r_delta = (RE + alt) * cos(lat)
    r_k = (RE + alt) * sin(lat)

    r_ecef = [r_delta * cos(lon), r_delta * sin(lon), r_k, 0, 0, 0]
    return array(r_ecef)


# %% ECEF -> ECI
def ecef2eci(x_ecef: ndarray, JD: float = 0) -> ndarray:
    """Convert an ECEF state vector into an ECI state vector.

    Args:
        x_ecef (ndarray): (6,) or (6, N) vector(s) in ECEF frame (km, km/s)
        JD (float, optional): Time since vernal equinox (since ECEF was aligned
            with ECI). Defaults to 0.

    Returns:
        ndarray: (6,) or (6, N) state vector(s) in ECI frame (km, km/s).  Returns
            (6,) if input is (6,1).

    Notes:
        - Does not account for Earth precession, nutation.
        - Passing in 6 vectors is an edge case which is not flagged as an error
            and outputs incorrect values, so ensure that the 0th dimension of the
            input array is the state vector.
    """
    x_ecef = _check_dimensions(x_ecef)

    # angular rate of Earth (rad/s)
    omega_vec = array([0, 0, OMEGA_EARTH])

    # rotation matrix from ECEF to ECI frame
    R = rot3(OMEGA_EARTH * JD)

    x_eci = zeros(x_ecef.shape)
    for i, vec in enumerate(x_ecef.transpose()):
        x_eci_i = _transport_theorem_posvel(vec, R, omega_vec)
        x_eci[:, i] = x_eci_i

    # convert to singleton dimension if single vector was input
    x_eci = x_eci.squeeze()

    return x_eci


# %% ECI -> ECEF
def eci2ecef(x_eci: ndarray, JD: float) -> ndarray:
    """Convert an ECI state vector into an ECEF state vector.

    Args:
        x_eci (ndarray): (6, N) or (6,) state vector(s) in ECI frame (km, km/s).
            If converting a single state vector, input can be 1D.
        JD (float): Time since vernal equinox (since ECEF was aligned with ECI)

    Returns:
        ndarray: (6, N) or (6,) state vector(s) in ECEF frame (km, km/s).
            Returns (6,) if input is (6,1).

    Notes:
        - Doesn't account for Earth precession, nutation.
        - Passing in 6 vectors is an edge case which is not flagged as an error
            and outputs incorrect values, so ensure that the 0th dimension of the
            input array is the state vector.
    """
    x_eci = _check_dimensions(x_eci)

    # angular rate of Earth (rad/s)
    # Use negative angular rate to convert from ECI to ECEF
    omega_vec = array([0, 0, -OMEGA_EARTH])

    R = rot3(-OMEGA_EARTH * JD)

    x_ecef = zeros(x_eci.shape)
    for i, vec in enumerate(x_eci.transpose()):
        x_ecef_i = _transport_theorem_posvel(vec, R, omega_vec)
        x_ecef[:, i] = x_ecef_i

    # convert to singleton dimension if single vector was input
    x_ecef = x_ecef.squeeze()

    return x_ecef


def _check_dimensions(x: ndarray) -> ndarray:
    """Checks dimensions of input array and returns a (6,N) array."""
    # reshape array if single-dimension
    if x.ndim == 1:
        x = reshape(x, (6, 1))

    # make array (6xN) if passed in as (Nx6)
    # note that this is ambiguous if you want to convert 6 vectors
    if x.shape[0] != 6:
        raise ValueError("Argument must be a (6,N) array.")
    return x


def _transport_theorem_posvel(x_frame0: ndarray, R: ndarray, omega: ndarray) -> ndarray:
    """Apply the transport theorem to a position+velocity state vector.

    This function applies the transport theorem to transform a state vector
    from one frame to another. The transformation
    is performed using a rotation matrix and an angular velocity vector.

    Args:
        x_frame0 (ndarray): The initial position and velocity vectors in
            the original frame. The first three elements represent the
            position vector and the last three elements represent the
            velocity vector.
        R (ndarray): The rotation matrix used to transform the vectors
            from the original frame to the new frame. Shape is (3, 3).
        omega (ndarray): The angular velocity vector of the new frame
            with respect to the original frame. Must have 3 elements.

    Returns:
        ndarray: The transformed position and velocity vectors in the
            new frame.

    Note:
        This function assumes that the input arrays are of appropriate
        sizes: x_frame0 should be a 1-D array of length 6, R should be
        a 2-D square matrix, and omega should be a 1-D array of length 3.
    """
    r0 = x_frame0[:3]
    v0 = x_frame0[3:]

    r1 = matmul(R, r0)

    v1 = matmul(R, v0) + cross(omega, matmul(R, r0))

    x1 = concatenate((r1, v1), axis=0)

    return x1


# %% 3-axis rotations
def rot1(theta: float) -> ndarray:
    """Generates rotation matrix about axis 1.

    Args:
        theta (float): angle (rad)

    Returns:
        ndarray: (3,3) rotation matrix
    """
    R1 = array(
        [
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)],
        ]
    )
    return R1


def rot2(theta: float) -> ndarray:
    """Generates rotation matrix about axis 2.

    Args:
        theta (float): angle (rad)

    Returns:
        ndarray: (3,3) rotation matrix
    """
    R2 = array(
        [
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ]
    )
    return R2


def rot3(theta: float) -> ndarray:
    """Generates rotation matrix about axis 3.

    Args:
        theta (float): angle (rad)

    Returns:
        ndarray: (3,3) rotation matrix
    """
    R3 = array(
        [
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R3


# %% Classical Orbital Elements -> ECI
def coe2eci(
    sma: float,
    ecc: float,
    inc: float,
    raan: float,
    argp: float,
    true_anom: float,
    mu: Optional[float] = mu,
) -> ndarray:
    r"""Convert a set of COEs to an ECI (J2000) position and velocity vector.

    References:
        :cite:t:vallado_2013_astro, Sections 2-6, Pgs 116-120

    Args:
        sma (float): semi-major axis, a (km).
        ecc (float): eccentricity, e[0,1).
        inc (float): inclination angle, i[0,\pi] in radians.
        raan (float): right ascension of the ascending node, \Omega[0,2\pi), in radians.
        argp (float): argument of perigee, \omega[0,2\pi), in radians.
        true_anom (float): true  anomaly (location) angle, f[0,2\pi), in radians.
        mu (float, optional): gravitational parameter of central body (km^3/sec^2).
            Defaults to :attr:.Earth.mu.

    Returns:
        ndarray: 6x1 ECI state vector (km; km/sec).
    """
    # %% Argument Checkers
    if sma < RE:
        raise ValueError("Semi-major axis is less than Earth radius.")
    if ecc < 0 or ecc >= 1:
        raise ValueError("Eccentricity is out of range 0<=ecc<1.")
    if inc < 0 or inc > pi:
        raise ValueError("Inclination is out of range 0<=inc<=pi.")
    if raan < 0 or raan >= (2 * pi):
        raise ValueError("RAAN is out of range 0<=raan<2*pi.")
    if argp < 0 or argp >= (2 * pi):
        raise ValueError("Argument of perigee is out of range 0<=argp<2*pi.")
    if true_anom < 0 or true_anom >= (2 * pi):
        raise ValueError("True anomaly is out of range 0<=true_anom<2*pi.")

    # Save cos(), sin() of anomaly angle, semi-parameter rectum
    cos_anom, sin_anom = cos(true_anom), sin(true_anom)
    p = sma * (1.0 - ecc**2)

    # Define PQW position, velocity vectors
    r_pqw = p / (1.0 + ecc * cos_anom) * array([cos_anom, sin_anom, 0.0])
    v_pqw = sqrt(mu / p) * array([-sin_anom, ecc + cos_anom, 0.0])

    # Define rotation matrix from PQW to ECI
    rot_pqw2eci = rot3(-raan).dot(rot1(-inc).dot(rot3(-argp)))

    return concatenate([rot_pqw2eci.dot(r_pqw), rot_pqw2eci.dot(v_pqw)], axis=0)
