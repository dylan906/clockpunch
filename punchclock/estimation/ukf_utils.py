from __future__ import annotations

from enum import IntEnum, auto

# Standard Library Imports
from typing import TYPE_CHECKING

# Third Party Imports
from numpy import (
    amin,
    arctan2,
    cos,
    diag,
    eye,
    fabs,
    fmod,
    pi,
    real,
    remainder,
    sign,
    sin,
    spacing,
)
from numpy.linalg import multi_dot
from scipy.linalg import LinAlgError, cholesky, eigvals, inv, norm, svd

if TYPE_CHECKING:
    # Third Party Imports
    from numpy import ndarray

TWOPI = 2.0 * pi


class IsAngle(IntEnum):
    R"""Enum class to determine if a measurement is an angle or not."""

    NOT_ANGLE = auto()
    R"""``int``: constant representing a measurement that is not an angle."""
    ANGLE_0_2PI = auto()
    R"""``int``: constant representing a measurement that is an angle and is valid :math:`[0, 2\pi)`."""
    ANGLE_NEG_PI_PI = auto()
    R"""``int``: constant representing a measurement that is an angle and is valid :math:`[-\pi, \pi)`."""


VALID_ANGLE_MAP: dict[IsAngle, tuple[float, float]] = {
    IsAngle.ANGLE_0_2PI: (0, 2.0 * pi),
    IsAngle.ANGLE_NEG_PI_PI: (-pi, pi),
}
R"""``dict[:class:.`IsAngle`, tuple[float, float]]``: maps :class:`.IsAngle` enums to their associated bounds."""


VALID_ANGULAR_MEASUREMENTS: tuple[IsAngle, ...] = tuple(VALID_ANGLE_MAP.keys())
R"""``tuple[:class:`.IsAngle`, ...]``: all valid angular measurement types categorized """


def wrapAngleNegPiPi(angle: float) -> float:
    R"""Force angle into range of :math:`(-\pi, \pi]`."""
    # Remainder takes sign of divisor (second arg)
    angle = remainder(angle, TWOPI)
    if fabs(angle) > pi:
        angle -= TWOPI * sign(angle)
    return angle


def wrapAngle2Pi(angle: float) -> float:
    R"""Force angle into range of :math:`[0, 2\pi)`."""
    # Fmod takes sign of dividend (first arg)
    angle = fmod(angle, TWOPI)
    if angle < 0:
        angle += TWOPI
    return angle


def angularMean(
    angles: ndarray,
    weights: ndarray | None = None,
    high: float = TWOPI,
    low: float = 0.0,
) -> float:
    R"""Calculate the (possibly weighted) mean of `angles` within a given range.

    The weighted angular mean is defined as

    .. math::

        \bar{\alpha}=\operatorname{atan2}\left(
            \sum_{j=1}^{n} \sin(\alpha_{j})w_{j}, \sum_{j=1}^{n} \cos(\alpha_{j})w_{j}
        \right)

    where :math:`\alpha_{j}` is a single angular value and :math:`w_{j}` is its corresponding
    weight. If no array of `weights` is provided, then
    :math:`w_{j} = 1 \quad\forall\quad j=1\ldots n`. This is by definition the circular mean.

    See Also:
        https://en.wikipedia.org/wiki/Circular_mean

    Args:
        angles (``ndarray``): Nx1 input array of angles, (radians).
        weights (``ndarray``, optional): Nx1 array of weights to compute a weighted mean. Defaults to
            ``None`` which means the angular values are equally weighted and the _circular_ mean
            is computed.
        high (``float``, optional): high boundary for mean range. Defaults to :math:`2\Pi`.
        low (``float``, optional): low boundary for mean range. Defaults to 0.0.

    Raises:
        :class:`.ShapeError`: raised if computing a weighted mean and the input array and the
            weight array do not have the same length.

    Returns:
        ``float``: the angular mean of the given angles.
    """
    sin_angles = sin((angles - low) * TWOPI / (high - low))
    cos_angles = cos((angles - low) * TWOPI / (high - low))

    if weights is not None:
        if angles.shape != weights.shape:
            msg = f"`angularMean()` wasn't passed arrays with equal length, {angles.shape} != {weights.shape}"
            print(msg)
            raise ValueError(msg)

        # Normalize weights
        weights = weights / norm(weights)
        # Inner product
        sin_mean = sin_angles.dot(weights)
        cos_mean = cos_angles.dot(weights)

    else:
        # Because of arctangent the mean == sum (equal weighting is redundant)
        sin_mean = sin_angles.sum()
        cos_mean = cos_angles.sum()

    # Determine the arctangent of the sine & cosine means, then rescale to [0, 2π]
    result_mean = wrapAngle2Pi(arctan2(sin_mean, cos_mean))
    # Rescale using the low, high values.
    return result_mean * (high - low) / TWOPI + low


def isPD(matrix: ndarray) -> bool:
    """Determine whether a matrix is positive-definite, via ``numpy.linalg.cholesky``.

    Args:
        matrix (``ndarray``): input matrix to be checked for positive definiteness.

    Returns:
        ``bool``: whether the given matrix is numerically positive definiteness.
    """
    try:
        cholesky(matrix)
        return True

    except LinAlgError:
        return False


def nearestPD(original_mat: ndarray) -> ndarray:
    R"""Find the nearest positive-definite matrix to input.

    References:
        #. :cite:t:`derrico_nearestspd`
        #. :cite:t:`higham_laa_1988_pd`

    Args:
        original_mat (``ndarray``): original matrix that is not positive definite.

    Returns:
        ``ndarray``: updated matrix, corrected to be positive definite.
    """
    # symmetrize matrix, perform singular value decomposition, compute symmetric polar factor
    sym_mat = (original_mat + original_mat.T) / 2
    _, singular, right_mat = svd(sym_mat)
    pol_factor = multi_dot((right_mat.T, diag(singular), right_mat))

    # Find A-hat in formula from paper, symmetrize it
    pd_mat = (sym_mat + pol_factor) / 2
    sym_pd_mat = (pd_mat + pd_mat.T) / 2

    # Return if positive-definite
    if isPD(sym_pd_mat):
        return sym_pd_mat

    _spacing = spacing(norm(original_mat))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # numpy cholesky will not. So where [1] uses `eps(min_eig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(min_eig)`, since `min_eig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # the order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    identity = eye(original_mat.shape[0])
    k = 1
    while not isPD(sym_pd_mat):
        min_eig = amin(real(eigvals(sym_pd_mat)))
        sym_pd_mat += identity * (-min_eig * k**2 + _spacing)
        k += 1

    return sym_pd_mat


def chiSquareQuadraticForm(residual: ndarray, covariance: ndarray) -> float:
    R"""Calculate the chi-square quadratic form of an :math:`n-`dimensional` random vector, :math:`x`.

    The chi-square quadratic form is calculated as follows:

    .. math::

        q &= (x - \bar{x})^{T} P^{-1} (x - \bar{x}) \\
        &= \tilde{x}^{T} P^{-1} \tilde{x}

    where :math:`x\in\mathbb{R}^{n}` is the random vector with mean :math:`\bar{x}` and covariance :math:`P`. The
    value :math:`\tilde{x} = (x - \bar{x})` is known as the **residual** vector of :math:`x`. This equation results
    in the scalar random value, :math:`q`, which is said to have a chi-square distribution with :math:`n`
    degrees of freedom. This distribution is typically written as:

    .. math::

        q \sim \chi^{2}_{n}

    with mean and variance:

    .. math::

        E[q] &= n \\
        E[(q - n)^{2}] &= 2n \\

    References:
        #. cite:t:`bar-shalom_2001_estimation`, Section 1.4.7, Eqn 1.4.17-1, Pg 57
        #. cite:t:`crassidis_2012_optest`, Section 4.3 & C.6

    Args:
        residual (``ndarray``): the residual vector of the random vector being tested,
            :math:`\tilde{x} = (x - \bar{x})`.
        covariance (``ndarray``): the covariance associated with the random vector being tested, :math:`P`.

    Returns:
        ``float``: :math:`q`, the chi-square quadratic form of the random vector :math:`x`.
    """
    return residual.T.dot(inv(covariance).dot(residual))


def findNearestPositiveDefiniteMatrix(covariance):
    """Finds the nearest PD matrix of the given covariance.

    This is primarily for numerically stabilizing covariances that become poorly conditioned. This
    function also logs the covariance for before & after the change.

    Args:
        covariance (numpy.ndarray): covariance matrix to be changed to PD

    Returns:
        ``ndarray``: cholesky factorization of the nearest PD matrix
    """
    # Factor the nearest positive definite matrix
    nearest_pd = nearestPD(covariance)
    return cholesky(nearest_pd)
