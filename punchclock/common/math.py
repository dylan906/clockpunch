"""Generic math functions module.

[1] D. A. Vallado and W. D. Mcclain, Fundamentals of astrodynamics and applications.
Hawthorne: Microcosm Press, 2013.
"""
# %% Imports
# Standard Library Imports
import warnings
from math import floor, log10

# Third Party Imports
from numpy import (
    amin,
    arccos,
    clip,
    cos,
    cross,
    diag,
    dot,
    exp,
    eye,
    fabs,
    log,
    log2,
    log10,
    matmul,
    ndarray,
    pi,
    real,
    sin,
    spacing,
    sqrt,
    squeeze,
    trace,
    transpose,
    vdot,
)
from numpy.linalg import LinAlgError, cholesky, det, inv, multi_dot, norm
from numpy.random import default_rng
from scipy.linalg import eigvals, svd

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.utilities import fpe_equals

# %% Constants
MU = getConstants()["mu"]
RE = getConstants()["earth_radius"]


# %% Functions
def nearestPD(original_mat: ndarray) -> ndarray:
    r"""Find the nearest positive-definite matrix to input.

    References:
        #. :cite:t:derrico_nearestspd
        #. :cite:t:higham_laa_1988_pd

    Args:
        original_mat (ndarray): original matrix that is not positive definite.

    Returns:
        ndarray: updated matrix, corrected to be positive definite.
    """
    # symmetrize matrix, perform singular value decomposition, compute symmetric
    # polar factor
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
    # The above is different from [1]. It appears that MATLAB's chol Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # numpy cholesky will not. So where [1] uses eps(min_eig) (where eps is Matlab
    # for np.spacing), we use the above definition. CAVEAT: our spacing
    # will be much larger than [1]'s eps(min_eig), since min_eig is usually on
    # the order of 1e-16, and eps(1e-16) is on the order of 1e-34, whereas
    # spacing will, for Gaussian random matrixes of small dimension, be on
    # the order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    identity = eye(original_mat.shape[0])
    k = 1
    while not isPD(sym_pd_mat):
        min_eig = amin(real(eigvals(sym_pd_mat)))
        sym_pd_mat += identity * (-min_eig * k**2 + _spacing)
        k += 1

    return sym_pd_mat


def isPD(matrix: ndarray) -> bool:
    """Determine whether a matrix is positive-definite, via numpy.linalg.cholesky.

    Args:
        matrix (ndarray): input matrix to be checked for positive definiteness.

    Returns:
        bool: whether the given matrix is numerically positive definiteness.
    """
    try:
        cholesky(matrix)
        return True

    except LinAlgError:
        return False


def logistic(
    x: float,
    x0: float = 0.0,
    k: float = 1.0,
    L: float = 1.0,
):
    """Logistic function.

    Args:
        x (float): Input to logistic function.
        x0 (float, optional): Value of x at sigmoid's midpoint. Defaults to 0.0.
        k (float, optional): Steepness parameter. Defaults to 1.0.
        L: (float, optional): Max value of output. Defaults to 1.0.

    Returns:
        float: Output.

    See Wikipedia for logistic function details.
    """
    return L / (1 + exp(-k * (x - x0)))


def saturate(
    values: list,
    setpoint: float,
    min_threshold: float = None,
    max_threshold: float = None,
) -> list:
    """Saturate values to min and/or max thresholds.

    Values are evaluated as: value <= setpoint. Values that satisfy the inequality
    are set to min_threshold; otherwise values are set to max_threshold.

    If a threshold is None, then values will not be saturated in that direction.

    Args:
        values (list): List of values.
        setpoint (float): The point that values will be evaluated against.
        min_threshold (float, optional): Values less than or equal to setpoint
            are set to min_threshold. Defaults to None.
        max_threshold (float, optional): Values greater than setpoint are set
            to max_threshold. Defaults to None.

    Returns:
        list: Saturated values.
    """
    new_values = [None] * len(values)
    for i, val in enumerate(values):
        if val <= setpoint:
            if min_threshold is None:
                new_values[i] = val
            else:
                new_values[i] = min_threshold
        else:
            if max_threshold is None:
                new_values[i] = val
            else:
                new_values[i] = max_threshold

    return new_values


# %% Linear function
def linear(x: float, m: float, b: float) -> float:
    """A simple linear function."""
    return m * x + b


# %% Get common logarithm
def find_exp(number) -> int:
    """Get the common logarithmm of the input."""
    base10 = log10(abs(number))
    return abs(floor(base10))


# %% Get vector normal
def normalVec(a: ndarray) -> ndarray:
    """Get a random unit vector that is normal to input vector."""
    assert a.ndim == 1

    rng = default_rng()

    a_hat = a / norm(a)

    b = rng.uniform(size=(len(a_hat)))
    b_hat = b / norm(b)
    while dot(a_hat, b_hat) == 1:
        # In case randomly-generated b is parallel to a, keep regenerating b until
        # they are not.
        b = rng.uniform(size=(len(a_hat)))
        b_hat = b / norm(b)

    c_hat = cross(a_hat, b_hat)

    return c_hat


# %% KL Divergence of Gaussian distributions
def kldGaussian(
    mu0: ndarray,
    mu1: ndarray,
    sigma0: ndarray,
    sigma1: ndarray,
    bits: bool = False,
) -> float:
    """Kullback-Leibler divergence of two Gaussian distributions.

    Args:
        mu0 (ndarray): Mean of distribution 0.
        mu1 (ndarray): Mean of distribution 1.
        sigma0 (ndarray): Covariance matrix of distribution 0.
        sigma1 (ndarray): Covariance matrix of distribution 1.
        bits (bool, optional): Set to True to return value in units of bits. Otherwise,
            return is in nats. Defaults to False.
    """
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence #noqa
    # NOTE: Returns nan if eitehr covariance matrix is singular.

    if mu0.shape[1] > mu0.shape[0]:
        mu0 = mu0.T
    if mu1.shape[1] > mu1.shape[0]:
        mu1 = mu1.T

    term1 = trace(matmul(inv(sigma1), sigma0))
    term2 = matmul(matmul(transpose(mu1 - mu0), inv(sigma1)), mu1 - mu0)
    term3 = len(mu0)
    term4 = log(det(sigma1) / det(sigma0))

    kld = float(0.5 * (term1 + term2 - term3 + term4))

    if bits is True:
        # get kld in bits vice nats
        kld = kld / log(2)

    return kld


# %% EntropyDiff
def entropyDiff(
    sigma_num: ndarray,
    sigma_den: ndarray,
    logbase: int | str = "e",
) -> float:
    """Calculate difference in entropy between two normal distributions.

    entropy = 0.5 * log(det(sigma_num) / det(sigma_den))

    where the base of the log function determines the units of the output.

    See: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Args:
        sigma_num (ndarray): Variance matrix in numerator.
        sigma_den (ndarray): Variance matrix in denominator.
        logbase (int | str, optional): [10 | 2 | "e"] Which base to use in the
            logarithmic function. Defaults to "e" (units are 'nats').

    Returns:
        float: Entropy
    """
    if (logbase == "e") or (logbase is None):
        logfunc = log
    elif logbase == 10:
        logfunc = log10
    elif logbase == 2:
        logfunc = log2

    warnings.filterwarnings("error")
    try:
        det_num = det(sigma_num)
        det_den = det(sigma_den)
        entropy = 0.5 * logfunc(det_num / det_den)
    except RuntimeWarning as ex:
        print(ex)
        print(f"{sigma_num=}")
        print(f"{sigma_den=}")

    warnings.resetwarnings()
    return entropy


def safeArccos(arg: float) -> float:
    r"""Safely perform an :math:`\arccos{}` calculation.

    It's possible due to floating point error/truncation that a value that should be
    1.0 or -1.0 could end up being e.g. 1.0000000002 or -1.000000002. These values result
    in ``numpy.arccos()`` throwing a domain warning and breaking functionality. ``safeArccos()``
    checks to see if the given ``arg`` is within the proper domain, and if not, will correct
    the value before performing the :math:`\arccos{}` calculation.

    Raises:
        ``ValueError``: if the given ``arg`` is *actually* outside the domain of :math:`\arccos{}`

    Args:
        arg (``ndarray``): value(s) to perform :math:`\arccos{}` calculation on

    Returns:
        (``ndarray``): result of :math:`\arccos{}` calculation on given ``arg``, radians
    """
    # Check for valid arccos domain
    if fabs(arg) <= 1.0:
        return arccos(arg)
    # else
    if not fpe_equals(fabs(arg), 1.0):
        msg = f"`safeArccos()` used on non-truncation/rounding error. Value: {arg}"
        raise ValueError(msg)

    return arccos(clip(arg, -1, 1))
