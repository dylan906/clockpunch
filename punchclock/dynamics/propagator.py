"""Propagator module."""
# %% Imports
# Standard Library Imports
from typing import Callable

# Third Party Imports
from numpy import ndarray, zeros
from scipy.integrate import solve_ivp


# %% Functions
def simplePropagate(
    func: Callable,
    x0: ndarray,
    t0: float,
    tf: float,
) -> ndarray:
    """Simple propagator.

    Notation:
        N = number of dimensions in state vector (elements in state)
        M = number of state vectors

    Args:
        func (`Callable`): Dynamics function.
        x0 (`ndarray`): (N, ) | (N, M) State initial condition(s). If only propagating
            one state vector, dimensions of input array can be 1D (N,) or 2D (N,1).
        t0 (`float`): Initial time (s)
        tf (`float`): Final time (s)

    Raises:
        ValueError: If `t0`==`tf`
        ValueError: If `x0` is poorly conditioned

    Returns:
        `ndarray`: (N, ) | (N, M) State(s) after propagation. Returns in same shape
            as input.
    """
    # Prevents IVP solver (and simplePropagate) from returning empty list.
    if t0 == tf:
        raise ValueError("Woah buddy, t0 and tf are equal.")

    # squeeze out singleton dimensions (if exist)
    x0 = x0.squeeze()

    if x0.ndim == 1:
        num_vectors = 1
    else:
        num_vectors = x0.shape[1]

    # initialize output
    y = zeros(x0.shape)
    # loop through initial condition vectors
    for i in range(num_vectors):
        if x0.ndim == 2:
            state = x0[:, i].squeeze()
        else:
            state = x0

        sol = solve_ivp(
            func,
            [t0, tf],
            state,
            t_eval=[tf],
            method="DOP853",
            atol=1e-10,
            rtol=1e-10,
        )

        # if sol.y == []:
        if sol.y.size == 0:
            raise ValueError(
                "solve_ivp did not converge, likely due to a bad initial condition."
            )

        if x0.ndim == 2:
            y[:, i] = sol.y.squeeze()
        else:
            y = sol.y.squeeze()

    return y
