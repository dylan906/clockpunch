"""Shortcut for building a UKF."""

# %% Imports
# Third Party Imports
from numpy import eye, ndarray

# Punch Clock Imports
from punchclock.dynamics.dynamics_classes import (
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter


# %% ezUKF
def ezUKF(params: dict) -> UnscentedKalmanFilter:
    """Shortcut function for creating a quick 6d, fully observable UKF.

    Recommended values for Q and R:
        Q = 0.001 * diag([1, 1, 1, 0.01, 0.01, 0.01])
        R = 0.1 * diag([1, 1, 1, 0.01, 0.01, 0.01])

    Args:
        params (dict): Includes the following keys:
        {
            "x_init" (ndarray (6,) ): initial state estimate,
            "p_init" (float | ndarray (6,6) ): initial estimate covariance
            "dynamics_type" (str): 'terrestrial' | 'satellite',
            "Q" (float | ndarray (6,6) ): process noise,
            "R" (float | ndarray (6,6) ): measurement noise,
        }

    Returns:
        UKF: An instance of UKF, see ukf.py for documentation.

    Notes:
        - If "Q", "R", or "p_init" are float or int, a (6, 6) diagonal array is
            generated with the input on the diagonal entries.
        - If using ndarray arguments, "Q", "R", and "p_init" must be
            well-conditioned.
        - Filter is initialized with time=0.
    """
    time = params.get("time", 0)

    if params["dynamics_type"] == "terrestrial":
        dynamics_model = StaticTerrestrial()
    elif params["dynamics_type"] == "satellite":
        dynamics_model = SatDynamicsModel()

    def fullObservable6D(state: ndarray):
        return state

    x_init = params["x_init"]

    Q = params["Q"]
    R = params["R"]
    p_init = params["p_init"]

    if type(Q) is float or int:
        Q = Q * eye(6)

    if type(R) is float or int:
        R = R * eye(6)

    if type(p_init) is float or int:
        p_init = p_init * eye(6)

    ukf = UnscentedKalmanFilter(
        time=time,
        est_x=x_init,
        est_p=p_init,
        dynamics_model=dynamics_model,
        measurement_model=fullObservable6D,
        q_matrix=Q,
        r_matrix=R,
    )

    return ukf
