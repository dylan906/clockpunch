"""Shortcut for building a UKF."""

# %% Imports
# Third Party Imports
import numpy.random
from numpy import diag, eye, ndarray
from numpy.random import default_rng

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
            "p_init" (float | ndarray (6,6) | dict ): initial estimate covariance
            "dynamics_type" (str): 'terrestrial' | 'satellite',
            "Q" (float | ndarray (6,6) | dict ): process noise,
            "R" (float | ndarray (6,6) | dict ): measurement noise,
        }

    Returns:
        UKF: An instance of UKF, see ukf.py for documentation.

    Notes:
        - If "Q", "R", or "p_init" are float or int, a (6, 6) diagonal array is
            generated with the input on the diagonal entries.
        - If using ndarray args, "Q", "R", and "p_init" must be
            well-conditioned.
        - If using dict args, format like:
            {
                "dist" (str): ["uniform", "normal"] Random distribution
                "params" (list): Params to use in distribution constructor
                "seed" (int, optional): Seed to use in random distribution. Defaults
                    to None.
            }
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

    if isinstance(Q, (float, int)):
        Q = Q * eye(6)

    if isinstance(R, (float, int)):
        R = R * eye(6)

    if isinstance(p_init, (float, int)):
        p_init = p_init * eye(6)

    params_subset = {
        k: v for k, v in params.items() if k in ["Q", "R", "p_init"]
    }
    derived_params = {
        "Q": Q,
        "R": R,
        "p_init": p_init,
    }
    for k, v in params_subset.items():
        if isinstance(v, dict):
            rng = default_rng(seed=v.get("seed", None))
            if v["dist"] == "uniform":
                rngFunc = rng.uniform
            elif v["dist"] == "normal":
                rngFunc = rng.normal
            rand_nums = rngFunc(*v["params"])
            derived_params[k] = diag(rand_nums)

    ukf = UnscentedKalmanFilter(
        time=time,
        est_x=x_init,
        est_p=derived_params["p_init"],
        dynamics_model=dynamics_model,
        measurement_model=fullObservable6D,
        q_matrix=derived_params["Q"],
        r_matrix=derived_params["R"],
    )

    return ukf
