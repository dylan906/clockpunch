"""Shortcut for building a UKF."""

# %% Imports
# Third Party Imports
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

    QRP = {k: v for k, v in params.items() if k in ["Q", "R", "p_init"]}
    derived_params = {
        "Q": None,
        "R": None,
        "p_init": None,
    }
    for k, v in QRP.items():
        if isinstance(v, dict):
            assert "dist" in v
            assert "params" in v
            assert len(v["params"]) == 2
            assert len(v["params"][0]) == 6
            assert len(v["params"][1]) == 6

            rand_diags = getRandomParams(**v)
            derived_params[k] = diag(rand_diags)
        elif isinstance(v, (float, int)):
            derived_params[k] = v * eye(6)
        elif isinstance(v, ndarray):
            assert v.shape == (6, 6)
            derived_params[k] = v

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


def getRandomParams(
    dist: str, params: list[list, list], seed: int = None
) -> ndarray:
    """Generate random numbers from a specified distribution.

    Args:
        dist (str): ["uniform" | "normal"]
        params (list[list, list]): Sub-lists contain parameters to input to distribution
            command. Lengths of sub-lists must be equal. See numpy.random for
            details.
        seed (int, optional): RNG seed. Defaults to None.

    Returns:
        ndarray: (N, ) Has length N = len(params[0])
    """
    rng = default_rng(seed=seed)
    if dist == "uniform":
        rngFunc = rng.uniform
    elif dist == "normal":
        rngFunc = rng.normal
    rand_nums = rngFunc(*params)
    return rand_nums
