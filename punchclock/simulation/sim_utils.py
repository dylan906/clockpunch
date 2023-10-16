"""Simulation utility functions."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import array, asarray, ndarray, pi, zeros
from numpy.random import default_rng
from ray.rllib.policy.policy import Policy as RayPolicy

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import coe2eci, ecef2eci, lla2ecef
from punchclock.policies.policy_base_class_v2 import CustomPolicy
from punchclock.policies.policy_builder import buildCustomPolicy
from punchclock.ray.build_ray_policy import buildCustomRayPolicy


# %% Functions
def genInitStates(
    num_initial_conditions: int,
    dist: str,
    dist_params: list[list],
    frame: str,
    seed: int = None,
) -> list[ndarray]:
    """Stochastically generate agent initial states in input frame, output in ECI.

    Args:
        num_initial_conditions (`int`): Number of initial condition arrays
            to generate (N or M).
        dist (`str`): Statistical distribution to use. Accepted arguments are
            ('normal' | 'uniform')
        dist_params (`list[list]`): Distribution parameters in a list of 6
            lists where the first value in each nested list is the first parameter
            of the distribution and the 2nd value is the second parameter.
            Each of the 6 nested lists corresponds to a dimension in the input
            frame state vector.
                - If `frame` == "ECI", each list is [I, J, K, dI, dJ, dK]
                - If `frame` == "LLA", each list is [lat, lon, alt, _, _, _]
                - If `frame` == "COE", each list is [SMA, e, i, RAAN, ARGP, TA]
            The values take on different meanings depending on the value of `dist`.
                - If `dist` == "normal", each nested list is [mean, std]
                - If `dist` == "uniform", each nested list is [low, high]
        frame (`str`): Acceptable arguments are ("ECI"| "COE" | "LLA"). Frame
            in which initial conditions are generated. If "LLA" is used, initial
            velocity is assumed to be 0 in the ECEF frame.
        seed (`int`, optional): RNG seed used to generate statistically distributed
            initial conditions. Set to value (other than None) to generate repeated
            results. Defaults to None.

    Returns:
        list[`ndarray`]: num_initial_conditions-long list of (6,1) arrays
            denoting ECI state of agent.

    Notes:
        - Units of `dist_params` for ECI: [km, km, km, km/s, km/s, km/s]
        - Units of `dist_params` for LLA: [rad, rad, km]
        - Units of `dist_params` for COE: [km, unitless, rad, rad, rad, rad]
        - See numpy.random documentation for details on distributions. This
            function supports only normal and uniform distributions.
        - Velocity arguments are ignored in `dist_params` if `frame` == "LLA"
        - Saturating SMA and eccentricity alters the distribution that initial
            conditions are generated. For SMA, the left tail of the PMF is
            more heavily weighted; for eccentricity, both tails are more heavily
            weighted. This means that the distributions are no longer "uniform"
            or "normal" (depending on input), but will be pretty close in most
            cases. The angular properties' distributions are not altered by
            wrapping because the distributions remain the same in the physical
            space.
    """
    # Set RNG seed. Explicitly assign and print for debugging.
    if seed is None:
        seed = default_rng().integers(9999999)

    print(f"RNG seed = {seed}")
    rng = default_rng(seed)

    # convert list of lists into 2d array
    dist_params = asarray(dist_params)

    # determine which statistical distribution to use and generate arrays
    if dist == "normal":
        ic_array = rng.normal(
            loc=dist_params[:, 0],
            scale=dist_params[:, 1],
            size=[num_initial_conditions, 6],
        )
    elif dist == "uniform":
        ic_array = rng.uniform(
            low=dist_params[:, 0],
            high=dist_params[:, 1],
            size=[num_initial_conditions, 6],
        )

    # convert initial conditions to ECI frame
    if frame == "ECI":
        ics_transformed = ic_array
    if frame == "LLA":
        # Velocities are ignored in the lla2ecef conversion (all agents will have 0
        # ECEF velocity, but non-zero ECI velocity)
        ics_transformed = array(
            [ecef2eci(lla2ecef(vec), JD=0) for vec in ic_array]
        )
    if frame == "COE":
        # wrap angle inputs and saturate SMA and eccentricity
        ic_dict = {
            "sma": ic_array[:, 0],
            "ecc": ic_array[:, 1],
            "inc": ic_array[:, 2],
            "raan": ic_array[:, 3],
            "argp": ic_array[:, 4],
            "ta": ic_array[:, 5],
        }
        ic_array = saturateCOEs(**ic_dict, return_array=True)
        # coe2eci optional 7th input is mu (not inputting here)
        ics_transformed = array([coe2eci(*vec) for vec in ic_array])

    # convert to list of (6,1) arrays for compatibility with Agent class
    ic_list = [a.reshape((6, 1)) for a in ics_transformed]
    return ic_list


def saturateCOEs(
    sma: list[float],
    ecc: list[float],
    inc: list[float],
    raan: list[float],
    argp: list[float],
    ta: list[float],
    RE: float = None,
    return_array: bool = False,
) -> dict | ndarray:
    """Wrap angles and saturate SMA and eccentricity of COEs to fit standard bounds.

    Args:
        sma (`list[float]`): Semi-major axis (km).
        ecc (`list[float]`): Eccentricity (unitless).
        inc (`list[float]`): Inclination (rad).
        raan (`list[float]`): Right ascension of ascending node (rad).
        argp (`list[float]`): Argument of perigee (rad).
        ta (`list[float]`): True anomaly (rad).
        RE (`float`, optional): Earth radius (km). Defaults to repo default value.
        return_array (`bool`, optional): If True, function returns COEs as an array,
            otherwise returns as a dict. Defaults to False.

    Returns:
        `dict | ndarray`: If dict, keys match arguments. If ndarray, rows correspond
            to arg list entries, columns correspond to [sma, ecc, inc, raan, argp,
            ta].

    Notes:
        - Saturates SMA lower bound at SMA >= Earth radius. No upper bound.
        - Saturates eccentricity lower bound at 0<=ecc<=0.99.
        - Inclination wrapped 0<=inc<=pi.
        - RAAN, argument of perigee, true anomaly wrapped at 0<=x<2*pi.
    """
    # Make sure lists are all same length
    it = iter([sma, ecc, inc, raan, argp, ta])
    the_len = len(next(it))
    if not all(len(entry) == the_len for entry in it):
        raise ValueError("not all lists have same length!")

    # Get default value of RE
    if RE is None:
        RE = getConstants()["earth_radius"]

    M = len(sma)

    # Loop through all COEs and saturate. Put COEs in an array be default for easier
    # to read looping.
    treated_coe_arr = zeros([M, 6])
    for i in range(M):
        # SMA
        if sma[i] < RE:
            treated_coe_arr[i, 0] = RE
        else:
            treated_coe_arr[i, 0] = sma[i]
        # eccentricity
        if ecc[i] < 0:
            treated_coe_arr[i, 1] = 0
        elif ecc[i] > 0.99:
            treated_coe_arr[i, 1] = 0.99
        else:
            treated_coe_arr[i, 1] = ecc[i]
        # Wrap angles
        #   inclination
        treated_coe_arr[i, 2] = inc[i] % (pi)
        #   other angles
        for j, x in enumerate([raan, argp, ta], start=3):
            treated_coe_arr[i, j] = x[i] % (2 * pi)

    # Return early if array-type return is set.
    if return_array:
        return treated_coe_arr

    # Make COE dict
    treated_coe_dict = {}
    treated_coe_dict["sma"] = list(treated_coe_arr[:, 0])
    treated_coe_dict["ecc"] = list(treated_coe_arr[:, 1])
    treated_coe_dict["inc"] = list(treated_coe_arr[:, 2])
    treated_coe_dict["raan"] = list(treated_coe_arr[:, 3])
    treated_coe_dict["argp"] = list(treated_coe_arr[:, 4])
    treated_coe_dict["ta"] = list(treated_coe_arr[:, 5])

    return treated_coe_dict


def buildCustomOrRayPolicy(
    config_or_path: dict | str,
) -> CustomPolicy | RayPolicy:
    """Build a custom or Ray policy depending on argument type.

    Args:
        config_or_path (`dict | str`): A CustomPolicy config or a path to a Ray
            checkpoint. See buildCustomRayPolicy and buildCustomPolicy for
            interface details.

    Returns:
        `CustomPolicy | RayPolicy`: A CustomPolicy if arg was a config; a RayPolicy
            if arg was a path.
    """
    assert isinstance(
        config_or_path, (str, dict)
    ), "config_or_path must be a dict or str"

    if isinstance(config_or_path, str):
        policy = buildCustomRayPolicy(config_or_path)
    elif isinstance(config_or_path, dict):
        policy = buildCustomPolicy(config_or_path)

    return policy
