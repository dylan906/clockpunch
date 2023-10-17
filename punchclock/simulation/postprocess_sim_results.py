"""Functions for loading and post-processing simulation results."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import array, max, mean
from pandas import DataFrame

# Punch Clock Imports
from punchclock.analysis_utils.utils import calc3dTr, countNullActions
from punchclock.common.utilities import actionSpace2Array


def addPostProcessedCols(
    df: DataFrame,
    info: dict,
) -> DataFrame:
    """Add columns to a SimResults DataFrame.

    Args:
        df (DataFrame): Results from a simulation.
        info (dict): Supplemental info. Used to generate data not available from
            SimResults. Expected keys: "seed".

    Columns:
        action_array (ndarray[int]): (N+1, M) Binary representation of df['action'].
        cov_tr (ndarray): (N, ) Trace of covariance matrices for all targets
            at that step.
        cov_mean (float): Mean of cov_tr.
        cov_max (float): Max of cov_tr.
        pos_cov_tr (ndarray): (N, ) Trace of positional covariance matrices
            for all targets at that step.
        pos_cov_mean (float): Mean of pos_cov_tr.
        pos_cov_max (float): Max of pos_cov_tr.
        vel_cov_tr (ndarray): (N, ) Trace of velocity covariance matrices for
            all targets at that step.
        vel_cov_mean (float): Mean of vel_cov_tr.
        vel_cov_max (float): Max of vel_cov_tr.
        null_action (int): Number of null actions taken at that step.
        cum_null_action (int): Cumulative version of null_action.
        seed (int | None): Seed used to generate agent initial conditions.

    """
    # Convert nested lists to arrays (if needed)
    df = convertNestedLists2Arrays(df)

    df["cov_tr"] = df["info_est_cov"].apply(calc3dTr)
    df["pos_cov_tr"] = df["info_est_cov"].apply(calc3dTr, pos_or_vel="pos")
    df["vel_cov_tr"] = df["info_est_cov"].apply(calc3dTr, pos_or_vel="vel")
    df["cov_mean"] = df["cov_tr"].apply(mean)
    df["cov_max"] = df["cov_tr"].apply(max)
    df["pos_cov_mean"] = df["pos_cov_tr"].apply(mean)
    df["pos_cov_max"] = df["pos_cov_tr"].apply(max)
    df["vel_cov_mean"] = df["vel_cov_tr"].apply(mean)
    df["vel_cov_max"] = df["vel_cov_tr"].apply(max)

    null_action_indx = df["info_num_targets"][0]
    df["null_action"] = df["action"].apply(
        countNullActions,
        null_action_indx=null_action_indx,
    )
    df["cum_null_action"] = df["null_action"].cumsum()

    # mc = MaskConverter(df["num_targets"][0], df["num_sensors"][0])
    # df["wasted_action"] = df.apply(
    #     lambda x: calcMissedOpportunities(
    #         action=x["action"],
    #         vis_map=x["vis_map_est"],
    #         mask_converter=mc,
    #     ),
    #     axis=1,
    # )
    # df["cum_wasted_action"] = df["wasted_action"].cumsum()

    # df["num_opportunities"] = df.apply(
    #     lambda x: countOpportunities(vis_map=x["vis_map_est"]),
    #     axis=1,
    # )
    # df["cum_opportunities"] = df["num_opportunities"].cumsum()

    df["action_array"] = df.apply(
        lambda x: actionSpace2Array(
            actions=x["action"],
            num_sensors=df["info_num_sensors"][0],
            num_targets=df["info_num_targets"][0],
        ),
        axis=1,
    )

    # df["action_mask_violations"] = df.apply(
    #     lambda x: countMaskViolations(
    #         x=x["action_array"],
    #         mask=x["action_mask"],
    #     ),
    #     axis=1,
    # )

    df["seed"] = info.get("seed", None)

    return df


def convertNestedLists2Arrays(df: DataFrame) -> DataFrame:
    """Convert nested lists to numpy arrays."""
    # Caused by using results directly rather than loading from .csv.
    # Does nothing if entries are already arrays.
    df["info_est_cov"] = df["info_est_cov"].apply(array)

    return df
