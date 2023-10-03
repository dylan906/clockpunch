"""Functions for loading and post-processing simulation results."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from gymnasium.spaces import flatten
from numpy import (
    all,
    array,
    asarray,
    count_nonzero,
    max,
    mean,
    multiply,
    ndarray,
    ones,
    sum,
    trace,
    vstack,
)
from pandas import DataFrame, read_csv

# Punch Clock Imports
from punchclock.common.utilities import (
    MaskConverter,
    actionSpace2Array,
    fromStringArray,
)


# %% Functions
def loadSimResults(fname: str, sim_name=None) -> DataFrame:
    """Load a SimResults .csv file as a DataFrame."""
    df = read_csv(fname, index_col=0)
    df = cleanupDF(df)
    return df


def cleanupDF(df):
    """Convert weirdly-formatted data to arrays."""
    for col_name, col_data in df.items():
        if isinstance(df[col_name][0], str):
            df[col_name] = col_data.apply(fromStringArray)
    return df


def countNullActions(x: ndarray, null_action_indx: int):
    """Count the number of instances of null_action_indx in x."""
    return count_nonzero(x == null_action_indx)


def calc3dTr(x: ndarray, pos_or_vel: str = None):
    """Get trace along 3d array.

    Assumes x is shape (A, B, B). Returns (A, ) shaped array.

    Optionally set pos_or_vel to "pos" or "vel" to get trace of positional or velocity
    components of matrix.
    """
    if pos_or_vel == "pos":
        x = x[:, :3, :3]
    elif pos_or_vel == "vel":
        x = x[:, 3:, 3:]
    return trace(x, axis1=1, axis2=2)


def countOpportunities(mask: ndarray[int] | list) -> int:
    """Count number of times agent had an opportunity to take an action.

    No distinction made on type of mask (e.g. visibility) and no accounting for
    inaction vs active action.

    A sensor can have no more than one opportunity per call. If there are multiple
    1s in a column, only 1 is counted.

    Args:
        mask (ndarray[int] | list): (A, M) binary action mask. Can be an
            (A, M) array or A-long list of M-long lists.

    Returns:
        int: Number sensors that had an available action (<=M).
    """
    # convert to array if arg is list
    mask = array(mask)
    assert mask.ndim == 2, "mask must have ndims == 2"

    any_avail = [1 for a in mask.T if 1 in a]

    return int(sum(sum(any_avail)))


def calcMissedOpportunities(
    action: ndarray[int] | list[int],
    mask: ndarray[int],
    mask_converter: MaskConverter,
) -> int:
    """Calculate number of instances of inaction when active actions were available.

    Inaction is defined as a value of `action` that is the max allowed valued. Active
    actions are any other value. Valid `actions` correspond to a 1 in `mask`;
    invalid actions correspond to a 0.

    Does not account for mask violations.

    Notation:
        N = number of targets
        M = number of sensors

    Args:
        action (ndarray[int] | list[int]): (M, ) array valued 0 to N. Values of
            N = inaction. Can be input as list (needed for loading .csv files with
            actions).
        mask (ndarray[int]): [(N, M) | (N + 1, M)] binary array action mask. May
            include inaction row.
        mask_converter (MaskConverter): See MaskConverter for details.

    Returns:
        int: Number of instances agent chose inaction while other actions were
            available (<=M).
    """
    assert mask_converter.action_space.contains(
        action
    ), "action not contained in mask_converter.action_space"
    assert mask.ndim == 2, "mask must be 2d"
    assert mask.shape in [
        (mask_converter.num_targets, mask_converter.num_sensors),
        (mask_converter.num_targets + 1, mask_converter.num_sensors),
    ]

    # convert action to ndarray if passed in as list
    action = asarray(action)

    if mask.shape[0] == mask_converter.num_targets:
        action_mask2d = mask_converter.appendInactionRowToActionMask(mask)

    action_flat = flatten(mask_converter.action_space, action)
    action_2d = mask_converter.convertActionMaskFrom1dTo2d(action_flat)

    missed_opps = 0
    for col_act, col_mask in zip(action_2d.T, action_mask2d.T):
        action_status = actionWasted(col_act, col_mask)
        missed_opps = missed_opps + int(action_status is True)

    return missed_opps


def actionWasted(action: ndarray[int], mask: ndarray[int]) -> bool:
    """Check if inaction was chosen over valid active action.

    Args:
        action (ndarray[int]): A single sensor's action array (1d array). Binary
            values.
        mask (ndarray[int]): A single sensor's action mask (1d array). Binary
            values.

    Returns:
        bool: True if inaction was chosen over valid active action, False otherwise.
    """
    if action[-1] == 1:
        num_possible_actions = count_nonzero(mask)
        if num_possible_actions > 1:
            status = True
        else:
            status = False
    else:
        status = False

    return status


def countMaskViolations(x: ndarray[int], mask: ndarray[int]) -> int:
    """Count number of instances of x violating mask.

    Entires of mask == 1 are allowed, entries of mask == 0  are disallowed. Any
    instance of a 1 in x where the same indexed value in mask == 0 counts as a
    violation.

    Args:
        x (ndarray[int]): Binary array, same size as mask.
        mask (ndarray[int]): Binary array, same size as x.

    Returns:
        int: Number of instances in which x(i, j) == 1 and mask(i, j) == 0.
    """
    assert x.shape == mask.shape
    assert all([b in [0, 1] for b in x.flat])
    assert all([b in [0, 1] for b in mask.flat])

    inv_mask = -1 * (mask - 1)
    violations_mat = multiply(x, inv_mask)

    return sum(violations_mat)


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

    df["cov_tr"] = df["est_cov"].apply(calc3dTr)
    df["pos_cov_tr"] = df["est_cov"].apply(calc3dTr, pos_or_vel="pos")
    df["vel_cov_tr"] = df["est_cov"].apply(calc3dTr, pos_or_vel="vel")
    df["cov_mean"] = df["cov_tr"].apply(mean)
    df["cov_max"] = df["cov_tr"].apply(max)
    df["pos_cov_mean"] = df["pos_cov_tr"].apply(mean)
    df["pos_cov_max"] = df["pos_cov_tr"].apply(max)
    df["vel_cov_mean"] = df["vel_cov_tr"].apply(mean)
    df["vel_cov_max"] = df["vel_cov_tr"].apply(max)

    null_action_indx = df["num_targets"][0]
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
            num_sensors=df["num_sensors"][0],
            num_targets=df["num_targets"][0],
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
    df["est_cov"] = df["est_cov"].apply(array)

    return df
