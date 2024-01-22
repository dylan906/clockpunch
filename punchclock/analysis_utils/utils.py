"""Analysis utils."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from gymnasium.spaces import flatten
from numpy import (
    all,
    array,
    asarray,
    count_nonzero,
    multiply,
    nan,
    ndarray,
    sum,
    trace,
)
from pandas import DataFrame, read_csv

# Punch Clock Imports
from punchclock.common.utilities import MaskConverter, fromStringArray


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


def checkNonTrivialOpportunities(mask: ndarray[int] | list) -> ndarray[int]:
    """Check which agents had more than one action available.

    Checks if there are more than one 1s in each column of the mask.

    Args:
        mask (ndarray[int] | list): A 2D array or list of lists representing
            the actions available to each agent. Each row corresponds to an
            agent and each column corresponds to an action. A value of 1
            indicates the action is available, and 0 indicates it is not.

    Returns:
        ndarray[int]: Whether or not each column of mask had more than one 1.

    Raises:
        AssertionError: If the input mask does not have exactly two dimensions.
    """
    # convert to array if arg is list
    mask = array(mask)
    assert mask.ndim == 2, "mask must have ndims == 2"

    sum_actions = sum(mask, axis=0)
    num_nontrivial = array([1 if a > 1 else 0 for a in sum_actions])

    return num_nontrivial.astype(int)


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
    else:
        action_mask2d = mask

    action_flat = flatten(mask_converter.action_space, action)
    action_2d = mask_converter.convertActionMaskFrom1dTo2d(action_flat)

    missed_opps = 0
    for col_act, col_mask in zip(action_2d.T, action_mask2d.T):
        action_status = checkSingleWastedAction(col_act, col_mask)
        missed_opps = missed_opps + int(action_status is True)

    return missed_opps


def checkSingleWastedAction(action: ndarray[int], mask: ndarray[int]) -> bool:
    """Check if inaction was chosen over valid active action.

    Args:
        action (ndarray[int]): (A, ) A single sensor's action array (1d array).
            Binary values.
        mask (ndarray[int]): (A, ) A single sensor's action mask (1d array).
            Binary values.

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

    Entires of mask == 1 are valid, entries of mask == 0  are invalid. Any
    instance of a 1 in x where the same indexed value in mask == 0 counts as a
    violation.

    Args:
        x (ndarray[int]): (A, B) Binary array, same size as mask.
        mask (ndarray[int]): (A, B) Binary array, same size as x.

    Returns:
        int: Number of instances in which x(i, j) == 1 and mask(i, j) == 0.
    """
    assert x.shape == mask.shape
    assert all([b in [0, 1] for b in x.flat])
    assert all([b in [0, 1] for b in mask.flat])

    inv_mask = -1 * (mask - 1)
    violations_mat = multiply(x, inv_mask)

    return sum(violations_mat)


def truncateDFColNames(df: DataFrame, char_lim: int = 18) -> DataFrame:
    """Truncate the column names of a DataFrame.

    Useful for printing DFs with long column names.

    Set char_lim to the number of characters to keep in column name.

    Set char_lim to negative to truncate from end instead of beginning.
    """
    if char_lim < 0:
        df_new = df.rename(columns=lambda x: x[char_lim:])
    else:
        df_new = df.rename(columns=lambda x: x[:char_lim])

    return df_new


def dropPartialTrials(df: DataFrame, group: str) -> DataFrame:
    """Remove rows corresponding to incomplete trials from a DataFrame.

    This function identifies and removes rows from the DataFrame that correspond to
    trials which were not completed. A trial is considered incomplete if it was
    restarted before completion, resulting in rows from the partially-complete trial
    being present in the DataFrame. The function operates by grouping the DataFrame
    by a specified column and calculating the difference in 'iterations_since_restore'
    for each group. Rows where this difference is not 1 or NaN are considered to
    be the start of a new trial, and all preceding rows in the same group are dropped.

    Args:
        df (DataFrame): The DataFrame from which to drop rows.
        group (str): The column name to group by when identifying incomplete trials.

    Returns:
        DataFrame: The DataFrame with rows from incomplete trials removed.

    Raises:
        AssertionError: If 'group' is not a column in 'df'.
    """
    assert group in df.columns
    df = deepcopy(df)
    # Need to make all rows have unique index value for diff to work
    df.reset_index(inplace=True)
    df["diff"] = df.groupby(group)["iterations_since_restore"].diff()
    cut_rows = df[~df["diff"].isin([1, nan])]  # cut this row and all before
    for g in cut_rows[group]:
        cut_idx = cut_rows[cut_rows[group] == g].index.values
        for sub_i in cut_idx:  # in case of multiple cut indices
            df_to_drop = df[(df[group] == g) & (df.index < sub_i)]
            df = df.drop(df_to_drop.index)

    # remove temporary columns
    df.drop(["diff", "index"], axis=1, inplace=True)
    # reset index to fill in gaps from dropped rows
    df.reset_index(inplace=True, drop=True)

    return df
