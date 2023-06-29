"""Policy utilities. A module for helper functions used by policy modules."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Tuple

# Third Party Imports
from numpy import (
    append,
    count_nonzero,
    log,
    multiply,
    ndarray,
    ones,
    shape,
    sqrt,
    sum,
    where,
)


# %% Functions
def appendSubsidyRow(
    rewards: ndarray,
    subsidy: float = 0,
):
    """Adds subsidy to rewards matrix to represent reward for inaction.

    Args:
        rewards (`ndarray`): (N, M) rewards matrix, where N is number of
            satellites and M is number of sensors.
        subsidy (`float`, optional): Subsidy amount. Defaults to 0.

    Returns:
        _type_: (N+1, M) subsidized reward matrix.
    """
    return append(rewards, subsidy * ones([1, shape(rewards)[1]]), axis=0)


def upperConfidenceBounds(
    t: float,
    Qt: float,
    Nt: int,
    c: float,
    max_reward: float,
) -> float:
    """Upper Confidence Bounds rule for a single sensor-target pair.

    Args:
        t (`float`): Current time.
        Qt (`float`): Unweighted value of action at time t.
        Nt (`int`): Number of times action has previously been activated.
        c (`float`): (c>0) Exploration parameter.
        max_reward (`float`): Use this value where otherwise Inf would be
            calculated.

    Returns:
        `float`: Value associated with action after UCB weighting.

    Reference: Sutton & Barto (2d Ed.) Sec 2.7.

    If an action has not been tried before (Nt(a)=0) then the value of that
        action is Inf.
    """
    if Nt == 0:
        # set value of actions that have not been sampled to Inf (do this to
        # avoid runtime errors if you let a division by zero pass through)
        value = max_reward
    elif t < 1:
        # if t<1, sqrt(log(t)) will be complex, so just set t=1, which means
        # log(t=1)=0, so value = Qt
        value = Qt
    else:
        # set values to vanilla UCB
        value = Qt + c * sqrt(log(t) / Nt)

    return value


def switchingCost(
    last_actions: ndarray[int],
    actions: ndarray[int],
    switch_cost: float,
):
    """Calculates switching cost for a set of actions.

    Args:
        last_actions (`ndarray[int]`): (N, M) array of 1s/0s
        actions (`ndarray[int]`): (N, M) array of 1s/0s
        switch_cost (`float`): Cost of switching one action

    Returns:
        `ndarray`: (N, M) array of switching cost given new actions

    Always returns negative value because switching is a cost, not a reward.
    """
    return -switch_cost * abs(actions - last_actions)


def multipleAssignmentCost(
    actions: ndarray[int],
    inaction_row: bool,
    cost: float,
) -> Tuple[float, dict]:
    """Computes cost from assigning multiple targets to sensors or vice versa.

    Args:
        actions (`ndarray[int]`): (N, M) | (N+1, M) Values are 0 or 1.
        inaction_row (`bool`): Set to True if extra row is included in `actions`
            for inaction.
        cost (`float`): Cost per multi-assignment (positive).

    Returns:
        penalty (`float`): Total reward incurred from multiple assignments, if
            any.
        penalties_report (`dict`): Itemizes the number of penalties for multiple
            sensor tasking (m on many), multiple target tasking (many on n), and
            total number of penalties.

    Always returns negative reward (aka cost).

    Each sensor or target that is tasked multiple times is penalized only once,
        regardless of how many times over 1 it is assigned. For example, the penalty
        for a sensor being tasked to 2 targets is the same as being tasked to 3
        targets.

    The penalty for a sensor being tasked to multiple targets is the same as being
        tasked to a target and inaction (if an inaction row is used).
    """
    assert cost >= 0, "Cost must be input as positive value."

    # Need to calculate two types of multi-taskings: a sensor tasked multiple times,
    # and a target that has multiple sensors tasked to it.

    # Calc number of taskings per sensor
    sum_rows = sum(actions, axis=0)
    reward_rows = -cost * (sum_rows > 1)
    num_mult_sensor_taskings = count_nonzero(sum_rows > 1)

    # remove inaction row if present
    if inaction_row is True:
        actions_no_subsidy = actions[:-1, :]
    else:
        actions_no_subsidy = actions

    # Calc number of taskings per target
    sum_columns = sum(actions_no_subsidy, axis=1)
    reward_cols = -cost * (sum_columns > 1)
    num_mult_target_taskings = count_nonzero(sum_columns > 1)

    # sum lists separately in case they are uneven length
    penalty = sum(reward_cols) + sum(reward_rows)

    # count total number of penalties and itemize in a dict
    num_penalties = num_mult_sensor_taskings + num_mult_target_taskings
    penalties_report = {
        "num_penalties": num_penalties,
        "num_mult_sensor_taskings": num_mult_sensor_taskings,
        "num_mult_target_taskings": num_mult_target_taskings,
    }

    return (penalty, penalties_report)


def nonVisibleAssignmentCost(
    actions: ndarray[int],
    vis_map: ndarray[int],
    cost: float,
) -> Tuple[float, int]:
    """Compute cost of assigning sensors to non-visible targets.

    Args:
        actions (`ndarray[int]`): (N, M) Values are 0 or 1.
        vis_map (`ndarray[int]`): (N, M) Visibility map. Values are 0 or 1, where
            1 indicates the m-n sensor-target pair are visible to each other.
        cost (`float`): Cost per instance of non-visible assignment. Positive
            value.

    Returns:
        penalties (`float`): Total negative reward incurred from multiple assignments,
            if any. Always negative.
        num_penalties (`int`): Number of non-visible penalties.

    If there is more than a single 1 per column of `actions`, then a penalty is
        applied for each instance of a non-visible assignment, even if the
        assignment is impossible (e.g.: one sensor assigned to multiple targets).
    """
    assert cost >= 0, "Cost must be input as positive value."

    # swap all 0s and 1s in vis map
    vis_map_inverted = where(
        (vis_map == 0) | (vis_map == 1), vis_map ^ 1, vis_map
    )
    penalties_matrix = multiply(actions, (vis_map_inverted))
    num_penalties = count_nonzero(penalties_matrix > 0)
    # penalties = multiply(actions, (-cost * vis_map_inverted))
    scaled_penalties = -cost * penalties_matrix
    penalties = sum(scaled_penalties)

    return (penalties, num_penalties)


def noAssignmentCost(
    actions: ndarray,
    cost: float = 100.0,
) -> float:
    """Computes cost of not assigning sensors to either targets or inaction.

    Args:
        actions (`ndarray`): (N+1, M) Action array of 1s and 0s. Includes inaction row.
        cost (`float`, optional): Cost per non-assignment (non-zero). Defaults to 100.0.

    Returns:
        `float`: Total reward incurred from non-assignments, if any.

    Always returns negative reward (aka cost).

    Total reward = -cost * number_of_non_tasked_sensors.

    Cost is not incurred for sensors tasked to inaction (a 1 in the N+1th row).

    """
    # Calc number of taskings for sensors
    sum_row = sum(actions, axis=0)

    # count number of zero-entries, multiply by cost
    reward_rows = -cost * count_nonzero(sum_row == 0)

    return reward_rows
