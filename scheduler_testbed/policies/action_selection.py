"""Action selection functions."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
import sys

# Third Party Imports
from numpy import argmax, asarray, ndarray, ones, zeros
from numpy.random import rand, randint, random

# Punch Clock Imports
from scheduler_testbed.common.math import saturate

# %% Global
MAX_FLOAT = sys.float_info.max


# %% Functions
def epsGreedy(
    Q: ndarray,
    epsilon: float,
) -> ndarray:
    """Epsilon-greedy action selection.

    Set epsilon=0 for pure greedy action selection. Set epsilon=1 for fully random
        selection.

    Args:
        Q (`ndarray`): (A, B) where A is the number of options to choose from,
            and B is the number of times a choice is made.
        epsilon (`float`): Probability of choosing random action, valued 0-1.
            Typically a low value.

    Returns:
        `ndarray`: (B, ) Actions selected. Valued 0 to (A-1).
    """
    num_choices = Q.shape[0]
    num_actions = Q.shape[1]
    # Make sure actions dtype=int, to interface with gym env.
    actions = zeros(num_actions, dtype=int)
    for i in range(num_actions):
        rand_num = rand()
        if rand_num >= epsilon:
            # choose max value of action for each sensor
            actions[i] = argmaxTieBreak(Q[:, i])
        else:
            actions[i] = randint(0, num_choices)

    return actions


def epsGreedyMask(
    Q: ndarray[float],
    epsilon: float,
    mask: ndarray[int] = None,
) -> ndarray[int]:
    """Epsilon-greedy, but random actions are restricted by a mask.

    Args:
        Q (`ndarray[float]`): (A, B) where A is the number of options to choose
            from, and B is the number of times a choice is made.
        epsilon (`float`): Probability of choosing random action, valued 0-1.
            Typically a low value.
        mask (`ndarray[int]`, optional): (A, B) Values are 0 or 1. Elements valued
            as 0 cannot be selected by a random action. Defaults to all 1s (no
            mask).

    Returns:
        `ndarray[int]`: (B, ) Actions selected. Valued 0 to (A-1).
    """
    num_choices = Q.shape[0]
    num_actions = Q.shape[1]
    # Make sure actions dtype=int, to interface with gym env.
    actions = zeros(num_actions, dtype=int)

    # Default mask
    if mask is None:
        mask = ones(shape=(num_choices, num_actions))

    mask_saturate = zeros(Q.shape)
    for i, col in enumerate(mask.T):
        mask_saturate[:, i] = saturate(
            col,
            setpoint=0,
            min_threshold=-MAX_FLOAT,
            max_threshold=0,
        )
    Q_saturate = Q + mask_saturate

    # Loop through value table and choose actions
    for i in range(num_actions):
        rand_num = rand()
        if rand_num >= epsilon:
            actions[i] = argmaxTieBreak(Q_saturate[:, i])
        else:
            actions[i] = argmaxTieBreak(mask_saturate[:, i])

    return actions


def argmaxTieBreak(value_list):
    """A random tie-breaking argmax.

    return the index of the largest value in the supplied list
    - arbitrarily select between the largest values in the case of a tie
    (the standard np.argmax just chooses the first value in the case of a tie)

    Source:
        https://gist.github.com/WhatIThinkAbout/d90154f7099065e0d601f30d0579b5e4
    """
    values = asarray(value_list)
    return argmax(random(values.shape) * (values == values.max()))
