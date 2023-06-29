"""Custom Gym utilities."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import argmax, int64, ndarray, zeros


# %% Chunker
def chunker(seq, size):
    """Divide a sequence into chunks of a given size."""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


# %% Convert Box action space to MultiDiscrete
def boxActions2MultiDiscrete(
    seq: ndarray[float],
    num_actions: int,
) -> ndarray[int]:
    """Convert actions in a `Box` space to actions in a `MultiDiscrete` space.

    Args:
        seq (`ndarray[float]`): (num_samples, (num_agents * num_actions)) or
            ((num_agents * num_actions), ) if only 1 sample.
        num_actions (`int`): Number of actions each agent can take. Must be identical
            for all agents.

    Returns:
        `ndarray[int]`: (num_samples, num_agents) or (num_agents) Valued at 0 to
            (num_actions - 1).
    """
    # Set number of samples and number of agents by checking dimensions of `seq`.
    # `seq` can be either a 1d or 2d array
    if seq.ndim > 1:
        num_samples = seq.shape[0]
        num_agents = int(seq.shape[1] / num_actions)
    else:
        num_samples = 1
        num_agents = int(seq.shape[0] / num_actions)

    # Instantiate actions array and loop through samples. Each sample is a subsequence
    # of `seq`. Within each `sub_seq`, find the actions corresponding to the max
    # value for each agent. There will be `num_agent`` max values to choose for each
    # `sub_seq`.
    new_actions = zeros((num_samples, num_agents), dtype=int64)
    for j in range(num_samples):
        if num_samples > 1:
            sub_seq = seq[j, :]
        else:
            sub_seq = seq

        groups = chunker(sub_seq, num_actions)

        for i, a in enumerate(groups):
            new_actions[j, i] = argmax(a)

    # squeeze out extra dimensions (doesn't do anything with multiple samples)
    return new_actions.squeeze()
