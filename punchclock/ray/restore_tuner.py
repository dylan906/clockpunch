"""Restore Tuner."""
# %% Imports
# Punch Clock Imports
from punchclock.ray.build_tuner import buildTuner

# %% Functions


def restoreTuner(checkpoint_dir: str, config: dict):
    """Build and restore tuner from checkpoint.

    Args:
        checkpoint_dir (`str`): Path to checkpoint directory.
        config (`dict`): Config used to build the tuner via `BuildTuner()`.

    Returns:
        `Tuner`: See Ray documentation for details.
    """
    # build tuner
    tuner = buildTuner(config=config)
    # Restore checkpoint
    tuner.restore(path=checkpoint_dir)

    return tuner
