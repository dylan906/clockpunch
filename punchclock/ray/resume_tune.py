"""Load an existing checkpoint and resume the tuning run."""
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
import ray
from ray import tuner

# %% Functions


def restoreTuner(checkpoint_dir: str, num_cpus: int):
    """Build and restore tuner from checkpoint.

    Args:
        checkpoint_dir (str): Path to checkpoint directory.
        num_cpus (int): Number of CPUs to use.

    Returns:
        Tuner: See Ray documentation for details.
    """
    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_cpus,
        num_gpus=0,
    )
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(num_cpus - 1)

    tuner.restore(path=checkpoint_dir)

    return tuner
