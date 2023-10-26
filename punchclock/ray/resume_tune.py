"""Load an existing checkpoint and resume the tuning run."""
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
import psutil
import ray
from ray.tune import Tuner

# %% Functions


def resumeTune(checkpoint_dir: str, trainable, num_cpus: int | None):
    """Build a Tuner and run .fit from an experiment checkpoint.

    Args:
        checkpoint_dir (str): Path to checkpoint directory.
        trainable: See Ray documentation for compatible formats.
        num_cpus (int): Number of CPUs to use.
    """
    assert isinstance(checkpoint_dir, str)
    assert isinstance(num_cpus, (int, type(None)))

    print("\nAttempting to resume tune run...")
    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_cpus,
        num_gpus=0,
    )
    if num_cpus is None:
        num_cpus_avail = psutil.cpu_count()
    else:
        num_cpus_avail = num_cpus

    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(num_cpus_avail - 1)

    tuner = Tuner.restore(trainable=trainable, path=checkpoint_dir)
    print(f"\nTuner restored: {tuner}")
    print("\nBeginning fit...")
    tuner.fit()
    print("\n...fit complete")
