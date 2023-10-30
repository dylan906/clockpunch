"""Load an existing checkpoint and resume the tuning run."""
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
import psutil
import ray
from ray.rllib.models import ModelCatalog
from ray.tune import Tuner
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.nets.lstm_mask import MaskedLSTM
from punchclock.ray.build_env import buildEnv

# %% Functions


def resumeTune(
    checkpoint_dir: str,
    trainable,
    num_cpus: int | None,
    resume_errored: bool = True,
    restart_errored: bool = False,
):
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

    # %% Register environment builder (via Ray, not Gym) and action mask model
    register_env("ssa_env", buildEnv)
    ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)
    ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

    if num_cpus is None:
        num_cpus_avail = psutil.cpu_count()
    else:
        num_cpus_avail = num_cpus

    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(num_cpus_avail - 1)

    tuner = Tuner.restore(
        trainable=trainable,
        path=checkpoint_dir,
        resume_errored=restart_errored,
        restart_errored=restart_errored,
    )
    print(f"\nTuner restored: {tuner}")
    print("\nBeginning fit...")
    tuner.fit()
    print("\n...fit complete")
