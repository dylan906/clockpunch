"""Module for building Ray tuner with SSAScheduler environment."""
# Using Ray with SLURM: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
# Tuner documentation: https://docs.ray.io/en/latest/tune/api_docs/execution.html?highlight=Tuner#tuner
# TunerInternal source: https://github.com/ray-project/ray/blob/0cd49130e669b296e3222c2d3eec8162ab34837f/python/ray/tune/impl/tuner_internal.py
# %% Imports
# Standard Library Imports
import os
import random
import string
from copy import deepcopy
from datetime import datetime

# Third Party Imports
import psutil
import ray
from numpy import ceil
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.tune import Tuner
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.utilities import printNestedDict
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.nets.lstm_mask import MaskedLSTM
from punchclock.ray.build_env import buildEnv


# %% Main Function
def buildTuner(
    config: dict,
    override_date: bool = False,
) -> Tuner:
    """Build a Ray Tuner with a SSAScheduler environment from primitives in a dict.

    Args:
        config (dict):
        {
            num_cpus (int | None): Number of CPUs to use in training run. If None,
                sets to maximum available CPUs on machine.
            trainable (str | Callable | ray.tune.trainable.trainable.Trainable): The
                trainable to be tuned. See Ray Tuner documentation for details.
            param_space (dict): Search space of the tuning job. See Ray Tuner documentation
                for details.
            tune_config (dict): Parameters that create a ray.tune.tune_config.TuneConfig
                object. Tuning algorithm specific configs. See Ray Tuner documentation for
                details.
            run_config (dict): Parameters that create a ray.air.config.RunConfig object.
                Runtime configuration that is specific to individual trials. See Ray Tuner
                documentation for details. At minimum, include a stop condition ("stop"),
                an experiment name ("name"), and a local directory ("local_dir") to save
                results.
        }
        override_date (bool, optional): If False, date-time string is appended
            to experiment name. Defaults to False.

    Returns:
        Tuner: See Ray documentation for details.

    Example for run_config:
        run_config = {
            "stop": {"episodes_total": 100},
            "name": "exp_name",
            "local_dir": "tests/ray"
        }
    """
    print("buildTuner started...")

    # get default values
    [num_cpus, config["param_space"]] = _getDefaults(
        param_space=config["param_space"],
        num_cpus=config.get("num_cpus", None),
    )
    # By default, add datetime to name string
    if override_date is False:
        config["run_config"] = _getExperimentName(config["run_config"])

    # %% Print other helpful info
    print(f"num_cpus = {num_cpus}")
    print("config file contents:")
    printNestedDict(config)

    # %% Register environment builder (via Ray, not Gym) and action mask model
    register_env("ssa_env", buildEnv)
    ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)
    ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

    # %% Initialize ray
    print("Initializing Ray environment...")
    # Must explicitly tell Ray resources available-- do NOT let Ray decide! Otherwise, weird
    # errors happen. Always set num_gpus to 0 (Ray default as well).
    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_cpus,
        num_gpus=0,
    )

    # Set number of pending trials manually to avoid Ray warning about large number of pending
    # trails potentially causing scheduling issues.
    num_workers = config["param_space"]["num_workers"]
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(num_workers)

    # %% Trainable
    trainable = config["trainable"]
    # %% Parameters for tuning run
    # Set some hard-coded params, then append the rest of the keys that were set via config.
    #
    # Examples of potential fields to include in param_space:
    #   "env_config": config["env_config"],
    #   "num_workers": num_workers,  # don't let Ray decide num_workers
    #   "lr": config["lr"],
    #   "model": {
    #           "fcnet_activation": "relu",
    #           "fcnet_hiddens": [tune.grid_search([100, 50]), tune.grid_search([50, 25])],
    #       },

    param_space = {
        "framework": "torch",
        "env": "ssa_env",
        "horizon": None,  # environment has its own horizon
        "ignore_worker_failures": True,
        "log_level": "DEBUG",
        "num_gpus": 0,
    }

    param_space.update(config["param_space"])
    print(f"Model: {param_space['model']}")

    # %% Tune config
    # algo_config = PPOConfig()
    tune_config = config.get("tune_config", {})
    tune_config = tune.tune_config.TuneConfig(**tune_config)

    # %% Run params
    # Hard code checkpoint at the end (True), and set the rest of the arguments via the input
    # config.

    # Testing hard-coded stopper
    # stopper = tune.stopper.TrialPlateauStopper(metric="episode_reward_mean")
    # config["run_config"].pop("stop")

    # Testing hard-coded failure_config. Try to set tune to retry trials if they error.
    failure_config = air.config.FailureConfig(max_failures=2)

    run_config = air.RunConfig(
        checkpoint_config=air.CheckpointConfig(
            # num_to_keep=2,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_at_end=True,
        ),
        failure_config=failure_config,
        # stop=stopper,
        **config["run_config"],
    )
    # %% Build and run tuner
    print("Initializing tuner...")
    # define tuner
    tuner = tune.Tuner(
        trainable=trainable,
        param_space=param_space,
        run_config=run_config,
        tune_config=tune_config,
    )
    print("Tuner instantiated")

    return tuner


def _getDefaults(
    param_space: dict,
    num_cpus: int | None,
) -> tuple[int, dict]:
    """Get default values for some parameters to override Ray defaults.

    Args:
        param_space (dict): Search space of the tuning job.
        num_cpus (int): Number of CPUs requested.

    Returns:
        num_cpus (int): Number of CPUs requested.
        param_space (dict): Same as input, but with some new/overridden entries.
    """
    # %% Get number of CPUs available on machine
    # Set desired number of CPUs to max available on machine by default or to
    # (max_available - num_cpus).
    num_cpus_avail = psutil.cpu_count()
    print(f"num_cpus on machine = {num_cpus_avail}")
    if num_cpus is None:
        num_cpus = num_cpus_avail
    elif num_cpus < 0:
        # If negative CPUs specified, then use the total available CPUs less the amount
        # specified.
        num_cpus = num_cpus_avail + num_cpus

    # %% Get number of workers
    # Assign default number of workers if none specified. Need to account for "num_workers"
    # not being in param_space because it is an optional input.

    if "num_workers" in param_space.keys():
        num_workers = param_space["num_workers"]
    else:
        num_workers = None

    if num_workers is None:
        # Need to reserve 1 CPU for overhead.
        num_workers = num_cpus - 1
    elif num_workers < 0:
        # If negative workers specified, use that amount less than (num_cpus-1).
        num_workers = num_cpus - 1 + num_workers

    param_space["num_workers"] = num_workers

    return (num_cpus, param_space)


def _getExperimentName(
    run_params: dict,
) -> dict:
    """Append a date-time string to the end of the experiment name.

    Args:
        run_params (dict): A dict that optionally contains "name" as a key.

    Returns:
        dict: Contains "key" as a dict with a str value.
    """
    new_run_params = deepcopy(run_params)
    rand_str = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=4)
    )
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If a name is provided, append the datetime string; if not provided, then
    # just make assign the datetime string as "name".
    if "name" in new_run_params.keys():
        new_run_params["name"] = (
            new_run_params["name"] + "_" + rand_str + "_" + datetime_str
        )
    else:
        new_run_params["name"] = rand_str + "_" + datetime_str

    return new_run_params
