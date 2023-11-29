"""Load experiment files."""
# %% Imports
# Standard Library Imports
from pathlib import Path, PosixPath
from typing import Tuple

# Third Party Imports
from pandas import read_csv
from pandas.core.frame import DataFrame
from ray import tune
from ray.air.result import Result
from ray.tune.result_grid import ResultGrid

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile


# %% loadExperiments
def loadExpResults(
    local_dir: str,
    exp_names: list[str],
) -> Tuple[list[str], list[ResultGrid], list[DataFrame], list[Result]]:
    """Load results from 1 or more experiments.

    Args:
        local_dir (`str`): Path to folder that contains experiment(s) results.
            All experiments must be in same parent folder.
        exp_names (list[str]): Name(s) of folder(s) of experiment(s).

    Returns:
        experiment_paths (`list[str]`): Full paths to all experiments.
        result_grids (`list[ResultGrid]`): Ray class containing results from a
            tune run.
        result_dfs (`list[DataFrame]`): Pandas DataFrames containing results from
            tune run.
        best_results (`list[Result]`): Ray class containing best result from all
            results in a ResultGrid.

    All returns have length = len(exp_names).
    """
    experiment_paths = [f"{local_dir}/{exp_name}" for exp_name in exp_names]
    result_grids = []
    result_dfs = []
    best_results = []
    for _, exp_path in enumerate(experiment_paths):
        print(f"Loading results from {exp_path}")
        restored_tuner = tune.Tuner.restore(
            exp_path, trainable="PPO", resume_unfinished=False
        )
        result_grid = restored_tuner.get_results()

        # Check if there have been errors
        if result_grid.errors:
            print("One of the trials failed!")
        else:
            print("No errors!")

        results_df = result_grid.get_dataframe()

        if False in results_df["done"].values:
            print("At least one trial did not complete")

        best_result = result_grid.get_best_result(
            metric="episode_reward_mean", mode="max"
        )
        print(
            results_df[
                [
                    "trial_id",
                    "training_iteration",
                    "episode_reward_mean",
                ]
            ]
        )
        result_grids.append(result_grid)
        result_dfs.append(results_df)
        best_results.append(best_result)

    return (experiment_paths, result_grids, result_dfs, best_results)


# %% Load Env and/or Trial Configs
def loadTrialConfigs(
    local_dir: str | PosixPath,
    exp_names: list[str | PosixPath],
) -> Tuple[list[dict], list[PosixPath], list[PosixPath], list[PosixPath]]:
    """Load trial configs from experiment directories.

    Args:
        local_dir (str | PosixPath): Directory containing all experiments.
        exp_names (list[str | PosixPath]): Directories of experiments

    Returns:
        trial_configs (list[dict]): Trial configs
        trial_paths (list[PosixPath]): Trial paths
        checkpoint_paths (list[PosixPath]): Policy checkpoint paths
        config_paths (list[PosixPath]): Trial config paths
    """
    experiment_paths = [Path(a) for a in exp_names]
    trial_paths = []
    checkpoint_paths = []
    for exp_path in experiment_paths:
        for name in exp_path.rglob("*/default_policy"):
            trial_paths.append(name.parents[2])
            checkpoint_paths.append(name)

    print("\n")

    [trial_configs, config_paths] = loadTrialConfigFilesAndPaths(trial_paths)

    print("Trial paths:")
    for x in trial_paths:
        print(f"  {x}")

    print("Trial config paths:")
    for x in config_paths:
        print(f"  {x}")

    print("Checkpoint paths:")
    for x in checkpoint_paths:
        print(f"  {x}")

    assert len(trial_configs) == len(trial_paths)
    assert len(trial_paths) == len(checkpoint_paths)
    assert len(checkpoint_paths) == len(config_paths)

    return trial_configs, trial_paths, checkpoint_paths, config_paths


def loadTrialConfigFilesAndPaths(
    trial_paths: list[str | PosixPath],
) -> Tuple[list[dict], list[PosixPath]]:
    """Load trial configs from list of trial paths."""
    trial_paths = [Path(a) for a in trial_paths]
    config_paths = []
    for tp in trial_paths:
        for name in tp.glob("*params.json"):
            config_paths.append(name)

    trial_configs = []
    for cp in config_paths:
        try:
            trial_configs.append(loadJSONFile(cp))
        except Exception as err:
            print(err)
            print("Trial did not have a config path")

    return trial_configs, config_paths


def loadEnvConfigs(
    local_dir: str,
    exp_names: list[str],
) -> list[dict]:
    """Load environment configs from experiment directories."""
    trial_configs, _, _, _ = loadTrialConfigs(local_dir, exp_names)
    env_configs = [tc["env_config"] for tc in trial_configs]

    return env_configs


def loadExpProgress(
    exp_path: PosixPath,
) -> tuple[list[DataFrame], list[dict]]:
    """Load experiment progress and config files.

    Ignores trials with empty folders. Returned list entries correspond to each
        other.

    Args:
        exp_path (PosixPath): Experiment directory.

    Returns:
        progress (list[DataFrame]): The trial's progress.csv file as a DataFrame.
            Each DataFrame corresponds to a trial in the experiment.
        params (list[dict]]): The trial configuration, as gotten from params.json.
            Each dict corresponds to a trial in the experiment.
    """
    progress = []
    params = []
    for name in exp_path.glob("*/*.csv"):
        # skips empty dirs
        trial_path = name.parent
        progress_path = next(trial_path.glob("*.csv"))
        params_path = next(trial_path.glob("params.json"))

        progress_dat = read_csv(progress_path)
        params_dat = loadJSONFile(params_path)

        progress.append(progress_dat)
        params.append(params_dat)

    return progress, params
