"""Monte Carlo config class."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from datetime import datetime
from pathlib import Path, PosixPath

# Punch Clock Imports
from punchclock.common.utilities import saveJSONFile


# %% Class
class MonteCarloConfig:
    """A class to input to MonteCarloRunner."""

    def __init__(
        self,
        num_episodes: int,
        policy_configs: list[dict | str | PosixPath],
        env_configs: list[dict],
        results_dir: str | PosixPath,
        trial_names: list = None,
        print_status: bool = False,
        num_cpus: int = None,
        multiprocess: bool = False,
        single_sim_mode: bool = False,
    ):
        """Initialize MC config.

        Args:
        num_episodes (int): Number of episodes per trial.
        policy_configs (list[dict | str | PosixPath]): One trial will be run for each
            entry in policies. If entry is a dict, must conform to buildCustomPolicy
            (see policy_builder.py). If entry is a str, a RayPolicy will be
            loaded from the checkpoint specified by the arg.
        env_configs (list[dict]): See buildEnv.
        results_dir (str | PosixPath): Where to save trial results. One DataFrame per
            trial will be saved here. Each file will have a unique name.
        trial_names (list, optional): Names of trials. If not specified,
            trials are assigned integer names starting at 0.
        print_status (bool, optional): Set to True to have status print while
            running simulations. Defaults to False.
        num_cpus (int, optional): Number of cpus to make available when running
            MC. Defaults to max available.
        multiprocess (bool, optional): Whether or not to multiprocess trials
            during sim run. Used for debugging. Defaults to False.
        single_sim_mode (bool, optional): Whether or not to use
            identical initial conditions between trials. If True, num_episodes
            must be 1, and all steps from all trials will be saved. Defaults
            to False.
        """
        assert isinstance(num_episodes, int)
        assert isinstance(policy_configs, list), "policy_configs must be a list"

        assert all(
            isinstance(pc, (str, dict, PosixPath)) for pc in policy_configs
        ), "All entries of policy_configs must be either a dict or str."
        assert isinstance(env_configs, list)
        assert all(isinstance(ec, dict) for ec in env_configs)
        assert isinstance(results_dir, (str, PosixPath))
        # convert results_dir to string so that it can be JSON serialized
        results_dir = str(results_dir)

        if trial_names is not None:
            assert isinstance(trial_names, list)
            assert len(trial_names) == len(policy_configs)

        for i, pc in enumerate(policy_configs):
            if isinstance(pc, PosixPath):
                policy_configs[i] = str(pc)

        self.config = {
            "num_episodes": num_episodes,
            "results_dir": results_dir,
            "trial_names": trial_names,
            "num_cpus": num_cpus,
            "multiprocess": multiprocess,
            "print_status": print_status,
            "env_configs": env_configs,
            "policy_configs": policy_configs,
            "single_sim_mode": single_sim_mode,
        }
        return

    def save(
        self,
        fpath: str | Path,
        append_timestamp: bool = False,
    ):
        """Save a Monte Carlo config as a JSON file.

        Args:
            fpath (str | Path): Exclude file extension
            append_timestamp (bool, optional): Append a time stamp to end of
                fpath. Defaults to False.
        """
        if isinstance(fpath, str):
            fpath = Path(fpath)

        if append_timestamp is True:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d-%H%M%S")
            fpath = fpath.joinpath("_" + date_time)

        saveJSONFile(str(fpath), self.config)
