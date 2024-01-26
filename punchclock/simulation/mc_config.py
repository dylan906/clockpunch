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
        static_initial_conditions: bool = False,
        save_last_step_only: bool = False,
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
        static_initial_conditions (bool | None, optional): Set to True to use
            same initial conditions for all episodes, using a seed generated
            by MonteCarloRunner on instantiation. Set to False to use
            randomly-generated ICs, where a new seed is generated every episode.
            Set to None to use env_config["seed"] to generate ICs. If
            static_initial_conditions is None and "seed" not in env_config,
            static_initial_conditions is reassigned to False. If
            static_initial_conditions is False and "seed" is in env_config,
            a new seed is generated every episode, ignoring env_config['seed'].
            Defaults to False.
        save_last_step_only (bool, optional): Set to True to save only the
            last step of each episode. Defaults to False.
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
            "static_initial_conditions": static_initial_conditions,
            "save_last_step_only": save_last_step_only,
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
