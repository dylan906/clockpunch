"""Monte Carlo config class."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from datetime import datetime

# Punch Clock Imports
from scheduler_testbed.common.utilities import saveJSONFile


# %% Class
class MonteCarloConfig:
    """A class to input to MonteCarloRunner."""

    def __init__(
        self,
        num_episodes: int,
        policy_configs: list[dict | str],
        env_config: dict,
        results_dir: str,
        trial_names: list = None,
        print_status: bool = False,
        num_cpus: int = None,
        multiprocess: bool = False,
        single_sim_mode: bool = False,
    ):
        """Initialize MC config.

        Args:
        num_episodes (`int`): Number of episodes per trial.
        policy_configs (`list[dict | str]`): One trial will be run for each
            entry in policies. If entry is a dict, must conform to buildCustomPolicy
            (see policy_builder.py). If entry is a str, a RayPolicy will be
            loaded from the checkpoint specified by the arg.
        env_config (`dict`): See buildEnv.
        results_dir (`str`): Where to save trial results. One DataFrame per
            trial will be saved here. Each file will have a unique name.
        trial_names (`list`, optional): Names of trials. If not specified,
            trials are assigned integer names starting at 0.
        print_status (`bool`, optional): Set to True to have status print while
            running simulations. Defaults to False.
        num_cpus (`int`, optional): Number of cpus to make available when running
            MC. Defaults to max available.
        multiprocess (`bool`, optional): Whether or not to multiprocess trials
            during sim run. Used for debugging. Defaults to False.
        single_sim_mode (`bool`, optional): Whether or not to use
            identical initial conditions between trials. If True, num_episodes
            must be 1, and all steps from all trials will be saved. Defaults
            to False.
        """
        assert all(isinstance(pc, (str, dict)) for pc in policy_configs)

        self.config = {
            "num_episodes": num_episodes,
            "results_dir": results_dir,
            "trial_names": trial_names,
            "num_cpus": num_cpus,
            "multiprocess": multiprocess,
            "print_status": print_status,
            "env_config": env_config,
            "policy_configs": policy_configs,
            "single_sim_mode": single_sim_mode,
        }
        return

    def save(
        self,
        fpath: str,
        append_timestamp: bool = False,
    ):
        """Save a Monte Carlo config as a JSON file.

        Args:
            fpath (`str`): Exclude file extension
            append_timestamp (`bool`, optional): Append a time stamp to end of
                fpath. Defaults to False.
        """
        if append_timestamp is True:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d-%H%M%S")
            fpath = fpath + "_" + date_time

        saveJSONFile(fpath, self.config)
