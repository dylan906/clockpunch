"""Monte Carlo sim runner module."""
# %% Imports
# Standard Library Imports
import itertools
import random
import string
from copy import copy, deepcopy
from datetime import datetime
from multiprocessing import Pool
from os import cpu_count, path
from pathlib import Path, PosixPath
from typing import Any, Tuple
from warnings import warn

# Third Party Imports
import ray
from numpy import iinfo
from numpy.random import default_rng
from pandas import DataFrame, concat
from ray.util.multiprocessing import Pool as RayPool

# Punch Clock Imports
from punchclock.common.utilities import saveJSONFile
from punchclock.environment.env import SSAScheduler
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.postprocess_sim_results import addPostProcessedCols
from punchclock.simulation.sim_runner import SimRunner
from punchclock.simulation.sim_utils import buildCustomOrRayPolicy


# %% Class
class MonteCarloRunner:
    """Used to run many simulations with iid environments."""

    def __init__(
        self,
        num_episodes: int,
        policy_configs: list[dict | str],
        results_dir: str,
        env_configs: list[dict] | dict = None,
        trial_names: list = None,
        print_status: bool = False,
        num_cpus: int = None,
        multiprocess: bool = True,
        static_initial_conditions: bool | None = False,
        save_last_step_only: bool = False,
        save_format: str = "pkl",
        env_config: dict = None,
        single_sim_mode=None,
    ):
        """Initialize MonteCarloRunner. Does not run a simulation.

        Number of trials = len(policy_configs) * len(env_configs)

        Args:
            num_episodes (int): Number of episodes per trial.
            policy_configs (list[dict | str]): One trial will be run for each
                entry in policies. If entry is a dict, must conform to buildCustomPolicy
                (see policy_utils module). If entry is a str, a RayPolicy will be
                loaded from the checkpoint specified by the arg.
            results_dir (str): Where to save trial results. One DataFrame per
                trial will be saved here. Each file will have a unique name.
            env_configs (list[dict] | dict): List of env configs to be run with
                all policies.
            trial_names (list, optional): Names of trials. If not specified,
                trials are assigned integer names starting at 0.
            print_status (bool, optional): Set to True to have status print while
                running simulations. Defaults to False.
            num_cpus (int, optional): Number of cpus to make available when running
                MC. Defaults to max available.
            multiprocess (bool, optional): Whether or not to multiprocess trials
                during sim run. Used for debugging. Defaults to True.
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
            save_format (str, optional): ["pkl" | "csv"] Only change from default
                for debugging. Saving as csv is not guaranteed to retain all sim
                data. Defaults to "pkl".
            env_config (dict, optional): Here for backward compatibility.
            single_sim_mode (optional): Here for backward compatibility.
        """
        if single_sim_mode is not None:
            raise ValueError(
                "The 'single_sim_mode' parameter is deprecated. If you set "
                "single_sim_mode to True, instead set static_initial_conditions "
                "to True and save_last_step_only to False. Otherwise, do the "
                "opposite. Delete single_sim_mode from your code."
            )

        # Assertions and warnings
        assert num_episodes > 0, "num_episodes must be > 0"
        assert save_format in [
            "pkl",
            "csv",
        ], "save_format must be in recognized list ['pkl', 'csv']"
        # If any env config in env_configs has "seed", then all will use the same seed.
        if any(["seed" in ec.keys() for ec in env_configs]):
            env_config_seed = [ec["seed"] for ec in env_configs if "seed" in ec.keys()][
                0
            ]
            assert isinstance(
                env_config_seed, int
            ), "If env_config contains 'seed' as a key, the value must be an integer."

            if static_initial_conditions in (True, None):
                # Change static_initial_conditions to True if None and env_config
                # provides seed.
                static_initial_conditions = True
                warn("""Using env_config['seed'] to build environments.""")
            elif static_initial_conditions is False:
                warn(
                    """static_initial_conditions is False, but env_config['seed']
                    is provided. Will generate random initial conditions (ignoring
                    env_config['seed']). Set static_initial_conditions = None to
                    use env_config['seed']."""
                )
        else:
            if static_initial_conditions is None:
                static_initial_conditions = False
                warn(
                    """static_initial_conditions is None, but no seed is provided
                    in env_config. Setting static_initial_conditions = False. Will
                    generate random initial conditions."""
                )

        assert isinstance(policy_configs, list)

        # TODO: Delete env_config after a version or two.
        if env_config is not None:
            warn("The arg 'env_config' is deprecated. Replace with 'env_configs'.")
            env_configs = [env_config]
        assert env_configs is not None
        if isinstance(env_configs, dict):
            env_configs = [env_configs]

        # Deepcopy args to prevent MCR from modifying args at higher scope.
        self.num_episodes = deepcopy(num_episodes)
        self.env_configs = deepcopy(env_configs)
        self.policy_configs = deepcopy(policy_configs)
        self.num_trials = len(self.policy_configs) * len(self.env_configs)
        self.print_status = deepcopy(print_status)
        self.results_dir = Path(results_dir)
        self.multiprocess = multiprocess
        self.static_initial_conditions = static_initial_conditions
        self.save_last_step_only = save_last_step_only
        self.save_format = save_format

        # Store config as dict for later saving as a file. Do this before inserting
        # defaults.
        self.config = {
            "num_episodes": deepcopy(num_episodes),
            "policy_configs": deepcopy(policy_configs),
            "env_configs": deepcopy(env_configs),
            "results_dir": deepcopy(results_dir),
            "trial_names": deepcopy(trial_names),
            "print_status": deepcopy(print_status),
            "num_cpus": deepcopy(num_cpus),
            "multiprocess": deepcopy(multiprocess),
            "static_initial_conditions": deepcopy(static_initial_conditions),
            "save_last_step_only": deepcopy(save_last_step_only),
            "save_format": deepcopy(save_format),
        }

        if num_cpus is None:
            self.num_cpus = cpu_count()
        else:
            assert num_cpus > 0, "num_cpus must be > 0"
            self.num_cpus = num_cpus

        # Make trial names integers if not specified in args.
        if trial_names is None:
            self.trial_names = [i for i in range(self.num_trials)]
        else:
            self.trial_names = deepcopy(trial_names)

        # max storable integer, used in rng generation
        self.max_int = iinfo(int).max
        # Initialize RNG for random initial conditions generator.
        self.rng = default_rng()
        # Store a fixed number for fixed initial conditions generator. Not allowed
        # to change between trials or episodes.
        self.fixed_seed = self.rng.integers(self.max_int)

        env_pol_pairs = list(
            itertools.product(self.config["env_configs"], self.config["policy_configs"])
        )
        self.trial_configs = []
        for i, tn in enumerate(self.trial_names):
            ec = env_pol_pairs[i][0]
            pc = env_pol_pairs[i][1]
            self.trial_configs.append(
                {
                    "trial_name": tn,
                    "env_config": ec,
                    "policy_config": pc,
                }
            )

        return

    def runMC(
        self,
        print_status: bool = None,
        multiprocess: bool = None,
    ) -> dict:
        """Run a Monte Carlo experiment.

        print_status (bool, optional): Set to override initialization value.
            Defaults to None.
        multiprocess (bool, optional): Set to override initialization value.
            Defaults to None.

        Returns data from the final time steps of all episodes. Does not retain
            information prior to the final step of each episode. Returns a dict
            with trial_names as keys. Each value is a num_episodes-long list. Each
            entry in list is a DataFrame that contains the final time step of
            information from the simulation.
        """
        # Override initialized print_status if set in arg.
        if print_status is not None:
            print_status = print_status
        else:
            print_status = self.print_status

        # Override initialized multiprocess if set in arg.
        if multiprocess is not None:
            multiprocess = multiprocess
        else:
            multiprocess = self.multiprocess

        # Each call of runMC() creates a new experiment directory within results_dir.
        [exp_dir, config_name] = self._makeDirs()

        # Save config to results_dir for traceability
        self._saveConfigs(
            exp_config_path=config_name,
            exp_config=self.config,
            trial_configs=self.trial_configs,
        )
        # saveJSONFile(str(config_name), self.config)

        # Initialize results dict; one entry per trial.
        self.results = {k: None for k in self.trial_names}

        # Start times for printing
        t_start = datetime.now()
        print(f"Monte Carlo simulation started at {t_start}")
        if print_status is True:
            print(f"Results will be saved to {self.results_dir}")

        # Args to pool.starmap must be pickleable and in a single list.
        pool_args = self._assemblePoolArgs(print_status)

        # Use multiprocessor (faster) or a loop (sequential) depending on setting.
        if multiprocess is True:
            pool_results = self.multiProcessTrials(pool_args, print_status)
        elif multiprocess is False:
            pool_results = self.serialProcessTrials(pool_args)

        # de-concat results and reassemble in a dict.
        for res in pool_results:
            data = res[0]
            name = res[1]
            self.results[name] = data

        t_finish = datetime.now()
        print(f"Monte Carlo simulation complete. Run time = {t_finish-t_start}")

        return deepcopy(self.exp_dir), deepcopy(self.results)

    def multiProcessTrials(
        self,
        pool_args: list,
        print_status: bool,
    ) -> list[Tuple]:
        """Run trials with multiprocessing.

        Args:
            pool_args (list): List of arguments for Pool.starmap() and/or
                ray.util.multiprocessing.Pool.starmap(). Same args as using serial
                processing.
            print_status (bool): Whether or not to print status.

        Returns:
            list[Tuple]: Each Tuple entry is 2-long, where 1st entry is trial
                results and 2nd entry is trial name.
        """
        # Split pool args into list for regular multiprocessing and another for
        # Ray multiprocessing. Then execute both lists and concatenate the results.
        standard_pool_args, ray_pool_args = self._splitPoolArgs(pool_args)

        if print_status is True:
            print("\nStandard multiprocessing beginning...")
        # Initialize multiprocessor pool for parallel sim running.
        # pool.starmap returns a list of 2-long tuples where the 1st entry is sim
        # results and the 2nd entry is trial name.
        pool = Pool(self.num_cpus)
        standard_pool_results = pool.starmap(self.runTrial, standard_pool_args)
        # shutdown pool process to free up workers
        pool.close()
        if print_status is True:
            print("Standard multiprocessing complete.")
            print("\nRay multiprocessing beginning...")

        # Ray Pool
        ray.init(num_cpus=self.num_cpus)
        ray_pool = RayPool()
        ray_pool_results = ray_pool.starmap(self.runTrial, ray_pool_args)
        ray_pool.close()
        ray.shutdown()

        if print_status is True:
            print("Ray multiprocessing complete.")

        pool_results = standard_pool_results + ray_pool_results

        return pool_results

    def serialProcessTrials(
        self,
        pool_args: list,
    ) -> list[Tuple]:
        """Run trials serially (one-by-one).

        Args:
            pool_args (list): List of arguments for self.runTrial(), where each
                entry is the ordered args expected by runTrial(). Same args as
                using multiprocessing.

        Returns:
            list[Tuple]: Each Tuple entry is 2-long, where 1st entry is trial
                results and 2nd entry is trial name.
        """
        # Loop through trials sequentially.
        pool_results = []
        for name, e_config, p_config, print_entry in pool_args:
            trial_res = self.runTrial(
                trial_name=name,
                env_config=e_config,
                policy_config_or_checkpoint=p_config,
                print_status=print_entry,
            )
            pool_results.append(trial_res)

        return pool_results

    def runTrial(
        self,
        trial_name: Any,
        env_config: dict,
        policy_config_or_checkpoint: dict | str,
        print_status: bool = False,
    ) -> Tuple[list[DataFrame], Any]:
        """Run multiple episodes of a simulation under a single policy.

        Saves a file in a specified format and directory (see __init__()).

        Args:
            trial_name (Any): Trial name.
            env_config (dict): See buildEnv.
            policy_config_or_checkpoint (dict | str): A CustomPolicy config or
                a path to a Ray checkpoint. See buildCustomRayPolicy and
                buildCustomPolicy for interface details.
            print_status (bool, optional): Set to True to print trial name and
                run time. Defaults to False.

        Returns:
            trial_results (list[DataFrame]): A num_episodes-long list of DataFrames.
                Each DataFrame is the results of the final step of a single episode.
            trial_name (Any): Same as argument. Primarily as a check to associate
                results with a name.
        """
        # Prints
        if print_status:
            print(f"Running trial {trial_name}...")
            t0 = datetime.now()

        # %% Build policy
        policy = buildCustomOrRayPolicy(config_or_path=policy_config_or_checkpoint)

        # Initialize trial_result as a dict with one entry, where the key is the
        # trial name. The entry is a list (length = num_episodes) of the final
        # step of dataframes.
        trial_result = [None for i in range(self.num_episodes)]

        for ep in range(self.num_episodes):
            [env, seed] = self._buildEnvironment(
                env_config=env_config,
                static_initial_conditions=self.static_initial_conditions,
            )
            info = {"seed": seed}

            # Build sim runner, then loop through num_episodes times.
            sim_runner = SimRunner(
                env=env,
                policy=policy,
                max_steps=env_config["horizon"],
            )

            sim_runner.reset()
            ep_results = sim_runner.runSim()
            df = ep_results.toDataFrame()
            df = addPostProcessedCols(df, info)

            if self.save_last_step_only is True:
                # Drop all but the last time step of the DF
                last_frame = df.tail(1)
                trial_result[ep] = deepcopy(last_frame)
            else:
                # Retain full DF
                trial_result[ep] = deepcopy(df)

        fpath = self._saveTrialResults(
            trial_result,
            trial_name,
            self.save_format,
        )

        if print_status:
            tf = datetime.now()
            print(f"Trial {trial_name} complete. Trial run time: {tf - t0}")
            print(f"Results saved to {fpath}")

        return trial_result, trial_name

    def _assemblePoolArgs(
        self,
        print_status: bool,
    ) -> list:
        """Assemble arguments for multiprocessing.Pool() into a list."""
        # Args to pool.starmap must be pickleable, meaning simple structures
        # (e.g. dicts). Args to starmap must be in a list in the expected order.

        # Get all combinations of env-policy pairs. Will be list with
        # len = num_trials = num_policies * num_envs
        env_pol_pairs = list(itertools.product(self.env_configs, self.policy_configs))
        env_configs = []
        policy_configs = []
        for ec, pc in env_pol_pairs:
            env_configs.append(deepcopy(ec))
            policy_configs.append(deepcopy(pc))

        print_list = [copy(print_status) for a in range(self.num_trials)]
        pool_args = [
            (a)
            for a in zip(
                self.trial_names,
                env_configs,
                policy_configs,
                print_list,
            )
        ]
        return pool_args

    def _buildEnvironment(
        self,
        env_config: dict,
        static_initial_conditions: bool,
    ) -> SSAScheduler:
        """Wrapper for buildEnv that handles initial conditions seed generation.

        |-----------------------|-----------------------------------------------------|
        |                       |             static_initial_conditions               |
        |                       |-----------------------------------------------------|
        |                       |   True                          | False             |
        |---------------|-------|---------------------------------|-------------------|
        | seed provided | True  | Use env_config["seed"]          | Generate new seed |
        | in env_config |-------|---------------------------------|-------------------|
        |               | False | Use un-changing self.fixed_seed | Generate new seed |
        |---------------|-------|---------------------------------|-------------------|
        """
        if static_initial_conditions is False:
            # generate new seed every time
            env_config["seed"] = self.rng.integers(self.max_int)
            print("Generating new env seed.")
        elif "seed" in env_config.keys():
            # use provided seed
            print("Using env_config-provided fixed seed.")
        elif "seed" not in env_config.keys():
            # use once-generated seed
            env_config["seed"] = self.fixed_seed
            print("Using MonteCarloRunner-generated fixed seed")
        else:
            raise Exception("There is a problem.")

        env = buildEnv(env_config)

        return env, env_config["seed"]

    def _makeDirs(
        self,
    ) -> Tuple[PosixPath, PosixPath]:
        """Generate config directory string and create directory if necessary.

        Returns:
            exp_dir (PosixPath): Experiment path.
            config_file_path (PosixPath): Config file path.
        """
        # Create results directory if it doesn't already exist. Each call of runMC()
        # creates a new directory within results_dir. Each subdirectory is time-stamped.
        # Trial results will be saved to each subdirectory.
        if not path.exists(self.results_dir):
            # makedirs(self.results_dir)
            self.results_dir.mkdir()

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
        self.exp_dir = deepcopy(self.results_dir)
        self.exp_dir = self.exp_dir.joinpath("exp_" + ts + "_" + rand_str)
        self.exp_dir.mkdir()
        print(f"Experiment dir created at: {str(self.exp_dir)}")

        self._makeTrialDirs(exp_dir=self.exp_dir)

        config_file_path = self.exp_dir.joinpath("exp_config").with_suffix(".json")

        return self.exp_dir, config_file_path

    def _makeTrialDirs(self, exp_dir: PosixPath):
        self.trial_dirs = []
        for tname in self.trial_names:
            trial_path = exp_dir.joinpath("trial_" + str(tname))
            trial_path.mkdir()
            self.trial_dirs.append(trial_path)
            print(f"Trial dir created at: {str(trial_path)}")

        return

    def _saveConfigs(
        self, exp_config_path: PosixPath, exp_config: dict, trial_configs: list[dict]
    ):
        saveJSONFile(str(exp_config_path), exp_config)

        for td, tc in zip(self.trial_dirs, trial_configs):
            trial_config_path = td.joinpath("trial_config").with_suffix(".json")
            saveJSONFile(str(trial_config_path), tc)

    def _saveTrialResults(
        self,
        trial_result: list,
        trial_name: str,
        file_format: str,
    ) -> str:
        """Save trial results to a .pkl or .csv.

        Saves result to experiment directory if one exists, otherwise to results
        directory (1 level up).

        Args:
            trial_result (list): Results from a trial
            trial_name (str): Name of trial
            file_format (str): 'pkl' | 'csv'

        Returns:
            str: Path result file was saved to
        """
        trial_df = self._convertResults2DF(trial_result, str(trial_name))
        if hasattr(self, "exp_dir"):
            trial_path = self.lookupTrialPath(str(trial_name), self.trial_dirs)
            fpath = trial_path.joinpath("results_" + str(trial_name)).with_suffix(
                "." + file_format
            )
        else:
            # This used for debugging the method alone, not a regular use case.
            fpath = self.results_dir.joinpath("results_" + str(trial_name)).with_suffix(
                "." + file_format
            )

        if file_format == "pkl":
            trial_df.to_pickle(fpath)
        elif file_format == "csv":
            trial_df.to_csv(fpath)

        return fpath

    def lookupTrialPath(self, trial_name: str, trial_dirs: list[PosixPath]):
        """Lookup trial path given a trial name."""
        for td in trial_dirs:
            if trial_name in td.name:
                return td

        return

    def _convertResults2DF(
        self,
        trial_results: list[DataFrame],
        trial_name: Any,
    ) -> DataFrame:
        """Convert a single trial's results into a DataFrame.

        Adds come Monte Carlo-specific info to the standard sim DataFrame.

        Args:
            trial_results (list[DataFrame]): A list of the DataFrames corresponding
                to time steps of episodes.
            trial_name (Any): Used as the entry in the "trial" column of the
                returned DataFrame.

        Returns:
            DataFrame: A DataFrame of the final time step of all episodes in
                a trial. The "step" index from the input DataFrames is moved to
                a column. Adds "trial" and "episode" columns.
        """
        ep_numbers = list(range(self.num_episodes))
        trial_df = concat(trial_results, keys=ep_numbers, names=["episode"])
        # Add new level to index named "trial"
        trial_df = concat({trial_name: trial_df}, names=["trial"])
        # Flatten multilayer index to all columns
        trial_df.reset_index(inplace=True)

        return trial_df

    def getDataFrame(self) -> DataFrame:
        """Get MC results as a DataFrame.

        Returns:
            DataFrame: Return has hierarchical index, level 0 = "trial",
                level 1 = "epsiode".
        """
        assert hasattr(self, "results"), "Monte Carlo has not been run yet."
        # Loop through trials and concatenate all episodes in a trial into a single
        # DataFrame, so each trial is represented by a single DataFrame. Then
        # combine all the trial DataFrames so that the entire experiment is represented
        # as a single DataFrame. The experiment DataFrame has hierarchical indexing,
        # where the trial name is the level 0 index.
        trial_df_list = []
        for trial, episode_list in self.results.items():
            trial_df = self._convertResults2DF(
                trial_results=episode_list,
                trial_name=trial,
            )
            trial_df_list.append(trial_df)

        # merge dataframes
        merged_df = concat(trial_df_list)
        return merged_df

    def _splitPoolArgs(self, pool_args: list[dict]) -> Tuple[list[dict], list[dict]]:
        """Split list of trial configs by custom or Ray policies.

        Args:
            pool_args (list[dict]): List of args for Pool().

        Returns:
            normal_args (list[dict]): Args for multiprocessing.Pool.starmap().
            ray_args (list[dict]): Args for ray.util.multiprocessing.Pool.starmap().
        """
        normal_args = []
        ray_args = []
        for config in pool_args:
            print(config)
            if isinstance(config[2], str):
                ray_args.append(config)
            else:
                normal_args.append(config)

        return normal_args, ray_args
