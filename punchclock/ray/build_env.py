"""Environment builder."""
# Used by Ray to register and build environments
# %% Imports
# Standard Library Imports
import json
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Third Party Imports
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium import wrappers as gym_wrappers

# Punch Clock Imports
from punchclock.common.utilities import array2List
from punchclock.environment import (
    info_wrappers,
    misc_wrappers,
    obs_wrappers,
    reward_wrappers,
)
from punchclock.environment.env import SSAScheduler, SSASchedulerParams

# %% Functions


# An environment builder function is required by RLlib. It simply takes in a dict and
# outputs an environment. In the case of SSAScheduler, we use SSASchedulerParams
# to build the environment, so the environment builder function here acts as a
# wrapper around SSASchedulerParams.
def buildEnv(env_config: dict) -> gym.Env:
    """Build SSAScheduler environment from config dict.

    Args:
        env_config (`dict`): A dict with the following structure:
        {
            "horizon" (`int`): See `SSASChedulerParams`,
            "agent_params" (`dict`): See `SSASChedulerParams`,
            "filter_params" (`dict`): See `SSASChedulerParams`,
            "time_step" (`float`, optional): See `SSASChedulerParams`,
            "seed" (`int`, optional): See `SSASChedulerParams`,
            "constructor_params" (`dict`, optional):
            {
                "wrappers" (`list[dict]`, optional): List of wrappers and their
                    associated configs to apply to base environment. Wrappers are
                    applied in the order in which they appear in the list. Each
                    entry is in the following format:
                        {
                            "wrapper": wrapper_name (`str`),
                            "wrapper_config": A `dict` containing optional arguments
                                for wrapper. Optional, defaults to {}.
                        }
            {
        }

    Returns:
        `env`: Gym environment.

    Example wrappers input:
        constructor_params = {
            "wrappers": [
                {
                    "wrapper": "filter_observation",
                    "wrapper_config": {"filter_keys": ["vis_map_est", "num_tasked"]},
                },
                {"wrapper": "float_obs"},
                {
                    "wrapper": "action_mask",
                    "action_mask_on": False,
                    },
            ]
        }
    """
    # %% Default constructor params
    # Create constructor params if not specified in arg. Set "wrappers" to [] if
    # not specified in arg; this makes other argument checking easier later.
    if "constructor_params" not in env_config.keys():
        env_config["constructor_params"] = {}

    if "wrappers" not in env_config["constructor_params"].keys():
        env_config["constructor_params"]["wrappers"] = []

    if "rescale_dict_obs" in [
        wpr["wrapper"] for wpr in env_config["constructor_params"]["wrappers"]
    ]:
        warnings.warn(
            """ Replace 'rescale_dict_obs' with 'linscale_dict_obs';
                'rescale_dict_obs' will be deprecated."""
        )

    # If an observation target_filter was set, make sure vis_map_est won't get filtered
    # out (if vis_map_est was not provided in the list of states to wrapper).
    wrapper_names = [
        a["wrapper"] for a in env_config["constructor_params"]["wrappers"]
    ]
    if "FilterObservation" in wrapper_names:
        filt_obs = [
            a
            for a in env_config["constructor_params"]["wrappers"]
            if a["wrapper"] == "FilterObservation"
        ][0]
        if "vis_map_est" not in filt_obs["wrapper_config"]["filter_keys"]:
            filt_obs["wrapper_config"]["filter_keys"].append("vis_map_est")
            warnings.warn(
                """'vis_map_est' not included in FilterObservation config.
            Appending to list of filters."""
            )
    # %% Build base environment
    # separate target_filter config from env config
    scheduler_config = deepcopy(env_config)
    scheduler_config.pop("constructor_params")

    env_params = SSASchedulerParams(**scheduler_config)
    env = SSAScheduler(env_params)

    # %% Wrap environment
    # Iterate along list of input wrappers and wrap the env according to the order
    # of the inputs. Order of wrappers matters.
    # Wrapper names must match the wrapper class name in the relevant module.
    for wrapper_dict in env_config["constructor_params"]["wrappers"]:
        wrapper_name = wrapper_dict["wrapper"]
        wrapper = getWrapper(wrapper_name)

        # Use blank dict for unprovided wrapper configs. Not all wrappers even
        # have configs, so need to have a default kwargs.
        kwargs = wrapper_dict.get("wrapper_config", {})
        env = wrapper(env, **kwargs)

    return env


def getWrapper(wrapper_name: str) -> Wrapper:
    """Get a Gymnasium wrapper class from a str of the wrapper.

    Args:
        wrapper_name (str): Name of wrapper.

    Returns:
        Wrapper: See Gymnasium documentation.
    """
    # Wrapper names must match the wrapper class name in the relevant module.
    # Try 3 modules to get wrappers.
    wrapper_modules = [
        gym_wrappers,
        obs_wrappers,
        reward_wrappers,
        misc_wrappers,
        info_wrappers,
    ]
    for wm in wrapper_modules:
        try:
            wrapper = getattr(wm, wrapper_name, {})
        except Exception:
            pass

        if wrapper != {}:
            break

    if wrapper == {}:
        # If wrapper not found in wrapper_modules, raise error
        raise ValueError(f"Wrapper '{wrapper_name}' not found.")

    return wrapper


def genConfigFile(
    config_dir: str | Path,
    config_file_name: str = None,
    num_cpus: int = None,
    trainable: str = None,
    param_space: dict = None,
    tune_config: dict = None,
    run_config: dict = None,
) -> dict:
    """Generate a config file to be used by buildTuner.

    By default, file is saves as 'config_YY-MM-DD_HH-MM-SS.json'.

    Args:
        config_dir (`str | Path`): Directory to save config file.
        config_file_name (`str`, optional): The name of the config file. Do not include
            path or file extension. Defaults to 'config_YY-MM-DD_HH-MM-SS'.
        num_cpus (`int`, optional): Number of CPUs to use in training run. Use
            None to use maximum available on machine. Defaults to None.
        trainable (`str`, optional): Defaults to None.
        param_space (`dict`): Search space of the tuning job. Must include the following
            keys:
            {
                "env_config": (`dict`) See `SSASchedulerParams`.
                }
        tune_config (`dict`, optional): Defaults to {}.
        run_config (`dict`, optional): Defaults to None.

    Returns:
        `dict`: Contains all of the arguments except `config_dir` and `config_file_name`.
    """
    # Check if environment config is in parameter space; we don't care about the contents
    # of env config, only that it exists, to stop really egregious arguments.
    if "env_config" not in param_space:
        raise ValueError("'env_config' is not in param_space")

    if isinstance(config_dir, str):
        config_dir = Path(config_dir)

    if tune_config is None:
        tune_config = {}

    # assemble data in dict
    config_data = {
        "num_cpus": num_cpus,
        "trainable": trainable,
        "param_space": param_space,
        "tune_config": tune_config,
        "run_config": run_config,
    }

    # Convert dict to json. The default setting is how to handle certain datatypes.
    json_object = json.dumps(config_data, default=array2List)

    if config_file_name is None:
        # set file path for default file name
        # get time stamp for file name
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = config_dir.joinpath("config_" + date_time).with_suffix(
            ".json"
        )
    else:
        # set file path for custom file name
        file_path = config_dir.joinpath(config_file_name).with_suffix(".json")

    # save json
    with open(str(file_path), "w") as outfile:
        outfile.write(json_object)

    print(f"Config file saved to {file_path} \n")
    return config_data
