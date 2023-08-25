"""Environment builder."""
# Used by Ray to register and build environments
# %% Imports
from __future__ import annotations

# Standard Library Imports
import json
import warnings
from copy import deepcopy
from datetime import datetime

# Third Party Imports
import gymnasium as gym
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation

# Punch Clock Imports
from punchclock.common.utilities import array2List
from punchclock.environment.env import SSAScheduler, SSASchedulerParams
from punchclock.environment.obs_wrappers import (
    ActionMask,
    Convert2dTo3dObsItems,
    ConvertCustody2ActionMask,
    ConvertObsBoxToMultiBinary,
    CopyObsItem,
    CustodyWrapper,
    DiagonalObsItems,
    FlatDict,
    FloatObs,
    LinScaleDictObs,
    MinMaxScaleDictObs,
    MultiplyObsItems,
    NestObsItems,
    SplitArrayObs,
    SqueezeObsItems,
    SumArrayWrapper,
    VisMap2ActionMask,
)
from punchclock.environment.reward_wrappers import (
    AssignObsToReward,
    NullActionReward,
    ThresholdReward,
    VismaskViolationReward,
)

# %% Functions


# An environment builder function is required by RLlib. It simply takes in a dict and
# outputs an environment. In the case of SSAScheduler, we use SSASchedulerParams
# to build the environment, so the environment builder function here acts as a
# wrapper around SSASchedulerParams.
def buildEnv(env_config: dict) -> gym.Env:
    """Build SSASCheduler environment from config dict.

    Args:
        env_config (`dict`): A dict with the following structure:
        {
            "horizon" (`int`): See `SSASChedulerParams`,
            "agent_params" (`dict`): See `SSASChedulerParams`,
            "filter_params" (`dict`): See `SSASChedulerParams`,
            "reward_params" (`dict`): See `SSASChedulerParams`,
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

    Supported wrappers:
        gym defaults: "filter_observation", "flatten_observation"
        Punch Clock: "float_obs", "action_mask", "flat_dict", "rescale_dict_obs"

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
    if "filter_observation" in wrapper_names:
        filt_obs = [
            a
            for a in env_config["constructor_params"]["wrappers"]
            if a["wrapper"] == "filter_observation"
        ][0]
        if "vis_map_est" not in filt_obs["wrapper_config"]["filter_keys"]:
            filt_obs["wrapper_config"]["filter_keys"].append("vis_map_est")
            warnings.warn(
                """'vis_map_est' not included in filter_observation config.
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
    # of the inputs. Use wrapper_map to map wrapper names (`str`s) to wrapper function
    # (`Callable`s). Order of wrappers matters.
    wrapper_map = {
        # Observation Space Wrappers
        "filter_observation": FilterObservation,
        "flatten_observation": FlattenObservation,
        "float_obs": FloatObs,
        # "action_mask": ActionMask,
        "flat_dict": FlatDict,
        "linscale_dict_obs": LinScaleDictObs,
        # "rescale_dict_obs" here for backward compatibility
        "rescale_dict_obs": LinScaleDictObs,
        "minmaxscale_dict_obs": MinMaxScaleDictObs,
        "splitarray_obs": SplitArrayObs,
        "custody": CustodyWrapper,
        "multiply_obs_items": MultiplyObsItems,
        "nest_obs_items": NestObsItems,
        "copy_obs_item": CopyObsItem,
        "vis_map_action_mask": VisMap2ActionMask,
        "convert_2d_to_3d_obs_items": Convert2dTo3dObsItems,
        "convert_custody_2_action_mask": ConvertCustody2ActionMask,
        "convert_obs_box_to_multibinary": ConvertObsBoxToMultiBinary,
        "squeeze_obs_items": SqueezeObsItems,
        "diagonal_obs_items": DiagonalObsItems,
        "sum_array_wrapper": SumArrayWrapper,
        # Reward Wrappers
        "assign_obs_to_reward": AssignObsToReward,
        "null_action_reward": NullActionReward,
        "threshold_reward": ThresholdReward,
        "vismask_violation_reward": VismaskViolationReward,
    }

    for wrapper_dict in env_config["constructor_params"]["wrappers"]:
        wrapper = wrapper_map[wrapper_dict["wrapper"]]
        # Use blank dict for unprovided wrapper configs. Not all wrappers even
        # have configs, so need to have a default kwargs.
        kwargs = wrapper_dict.get("wrapper_config", {})
        env = wrapper(env, **kwargs)

    return env


def genConfigFile(
    config_dir: str,
    param_space: dict,
    config_file_name: str = None,
    num_cpus: int = None,
    trainable: str = None,
    tune_config: dict = None,
    run_config: dict = None,
) -> dict:
    """Generate a config file for a ML tuning run.

    By default, file is saves as 'config_YY-MM-DD_HH-MM-SS.json'.

    Args:
        config_dir (`str`): Directory to save config file.
        param_space (`dict`): Search space of the tuning job. Must include the following
            keys:
            {
                "env_config": (`dict`) See `SSASchedulerParams`.
                }
        config_file_name (`str`, optional): The name of the config file. Do not include
            path or file extension. Defaults to 'config_YY-MM-DD_HH-MM-SS'.
        num_cpus (`int`, optional): Number of CPUs to use in training run. Use
            None to use maximum available on machine. Defaults to None.
        trainable (`str`, optional): Defaults to None.
        tune_config (`dict`, optional): Defaults to {}.
        run_config (`dict`, optional): Defaults to None.

    Returns:
        `dict`: Contains all of the arguments except `config_dir` and `config_file_name`.
    """
    # Check if environment config is in parameter space; we don't care about the contents
    # of env config, only that it exists, to stop really egregious arguments.
    if "env_config" not in param_space:
        raise ValueError("'env_config' is not in param_space")

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
        file_path = config_dir + "/config_" + date_time + ".json"
    else:
        # set file path for custom file name
        file_path = config_dir + "/" + config_file_name + ".json"

    # save json
    with open(file_path, "w") as outfile:
        outfile.write(json_object)

    print(f"Config file saved to {file_path} \n")
    return config_data
