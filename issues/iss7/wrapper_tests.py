"""Different wrapper configs for issue #7."""
# Third Party Imports
from gymnasium.spaces.utils import flatten_space
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_tuner import buildEnv

# %% Env config
# Most of this is boilerplate; the important dict is
# env_config["constructor_params"]["wrappers"]

env_config = {
    "horizon": 10,
    "agent_params": {
        "num_sensors": 2,
        "num_targets": 4,
        "sensor_dynamics": "terrestrial",
        "target_dynamics": "satellite",
        "sensor_dist": None,
        "target_dist": "normal",
        "sensor_dist_frame": None,
        "target_dist_frame": "COE",
        "sensor_dist_params": None,
        "target_dist_params": [
            [5000, 200],
            [0, 0],
            [0, 3.141592653589793],
            [0, 6.283185307179586],
            [0, 6.283185307179586],
            [0, 6.283185307179586],
        ],
        "fixed_sensors": [
            [6378.1363, 0.0, 0.0, 0.0, 0.46369050901000003, 0.0],
            [0.0, 6378.1363, 0.0, -0.46369050901000003, 0.0, 0.0],
        ],
        "fixed_targets": None,
    },
    "filter_params": {
        "Q": [
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e-05, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1e-05, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1e-05],
        ],
        "R": [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.001],
        ],
        "p_init": [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        ],
    },
    "reward_params": {
        "reward_func": "Threshold",
        "obs_or_info": "obs",
        "metric": "num_tasked",
        "preprocessors": ["min"],
        "metric_value": 3,
        "inequality": ">",
        "penalty": 1,
    },
    "time_step": 100,
    "constructor_params": {
        # Comment out individual or combinations of wrappers to test
        "wrappers": [
            {
                "wrapper": "filter_observation",
                "wrapper_config": {"filter_keys": ["vis_map_est"]},
            },
            {"wrapper": "action_mask"},
            {"wrapper": "flat_dict"},
        ]
    },
}

# %% Register env and model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

# %% Build env, check that it works
env = buildEnv(env_config)
env.reset()
env.step(env.action_space.sample())
print(f"\nobs space = {env.observation_space}")
print(f"action space = {env.action_space}")

# %% Build Algo
# num_inputs = flatten_space(env.observation_space.spaces["observations"]).shape[
#     0
# ]
# print(f"num_inputs = {num_inputs}")
# num_outputs = flatten_space(env.action_space).shape[0]
# print(f"num_outputs = {num_outputs}")

algo_config = (
    ppo.PPOConfig()
    .training(
        model={
            "custom_model": MyActionMaskModel,
            # "fcnet_hiddens": [8, 10],
            # "custom_model_config": {
            #     "num_inputs": num_inputs,
            #     "num_outputs": num_outputs,
            # },
        }
    )
    .environment(
        env="my_env",
        env_config=env_config,
    )
    .framework("torch")
)
algo_config.validate()
algo = algo_config.build()
algo.training_step()
