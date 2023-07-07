"""Test for build_tuner.py."""
# NOTE: This test uses a config file in the test directory ("config_test.json")
# NOTE: This test generates a set of files in a directory specified by config_test.json
# %% Imports
# Standard Library Imports
import os
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
import ray
from gymnasium.spaces.utils import flatten, flatten_space
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.stopper import MaximumIterationStopper
from torch import tensor

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile, printNestedDict
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_env import buildEnv
from punchclock.ray.build_tuner import (
    _getDefaults,
    _getExperimentName,
    appendNewKeys,
    buildTuner,
)

# %% Load json file
# path to the folder this script is contained
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"dir path: {dir_path}")

# config file name
config_name = "config_test.json"

file_path = dir_path + "/" + config_name
print(f"file path: {file_path}")

config = loadJSONFile(file_name=file_path)

# %% Test appendNewKeys
print("\nTest appendNewKeys...")
dict1 = {
    "key1": 0,
    "key2": "g",
}
dict2 = {
    "key1": 4,
    "key3": "foo",
}
print(f"dict1 = {dict1}")
print(f"dict2 = {dict2}")

out_dict = appendNewKeys(base_dict=dict1, new_dict=dict2)
print(f"out_dict = {out_dict}")
# %% Test _getDefaults
print("\nTest _getDefaults...")
[num_cpus, param_space_out] = _getDefaults(
    param_space=config["param_space"],
    num_cpus=config["num_cpus"],
)

print(f"num_cpus = {num_cpus}")
print("param_space =")
printNestedDict(param_space_out)

# %% Test _getExperimentName
print("\nTest _getExperimentName...")
# test without name provided
run_params_in = {}
run_params_out = _getExperimentName(run_params_in)
print(f"run_params_in = {run_params_in}")
print(f"run_params_out = {run_params_out}")
# test with name provided
run_params_in = {"name": "name"}
run_params_out = _getExperimentName(run_params_in)
print(f"run_params_in = {run_params_in}")
print(f"run_params_out = {run_params_out}")

# %% Model/environment interface tests
print("\nInterface tests...")
# See Ray example: https://github.com/ray-project/ray/blob/877770e2d5c1d6e76d68998b0520390986318787/rllib/examples/action_masking.py
# See Algorithms doc: https://docs.ray.io/en/latest/rllib/package_ref/algorithm.html#defining-algorithms-with-the-algorithmconfig-class
# Manually build algo under 3 conditions:
#  1. Just the custom env (use Ray default model)
#  2. Just the custom model (use RandomEnv as env)
#  3. Both the custom env and custom model
# The purpose of the above tests is to test interfaces (assumes custom model and
# custom env pass tests in isolation).

# register env and model; build environments
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)
test_env = buildEnv(env_config=config["param_space"]["env_config"])
print(f"observations space = {test_env.observation_space}")
print(f"action space = {test_env.action_space}")
rand_env = RandomEnv(
    {
        "observation_space": deepcopy(test_env.observation_space),
        "action_space": deepcopy(test_env.action_space),
    }
)
# Check that custom and rand environments have same obs/action spaces
print("Random environment checks:")
print(
    f"    Observation spaces match? {test_env.observation_space == rand_env.observation_space}"
)
print(
    f"    Action spaces match? {test_env.action_space == rand_env.action_space}"
)

# # Disable preprocessor
# config["param_space"]["model"]["_disable_preprocessor_api"] = True

# Set algo configs (put all configs in dict for easier looping through tests)
print("Build algo configs...")
algo_configs = {}
algo_configs["custom_env"] = (
    ppo.PPOConfig()
    .training()
    .environment(
        env="my_env",
        env_config=config["param_space"]["env_config"],
    )
    .framework("torch")
)
algo_configs["custom_model"] = (
    ppo.PPOConfig()
    .training(
        model={**config["param_space"]["model"]},
    )
    .environment(
        env=RandomEnv,
        env_config={
            "observation_space": rand_env.observation_space,
            "action_space": rand_env.action_space,
        },
    )
    .framework("torch")
)
algo_configs["full_custom"] = (
    ppo.PPOConfig()
    .training(
        model={**config["param_space"]["model"]},
        # train_batch_size=1,
        # sgd_minibatch_size=1,
    )
    .environment(
        env="my_env",
        env_config=config["param_space"]["env_config"],
    )
    .framework("torch")
    # .rollouts(rollout_fragment_length="auto")
)

# build algos from algo configs
print("Build algos...")
algos = {k: v.build() for (k, v) in algo_configs.items()}

# Check that observations flow through algo policies correctly. Use sample observation
# from test environment as input to policies. The sample should be the same for
# both the custom env and random env (since the random env was instantiated by
# copying the custom env observation and action spaces).

raw_ob = test_env.observation_space.sample()
print(f"    raw observation = {raw_ob}")

actions = {k: v.compute_single_action(raw_ob) for (k, v) in algos.items()}
print(f"    algo actions = {actions}")

results = {}
for k, v in algos.items():
    results[k] = v.training_step()
    print(f"{k} Training step successful")

fit_results = {}
for k, v in algo_configs.items():
    tuner = tune.Tuner(
        "PPO",
        param_space=v.to_dict(),
        run_config=air.config.RunConfig(
            # stop=MaximumIterationStopper(max_iter=1)
            stop={"timesteps_total": 10}
        ),
    )
    fit_results[k] = tuner.fit()
    print(f"{k} Tune successful")

# %% Test buildTuner
print("\nTest buildTuner...")
# Test may take a minute
tuner = buildTuner(config, override_date=True)
print(f"tuner = {tuner}")

# %% Test fit
print("Running fit...")
tuner.fit()
print("...fit complete.")

# %%
print("done")
