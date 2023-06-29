"""Test for mc_config.py."""
# NOTE: This script generates a .json file.
# NOTE: This script requires config_env.json to run
# %% Imports
# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from numpy.random import default_rng

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.policies.policy_builder import (
    buildCustomPolicy,
    buildSpaceConfig,
)
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.mc_config import MonteCarloConfig
from punchclock.simulation.sim_utils import buildCustomOrRayPolicy

# %% Load env config
print("\nLoad environment config...")
env_config = loadJSONFile("tests/simulation/data/config_env.json")
# env_config["seed"] = 9793741
env_config["horizon"] = 2
# %% Build policy configs
print("\nBuild policy configs...")

# For Ray policies, configs are just checkpoint paths. For CustomPolicies, configs
# are dicts.

checkpoint_path = "tests/simulation/data/test_checkpoint/checkpoint_000200/policies/default_policy"
results_dir = "tests/simulation/data/mc_results"

test_env = buildEnv(env_config)
obs_space_config = buildSpaceConfig(test_env.env.observation_space).toDict()
act_space_config = buildSpaceConfig(test_env.env.action_space).toDict()

rand_pol_config = {
    "policy": "RandomPolicy",
    "observation_space": obs_space_config,
    "action_space": act_space_config,
}

greedy_pol_config = {
    "policy": "GreedyCovariance",
    "observation_space": obs_space_config,
    "action_space": act_space_config,
    "epsilon": 0.5,
    "mode": "position",
}

policy_configs = [checkpoint_path, rand_pol_config, greedy_pol_config]
# policy_configs = [checkpoint_path]

# %% Initialize MCConfig
print("\nInitialize MCConfig...")
mc_config = MonteCarloConfig(
    num_episodes=1,
    policy_configs=policy_configs,
    env_config=env_config,
    results_dir=results_dir,
    print_status=True,
)

# %% Save MCConfig as json
print("\nSaving MCConfig...")
results_path = "tests/simulation/data/test_mc_config"
mc_config.save(results_path, append_timestamp=False)

# %% Load MC Config and check that policies can be built
print("\nLoading JSON...")
loaded_config = loadJSONFile(results_path + ".json")

for p_config in loaded_config["policy_configs"]:
    policy = buildCustomOrRayPolicy(p_config)
    print(f"policy = {policy}")
# %% done
print("done")
