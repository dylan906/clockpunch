"""Test for mc_config.py."""
# NOTE: This script generates a file: "data/config_mc.json""
# NOTE: This script requires "data/config_env.json" to run
# NOTE: This script requires a Ray checkpoint in "/data/test_checkpoint2/" to run
# NOTE: If this script isn't working, try regenerating the Ray checkpoint by running
# gen_checkpoint.py (in "data/")
# %% Imports
# Standard Library Imports
import os
from copy import deepcopy
from pathlib import Path

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
fpath = Path(__file__).parent
env_config_path = fpath.joinpath("data/config_env").with_suffix(".json")
env_config = loadJSONFile(env_config_path)
env_config["horizon"] = 2

env_config2 = deepcopy(env_config)
env_config2["horizon"] = 4
# %% Build policy configs
print("\nBuild policy configs...")

# For Ray policies, configs are just checkpoint paths. For CustomPolicies, configs
# are dicts.
checkpoint_path = fpath.joinpath(
    # "data/test_checkpoint2/test_trial/PPO_ssa_env_6a185_00000_0_2023-09-20_12-40-31/checkpoint_000001/policies/default_policy"
    "data/test_checkpoint2/test_trial/PPO_ssa_env_172e4_00000_0_2023-10-17_17-28-18/checkpoint_000001/policies/default_policy"  # noqa
)
results_dir = fpath.joinpath("data/mc_results")

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
    env_configs=[env_config, env_config2],
    results_dir=results_dir,
    print_status=True,
)

# %% Save MCConfig as json
print("\nSaving MCConfig...")
results_path = fpath.joinpath("data/config_mc").with_suffix(".json")
mc_config.save(results_path, append_timestamp=False)

# %% Load MC Config and check that policies can be built
print("\nLoading JSON...")
loaded_config = loadJSONFile(results_path)

for p_config in loaded_config["policy_configs"]:
    policy = buildCustomOrRayPolicy(p_config)
    print(f"policy = {policy}")

# %% done
print("done")
