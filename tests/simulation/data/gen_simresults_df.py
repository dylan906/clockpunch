"""Generate a data file for test_postprocess_sim_results.py to ingest."""
# %% Imports

# Standard Library Imports
import os

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.policies.random_policy import RandomPolicy
from punchclock.ray.build_env import buildEnv
from punchclock.simulation.sim_runner import SimRunner

# %% Load env config
print("\nLoad environment config...")
fpath = os.path.dirname(os.path.realpath(__file__))
env_config_path = fpath + "/config_env.json"
env_config = loadJSONFile(env_config_path)

# loaded env has FlatDict wrapper at end of wrappers, but this test operates on
# a lower level (SimRunner), that doesn't interface with FlatDict. So delete that
# wrapper from config before building env.
del env_config["constructor_params"]["wrappers"][-1]

# %% Build policy
env = buildEnv(env_config)

policy = RandomPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    use_mask=True,
)

# %% Build and run SimRunner
simrunner = SimRunner(env=env, policy=policy, max_steps=env_config["horizon"])
results = simrunner.runSim()
df = results.toDataFrame()

# %% Save dataframe as pickle
fpath = os.path.dirname(os.path.realpath(__file__))
savepath = fpath + "/simresults_df.pkl"
df.to_pickle(savepath)

# %% done
print("done")
