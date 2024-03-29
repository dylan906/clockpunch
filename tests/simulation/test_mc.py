"""Tests for mc.py."""
# NOTE: Running script saves multiple files.
# NOTE: This script requires "config_mc.json" to run
# NOTE: If this test is not running correctly, first try running test_mc_config.py
# to regenerate config_mc.json.
# %% Imports
# Standard Library Imports
import warnings
from copy import deepcopy
from multiprocessing import active_children
from pathlib import Path

# Third Party Imports
from pandas import DataFrame, concat, read_csv, read_pickle

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.simulation.mc import MonteCarloRunner

# %% Load MC Config
fpath = Path(__file__).parent
mc_config_path = fpath.joinpath("data/config_mc").with_suffix(".json")
mc_config_loaded = loadJSONFile(mc_config_path)

mc_config_loaded["num_episodes"] = 2
print(mc_config_loaded)
mc_config_loaded["print_status"] = True

# %% Test MC Runner
print("\nTest initialization...")
mcr = MonteCarloRunner(**mc_config_loaded)
print(f"Monte Carlo runner = {mcr}")
print(f"active children: {active_children()}")

# %% Test runTrial
print("\nTest runTrial...")
mcr2 = deepcopy(mcr)
mcr2.runTrial(
    trial_name=mcr2.trial_names[0],
    env_config=deepcopy(mcr2.env_configs[0]),
    policy_config_or_checkpoint=deepcopy(mcr2.policy_configs[0]),
)
print(f"active children: {active_children()}")

# %% Test runMC without multiprocess (and stochastic random conditions)
print("\nTest runMC without multiprocessor...")
mcr3 = deepcopy(mcr)
exp_dir, mc_results = mcr3.runMC(multiprocess=False)
print(f"Monte Carlo results = {mc_results}")
print(f"active children: {active_children()}")
combined_df = DataFrame()
for path in exp_dir.glob("*/*.pkl"):
    loaded_df = read_pickle(path)
    combined_df = concat([combined_df, loaded_df])
print("Loaded results DF:\n", combined_df[["trial", "episode", "step", "seed"]])

# %% runMC with fixed initial conditions
print("\nTest runMC with fixed initial conditions...")
mc_config4 = deepcopy(mc_config_loaded)
mc_config4["num_episodes"] = 1
mc_config4["static_initial_conditions"] = True
mc_config4["print_status"] = False
mcr4 = MonteCarloRunner(**mc_config4)
exp_dir, results = mcr4.runMC()
combined_df = DataFrame()
for path in exp_dir.glob("*/*.pkl"):
    loaded_df = read_pickle(path)
    combined_df = concat([combined_df, loaded_df])
print("Loaded results DF:\n", combined_df[["trial", "episode", "step", "seed"]])

# %% runMC save as csv
print("\nTest runMC with save_format as csv...")
mc_config4b = deepcopy(mc_config_loaded)
mc_config4b["save_format"] = "csv"
mcr4b = MonteCarloRunner(**mc_config4b)
exp_dir, results = mcr4b.runMC()
combined_df = DataFrame()
for path in exp_dir.glob("*/*.csv"):
    loaded_df = read_csv(path)
    combined_df = concat([combined_df, loaded_df])
print("Loaded results DF:\n", combined_df[["trial", "episode", "step", "seed"]])

# %% Test Error and warning catchers
print("\nTest error and warning catchers...")
mc_config5 = deepcopy(mc_config_loaded)

# Test <1 number of episodes
print("  \nTest <1 number of episodes...")
mc_config5["num_episodes"] = 0
try:
    mcr = MonteCarloRunner(**mc_config5)
except Exception as e:
    print(e)

# Test fixed initial conditions but with >1 episodes
print("  \nTest fixed initial conditions but with >1 episodes...")
mc_config5["num_episodes"] = 3
mc_config5["static_initial_conditions"] = True
try:
    mcr = MonteCarloRunner(**mc_config5)
except Exception as e:
    print(e)

# Test Set env_config["seed"] to improper value
print("  \nTest Set env_config['seed'] to improper value...")
mc_config5["static_initial_conditions"] = False
mc_config5["env_configs"][0]["seed"] = None
try:
    mcr = MonteCarloRunner(**mc_config5)
    print("Test failed")
except Exception as e:
    print(e)
    print("Test passed")

# Set env_config seed to real number and static_initial_conditions to False
# (should get 1 warning)
print("  \nSet env_config seed to real number and static_initial_conditions to False")
mc_config5["env_configs"][0]["seed"] = 1
mc_config5["static_initial_conditions"] = False
with warnings.catch_warnings(record=True) as w:
    mcr = MonteCarloRunner(**mc_config5)
    assert len(w) == 1

# Set env_config seed to real number and static_initial_conditions to True
# (should get 1 warning)
print("  \nSet env_config seed to real number and static_initial_conditions to True")
mc_config5["static_initial_conditions"] = True
with warnings.catch_warnings(record=True) as w:
    mcr = MonteCarloRunner(**mc_config5)
    assert len(w) == 1

# %% Test runMC w/ multiprocess
# # NOTE: This test hangs on loading the Ray policy.
print("\nTest runMC w/ multiprocessor...")
mcr1 = deepcopy(mcr)
mc_results = mcr1.runMC(multiprocess=True)
print(f"Monte Carlo results = {mc_results}")

print("\nTest getDataFrame")
results_df = mcr1.getDataFrame()
print(f"results_df = {results_df}")
print(f"results_df.columns = {results_df.columns}")
# %% Done
print("done")
