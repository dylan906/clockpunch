"""Tests for postprocess_sim_results.py."""
# NOTE: This test requires a data file "/simresults_df.pkl"
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
from pandas import read_pickle

# Punch Clock Imports
from punchclock.simulation.postprocess_sim_results import addPostProcessedCols

# %% Load dataframe
fpath = os.path.dirname(os.path.realpath(__file__))
df_path = fpath + "/simresults_df.pkl"
df = read_pickle(df_path)

# %% addPostProcessedCols
print("\nTest addPostProcessedCols...")
df_processed = addPostProcessedCols(df=df, info={})
print(f"cols = {df_processed.columns}")
print(
    df_processed[
        [
            "cov_tr",
            "cov_mean",
            "cov_max",
            "null_action",
            "cum_null_action",
        ]
    ]
)


# %% Done
print("done")
