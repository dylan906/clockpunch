"""Tests for postprocess_sim_results.py."""
# NOTE: This test requires a data file "/simresults_df.pkl"
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
from numpy import array
from numpy.random import randint
from pandas import read_pickle

# Punch Clock Imports
from punchclock.analysis_utils.postprocess_sim_results import (
    addPostProcessedCols,
    calc3dTr,
    calcMissedOpportunities,
    countMaskViolations,
    countNullActions,
    countOpportunities,
)
from punchclock.common.utilities import MaskConverter

# %% Load test sim results
print("\nTest sim results...")

fpath = os.path.dirname(os.path.realpath(__file__))
df_path = fpath + "/simresults_df.pkl"
df = read_pickle(df_path)

# %% Count Null Actions
print("\nTest countNullActions...")
null_indx = 3
actions = array([1, 3, 0, 5])
num_null_acts = countNullActions(x=actions, null_action_indx=null_indx)
print(f"null index = {null_indx}")
print(f"actions = {actions}")
print(f"num null actions = {num_null_acts}")

# %% get3dCovTr
print("\nTest calc3dTr...")
x = randint(0, 2, size=(3, 2, 2))
trace = calc3dTr(x)
print(f"x = \n{x}")
print(f"3d tr(x) = {trace}")
# %% countOpportunities
print("\nTest countOpportunities...")
vis_map = array([[1, 1, 1]])
opps = countOpportunities(vis_map)
print(f"vis_map = {vis_map}")
print(f"opportunities = {opps}")

vis_map = array([[0, 0, 0]])
opps = countOpportunities(vis_map)
print(f"vis_map = {vis_map}")
print(f"opportunities = {opps}")

vis_map = array([[0, 1, 0]])
opps = countOpportunities(vis_map)
print(f"vis_map = {vis_map}")
print(f"opportunities = {opps}")

vis_map = array([[0, 1, 0], [1, 0, 1]])
opps = countOpportunities(vis_map)
print(f"vis_map = \n{vis_map}")
print(f"opportunities = {opps}")

# %% calcMissed Opportunities
print("\nTest calcMissedOpportunities...")
mc = MaskConverter(
    num_targets=df["num_targets"][0],
    num_sensors=df["num_sensors"][0],
)

# Set actions to inaction
# Set mask to all-visible in one column, all non-visible in other column
action = mc.action_space.sample()
action.fill(mc.action_space.nvec[0] - 1)
vis_map = mc.vis_mask_space.sample()
vis_map[:, 0].fill(1)
vis_map[:, 1].fill(0)
missed_opps = calcMissedOpportunities(
    action=action,
    vis_map=vis_map,
    mask_converter=mc,
)
print(f"action = {action}")
print(f"vis_map = \n{vis_map}")
print(f"missed opportunities = {missed_opps}\n")

# Try with active actions (missed_opps should be 0)
action.fill(0)
missed_opps = calcMissedOpportunities(
    action=action,
    vis_map=vis_map,
    mask_converter=mc,
)
print(f"action = {action}")
print(f"vis_map = \n{vis_map}")
print(f"missed opportunities = {missed_opps}")

# Test with list-action
action.fill(mc.action_space.nvec[0] - 1)
action = action.tolist()
missed_opps = calcMissedOpportunities(
    action=action,
    vis_map=vis_map,
    mask_converter=mc,
)
print(f"action = {action}")
print(f"missed opportunities = {missed_opps}")

# %% countMaskViolations
print("\nTest countMaskViolations...")
x = randint(0, 2, size=(2, 2))
mask = randint(0, 2, size=(2, 2))
vio = countMaskViolations(x, mask)
print(f"x = \n{x}")
print(f"mask = \n{mask}")
print(f"violations = {vio}")

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
            "wasted_action",
            "cum_wasted_action",
            "num_opportunities",
            "cum_opportunities",
            "action_mask_violations",
        ]
    ]
)


# %% Done
print("done")
