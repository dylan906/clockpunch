"""Test analysis_utils/utils.py."""
# %% Imports

# Third Party Imports
from numpy import array
from numpy.random import randint

# Punch Clock Imports
from punchclock.analysis_utils.utils import (
    calc3dTr,
    calcMissedOpportunities,
    countMaskViolations,
    countNullActions,
    countOpportunities,
)
from punchclock.common.utilities import MaskConverter

# %% Count Null Actions
print("\nTest countNullActions...")
null_indx = 3
actions = array([1, 3, 0, 5])
num_null_acts = countNullActions(x=actions, null_action_indx=null_indx)
print(f"null index = {null_indx}")
print(f"actions = {actions}")
print(f"num null actions = {num_null_acts}")

# %% calc3dTr
print("\nTest calc3dTr...")
x = randint(0, 2, size=(3, 2, 2))
trace = calc3dTr(x)
print(f"x = \n{x}")
print(f"3d tr(x) = {trace}")
# %% countOpportunities
print("\nTest countOpportunities...")
mask = array([[1, 1, 1]])
opps = countOpportunities(mask)
print(f"mask = {mask}")
print(f"opportunities = {opps}")

mask = array([[0, 0, 0]])
opps = countOpportunities(mask)
print(f"mask = {mask}")
print(f"opportunities = {opps}")

mask = array([[0, 1, 0]])
opps = countOpportunities(mask)
print(f"mask = {mask}")
print(f"opportunities = {opps}")

mask = array([[0, 1, 0], [1, 0, 1]])
opps = countOpportunities(mask)
print(f"mask = \n{mask}")
print(f"opportunities = {opps}")

mask = array([[1, 0, 0], [1, 0, 0]])
opps = countOpportunities(mask)
print(f"mask = \n{mask}")
print(f"opportunities = {opps}")

# %% calcMissed Opportunities
print("\nTest calcMissedOpportunities...")
mc = MaskConverter(
    num_targets=3,
    num_sensors=2,
)

# Set actions to inaction
# Set mask to all-visible in one column, all non-visible in other column
action = mc.action_space.sample()
action.fill(mc.action_space.nvec[0] - 1)
mask = mc.vis_mask_space.sample()
mask[:, 0].fill(1)
mask[:, 1].fill(0)
missed_opps = calcMissedOpportunities(
    action=action,
    mask=mask,
    mask_converter=mc,
)
print(f"action = {action}")
print(f"mask = \n{mask}")
print(f"missed opportunities = {missed_opps}\n")

# Try with active actions (missed_opps should be 0)
action.fill(0)
missed_opps = calcMissedOpportunities(
    action=action,
    mask=mask,
    mask_converter=mc,
)
print(f"action = {action}")
print(f"mask = \n{mask}")
print(f"missed opportunities = {missed_opps}")

# Test with list-action
action.fill(mc.action_space.nvec[0] - 1)
action = action.tolist()
missed_opps = calcMissedOpportunities(
    action=action,
    mask=mask,
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
