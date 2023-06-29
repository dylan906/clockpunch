"""Test for metrics.py."""
# %% Imports


# Third Party Imports
from numpy import array, diag, eye, ones, zeros

# Punch Clock Imports
from scheduler_testbed.common.metrics import (
    TaskingMetricTracker,
    meanVarUncertainty,
)

# %% Test meanUncertainty
print("\nTest meanUncertainty...")
print("  ...with 2D inputs")
cov_block_a = eye(3)
cov_block_b = 0.1 * eye(3)

cov1 = zeros([6, 6])
cov1[:3, :3] = cov_block_a
cov1[3:, 3:] = cov_block_b

cov2 = 2 * cov1

# test with default input
covs = [cov1, cov2]
mean_unc = meanVarUncertainty(covs=covs)
print(f"covariances = \n{covs}")
print(f"mean uncertainty = {mean_unc}")

# test with optional input
mean_unc = meanVarUncertainty(covs=covs, pos_vel="velocity")
print(f"covariances = \n{covs}")
print(f"mean uncertainty = {mean_unc}")

# test incorrectly formatted input
try:
    mean_unc = meanVarUncertainty(covs=cov1)
except TypeError:
    print("error occured")

print("  ...with 1D inputs")
cov_diag1 = diag(cov1)
cov_diag2 = diag(cov2)
covs = [cov_diag1, cov_diag2]
mean_unc = meanVarUncertainty(covs)
print(f"covariances = \n{covs}")
print(f"mean uncertainty = {mean_unc}")


# %% initialize Tracker with 2 targets, 4 sensors
print("\nInitialize Tracker...")
target_ids = ["A", "B"]
sensor_ids = [0, 1, 2, 3]
tracker = TaskingMetricTracker(sensor_ids, target_ids)
# %% Test _calcNonVisTaskings
print("\nTest _calcNonVisTaskings()...")
tracker.reset()
vis_mask = zeros([2, 4])
vis_mask[0, 2] = 1
actions = array([0, 0, 1, 2])
[taskings, by_sensor] = tracker._calcNonVisTaskings(vis_mask, actions)
print(
    f"actions = {actions}"
    f"vis_mask = {vis_mask}"
    f"non-vis taskings = {taskings}"
    f"non-vis taskings by sensor = {by_sensor}"
)

# %% Task inaction-- choose max value in action array (2 targets, so inaction = 2)
print("\nTest tasking inaction...")
tracker.reset()
actions = array([2, 2, 2, 2])
vis_mask_truth = ones([2, 4])
vis_mask_est = ones([2, 4])
[
    unique_taskings,
    targets_tasked,
    non_vis_est,
    non_vis_truth,
    multiple_taskings,
    non_vis_by_sensor_est,
    non_vis_by_sensor_truth,
] = tracker.update(
    actions,
    vis_mask_est=vis_mask_est,
    vis_mask_truth=vis_mask_truth,
)

print("Inaction:")
print(
    f"  actions = {actions}\n"
    f"  unique_taskings = {unique_taskings} \n"
    f"  targets tracked = {targets_tasked}\n"
    f"  multiple taskings = {multiple_taskings}\n"
)

# %% Task target back-to-back
print("\nTest tasking same target back-to-back...")
# Task same target back-to-back. Unique_taskings and targets_tracked should change
# after the first tasking, but stay the same after the 2nd tasking.
print("Same target twice in a row:")
tracker.reset()
actions = array([0, 2, 2, 2])
for _ in range(2):
    [
        unique_taskings,
        targets_tasked,
        non_vis_est,
        non_vis_truth,
        multiple_taskings,
        non_vis_by_sensor_est,
        non_vis_by_sensor_truth,
    ] = tracker.update(
        actions=actions,
        vis_mask_est=vis_mask_est,
        vis_mask_truth=vis_mask_truth,
    )
    print(
        f"  actions = {actions}  \n"
        f"  unique_taskings = {unique_taskings} \n"
        f"  targets tracked = {targets_tasked} \n"
    )

# Task target multiple times in same instance
print("Same target by 2 sensors:")
tracker.reset()
actions = array([1, 1, 2, 2])
[
    unique_taskings,
    targets_tasked,
    non_vis_est,
    non_vis_truth,
    multiple_taskings,
    non_vis_by_sensor_est,
    non_vis_by_sensor_truth,
] = tracker.update(
    actions,
    vis_mask_est=vis_mask_est,
    vis_mask_truth=vis_mask_truth,
)
print(
    f"  actions = {actions} \n"
    f"  unique_taskings = {unique_taskings} \n"
    f"  targets tracked = {targets_tasked}\n"
    f"  multiple taskings = {multiple_taskings}\n"
)

# # task empty target list
# tracker.reset()
# actions = array([])
# [
#     unique_taskings,
#     targets_tasked,
#     non_vis_est,
#     non_vis_truth,
#     multiple_taskings,
#     non_vis_by_sensor_est,
#     non_vis_by_sensor_truth,
# ] = tracker.update(
#     actions,
#     vis_mask_est=vis_mask_est,
#     vis_mask_truth=vis_mask_truth,
# )
# print("Empty action array:")
# print(
#     f"  unique_taskings = {unique_taskings} \n"
#     f"  targets tracked = {targets_tasked}\n"
#     f"  multiple taskings = {multiple_taskings}\n"
# )

# %% Task non-visible targets
print("\nTask non-visible targets...")

# Set estimated and truth visibility masks differently. Est and truth tasking
# counts should output differently.
# Test with two visibility mask configs:
#   1. estimated mask to all-visible, truth mask is all-not-visible.
#   2. mixed-values in both visibility masks

actions = array([0, 0, 1, 1])
vis_mask_est = ones([2, 4])
vis_mask_truth = zeros([2, 4])
print(f"actions = {actions}")
for _ in range(2):
    tracker.reset()

    [
        unique_taskings,
        targets_tasked,
        non_vis_est,
        non_vis_truth,
        multiple_taskings,
        non_vis_by_sensor_est,
        non_vis_by_sensor_truth,
    ] = tracker.update(
        actions,
        vis_mask_est=vis_mask_est,
        vis_mask_truth=vis_mask_truth,
    )
    print("Non-visible taskings:")
    print(
        f"  vis_mask_est = \n{vis_mask_est}\n"
        f"  vis_mask_truth = \n{vis_mask_truth}\n"
        f"  non-vis taskings (est) = {non_vis_est}\n"
        f"  non-vis taskings by sensor (est) = {non_vis_by_sensor_est}\n"
        f"  non-vis taskings (truth) = {non_vis_truth}\n"
        f"  non-vis taskings by sensor (truth) = {non_vis_by_sensor_truth}\n"
    )
    # set visibility masks to mixed values for 2nd test
    vis_mask_est[0, 0] = 0
    vis_mask_truth[0, 0] = 1


# %% Multiple instances of multi-tasking
print("\nTest multiple instances of multi-tasking...")
# Task 2 sensors to Target A and 2 sensors to Target B. Expect multiple_taskings
# and unique_taskings to increase by 2.
tracker.reset()
actions = array([0, 0, 1, 1])
[
    unique_taskings,
    targets_tasked,
    non_vis_est,
    non_vis_truth,
    multiple_taskings,
    non_vis_by_sensor_est,
    non_vis_by_sensor_truth,
] = tracker.update(
    actions,
    vis_mask_est=vis_mask_est,
    vis_mask_truth=vis_mask_truth,
)
print("\nMultiple multi-taskings (at same time):")
print(
    f"  actions = {actions} \n"
    f"  unique_taskings = {unique_taskings} \n"
    f"  targets tracked = {targets_tasked}\n"
    f"  multiple taskings = {multiple_taskings}\n"
)
# %% done
print("done")
