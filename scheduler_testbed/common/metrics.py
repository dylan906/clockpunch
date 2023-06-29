"""Metrics to measure policy performance."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Tuple

# Third Party Imports
from numpy import (
    count_nonzero,
    mean,
    multiply,
    ndarray,
    stack,
    sum,
    trace,
    unique,
    zeros,
)


# %% Mean Uncertainty
def meanVarUncertainty(
    covs: list[ndarray],
    pos_vel: str = "position",
) -> float:
    """Calculate mean variance of error covariance matrices or diagonals.

    Args:
        covs (`list[ndarray]`): Each array must be (6, 6) or (6,). The first 3 diagonals
            elements (for a 2D input) or first 3 elements (for a 1D input) are position error
            covariance, the later 3 diagonal elements (or elements) are velocity error covariance.
        pos_vel (`str`, optional): "position" (default) | "velocity". Which parameter to
            calculate mean of.

    Returns:
        `float`: `mean(tr(cov_i))`, where i is a single covariance matrix, and the mean is
            taken over `i in 1:N` matrices.

    Example 1:
        covs = array([
                        [
                        [1., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0.1, 0., 0.],
                        [0., 0., 0., 0., 0.1, 0.],
                        [0., 0., 0., 0., 0., 0.1]
                        ],
                        [
                        [2., 0., 0., 0., 0., 0.],
                        [0., 2., 0., 0., 0., 0.],
                        [0., 0., 2., 0., 0., 0.],
                        [0., 0., 0., 0.2, 0., 0.],
                        [0., 0., 0., 0., 0.2, 0.],
                        [0., 0., 0., 0., 0., 0.2]
                        ]
                    ])
        meanVarUncertainty(covs) # returns 4.5

    Example 2:
        covs = array(
                    [[1, 1, 1, 0.1, 0.1, 0.1],
                    [2, 2, 2, 0.2, 0.2, 0.2]]
                    )
        meanVarUncertainty(covs) # returns 4.5


    """
    if type(covs) != list:
        raise TypeError("`covs` must be a list of arrays.")

    # If full covariance matrices are input, strip out the off-diagonals, and get the relevant
    # diagonals.
    if covs[0].ndim == 2:
        if pos_vel == "position":
            covs_stripped = [cov[:3, :3] for cov in covs]
        elif pos_vel == "velocity":
            covs_stripped = [cov[3:, 3:] for cov in covs]

        covs_stacked = stack(covs_stripped, axis=0)
        covs_traces = trace(covs_stacked, axis1=1, axis2=2)
    # If covariance diagonals are input, get the relevant diagonals.
    elif covs[0].ndim == 1:
        if pos_vel == "position":
            covs_stripped = [cov[:3] for cov in covs]
        elif pos_vel == "velocity":
            covs_stripped = [cov[3:] for cov in covs]
        # Set axis so that the elements of individual covariance matrices are summed, but
        # that the matrix traces are not added together.
        covs_traces = sum(covs_stripped, axis=1)

    mean_var = mean(covs_traces)

    return mean_var


# %% Unique Tasking Tracker
class TaskingMetricTracker:
    """Tracks persistent metrics related to sensor-to-target tracking.

    Attributes:
        targets_tasked (`list`): Non-repeating IDs of targets that have been tasked.
            Starts empty.
        unique_tasks (`int`): Number of unique targets tasked. Starts at 0.
        non_vis_tasked_est (`int`): Number of times any sensor has been tasked
            to an estimated non-visible target. Starts at 0.
        non_vis_tasked_truth (`int`): Number of times any sensor has been tasked
            to a truly non-visible target. Starts at 0.
        non_vis_by_sensor_est (`list[int]`): (M,) Number of times each sensor has
            been tasked to an estimated non-visible target. Starts at [0, ..., 0].
        non_vis_by_sensor_truth (`list[int]`): (M,) Number of times each sensor has
            been tasked to a truly non-visible target. Starts at [0, ..., 0].
        multiple_taskings (`int`): Number of times multiple sensors have been tasked to the
            same target. Any number of sensors >1 tasked to a single target counts as +1
            to the increment. Multiple multi-assignments count as multiple increments (e.g.
            if sensors A and B are assigned to target 1 and sensors C and D are assigned
            to target 2, `multiple_taskings` increments by +2). Starts at 0.

    Notes:
        - M = number of sensors; N = Number of targets
        - Attributes cannot decrease unless `TaskingMetricTracker` is reset via
            `TaskingMetricTracker.reset()`.
    """

    def __init__(
        self,
        sensor_ids: list,
        target_ids: list,
    ):
        """Initialize `TaskingMetricTracker`.

        Args:
            sensor_ids (`list`): List of sensor IDs.
            target_ids (`list`): List of target IDs.
        """
        self.sensor_ids = sensor_ids
        self.target_ids = target_ids
        self.num_sensors = len(sensor_ids)
        self.num_targets = len(target_ids)
        self.reset()

        return

    def reset(self):
        """Reset tracked targets to empty set and unique tasks to 0."""
        # _target_set is private attr used in getting unique targets.
        # targets_tasked is public attr that has same entries as _target_set,
        # but is a list.
        self._target_set = set()
        self.targets_tasked = []
        self.unique_tasks = 0
        self.non_vis_tasked_est = 0
        self.non_vis_tasked_truth = 0
        self.non_vis_by_sensor_est = zeros([self.num_sensors], dtype=int)
        self.non_vis_by_sensor_truth = zeros([self.num_sensors], dtype=int)
        self.multiple_taskings = 0

        return

    def update(
        self,
        actions: ndarray,
        vis_mask_est: ndarray,
        vis_mask_truth: ndarray,
    ) -> Tuple[int, set, int, int, int, list[int], list[int]]:
        """Update tracker with targets tasked.

        Args:
            actions (`ndarray`): (M,) Multi discrete action array valued 0-N.
                Values of N correspond to inaction.
            vis_mask_est (`ndarray`): (N, M) The estimated visibility mask. Values
                are binary (0/1). 1 indicates the sensor-target pair can see each
                other. The order of rows and columns must match order of
                `self.target_ids` and `self.sensor_ids`, respectively.
            vis_mask_truth (`ndarray`): (N, M) The truth visibility mask. Values
                are binary (0/1). 1 indicates the sensor-target pair can see each
                other. The order of rows and columns must match order of
                `self.target_ids` and `self.sensor_ids`, respectively.
        Returns:
            unique_tasks (`int`): Number of unique targets tasked.
            targets_tasked (`list`): IDs of targets that have been tasked.
            non_vis_tasked_est (`int`): Number of times any sensor has been tasked
                to an estimated non-visible target.
            non_vis_tasked_truth (`int`): Number of times any sensor has been
                tasked to a truly non-visible target.
            multiple_taskings (`int`): Number of times multiple sensors have been
                tasked to the same target.
            non_vis_by_sensor_est (`list[int]`): (M,) Number of times each sensor
                has been tasked to an estimated non-visible target.
            non_vis_by_sensor_truth (`list[int]`): (M,) Number of times each sensor
                has been tasked to a truly non-visible target.
        Notes:
            - Be consistent in format between subsequent updates. For example,
                if a target ID is an `int` the first time it is input, do not
                switch to a `str` later.
        """
        # strip out "inactions" (sensors not tasked to targets)
        actions_targets_only = actions[actions < self.num_targets]
        # %% Unique targets update

        # get IDs of all tasked targets
        tasked_ids = [self.target_ids[i] for i in actions_targets_only]

        # convert to set to eliminate duplicate entries and enable set operations
        tasked_ids = set(tasked_ids)

        # get items in target_ids that have not already been tasked
        new_list = tasked_ids.difference(self._target_set)

        # add to length of unique tasks
        self.unique_tasks += len(new_list)

        # update list of unique targets tracked
        self._target_set = self._target_set.union(new_list)
        self.targets_tasked = list(self._target_set)

        # %% Non-visible tasked update
        [
            non_vis_tasked_est_now,
            non_vis_by_sensor_est_now,
        ] = self._calcNonVisTaskings(vis_mask_est, actions)
        self.non_vis_tasked_est += non_vis_tasked_est_now
        self.non_vis_by_sensor_est += non_vis_by_sensor_est_now

        [
            non_vis_tasked_truth_now,
            non_vis_by_sensor_truth_now,
        ] = self._calcNonVisTaskings(vis_mask_truth, actions)
        self.non_vis_tasked_truth += non_vis_tasked_truth_now
        self.non_vis_by_sensor_truth += non_vis_by_sensor_truth_now
        # %% Multiple taskings update
        # Count number of times each target was tasked at this time
        [_, counts] = unique(
            actions_targets_only,
            return_counts=True,
        )
        # Get the number of instances of multi-targeting at this time
        num_mult_taskings = len(counts[counts > 1])
        self.multiple_taskings += num_mult_taskings

        return (
            self.unique_tasks,
            self.targets_tasked,
            self.non_vis_tasked_est,
            self.non_vis_tasked_truth,
            self.multiple_taskings,
            self.non_vis_by_sensor_est,
            self.non_vis_by_sensor_truth,
        )

    def _calcNonVisTaskings(
        self,
        vis_mask: ndarray[int],
        actions: ndarray[int],
    ) -> Tuple[int, list[int]]:
        """Calculate sensor-target non-visible taskings.

        Args:
            vis_mask (`ndarray[int]`): (N, M) visibility mask valued at 0 or 1.
            actions (`ndarray[int]`): (M, ) Multi discrete action array valued 0-N.

        Returns:
            non_vis_taskings (`int`): Number of instances of any sensor being
                tasked to a non-visible target at a single time.
            non_vis_by_sensor (`list[int]`): (M,) Number of instances of each
                sensor being tasked to a non-visible target at a single
                time.
        """
        assert len(actions) == self.num_sensors
        assert vis_mask.shape == (self.num_targets, self.num_sensors)

        # Transform MultiDiscrete action array into 2D binary (valued at 0 or 1)
        # array (without inaction row). Rows = targets, columns = sensors.
        actions_array = zeros([self.num_targets, self.num_sensors])
        for sens, targ in enumerate(actions):
            # skip if action is non-target (inaction)
            if targ < self.num_targets:
                actions_array[int(targ), sens] = 1

        # mask actions by visibility status
        masked_actions = multiply(actions_array, vis_mask)
        # get difference between actions array and masked actions array (will be array of
        # 1s and 0s)
        diff_actions = actions_array - masked_actions
        # count number of 1s in differenced array, which equals number of non-visible taskings
        num_non_vis_taskings = count_nonzero(diff_actions == 1)

        non_vis_sensors = sum(diff_actions, axis=0, dtype=int)

        return (num_non_vis_taskings, non_vis_sensors)
