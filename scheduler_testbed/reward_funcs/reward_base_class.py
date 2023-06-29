"""Reward function base class."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from typing import Tuple

# Third Party Imports
from numpy import count_nonzero, ndarray

# Punch Clock Imports
from scheduler_testbed.common.utilities import actionSpace2Array
from scheduler_testbed.policies.policy_utils import (
    multipleAssignmentCost,
    nonVisibleAssignmentCost,
)


# %% Base class
class RewardFunc:
    """Base class for reward functions."""

    def __init__(
        self,
        penalties: dict = None,
        subsidies: dict = None,
    ):
        """Initialize base RewardFunc.

        Args:
            penalties (`dict`, optional): Values for base penalties (non-visible
                assignment and multi-assignment). Optionally enter a dict with
                the following structure:
                    {
                        "multi_assignment": float (default 0),
                        "non_vis_assignment": float (default 0)
                    }
            subsidies (`dict`, optional): Values for base subsidies. Optionally
                enter a dict with the following structure:
                    {
                        "inaction": float (default 0),
                        "active_action": float (default 0)
                    }

        """
        # default penalties
        default_multi_assignment = 0
        default_non_vis_assignment = 0
        if penalties is None:
            penalties = {
                "multi_assignment": default_multi_assignment,
                "non_vis_assignment": default_non_vis_assignment,
            }
        if "multi_assignment" not in penalties:
            penalties["multi_assignment"] = default_multi_assignment
        if "non_vis_assignment" not in penalties:
            penalties["non_vis_assignment"] = default_non_vis_assignment

        self.penalties = penalties

        # default subsidies
        default_inaction = 0
        default_active_action = 0
        if subsidies is None:
            subsidies = {
                "inaction": default_inaction,
                "active_action": default_active_action,
            }
        if "inaction" not in subsidies:
            subsidies["inaction"] = default_inaction
        if "active_action" not in subsidies:
            subsidies["active_action"] = default_active_action

        self.subsidies = subsidies

        return

    def reset(self):
        """Overwrite."""

    def calcReward(self, obs, info: dict, actions: ndarray[int]) -> float:
        """Overwrite."""
        reward = 0
        return reward

    def calcSubsidies(
        self,
        info: dict,
        actions: ndarray[int],
    ) -> Tuple[float, dict]:
        """Calculate subsidies.

        Args:
            info (`dict`): Info from environment.
            actions (`ndarray[int]`): (M, ) Actions valued 0 to N, where N indicates
                inaction.

        Returns:
            subsidies (`float`): Subsidy >= 0.
            subsidy_report (`dict`): Contains number of subsidies itemized by type
                of subsidy (e.g. inaction, active_action). Used mainly for debugging.
                If a subsidy value is set to 0 during reward_func initialization,
                then instances of that event happening are not counted in the subsidy
                report.
        """
        # Get number of inactions and active actions, then multiply each by the
        # amount of the appropriate subsidy value, according to self.subsidies.
        inaction = info["num_targets"]
        num_inaction = count_nonzero(actions == inaction)
        num_active_actions = count_nonzero(actions != inaction)

        subsidy_inaction = self.subsidies["inaction"] * num_inaction
        subsidy_active_action = (
            self.subsidies["active_action"] * num_active_actions
        )
        subsidy = sum([subsidy_inaction, subsidy_active_action])

        # Only count instances of subsidies if non-zero subsidy value is set
        num_inaction_subsidies = num_inaction * zeroIfZero(
            self.subsidies["inaction"]
        )
        num_active_action_subsidies = num_active_actions * zeroIfZero(
            self.subsidies["active_action"]
        )
        subsidy_report = {
            "num_inaction": num_inaction_subsidies,
            "num_active_action": num_active_action_subsidies,
            "num_subsidies_total": sum(
                [num_inaction_subsidies, num_active_action_subsidies]
            ),
        }

        return (subsidy, subsidy_report)

    def calcPenalties(
        self,
        info: dict,
        actions: ndarray[int],
    ) -> Tuple[float, dict]:
        """Calculates penalties (negative reward).

        Optionally overwrite.

        This function is the default for `RewardFunc` base class. Replace if you
        want to include more than just multi-assignment and non-visible penalties.

        Notation:
            N = number of targets
            M = number of sensors

        Args:
            info (`dict`): Info from environment.
            actions (`ndarray[int]`): (M, ) Actions valued 0 to N, where N indicates
                inaction.

        Returns:
            penalties (`float`): Penalty <= 0.
            penalty_report (`dict`): Contains number of penalties itemized by type
                of penalty (e.g. non-visible, multi-assignment). Used mainly for
                debugging.
        """
        true_vis_map = info["vis_map_truth"]
        num_sensors = info["num_sensors"]
        num_targets = info["num_targets"]

        # convert actions array to 2D with 1s and 0s
        actions_2d_array = actionSpace2Array(
            actions=actions,
            num_sensors=num_sensors,
            num_targets=num_targets,
        )
        # get multi-assignment penalties
        [penalty_ma, ma_penalty_report] = multipleAssignmentCost(
            actions=actions_2d_array,
            inaction_row=True,
            cost=self.penalties["multi_assignment"],
        )
        # Get non-visible assignment penalties. Note that we pass an (N, M) array to
        # nonVisibleAssignmentCost-- we don't pass in the inaction row.
        [penalty_nv, num_penalties_nv] = nonVisibleAssignmentCost(
            actions=actions_2d_array[:-1, :],
            vis_map=true_vis_map,
            cost=self.penalties["non_vis_assignment"],
        )

        # total penalties
        penalties = sum([penalty_ma, penalty_nv])

        # Build penalties report (for debugging and deeper insight). Delete 'total'
        # item from multi-assignment report.
        penalty_report = deepcopy(ma_penalty_report)
        del penalty_report["num_penalties"]
        penalty_report["num_non_vis_taskings"] = num_penalties_nv
        penalty_report["num_penalties_total"] = sum(penalty_report.values())

        return (penalties, penalty_report)

    def calcNetReward(self, obs, info: dict, actions: ndarray[int]) -> float:
        """Don't overwrite."""
        reward = self.calcReward(obs, info, actions)
        [subsidies, _] = self.calcSubsidies(info, actions)
        [penalties, _] = self.calcPenalties(info, actions)

        return sum([reward, subsidies, penalties])


def zeroIfZero(val: float) -> int:
    """Return 0 if val is 0, and 1 otherwise."""
    if val == 0:
        return 0
    else:
        return 1
