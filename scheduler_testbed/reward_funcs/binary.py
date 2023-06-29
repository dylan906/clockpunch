"""Binary reward function."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from warnings import warn

# Third Party Imports
from numpy import count_nonzero, ndarray

# Punch Clock Imports
from scheduler_testbed.common.utilities import MaskConverter, actionSpace2Array
from scheduler_testbed.reward_funcs.reward_base_class import RewardFunc


# %% Class
class BinaryReward(RewardFunc):
    """Grants reward for tasking a sensor to a visible target, and a penalty otherwise.

    - Can set static reward and penalty.
    - Tasking to null action is not rewarded nor penalized.
    - Subsidies and penalties from the RewardFunc base class still apply (defaults
        to None).
    - Does not have a reset() method.
    """

    def __init__(
        self,
        reward: float = 1,
        penalty: float = 0,
        penalties: dict = None,
        subsidies: dict = None,
    ):
        """Initialize a BinaryReward reward function.

        Args:
            reward (`float`, optional): Reward for tasking a sensor to a visible
                target. Defaults to 1.
            penalty (`float`, optional): Penalty for tasking a sensor to a non-visible
                target. Defaults to 0.
            penalties (`dict`, optional): See RewardFunc for details. Defaults
                to None.
            subsidies (`dict`, optional): See RewardFunc for details. Defaults
                to None.
        """
        if penalties is not None:
            if penalties["non_vis_assignment"] != 0 and penalty != 0:
                warn(
                    "Non-zero values were set for non_vis_assignment penalty and"
                    "BinaryReward penalty. Non-visible assignments will be double"
                    "-penalized."
                )
        if subsidies is not None:
            if subsidies["active_action"] != 0 and reward != 0:
                warn(
                    "Non-zero values were set for active_action subsidy and"
                    "BinaryReward subsidy. Active actions will be double-rewarded."
                )

        super().__init__(penalties=penalties, subsidies=subsidies)
        assert reward >= 0
        assert penalty >= 0

        self.reward = reward
        self.penalty = penalty

    def calcReward(
        self,
        obs: dict,
        info: dict,
        actions: ndarray[int],
    ) -> float:
        """Calculate reward.

        Args:
            obs (`dict`, optional): Includes "action_mask" as a key.
            info (`dict`): Includes "num_sensors", "num_targets", and "vis_map_truth"
                as keys.
            actions (`ndarray[int]`): (M, ) Values are 0 to N, where N indicates
                null action.

        Returns:
            `float`: Reward for actions.
        """
        num_targets = info["num_targets"]
        num_sensors = info["num_sensors"]
        if not hasattr(self, "mask_converter"):
            # Intialize mask converter if not already (can't do this on class
            # initialization b/c we don't know the number of agents yet)
            self.mask_converter = MaskConverter(
                num_targets=num_targets,
                num_sensors=num_sensors,
            )

        [_, penalty_report] = self.calcPenalties(
            info=info,
            actions=actions,
        )
        penalty_non_vis = self.penalty * penalty_report["num_non_vis_taskings"]

        # action mask: ((N+1)*M, ) array
        action_mask = obs["action_mask"]
        # actions_2d: (N+1, M) array
        actions_2d = actionSpace2Array(actions, num_sensors, num_targets)
        # actions_1d: ((N+1)*M, ) array
        actions_1d = self.mask_converter.convertActionMaskFrom2dTo1d(actions_2d)

        # Need number of actions that DON'T violate action mask, but also are to
        # real targets (not inaction).
        #   1. Remove inactions from actions and mask (gives two binary arrays)
        #   2. Add both arrays (gives array of 0s, 1s, 2s)
        #   3. Count number of 2s, corresponds to number of actions that where
        #   valid according to mask
        #   4. Multiple count of valid actions by reward

        # actions_stripped and mask_stripped: ((N*M), ) array
        actions_stripped = self.mask_converter.removeInactionsFrom1dMask(
            mask1d=actions_1d
        )
        mask_stripped = self.mask_converter.removeInactionsFrom1dMask(
            mask1d=action_mask
        )
        num_valid_actions = count_nonzero(
            (actions_stripped + mask_stripped) == 2
        )
        reward_valid_actions = self.reward * num_valid_actions

        return reward_valid_actions - penalty_non_vis
