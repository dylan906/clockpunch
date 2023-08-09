"""Greedy covariance policy v2."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Tuple
from warnings import warn

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import MultiBinary
from numpy import diagonal, multiply, ndarray, zeros

# Punch Clock Imports
from punchclock.common.utilities import MaskConverter
from punchclock.policies.action_selection import epsGreedyMask
from punchclock.policies.policy_base_class_v2 import CustomPolicy

# %% Class


class GreedyCovariance(CustomPolicy):
    """Tasks maximum trace of covariance matrices with e-greedy action selection.

    Use `mode` argument on initialization to set whether reward/value functions
        take trace of position, velocity, or combined covariance matrices.

    Notation:
        M = number of sensors
        N = number of targets

    Observation Space: A Dict with the following structure:
        Dict({
            "observations": Dict({
                "est_cov": Box(low=-inf, high=inf, shape=(N, 6, 6)),
                "vis_map_est": MultiBinary((N, M))
            }),
            "action_mask": MultiBinary((N+1, M)),
        })
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.MultiDiscrete,
        epsilon: float = 0,
        mode: str = "position",
        subsidy: float = 0,
    ):
        """Initialize GreedyCovariance policy.

        Args:
            observation_space (`gym.spaces.Dict`): Observation space.
            action_space (`gym.spaces.MultiDiscrete`): Action space.
            epsilon (`float`, optional): Probability of choosing random action.
                Valued 0-1. Defaults to 0 (no random actions will be taken).
            mode (`str`, optional): Choose one of ("position" | "velocity" | "both").
                Determines the reward/value function used. The reward/value function
                is the sum of traces of covariance matrices. The value of `mode`
                chooses which component of the covariance matrix to use. Defaults
                to "position".
                    If mode == "position", reward = trace(cov[:3]),
                    If mode == "velocity", reward = trace(cov[3:]),
                    If mode == "both", reward = trace(cov).
            subsidy (`float`, optional): Reward for inaction. Defaults to 0.
        """
        # inherit from base class
        super().__init__(
            observation_space=observation_space, action_space=action_space
        )
        assert all(
            key in observation_space.spaces["observations"].spaces
            for key in ["est_cov", "vis_map_est"]
        ), """observation_space.spaces['observations'].spaces must contain
            ['est_cov', 'vis_map_est']"""
        assert (
            len(
                observation_space.spaces["observations"].spaces["est_cov"].shape
            )
            == 3
        ), "'est_cov' must be 3d."
        assert (
            observation_space.spaces["observations"].spaces["est_cov"].shape[1]
            == observation_space.spaces["observations"]
            .spaces["est_cov"]
            .shape[2]
            == 6
        ), "'est_cov' must have shape [N, 6, 6]."
        assert isinstance(
            observation_space.spaces["observations"].spaces["vis_map_est"],
            MultiBinary,
        ), "'vist_map_est' must be MultiBinary."

        self.num_sensors = len(self.action_space.nvec)
        # convert to int manually because nvec has a non-standard int type
        self.num_targets = int(self.action_space.nvec[0] - 1)

        assert observation_space.spaces["observations"].spaces[
            "vis_map_est"
        ].shape == (self.num_targets, self.num_sensors)

        if subsidy > 0:
            warn(
                f"Subsidy > 0 used (subsidy = {subsidy}). Make sure you used a "
                "small enough positive value so that the subsidized action does not "
                "beat out the unsubsidized actions."
            )

        self.epsilon = epsilon
        self.mode = mode
        self.subsidy = subsidy
        self.mask_converter = MaskConverter(
            num_targets=self.num_targets,
            num_sensors=self.num_sensors,
        )

    def computeAction(self, obs: dict) -> ndarray[int]:
        """E-greedy action selection.

        Args:
            obs (`dict`): Must contain "est_cov", "vis_map_est", and "action_mask"
                in keys.

        Returns:
            `ndarray[int]`: (M, ) Valued 0-N denoting actions. N = inaction.
        """
        [cov, vis_map, mask1d] = self.getCovVisMask(obs)
        mask2d = self.mask_converter.convertActionMaskFrom1dTo2d(mask1d)
        cov_diags = diagonal(cov, axis1=1, axis2=2)
        Q = self.calcQ(cov_diags, vis_map)
        action = epsGreedyMask(
            Q=Q,
            epsilon=self.epsilon,
            mask=mask2d,
        )
        # is_valid = isActionValid(mask=mask2d, action=action)
        # print(f"policy action is valid? {is_valid}")

        return action

    def getCovVisMask(
        self, obs: dict
    ) -> Tuple[ndarray[float], ndarray[int], ndarray[int]]:
        """Get covariance diagonals, visibility map, and action mask.

        Args:
            obs (`dict`): Must follow this structure:
            {
                "observations": {
                    "est_cov": values,
                    "vis_map": values
                }
                "action_mask": values
            }

        Returns:
            cov (`ndarray[float]`): (6, N) Diagonals of covariance matrices.
            vis_map (`ndarray[int]`): (N, M) Sensor-target visibility map. Values
                are 0 or 1.
            action_mask (`ndarray[int]`): ((N * M) + M, ) Binary values where 0
                indicates a masked (forbidden) action.
        """
        cov = obs["observations"]["est_cov"]
        vis_map = obs["observations"]["vis_map_est"]
        action_mask = obs["action_mask"]

        return (cov, vis_map, action_mask)

    def calcQ(
        self,
        cov_diags: ndarray[float],
        vis_map: ndarray[int],
    ) -> ndarray[float]:
        """Calculate estimated action-value (Q).

        Args:
            cov_diags (`ndarray[float]`): (6, N), Diagonals of covariance matrices
                of N targets.
            vis_map (`ndarray[int]`): (N, M), 0 or 1 -valued visibility map indicating
                sensor-target pairs' visibility status (1=visible).

        Returns:
            Q (`ndarray[float]`): (N+1, M), Estimated reward, including subsidies.
        """
        # Check which mode policy is, then strip covariance matrices appropriately.
        if self.mode == "position":
            cov_cropped = cov_diags[:3, :]
        elif self.mode == "velocity":
            cov_cropped = cov_diags[3:, :]
        elif self.mode == "both":
            cov_cropped = cov_diags

        # Q table:
        # Initialize Q array [num_targ+1 x num_sensors] array where each
        # value is the reward from tasking that satellite-sensor pair.
        Q = zeros([self.num_targets + 1, self.num_sensors])
        # assign last row value to subsidy
        Q[-1, :] = self.subsidy

        for sens in range(self.num_sensors):
            for targ in range(self.num_targets):
                # Use sum of cropped covariance
                Q[targ, sens] = sum(cov_cropped[:, targ])

        # convert Q-values to 0 for non-visible target-sensor pairs (leave subsidy
        # row alone)
        Q[:-1, :] = multiply(Q[:-1, :], vis_map)

        return Q
