"""Custom Policy (v2) module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import final

# Third Party Imports
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete
from numpy import ndarray


# %% Class
class CustomPolicy(ABC):
    """Base class for deterministic policies."""

    def __init__(
        self,
        observation_space: Dict,
        action_space: MultiDiscrete,
    ):
        """Initialize base class.

        Environment must have a Dict observation space and MultiDiscrete action
        space. Items in observation space must be "observations" and "action_mask",
        where the value of "observations" is a Dict and the value of "action_mask"
        is a gym.spaces.Box.
        """
        assert isinstance(
            observation_space,
            Dict,
        ), "Observation space must be a Dict."

        assert (
            "observations" in observation_space.spaces
        ), "Observation space must contain 'observations'."
        assert (
            "action_mask" in observation_space.spaces
        ), "Observation space must contain 'action_mask'."
        assert isinstance(
            observation_space.spaces["observations"], Dict
        ), "observation_space['observations'] must be a Dict."
        assert isinstance(
            observation_space.spaces["action_mask"],
            MultiBinary,
        ), """observation_space['action_mask'] must be MultiBinary."""

        assert isinstance(
            action_space,
            MultiDiscrete,
        ), "Action space must be MultiDiscrete."
        assert all(
            action_space.nvec == action_space.nvec[0]
        ), """All values in action_space.nvec must be same."""

        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def computeAction(self, obs: dict) -> ndarray[int]:
        """Overwrite this method. This is the public-facing way to calculate actions.

        Args:
            obs (`dict`): A single observation.

        Returns:
            `ndarray[int]`: (M, ) MultiDiscrete actions valued 0-N denoting actions.
                N = inaction.
        """
        action = None

        return action

    @final
    def _computeSingleAction(self, obs: dict) -> ndarray[int]:
        """Don't overwrite this method.

        This method wraps computeAction() and is called by simulation runner.
        """
        assert self.observation_space.contains(
            obs
        ), "Observation must be contained in observation space."

        action = self.computeAction(obs)
        assert self.action_space.contains(
            action
        ), "Action must be contained in action space."
        return action
