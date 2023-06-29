"""Custom Policy (v2) module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
import gymnasium as gym
from numpy import ndarray


# %% Class
class CustomPolicy:
    """Base class for deterministic policies."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.MultiDiscrete,
    ):
        """Initialize base class.

        Environment must have a Dict observation space and MultiDiscrete action
        space. Items in observation space must be "observations" and "action_mask",
        where the value of "observations" is a Dict and the value of "action_mask"
        is a gym.spaces.Box.
        """
        assert isinstance(
            observation_space,
            gym.spaces.Dict,
        ), "Observation space must be a gym.spaces.Dict."

        assert (
            "observations" in observation_space.spaces.keys()
        ), "Observation space must contain 'observations' as a key."
        assert (
            "action_mask" in observation_space.spaces.keys()
        ), "Observation space must contain 'action_mask' as a key."
        assert isinstance(
            observation_space.spaces["observations"], gym.spaces.Dict
        ), "observation_space['observations'] must be a gym.spaces.Dict."
        assert isinstance(
            observation_space.spaces["action_mask"], gym.spaces.Box
        ), "observation_space['action_mask'] must be a gym.spaces.Box."

        assert isinstance(
            action_space,
            gym.spaces.MultiDiscrete,
        ), "Action space must be a gym.spaces.MultiDiscrete."

        self.observation_space = observation_space
        self.action_space = action_space

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
