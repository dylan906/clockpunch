"""Random Policy."""
# %% Imports
from __future__ import annotations

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
from numpy import int64, ndarray, zeros

# Punch Clock Imports
from scheduler_testbed.policies.action_selection import epsGreedyMask
from scheduler_testbed.policies.policy_base_class_v2 import CustomPolicy


# %% random policy class
class RandomPolicy(CustomPolicy):
    """Choose random action from action space with optional action mask.

    Notation:
        N = number of targets
        M = number of sensors
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.MultiDiscrete,
        use_mask: bool = False,
    ):
        """Initialize policy.

        Args:
            observation_space (`gym.spaces.Dict`): Observation space.
            action_space (`gym.spaces.MultiDiscrete`): Action space.
            use_mask (`bool`, optional): Whether or not policy uses action masking.
                Defaults to False.
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space
        )
        self.use_mask = use_mask
        self.num_sensors = len(self.action_space.nvec)
        # Calculating num_targets from action_space.nvec[0] assumes same number
        # of available actions for all sensors.
        self.num_targets = self.action_space.nvec[0] - 1

        # Set some parameters only if using action mask.
        if use_mask is True:
            # mask_1d_space is 1d Box; action space is MD
            self.mask_1d_space = flatten_space(self.action_space)

            # 2d mask space is transpose of usual format; this is for easier
            # transformation of actions in computeAction(). 2d mask is [M, (N+1)].
            self.mask_2d_space = Box(
                0,
                2,
                shape=(
                    self.num_sensors,
                    self.num_targets + 1,
                ),
                dtype=int64,
            )

            assert (
                "action_mask" in self.observation_space.sample()
            ), "The key 'action_mask' must be in environment observation space."
        else:
            self.mask_1d_space = None
            self.mask_2d_space = None

    def computeAction(self, obs: dict) -> ndarray[int]:
        """Compute a single action from a single observation.

        Args:
            obs (`dict`): A single observation. If action masking is enabled, obs
                must contain "action_mask" as a key. Action mask is a ( (N+1) * M, )
                binary array, representing the flattened action space.

        Returns:
            `ndarray[int]`: (M, ) MultiDiscrete actions valued 0-N denoting actions.
                N = inaction.
        """
        # Assign arbitrary values for Q to interface with epsGreedyMask
        Q = zeros(shape=(self.num_targets + 1, self.num_sensors))
        if self.use_mask is False:
            mask = None
        else:
            mask_flat = obs["action_mask"]
            # Convert mask from 1d array to 2d array
            mask = unflatten(self.mask_2d_space, mask_flat).transpose()

        action = epsGreedyMask(Q=Q, epsilon=1, mask=mask)
        action_flat = flatten(self.action_space, action)

        if self.use_mask is True:
            # Make sure actions don't violate action mask
            assert (
                action_flat <= mask_flat
            ).all(), "Action violates action mask."

        return action
