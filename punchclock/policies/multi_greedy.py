"""MultiGreedy policy."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from warnings import warn

# Third Party Imports
from gymnasium.spaces import Dict, MultiDiscrete
from numpy import ndarray, ones, vstack

# Punch Clock Imports
from punchclock.policies.action_selection import epsGreedyMask
from punchclock.policies.policy_base_class_v2 import CustomPolicy


# %% MultiGreedy
class MultiGreedy(CustomPolicy):
    """Chooses 1 action per column in an observed value table.

    Selections actions corresponding to max value (in an e-greedy sense) in each
    column. Action mask and subsidy actions optional.

    Example (no subsidy or action mask):
        obs = {
            "a": array([
                [1, 2, 1],
                [3, 0, 0],
            ])
        }

        action = array([1, 0, 0])

    Example (subsidy = 0.1, no action mask)
        obs = {
            "a": array([
                [1, -1, 0],
                [3,  0, 0],
            ])
        }

        subsidized_obs = {
            "a": array([
                [1,  -1,   0  ],
                [3,   0,   0  ],
                [0.1, 0.1, 0.1]
            ])
        }

        action = array([1, 2, 2])

    Example (subsidy = 0.1, with action mask)
        obs = {
            "a": array([
                [1, -1, 1],
                [3,  4, 2],
            ])
        }

        mask = array([
            [1, 1, 0],
            [0, 0, 0]
        ])

        subsidized_obs = {
            "a": array([
                [1,  -1,   0  ],
                [0,   0,   0  ],
                [0.1, 0.1, 0.1]
            ])
        }

        action = array([0, 2, 2])

    """

    def __init__(
        self,
        observation_space: Dict,
        action_space: MultiDiscrete,
        key: str,
        epsilon: float = 0,
        subsidy: float = 0,
        use_mask: bool = True,
    ):
        """Initialize MultiGreedy policy.

        Args:
            observation_space (Dict): See CustomPolicy for requirements.
            action_space (MultiDiscrete): See CustomPolicy for requirements
            key (str): Corresponds to a 2D array in observation_space["observations"].
            epsilon (float, optional): Probability of choosing random action.
                Valued 0-1. Defaults to 0 (no random actions will be taken).
            subsidy (float, optional): Reward for inaction. Inaction corresponds
                to the bottom row of key.value, which is automatically appended
                if key.value has 1-less row than action_mask. Subsidy not used if
                action_mask is same shape as key.value. Defaults to 0.
            use_mask (bool, optional): Whether or not to use action mask. If used,
                "action_mask" must be a key in observation_space. Defaults to True.
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space
        )
        assert (
            key in observation_space.spaces["observations"].spaces
        ), f"""observation_space.spaces['observations'] must contain {key}."""
        assert (
            len(observation_space.spaces["observations"].spaces[key].shape) == 2
        ), f"""observation_space['observations']['{key}'] must be 2d."""

        self.num_sensors = len(self.action_space.nvec)
        # convert to int manually because nvec has a non-standard int type
        self.num_targets = int(self.action_space.nvec[0] - 1)

        if subsidy > 0:
            warn(
                f"""Subsidy > 0 used (subsidy = {subsidy}). Make sure you used a
                small enough positive value so that the subsidized action does not
                beat out the unsubsidized actions."""
            )

        if observation_space.spaces["observations"].spaces[key].shape[0] == (
            observation_space.spaces["action_mask"].shape[0] - 1
        ):
            print(
                f"""observation_space['observations']['{key}'] has 1 less row than
                action_mask. Policy will append subsidy row to
                obs['observations']['{key}']."""
            )
            self.append_subsidy = True
        else:
            self.append_subsidy = False

        self.key = key
        self.epsilon = epsilon
        self.subsidy = subsidy
        self.use_mask = use_mask

    def computeAction(self, obs: dict) -> ndarray[int]:
        """E-greedy action selection.

        Args:
            obs (dict): Must contain ["observations"][self.key], where key is specified
                at instantiation. The value associated with self.key in the observation
                must be a 2d array. If action mask is enabled, must contain
                ["action_mask"] as a 2d array.

        Returns:
            ndarray[int]: _description_

        Notation:
            Shape of value array (in observation): (A or A-1, B)
            Shape of action mask: (A, B)

        Value array and action mask must have same number of columns. If number
            of rows in value array is (A-1), a subsidy row is appended prior to
            selecting action. All entries in subsidy row have value self.subsidy,
            which is set at instantiation.
        """
        Q = obs["observations"][self.key]

        if self.use_mask is False:
            action_mask = None
        else:
            action_mask = obs["action_mask"]

        # If Q has 1-less row than action mask, append subsidy row to Q
        if self.append_subsidy is True:
            Q = vstack((Q, self.subsidy * ones((1, self.num_sensors))))

        action = epsGreedyMask(
            Q=Q,
            epsilon=self.epsilon,
            mask=action_mask,
        )
        return action
