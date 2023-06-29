"""Upper Confidence Bounds policy."""
# %% Imports
from __future__ import annotations

# Third Party Imports
import gymnasium as gym
from numpy import concatenate, multiply, ndarray, ones, vstack, zeros

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.policies.action_selection import epsGreedyMask
from punchclock.policies.policy_base_class_v2 import CustomPolicy
from punchclock.policies.policy_utils import (
    appendSubsidyRow,
    upperConfidenceBounds,
)


# %% Class
class UpperConfidenceBounds(CustomPolicy):
    """Upper Confidence Bounds (UCB) policy.

    Selects actions based on a combination of immediate reward and a factor based on the
    number of times the action has previously been selected and the simulation time.

    Reference: "Reinforcement Learning, an Introduction, 2d Edition", by Richard Sutton
    and Andrew Barto, section 2.7.

    Assumes evenly spaced time steps.

    Default penalties inherited from base class.

    Notation:
        - M = number of sensors
        - N = number of targets
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.MultiDiscrete,
        exploration_param: float,
        max_reward: float,
        epsilon: float = 0.0,
        num_previous_actions: ndarray[int] = None,
        inaction_subsidy: float = 1e-3,
        Q_initial: ndarray[float] = None,
        action_initial: ndarray[int] = None,
    ):
        """Initialize UpperConfidenceBounds (UCB) policy.

        Args:
            observation_space (`gym.spaces.Dict`): See CustomPolicy.
            action_space (`gym.spaces.MultiDiscrete`): See CustomPolicy.
            exploration_param (`float`): Must be greater than 0.
            max_reward (`float`): Value assigned to targets that have never been
                tasked.
            epsilon (`float`, optional): Probability of choosing random action.
                Valued 0-1. Defaults to 0 (no random actions will be taken).
            num_previous_actions (`ndarray[int]`, optional): (N, ) Number of times
                each target has previously been tasked. Defaults to zeros(N).
            inaction_subsidy (`float`, optional): Value of not tasking a sensor. Defaults to 1e-3.
            Q_initial (`ndarray[float]`, optional): (N+1, M) Initial values of
                sensor-target pairs, with the bottom row being the value of inaction.
                Defaults to uniform random values on the interval [0, 1) for entries
                on rows 0 to N-1, and the value of subsidy for entries in row N.
            action_initial (`ndarray[int]`, optional): (M, ) Initial actions, valued
                0 to N, where N indicates inaction. Defaults to all sensors being
                inactive (N * ones(M)).
        """
        # inherit from base class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

        self.num_sensors = len(action_space)
        self.num_targets = action_space.nvec[0]

        # default Nt (number of times targets previously tasked)
        if num_previous_actions is None:
            num_previous_actions = zeros(self.num_targets, dtype=int)

        self.Nt = num_previous_actions
        self.c = exploration_param
        self.max_reward = max_reward
        self.epsilon = epsilon
        self.inaction_subsidy = inaction_subsidy
        # current step of policy
        self.t_step = 0

    # %% Methods
    def computeAction(
        self,
        obs: dict,
    ) -> ndarray[int]:
        """Compute single action.

        Args:
            observation (`dict`): See CustomPolicy.
        Returns:
            action (`ndarray[int]`): (M, ) Chosen actions valued 0 to N, where
              N is inaction.
        """
        # Increment time step (assumes evenly spaced time steps, otherwise would need to
        # have current time as an input).
        self.t_step += 1
        # self.cum_reward += previous_reward

        # get vis map from observation
        vis_map = obs["observations"]["vis_map_est"]

        # update value estimates and choose action
        self.Q = self.calcQ(vis_map)
        self.action = self.chooseAction(self.Q, vis_map)

        # update number of times tasked
        actions_no_subsidy = self.action[self.action < self.num_targets]
        for i in actions_no_subsidy:
            self.Nt[i] += 1

        # return self.Q, self.cum_reward, self.action
        return self.action

    def chooseAction(
        self,
        Q: ndarray[float],
        vis_map: ndarray[int],
    ) -> ndarray[int]:
        """Choose actions given action-values.

        Uses epsilon-greedy action selector.

        Args:
            Q (`ndarray[float]`): (N+1, M) Estimated action-values.

        Returns:
            `ndarray[int]`: (M, ) Values indicate targets, where M is inaction.
                Valued 0 to M.
        """
        # Append null action row to vis_map to create 2d action mask
        action_mask = vstack(
            [
                vis_map,
                ones((1, self.num_sensors)),
            ]
        ).astype(int)
        actions = epsGreedyMask(Q=Q, epsilon=self.epsilon, mask=action_mask)

        return actions

    def calcQ(self, vis_map: ndarray[int]) -> ndarray[float]:
        """Estimate action-value table with UCB method.

        Arguments:
            vis_map (`ndarray[int]`): (N, M), 0 or 1 -valued visibility map indicating
                sensor-target pairs' visibility status (1=visible).

        Returns:
            `ndarray[float]`: (N+1, M) Estimated values for all sensor-target pairs
                and inaction row (N+1).
        """
        # Q table:
        # Initialize Q array [num_targ+1 x num_sensors] array where each
        # value is the reward from tasking that satellite-sensor pair.
        # All unweighted values are 1 (pre-UCB weighting) except the bottom row,
        # which is subsidy row.
        Q = ones([self.num_targets, self.num_sensors])
        subsidy_array = self.inaction_subsidy * ones([1, self.num_sensors])
        Q = concatenate((Q, subsidy_array))

        # Weight by UCB (not looping through subsidy row)
        for col in range(self.num_sensors):
            for row in range(self.num_targets):
                # get unweighted pair value
                Q_pair = Q[row, col]
                # get number of times target has previously been tasked
                Nt_target = self.Nt[row]

                Q[row, col] = upperConfidenceBounds(
                    t=self.t_step,
                    Qt=Q_pair,
                    Nt=Nt_target,
                    c=self.c,
                    max_reward=self.max_reward,
                )

        # convert Q-values to 0 for non-visible target-sensor pairs (leave subsidy
        # row alone)
        Q[:-1, :] = multiply(Q[:-1, :], vis_map)

        return Q

    # def calcPotentialReward(
    #     self,
    #     sensors: list[Sensor],
    #     targets: list[Target],
    #     vis_map: ndarray[int],
    # ) -> ndarray[float]:
    #     """Calculate reward for all sensor-target pairs at a time step.

    #     - Includes visibility masking and subsidies.
    #     - Does not include penalties.
    #     - Sensors and targets not explicitly used in this function, but required for base
    #         class.

    #     Args:
    #         sensor (`list[Sensor]`): A list of sensors. Not used.
    #         target (`list[Target]`): A list of targets. Not used.
    #         vis_map (`ndarray[int]`): (N, M) Valued 0/1 where 1 indicates the sensor-target
    #             pair can see each other.

    #     Returns:
    #         `ndarray[float]`: (N+1, M) Rewards matrix for sensor-target taskings.
    #     """
    #     rewards = zeros([self.num_targets, self.num_sensors])
    #     for col in range(self.num_sensors):
    #         for row in range(self.num_targets):
    #             # get value of sensor-target pair from Q table
    #             Q_pair = self.Q[row, col]
    #             Nt_target = self.Nt[row]

    #             # calculate reward
    #             rewards[row, col] = upperConfidenceBounds(
    #                 t=self.t_step,
    #                 Qt=Q_pair,
    #                 Nt=Nt_target,
    #                 c=self.c,
    #                 max_reward=self.max_reward,
    #             )

    #     masked_rewards = multiply(rewards, vis_map)
    #     subsidized_rewards = appendSubsidyRow(
    #         rewards=masked_rewards,
    #         subsidy=self.subsidy,
    #     )

    #     return subsidized_rewards
