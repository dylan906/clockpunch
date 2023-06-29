"""NeedParameter policy."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
# from numpy import multiply as elementalMultiply
from numpy import multiply, ndarray, ones, zeros
from numpy.random import rand, randint

# Punch Clock Imports
from punchclock.common.agents import Sensor, Target
from punchclock.policies.action_selection import epsGreedy
from punchclock.policies.policy_base_class import Policy
from punchclock.policies.policy_utils import appendSubsidyRow


# %% Class
class NeedParameter(Policy):
    """Maximize probability of successful observation, with binary need parameter overlay.

    Reward function (sum for all targets i): Q = sumOverTargets( mu[i] * p[i] )

    Default penalties inherited from base class.

    Notation:
        M = number of sensors
        N = number of targets
        mu = Need parameter. 0 or 1 indicating whether (1) or not (0) a given target needs
        to be tasked.
        p = Probability of successful observation. Valued 0 to 1.

    Modified from "Space Situational Awareness Sensor Tasking: Comparison of Machine
    Learning with Classical Optimization Methods", by Little and Frueh, 2020.
    """

    def __init__(
        self,
        initial_p: list,
        num_sensors: int,
        num_targets: int,
        epsilon: float = 0,
        debug: bool = False,
        subsidy: float = 1e-3,
        Q_initial: ndarray[float] = None,
        action_initial: ndarray[int] = None,
        penalties: dict = None,
    ):
        """Initialize `NeedParameter` policy.

        Args:
            initial_p (`list`): N-long list of probabilities of a successful observation.
                Valued 0-1.
            num_sensors (`int`): Number of sensors.
            num_targets (`int`): Number of targets.
            epsilon (`float`, optional): Parameter in e-greedy action selection. Larger values
                correspond to more randomness. Valued 0-1. Set to 0 for fully greedy. Defaults
                to 0.
            debug (`bool`, optional): Set to `True` to randomly evolve `mu` and set `p` to
                ones. Defaults to `False`.
            subsidy (`float`, optional): Value of not tasking a sensor. Defaults to 1e-3.
            Q_initial (`ndarray[float]`, optional): (N+1, M) Initial values of sensor-target
                pairs, with the bottom row being the value of inaction.
            action_initial (`ndarray[int]`, optional): (M, ) Initial actions, valued 0 to
                N, where N indicates inaction.
            penalties (`dict`, optional): This policy has a single penalty for multi-target
                assignment. Recommend setting to large value. Defaults to:
                {
                    "multi_assignment": 1e6,
                }

        Notation:
            M = number of sensors
            N = number of targets
            mu = 0 or 1 indicating whether (1) or not (0) a given target needs to be tasked.

        mu and p are N-long lists.
        """
        # inherit from base class
        super().__init__(
            Q_initial=Q_initial,
            action_initial=action_initial,
            subsidy=subsidy,
            num_sensors=num_sensors,
            num_targets=num_targets,
            penalties=penalties,
        )

        self.p = initial_p
        self.epsilon = epsilon
        self.debug = debug

        # cumulative reward
        self.cum_reward = 0
        # need parameters all initialize as 1
        self.mu = list(ones(self.num_targets, dtype=int))

    def reset(self):
        """Reset policy initial values."""
        self.Q = deepcopy(self.Q_init)
        self.action = deepcopy(self.action_initial)
        self.mu = list(ones(self.num_targets, dtype=int))

    def update(
        self,
        obs: dict,
        reward: float,
        debug=False,
    ) -> tuple[ndarray[float], float, ndarray[int]]:
        """Update policy.

        Normally `mu` and `p` updates would be included in the observation, but in cases
        of debugging, you can set `debug=True` to randomly evolve `mu` and set `p` to ones.
        You can also set `debug` when initializing the policy (see `__init__`).

        Args:
            obs (`dict`): See SSAScheduler observation space for description.
            reward (`float`): Reward received from last action.
            debug (`bool`): Set to `True` to randomly evolve `mu` and set `p` to ones. Otherwise,
                mu and p do not change.

        Returns:
            Q (`ndarray[float]`): (N+1, M), Estimated reward, including subsidies.
            cum_reward (`float`): Cumulative reward received.
            action (`ndarray[int]`): Chosen actions.
        """
        vis_map = obs["vis_map_est"]

        if (self.debug is True) or (debug is True):
            # randomly evolve mu and randomly regenerate p (for debugging)
            mu = self._randomizeMu(self.mu, self.num_sensors)
            p = self._regenP(ones_flag=True)
        else:
            mu = self.mu
            p = self.p

        self.cum_reward += reward
        self.mu = mu
        self.p = p
        self.Q = self.calcQ(vis_map)
        self.action = self.chooseAction(self.Q)

        return (self.Q, self.cum_reward, self.action)

    def chooseAction(self, Q: ndarray) -> ndarray:
        """Choose actions given action-values.

        Uses epsilon-greedy action selector.

        Args:
            Q (`ndarray`): (N+1, M) Estimated action-values.

        Returns:
            `ndarray`: (M, ) Values indicate targets, where M is inaction. Valued 0 to M.
        """
        actions = epsGreedy(Q=Q, epsilon=self.epsilon)

        return actions

    def calcQ(self, vis_map) -> ndarray:
        """Calculate estimated action-value (Q).

        Args:
            vis_map (`ndarray`): (N, M), 0 or 1 -valued visibility map indicating sensor-target
                pairs' visibility status (1=visible).

        Returns:
            Q (`ndarray`): (N+1, M), Estimated reward, including subsidies.
        """
        # Q table:
        # Initialize Q array [num_targ+1 x num_sensors] array where each
        # value is the reward from tasking that satellite-sensor pair.
        Q = zeros([self.num_targets + 1, self.num_sensors])
        # assign last row value to subsidy
        Q[-1, :] = self.subsidy

        for sens in range(self.num_sensors):
            for targ in range(self.num_targets):
                Q[targ, sens] = self.mu[targ] * self.p[targ]

        # convert Q-values to 0 for non-visible target-sensor pairs (leave subsidy row alone)
        Q[:-1, :] = multiply(Q[:-1, :], vis_map)

        return Q

    def calcPotentialReward(
        self,
        sensors: list[Sensor],
        targets: list[Target],
        vis_map: ndarray[int],
    ) -> ndarray[float]:
        """Calculate reward for all sensor-target pairs at a time step.

        - Includes visibility masking and subsidies.
        - Does not include penalties.
        - Sensors and targets not explicitly used in this function, but required for base
            class.

        Args:
            sensor (`list[Sensor]`): A list of sensors. Not used.
            target (`list[Target]`): A list of targets. Not used.
            vis_map (`ndarray[int]`): (N, M) Valued 0/1 where 1 indicates the sensor-target
                pair can see each other.

        Returns:
            `ndarray[float]`: (N+1, M) Rewards matrix for sensor-target taskings.
        """
        rewards = zeros([self.num_targets, self.num_sensors])
        for j in range(self.num_sensors):
            for i in range(self.num_targets):
                r = self.mu[i] * self.p[i]
                rewards[i, j] = r

        # mask visibility
        masked_rewards = multiply(rewards, vis_map)

        subsidized_rewards = appendSubsidyRow(
            rewards=masked_rewards,
            subsidy=self.subsidy,
        )

        return subsidized_rewards

    def getNetReward(
        self,
        sensors: list[Sensor],
        targets: list[Target],
        actions: ndarray[int],
        vis_map_truth: ndarray[int],
        vis_map_est: ndarray[int],
        debug: bool = False,
    ) -> float:
        """Calculate true net reward received, including penalties and subsidies.

        Change all entries of `self.mu==1` to 0 iff indices are includes in actions. Does
            not affect entries of mu that are already 0.

        Args:
            sensors (`list[Sensor]`): _description_
            targets (`list[Target]`): _description_
            vis_map_truth (`ndarray[int]`): (N, M) True visibility map. All values are
                0/1, where 1 indicated the m-n sensor-target pair are visible to each other.
            vis_map_est (`ndarray[int]`): (N, M) True visibility map. All values are
                0/1, where 1 indicated the m-n sensor-target pair are visible to each other.
            actions (`ndarray[int]`): (M, ) Actions valued 0 to N, where N indicates inaction.
            debug (`bool`, optional): Set to `True` to skip calculating policy-specific potential
                rewards and just set all potential rewards to 1. Defaults to `False`.


        Returns:
            `float`: Net reward (includes penalties)
        """
        # calculate reward
        net_reward = super().getNetReward(
            sensors=sensors,
            targets=targets,
            vis_map_truth=vis_map_truth,
            vis_map_est=vis_map_est,
            actions=actions,
            debug=debug,
        )

        # In debug mode we don't do any policy-specific rewards; this is just for testing
        # policy base class.
        if debug is False:
            # flip need parameters to 0 if tasked
            for act in actions:
                # skip if inaction is chosen
                if act > (len(targets) - 1):
                    continue
                self.mu[act] = 0

        return net_reward

    def _randomizeMu(self, mu: list[int], num_sensors: int) -> list[int]:
        """Randomly convert M values of mu from 1 to 0.

        Used for debugging. If all elements of mu are 0, does nothing.

        Arguments:
            mu (`list[int]`): N-long list of deed parameters (mu). Valued 0 or 1.

        Returns:
            `list[int]`: N-long list of deed parameters (mu). Valued 0 or 1.
        """
        # deepcopy to protect scope
        mu = deepcopy(mu)
        # Loop through sensors
        for _ in range(num_sensors):
            # set a random element of mu to 0
            while True:
                index = randint(0, high=self.num_targets)
                # If all of mu is 0, break loop
                if all(val == 0 for val in mu):
                    break
                # If chosen element is already 0, continue loop
                elif mu[index] == 0:
                    continue
                # Otherwise, set mu[index] to 0
                else:
                    mu[index] = 0
                    break

        # # NOTE: Delete ME!
        # mu = list(ones(len(mu), dtype=int))
        return mu

    def _regenP(self, ones_flag: bool) -> list[float]:
        """Generate prob of successful observation (p) randomly or as ones.

        Args:
            ones_flag (`bool`):
                If `True`, p= ones(num_targets).
                If `False`, p = rand(num_targets).

        Returns:
            `list[float]`: Probability of detection for N targets. Valued 0 to 1.

        Used for debugging.
        """
        if ones_flag is False:
            # randomly generate p
            p = rand(self.num_targets)
        else:
            # set p to all ones
            p = ones(self.num_targets)

        return p
