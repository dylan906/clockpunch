"""Module for things to run simulations."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Tuple, Union

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces.utils import flatten
from numpy import (
    array,
    bool_,
    dstack,
    float32,
    float64,
    fromstring,
    int64,
    ndarray,
)
from pandas import DataFrame, concat, json_normalize
from ray.rllib.policy.policy import Policy as RayPolicy
from torch import Tensor, tensor

# Punch Clock Imports
from punchclock.common.agents import Agent
from punchclock.environment.env import SSAScheduler
from punchclock.environment.misc_wrappers import IdentityWrapper
from punchclock.environment.wrapper_utils import (
    getNumWrappers,
    getWrapperList,
    getXLevelWrapper,
)
from punchclock.policies.policy_base_class_v2 import CustomPolicy


# %% Class
class SimRunner:
    """Simulation runner for `SSAScheduler` environment.

    Environment must be wrapped by at least one wrapper. One of the wrappers must
    be include an action mask, which may be placed anywhere in the stack of wrappers.
    The structure for an action mask wrapper is defined below. If env has only a
    single wrapper, it must be an action mask. Top wrapper observation space must
    be a Dict with "observations" and "action_mask" as keys (an action mask wrapper).

    Action mask wrapper requirements:
        - Wrapper must be an observation space wrapper.
        - Wrapped observation space must be in format:
            env.observation_space = Dict({
                "action_mask": Box(),
                "observations": Dict()
            })

    Works with RLLib policy or CustomPolicy.
    """

    def __init__(
        self,
        env: SSAScheduler,
        policy: Union[RayPolicy, CustomPolicy],
        max_steps: int,
    ):
        """Initialize SimRunner.

        Args:
            env_params (`SSAScheduler`): Must be wrapped. One of the wrappers must
                be be an action mask wrapper. Observation space must be a
                gym.spaces.Dict. and the keys "observations" and "action_mask".
                Base environment must be SSAScheduler.
            policy (`Union[RayPolicy, CustomPolicy]`): Can be either an Ray policy
                or a CustomPolicy.
            max_steps (`int`): Number of steps to take in simulation. Can be greater
                than the max number of steps before an environment resets.

        Attributes:
            env (`SSAScheduler`): Gym environment used in simulation.
            wrappers (`list[gym.Wrapper]`): List of wrappers around env.
            num_env_wrappers (`int`): Number of wrappers around env.
            policy (`Union[RayPolicy, CustomPolicy]`): Policy used to control actions.
            max_steps (`int`): Number of steps to take when running simulation.
        """
        # %% Argument checks
        self.wrappers = getWrapperList(env)

        assert isinstance(env, gym.Wrapper), "Environment must be wrapped."
        assert isinstance(
            env.unwrapped, SSAScheduler
        ), "Base environment must be SSAScheduler."
        assert (
            "observations" in env.observation_space.spaces
        ), "Environment observation space must contain 'observations' as a key."
        assert (
            "action_mask" in env.observation_space.spaces
        ), "Environment observation space must contain 'action_mask' as a key."
        assert isinstance(
            policy, (RayPolicy, CustomPolicy)
        ), "Policy must be RayPolicy or CustomPolicy"
        if isinstance(policy, CustomPolicy):
            assert (
                IdentityWrapper in self.wrappers
            ), """When using a CustomPolicy, env must have IdentityWrapper in the
            wrapper stack."""
        # %% Set class attributes
        self.env = env
        self.max_steps = max_steps
        self.policy = policy
        self.num_env_wrappers = getNumWrappers(env)
        # reset all time-varying parameters
        self.reset()

    def _computeSingleActionWrapper(self, obs: dict) -> ndarray[int]:
        """Function used to wrap compute_single_action method.

        Used for Ray policies and custom policies.

        Recurrent policies are handled differently from non-recurrent.
        """
        assert isinstance(obs, dict)
        if isinstance(self.policy, CustomPolicy):
            action = self.policy._computeSingleAction(obs=obs)
        elif self.policy.is_recurrent():
            # recurrent Ray policies
            rnn_state = getattr(
                self, "rnn_state", self.policy.get_initial_state()
            )

            action, rnn_state, _ = self.policy.compute_single_action(
                obs=obs,
                state=rnn_state,
                explore=False,
            )
            self.rnn_state = rnn_state
        else:
            # non-recurrent Ray policies
            obs_tensor = self._convertObs2Tensor(obs)
            action = self.policy.compute_single_action(
                obs=obs_tensor, explore=False
            )

        return action

    def _convertObs2Tensor(self, obs: dict) -> Tensor:
        """Flatten a dict obs and convert to Tensor."""
        obs = flatten(self.env.observation_space, obs)
        obs_tensor = tensor(obs)
        return obs_tensor

    def _getObs(
        self,
        stop_at_identity_wrapper: bool = False,
        identity_wrapper_id=None,
    ) -> OrderedDict:
        """Get observation from top-level/intermediate wrapped or bare environment.

        Handles getting observations through multiple layers of wrappers.

        Args:
            stop_at_identity_wrapper (`bool`, optional): Set to True to return
                observation from IdentityWrapper, regardless of any wrappers above
                it. Defaults to False.
            identity_wrapper_id (optional): Identifier to use if
                stop_at_identity_wrapper is True and there are multiple IdentityWrappers
                on environment. Defaults to None.

        """
        if isinstance(self.env, gym.Wrapper):
            # Get unwrapped observation, then pass through layer(s) of wrappers
            # from bottom-up to transform to top-level env observation space.
            obs = self.env.unwrapped._getObs()
            for i in range(1, self.num_env_wrappers + 1):
                env = getXLevelWrapper(
                    deepcopy(self.env), self.num_env_wrappers - i
                )
                # print(env)
                # if isinstance(env, ObservationWrapper):
                if hasattr(env, "observation"):
                    # for wrappers that have modify observation, do it;
                    # otherwise obs_in = obs_out
                    obs = env.observation(obs)

                if stop_at_identity_wrapper is True:
                    # Break for loop if at IdentityWrapper, skip any higher
                    # levels of wrappers.
                    if (
                        self._isWrapperIdentity(env, identity_wrapper_id)
                        is True
                    ):
                        break

            obs = OrderedDict(obs)

        else:
            # If bare environment, just use _getObs() once and output
            obs = self.env._getObs()

        assert isinstance(obs, OrderedDict)

        return obs

    def _isWrapperIdentity(self, env: gym.Env, wrapper_id: Any = None) -> bool:
        """Check if env top level wrapper is IdentityWrapper with optional id.

        Args:
            env (gym.Env): A gym environment.
            wrapper_id (Any, optional): Wrapper must also have matching id. Defaults
                to None.

        Returns:
            bool: Returns True if env top level wrapper is IdentityWrapper.
        """
        if isinstance(env.observation_space, gym.spaces.Dict) and isinstance(
            env, IdentityWrapper
        ):
            if wrapper_id is None:
                checkwrap = True
            else:
                if env.id == wrapper_id:
                    checkwrap = True
                else:
                    checkwrap = False
        else:
            checkwrap = False

        return checkwrap

    def _getInfo(self):
        """Abstraction for getting info from wrapped environment."""
        # Drill down through wrappers until find one that outputs info. Assumes
        # wrappers that modify info have '_getInfo()` method. Base env SSAScheduler
        # has _getInfo().
        for i in range(0, self.num_env_wrappers + 1):
            env = getXLevelWrapper(deepcopy(self.env), i)
            if hasattr(env, "_getInfo"):
                info = env._getInfo()
                break
            else:
                info = None

        if info is None:
            Exception("Base environment and wrappers do not have _getInfo().")

        return info

    def reset(self):
        """Reset time-varying attributes."""
        self.env.reset()
        # %% (re-)Initialize outputs
        self.reward_hist = [None] * self.max_steps
        # Store reward as numpy float (vs python float) for consistency with the
        # rest of a sim run.
        self.reward_hist[0] = float64(0.0)
        self.done_hist = [None] * self.max_steps
        self.done_hist[0] = False
        self.action_hist = [None] * self.max_steps
        self.obs_hist = [None] * self.max_steps

        # Getting initial obs and info requires tailored methods. Obs structure
        # differs depending on Custom vs Ray policy.
        if isinstance(self.policy, CustomPolicy):
            self.obs_hist[0] = self._getObs(stop_at_identity_wrapper=True)
        else:
            self.obs_hist[0] = self._getObs()

        self.info_hist = [None] * self.max_steps
        self.info_hist[0] = self._getInfo()

        action_init = self._computeSingleActionWrapper(self.obs_hist[0])
        self.action_hist[0] = getActionArray(action_init)

    def step(
        self,
        action: ndarray[int],
    ) -> Tuple[Any, float, bool, dict, ndarray]:
        """Step simulation forward by 1 time step.

        Args:
            action (`ndarray[int]`): MultiDiscrete action array.

        Returns:
            obs (`Any`): Type is dependent on wrappers used on self.env.
            reward (`float`): Reward for single step.
            done (`bool`): Whether or not the env is done.
            info (`dict`): Info for single step.
            next_action (`ndarray`): Policy's action based on argument action.
        """
        # step environment
        [obs, reward, done, truncated, info] = self.env.step(action)
        if isinstance(self.policy, CustomPolicy):
            # Overwrite obs if using CustomPolicy. CustomPolicy requires obs =
            # {"observations": Dict, "action_mask": Box}
            obs = self._getObs(stop_at_identity_wrapper=True)

        # Convert to OrderedDict if env outputs regular dict (happens with wrapped
        # environments).
        obs = OrderedDict(obs)

        # Get single step action from policy
        next_action = self._computeSingleActionWrapper(obs)
        next_action = getActionArray(next_action)

        return (obs, reward, done, info, next_action)

    def runSim(self) -> SimResults:
        """Run through self.max_steps of a simulation and return results.

        Resets environment upon call.

        Returns:
            `SimResults`: A dataclass with the following entries:
                actions (`ndarray`): shape [T, M]. Time history of actions.
                agents (`list[Agent]`): (N+M)-long list of `Agent` objects at the
                    final state of the simulation.
                info (`list`): T-long list of info following Gym environment info
                    format.
                obs (`list`): T-long list of observations following Gym environment
                    observation space format.
                reward (`ndarray`): shape [T,]. Reward received at each time instance.
        """
        # %% Loop through time
        self.reset()

        # set initial actions
        action = self.action_hist[0]
        for i in range(1, self.max_steps):
            # output action is for next step
            [observation, reward, done, info, action] = self.step(action)

            # record entries in _hist variables
            self.obs_hist[i] = observation
            self.reward_hist[i] = reward
            self.done_hist[i] = done
            self.info_hist[i] = info
            self.action_hist[i] = action

        return SimResults(
            obs=self.obs_hist,
            reward=self.reward_hist,
            done=self.done_hist,
            info=self.info_hist,
            actions=self.action_hist,
        )


@dataclass
class SimResults:
    """Container for simulation results."""

    obs: list
    reward: list
    done: list
    info: list
    actions: list

    def toDataFrame(self) -> DataFrame:
        """Convert results into a pandas DataFrame.

        Recommended use is to save DataFrame as .pkl.
        """
        results_df = results2DF(results=self)

        return results_df

    def toPrimitiveSimResults(self) -> dict:
        """Convert sim results to a PrimitiveSimResults.

        PrimitiveSimResults is easier to save as a csv.

        Returns:
            `PrimitiveSimResults`: See PrimitiveSimResults.
        """
        results_dict = self._recursivelyConvertDicts2JSONable(self.__dict__)
        return PrimitiveSimResults(**results_dict)

    def _recursivelyConvertDicts2JSONable(self, in_dict: dict) -> dict:
        """Recursively convert dict entries into JSON-able objects.

        Args:
            in_dict (`dict`): Has mixed-type entries that are not necessarily JSON-able.

        Returns:
            `dict`: Same keys as input, but with values that are JSON-able.
        """
        out = {}
        # Loop through key-value pairs of in_dict. If a value is a dict, then recurse.
        # Otherwise, convert value to a JSON-able type. Special handling if the
        # value is a `list`. Lists of dicts are recursed; lists of non-dicts and
        # empty lists are converted to JSON-able as normal.
        for k, v in in_dict.items():
            if isinstance(v, dict):
                out[k] = self._recursivelyConvertDicts2JSONable(v)
            elif isinstance(v, list):
                if len(v) == 0:
                    out[k] = [self._convert2JSONable(a) for a in v]
                elif isinstance(v[0], dict):
                    out[k] = [
                        self._recursivelyConvertDicts2JSONable(a) for a in v
                    ]
                else:
                    out[k] = [self._convert2JSONable(a) for a in v]
            else:
                out[k] = self._convert2JSONable(v)
        return out

    def _convert2JSONable(self, entry: Any) -> list:
        """Convert a non-serializable object into a JSON-able type."""
        if isinstance(entry, ndarray):
            # numpy arrays need their own tolist() method to convert properly.
            out = entry.tolist()
        elif isinstance(entry, set):
            out = list(entry)
        elif isinstance(entry, (float32, int64)):
            out = entry.item()
        else:
            out = entry

        return out

    def _convertActions2Lists(self, actions: list[ndarray]) -> list[list]:
        nested_lists = []
        for entry in actions:
            nested_lists.append(entry.tolist())

        return nested_lists


@dataclass
class PrimitiveSimResults:
    """A version of SimResults with only primitive dtypes.

    Good for saving a DataFrame as a .csv.
    """

    obs: list
    reward: list
    done: list
    info: list
    actions: list

    def toDataFrame(self) -> DataFrame:
        """Convert to a DataFrame."""
        results_df = results2DF(results=self)

        return results_df


def results2DF(results: SimResults | PrimitiveSimResults) -> DataFrame:
    """Convert SimResults or PrimitiveSimResults to a DataFrame."""
    obs_df = json_normalize(results.obs)
    info_df = json_normalize(results.info)
    done_df = json_normalize([{"done": v} for v in results.done])
    action_df = json_normalize([{"action": v} for v in results.actions])
    reward_df = json_normalize([{"reward": v} for v in results.reward])

    results_df = concat(
        [
            obs_df,
            info_df,
            done_df,
            action_df,
            reward_df,
        ],
        axis=1,
    )

    # Name index (initialized w/o name)
    results_df.index.rename("step", inplace=True)

    return results_df


def concatenateStates(list_of_agents: list[Agent]) -> ndarray:
    """Extract eci states from agents.

    Args:
        list_of_agents (`list[Agent]`): List of Agents.

    Returns:
        `ndarray`: (6, num_agents) Column i is the ECI state of agent i.
    """
    states = [agent.eci_state for agent in list_of_agents]
    # convert list to array, squeeze out singleton 3rd dimension, transpose to
    # put states in 0th dimension
    states = array(states).squeeze().transpose()
    return states


def getActionArray(action: Union[Tuple, ndarray]) -> ndarray:
    """Check if action is a tuple and extract the array if needed.

    If argument is an `ndarray`, return is same as input.
    """
    # Get array from action if next_action is a tuple. This happens when policy
    # is a Torch policy.
    if isinstance(action, tuple):
        array_action = action[0]
    elif isinstance(action, ndarray):
        array_action = action

    return array_action


def formatSimResultsDF(df: DataFrame, print_output: bool = False) -> DataFrame:
    """Format a SimResults-generated DataFrame to usable dtypes."""
    conversion_map = {
        1: OneDStr2Array,
        2: TwoDStr2Array,
        3: ThreeDStr2Array,
    }

    for col_name, col_data in df.items():
        if col_data.dtype == "O":
            dims = getDimensions(col_data[0])
            converter = conversion_map[dims]
            df[col_name] = df[col_name].apply(converter)

        if print_output is True:
            print(df[col_name])

        # `bool_` is a numpy type, not the same as `bool`
        assert isinstance(
            df[col_name][0],
            (
                Number,
                bool_,
                ndarray,
            ),
        ), "Something went wrong when converting dtypes from str."

    return df


def getDimensions(string: str) -> int:
    """Get dimensions of string-represented array."""
    # Assumes the number of "[" at the beginning of string corresponds to dimensions
    # of array.
    dims = string.count("[", 0, 3)

    return dims


def OneDStr2Array(x: str) -> ndarray:
    """Chop off brackets from string, then convert to ndarray."""
    # action input in format "[# # #]" or "[#, #, #]"
    sep = getSeparator(x)

    out = fromstring(x[1:-1], sep=sep)

    return out


def TwoDStr2Array(x_in: str) -> ndarray:
    """Convert 2d string representing array into 2d array.

    Argument format: '[[#, ..., #], ..., [#, ..., #]]'
    """
    # Crop outer brackets, replace left '[' with blanks, then split on new line
    # separator. Convert resulting string to arrays. Separator and new line separator
    # may vary depending on argument.
    newline_sep = getNewlineSep(x_in)

    x = x_in[1:-1]
    x = [a.strip() + "]" for a in x.split(newline_sep) if a]
    # replace inserted last ']'
    x[-1] = x[-1][:-1]

    list_of_arrays = [OneDStr2Array(a) for a in x]
    array2d = array(list_of_arrays)

    return array2d


def ThreeDStr2Array(x_in: str) -> ndarray:
    """Convert 3d string representing array into 3d array.

    Argument format: '[[[#, ..., #], ..., [#, ..., #]], ... ]]]'
    """
    newline_sep = getNewlineSep(x_in)

    x = x_in[1:-1]
    x = [a.strip() + "]" for a in x.split("]" + newline_sep) if a]
    list_of_arrays = [TwoDStr2Array(a) for a in x]

    array3d = dstack(list_of_arrays)

    return array3d


def getSeparator(s: str) -> str:
    """Get separator."""
    if "," in s:
        sep = ","
    else:
        sep = " "

    return sep


def getNewlineSep(s: str) -> str:
    """Get new line separator."""
    if "]\n" in s:
        sep = "]\n"
    else:
        sep = "],"

    return sep
