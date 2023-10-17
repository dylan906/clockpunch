"""Misc wrappers."""
# %% Imports
# Standard Library Imports
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, Tuple, final
from warnings import warn

# Third Party Imports
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete, Space
from numpy import append, array, int8, int64, ones

# Punch Clock Imports
from punchclock.analysis_utils.utils import countMaskViolations
from punchclock.common.custody_tracker import CustodyTracker
from punchclock.common.utilities import actionSpace2Array, getInfo
from punchclock.environment.wrapper_utils import (
    OperatorFuncBuilder,
    SelectiveDictProcessor,
    binary2ActionMask,
)
from punchclock.policies.policy_builder import buildSpace


# %% RandomInfo
class RandomInfo(Wrapper):
    """Appends items to an env's info.

    Used for development.
    """

    def __init__(self, env, info_space: Dict = None):
        """Wrap an env."""
        super().__init__(env)
        if info_space is None:
            info_space = Dict({0: Box(0, 1)})

        assert isinstance(info_space, Dict)
        assert all([isinstance(i, Space) for i in info_space.values()])
        self.info_space = info_space

    def step(self, action):
        """Step an env."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        new_info = self.info_space.sample()
        infos.update(new_info)
        return (observations, rewards, terminations, truncations, infos)

    def reset(self, seed: int | None = None, options=None):
        """Reset me bro."""
        obs = self.observation_space.sample()
        info = self.info_space.sample()

        return obs, info


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    # NOTE: SimRunner is hugely dependent on this class. Be careful about modifying
    # it.

    def __init__(self, env: Env, id: Any = None):  # noqa
        """Wrap environment with IdentityWrapper.

        Args:
            env (Env): A Gymnasium environment.
            id (Any, optional): Mainly used to distinguish between multiple instances
                of IdentityWrapper. Defaults to None.
        """
        super().__init__(env)

        self.id = id
        return

    def observation(self, obs):
        """Pass-through observation."""
        return obs


# %% Base class: ModifyObsOrInfo
class ModifyObsOrInfo(Wrapper):
    """Base wrapper used for wrapper that modify either info or obs."""

    def __init__(self, env: Env, obs_info: str):
        """Initialize wrapper.

        Args:
            env (Env): Must have Dict observation space.
            obs_info (str): ["obs" | "info"] The wrapper modifies either the
                observation or the info (not both).
        """
        super().__init__(env)
        assert isinstance(
            env.observation_space, Dict
        ), "env.observation_space must be a gymnasium.spaces.Dict."
        assert obs_info in [
            "info",
            "obs",
        ], "obs_info must be either 'obs' or 'info'."

        self.obs_info = obs_info

    @abstractmethod
    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Child classes require this function.

        Args:
            obs (OrderedDict): Unwrapped observation.
            info (dict): Unwrapped info.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", this is the modified
                observation. Otherwise, same as input.
            new_info (dict): If self.obs_info == "info", this is the modified
                info. Otherwise, same as input.
        """
        return new_obs, new_info  # noqa

    @final
    def reset(
        self, seed: int | None = None, options=None
    ) -> Tuple[OrderedDict, dict]:
        """Reset env."""
        obs, info = super().reset(seed=seed, options=options)
        new_obs, new_info = self.modifyOI(
            obs=deepcopy(obs), info=deepcopy(info)
        )
        self.info = deepcopy(info)
        return new_obs, new_info

    @final
    def step(self, action: Any) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Step environment."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        new_obs, new_info = self.modifyOI(obs=observations, info=infos)
        self.info = deepcopy(new_info)
        return (new_obs, rewards, terminations, truncations, new_info)

    @final
    def observation(self, observation: OrderedDict) -> OrderedDict:
        """Modify observation only if obs_info == "obs"."""
        if self.obs_info == "obs":
            info = deepcopy(self.info)
            new_obs, _ = self.modifyOI(obs=observation, info=info)
        else:
            new_obs = observation

        return new_obs


# %% CopyObsInfoItem
class CopyObsInfoItem(ModifyObsOrInfo):
    """Copy an item from and obs/info to obs/info.

    Overwrites existing item, if it already exists.

    Environment must have Dict observation space.
    """

    def __init__(
        self,
        env: Env,
        copy_from: str,
        copy_to: str,
        from_key: str,
        to_key: str = None,
        info_space_config: dict = None,
    ):
        """Wrap environment with CopyObsInfoItem wrapper.

        Args:
            env (Env): Gymnasium environment with Dict observation space.
            copy_from (str): ["obs" | "info"] The source of the item.
            copy_to (str): ["obs" | "info"] The place to copy the source item to.
            from_key (str): Must be in copy_from associated dict.
            to_key (str, optional): Key assigned to copied value. If None, from_key
                will be used. Defaults to None.
            info_space_config (dict, optional): Required if copy_from/copy_to
                are "info"/"obs". Ignored otherwise. Config to build a space
                corresponding to source item in info. See buildSpace for details.
                Format is:
                    {
                        "space": Name of space class (e.g. "MultiDiscrete"),
                        kwargs: kwargs used for the desired space,
                    }
        """
        for k in [copy_from, copy_to]:
            assert k in ["info", "obs"]

        super().__init__(env=env, obs_info=copy_to)

        if copy_from == "info":
            copy_source = getInfo(env)
        elif copy_from == "obs":
            copy_source = deepcopy(env.observation_space.spaces)

        if copy_to == "info":
            copy_destination = getInfo(env)
        elif copy_to == "obs":
            copy_destination = deepcopy(env.observation_space)

        assert (
            from_key in copy_source
        ), f"{from_key} is not a key in {copy_from}."

        if to_key is None:
            # default key
            to_key = from_key

        if to_key in copy_destination:
            warn(
                f"{to_key} is already in {copy_destination}. Value will be overwritten."
            )

        if copy_from == "info" and copy_to == "obs":
            assert (
                info_space_config is not None
            ), "info_space_config is required if copy from info into obs."

        self.copy_from = copy_from
        self.copy_to = copy_to
        self.from_key = from_key
        self.to_key = to_key

        # build new observation space, if required
        if self.copy_to == "obs":
            new_obs_space = deepcopy(env.observation_space)
            if self.copy_from == "info":
                self.info_item_space = buildSpace(
                    space_config=info_space_config
                )
                new_obs_space[self.to_key] = self.info_item_space
            elif self.copy_from == "obs":
                new_obs_space[self.to_key] = copy_source[self.from_key]

            self.observation_space = new_obs_space

    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Modify unwrapped obs/info, output wrapped obs/info.

        Args:
            obs (OrderedDict): Observation.
            info (dict): Info.

        Returns:
            new_obs (OrderedDict): If self.copy_to == "obs", this is the modified
                observation. Otherwise, same as input.
            new_info (dict): If self.copy_to == "info", this is the modified
                info. Otherwise, same as input.
        """
        new_item = self._getSourceItem(info=info, obs=obs)
        info_new, obs_new = self._copySourceToDestinationItem(
            info=info,
            obs=obs,
            source_item=new_item,
        )

        return obs_new, info_new

    def _getSourceItem(self, info: dict, obs: dict) -> dict:
        """Copy an item from obs or info."""
        if self.copy_from == "info":
            source = deepcopy(info)
        elif self.copy_from == "obs":
            source = deepcopy(obs)

        source_value = source[self.from_key]
        new_item = {self.from_key: source_value}

        return new_item

    def _copySourceToDestinationItem(
        self,
        info: dict,
        obs: dict,
        source_item: dict,
    ) -> Tuple[dict, dict]:
        """Copy source item to either info or dict, return updated info/dict.

        One of the returns (info or obs) is modified. The other return is same
        as input.

        Returns:
            info (dict): info
            obs (dict): observation
        """
        destination_item = {self.to_key: source_item[self.from_key]}
        if self.copy_to == "info":
            info = deepcopy(info)
            info.update(destination_item)
        elif self.copy_to == "obs":
            obs = deepcopy(obs)
            obs.update(destination_item)

        return info, obs


# %% OperatorWrapper
class OperatorWrapper(ModifyObsOrInfo):
    """Apply a function from operator package to obs or info.

    See OperatorFuncBuilder for details.
    """

    def __init__(
        self,
        env: Env,
        obs_or_info: str,
        func_str: str,
        key: str,
        copy_key: str = None,
        a: Any = None,
        b: Any = None,
        b_key: str = None,
    ):
        """Initialize wrapper with modifying function from operator package.

        Args:
            env (Env): Must have Dict observation space.
            obs_or_info (str): ["obs" | "info"] Whether this wrapper modifies
                observation or info.
            func_str (str): Str representation of a function from the operator
                package.
            key (str): Key from observation or info (whichever is indicated by
                obs_or_info).
            copy_key (str, optional): New key to create (in obs or info) with
                transformed value. If None, transformed value overwrites unwrapped
                value. Defaults to None.
            a (Any, optional): Provide only if fixing the first arg of func. Defaults
                to None.
            b (Any, optional): Provide only if fixing the second arg of func.
                Defaults to None.
            b_key (str, optional): Use if the second arg of function is another
                item in info/obs. Defaults to None.
        """
        super().__init__(env=env, obs_info=obs_or_info)

        self.func = OperatorFuncBuilder(func_str=func_str, a=a, b=b)

        self.obs_or_info = obs_or_info
        self.key = key

        if copy_key is None:
            copy_key = key
        self.copy_key = copy_key

        if b_key is None:
            b_key = ""
        self.b_key = b_key

    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Modify observation or info.

        Args:
            obs (OrderedDict): Unwrapped obs.
            info (dict): Unwrapped info.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", this is the modified
                observation. Otherwise, same as input.
            new_info (dict): If self.obs_info == "info", this is the modified
                info. Otherwise, same as input.
        """
        # Get the appropriate item from the appropriate dict.
        # Then transform the item via the operation function defined on instantiation.
        # Then update the appropriate dict and output obs and info.
        if self.obs_info == "obs":
            thing = obs[self.key]
            secondary = obs.get(self.b_key)
        elif self.obs_info == "info":
            thing = info[self.key]
            secondary = info.get(self.b_key)

        if secondary is None:
            thing_trans = self.func(thing)
        else:
            thing_trans = self.func(thing, secondary)

        if self.obs_info == "obs":
            obs.update({self.copy_key: thing_trans})
        elif self.obs_info == "info":
            info.update({self.copy_key: thing_trans})

        return obs, info


# %% MaskViolationChecker
class MaskViolationChecker(Wrapper):
    """Check if action violates action mask and warn if so."""

    def __init__(self, env: Env, mask_key: str, debug: bool = False):
        """Wrap environment.

        Args:
            env (Env): Must have MultiDiscrete action space of shape [N+1] * M.
            mask_key (str): Key in info corresponding to action mask. Action mask
                must be binary and shape (N+1, M).
            debug (bool, optional): If True, skip mask checking. Wrapper becomes
                a pass-through. Defaults to False.
        """
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a MultiDiscrete."

        super().__init__(env)
        self.debug = debug
        self.mask_key = mask_key
        self.previous_mask = None
        self.num_sensors = len(env.action_space.nvec)
        # Assumes action space includes inaction
        self.num_targets = env.action_space.nvec[0] - 1

    def step(self, action: Any) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Step environment."""
        if (self.previous_mask is not None) and (self.debug is False):
            action_2d = actionSpace2Array(
                actions=action,
                num_sensors=self.num_sensors,
                num_targets=self.num_targets,
            )
            num_violations = countMaskViolations(
                x=action_2d, mask=self.previous_mask
            )
            if num_violations > 0:
                warn(
                    f"""Action mask violated.
                     action = {action}
                     action_mask = {self.previous_mask}"""
                )

        (obs, reward, termination, truncation, info) = self.env.step(action)
        self.previous_mask = deepcopy(info[self.mask_key])
        return obs, reward, termination, truncation, info


# %% getIdentityWrapper
# This was in wrapper_utils.py, but that causes circular import errors, so moving
# here for now. Cludge solution.
def getIdentityWrapperEnv(env: Env) -> Env:
    """Get the IdentityWrapper level of an env, if one exists.

    Args:
        env (Env): A Gymnasium environment.

    Raises:
        Exception: If there is no IdentityWrapper in the stack of wrappers, raises
            an Exception.

    Returns:
        Env: Returns the environment with IdentityWrapper at the top level. All
            wrappers above IdentityWrapper are discarded.
    """
    env_temp = deepcopy(env)
    while not isinstance(env_temp, IdentityWrapper):
        if env_temp == env_temp.unwrapped:
            raise Exception(f"No IdentityWrapper in {env}")

        env_temp = getattr(env_temp, "env", {})

    return env_temp


# %% CustodyWrapper
class CustodyWrapper(ModifyObsOrInfo):
    """Add 'custody' as an item to a Dict observation space or info.

    Custody entry is a MultiBinary space with shape (N,), where N is the number
    of targets. Does not modify other items in observation/info, just adds "custody"
    to end of (ordered) dict.

    Unwrapped env must have `key` in observation space or info (depending on value
    of `obs_info`). Corresponding item must be a 3d Box space.
    """

    def __init__(
        self,
        env: Env,
        obs_info: str,
        key: Any,
        config: dict = None,
        target_names: list = None,
        initial_status: list[bool] = None,
    ):
        """Wrap environment with CustodyWrapper.

        Args:
            env (Env): Must have a Dict observation space.
            obs_info (str): ["obs" | "info"] The wrapper modifies either the observation
                or the info (not both).
            key (Any): A key contained in the observation space or info (depending
                on value of `obs_info`). The value corresponding to this key must
                conform to interface expected in CustodyTracker via `config`.
            config (dict, optional): See CustodyTracker for details. Defaults to None.
            target_names (list, optional): Target names. Used for debugging. Must
                have length == env.action_space.nvec[0]. Defaults to None.
            initial_status (list[bool], optional): See CustodyTracker for details.
                Defaults to None.
        """
        super().__init__(env=env, obs_info=obs_info)

        assert (
            key in env.observation_space.spaces
        ), f"{key} must be in env.observation_space."
        if obs_info == "obs":
            assert (
                "custody" not in env.observation_space.spaces
            ), "'custody' is already in env.observation_space."
        elif obs_info == "info":
            info_test = getInfo(env)
            assert "custody" not in info_test, "'custody' is already in info."

        assert isinstance(
            env.action_space, MultiDiscrete
        ), "Action space must be MultiDiscrete."

        num_targets = env.action_space.nvec[0] - 1

        if target_names is not None:
            assert (
                len(target_names) == num_targets
            ), "num_targets must equal len(target_names) (if target_names is not None)."

        # make wrapper
        self.key = key

        self.custody_tracker = CustodyTracker(
            config=config,
            num_targets=num_targets,
            target_names=target_names,
        )

        if obs_info == "obs":
            # Update observation space, maintain order, append "custody" to end.
            new_space = OrderedDict({**env.observation_space})
            new_space["custody"] = MultiBinary(num_targets)
            self.observation_space = Dict(new_space)

    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Append custody entry to observation or info dict.

        Custody is a (N,) binary array where 1 indicates the n-th target is in
        custody.

        Args:
            obs (OrderedDict): Unwrapped observation. If `self.obs_info` ==
                "obs", must contain self.key.
            info (dict): Unwrapped info. If `self.obs_info` == "info", must
                contain self.key.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", same as input obs
                but with "custody" item appended. Otherwise, same as input obs.
            new_info (dict): If self.obs_info == "info", same as input info but
                with "custody" item appended. Otherwise, same as input info.
        """
        if self.obs_info == "obs":
            relevant_dict = obs
        elif self.obs_info == "info":
            relevant_dict = info

        custody_input = relevant_dict[self.key]

        # custody_tracker outputs custody status as a list of bools; convert to
        # a 1d array of ints. Use int8 for dtype-> this is the default dtype of
        # MultiBinary space. Make sure "custody" is added at end of OrderedDict
        # observation.
        custody = array(self.custody_tracker.update(custody_input)).astype(int8)
        new_item = {"custody": custody}

        if self.obs_info == "obs":
            new_obs = deepcopy(obs)
            new_obs.update(new_item)
            new_info = info
        elif self.obs_info == "info":
            new_obs = obs
            new_info = deepcopy(info)
            new_info.update(new_item)

        return new_obs, new_info


# %% TruncateIfNoCustody
class TruncateIfNoCustody(Wrapper):
    """Truncate environment if custody array is all zeros."""

    def __init__(self, env: Env, obs_info: str, key: str):
        """Wrap environment that already has custody array.

        Args:
            env (Env): Gymnasium environment.
            obs_info (str): ["obs" | "info"] Whether custody array is in observation
                or info.
            key (str): Key corresponding to custody array.
        """
        super().__init__(env)
        if obs_info == "info":
            the_dict = getInfo(env)
        elif obs_info == "obs":
            the_dict = env.observation_space.spaces

        assert key in the_dict

        self.obs_info = obs_info
        self.key = key

    def step(self, action) -> Tuple[OrderedDict, float, bool, bool, dict]:
        """Set truncate to True if custody array is all zeros.

        Does not modify obs, reward, termination, info.
        """
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        relevant_item = self._getSourceItem(info=infos, obs=observations)
        custody = relevant_item[self.key]
        if all([c == 0 for c in custody]):
            truncations = True

        return (observations, rewards, terminations, truncations, infos)

    def _getSourceItem(self, info: dict, obs: dict) -> dict:
        """Copy an item from obs or info."""
        if self.obs_info == "info":
            source = deepcopy(info)
        elif self.obs_info == "obs":
            source = deepcopy(obs)

        source_value = source[self.key]
        new_item = {self.key: source_value}

        return new_item


# %% ConvertCustody2ActionMask
class ConvertCustody2ActionMask(ModifyObsOrInfo):
    """Convert a MultiBinary custody array to a MultiBinary action mask.

    Assumes inaction is a valid action.

    Notation:
        M: number of sensors
        N: number of targets
        custody array: N-long, 1d array, where a 1 indicates the n'th target is
            in custody, and 0 otherwise.
        action mask: (N+1, M) binary array, where a 1 indicates the (n, m)'th
            target-sensor pair is a valid action. The extra "+1" entries denote
            inaction, which is always valid.

    Example (M=2):
        env.observation_space = Dict(
            {
                "custody": MultiBinary(3),
            }

        wrapped_env = ConvertCustody2ActionMask(
            env,
            key = "custody",
            )

        wrapped_env.observation_space = Dict(
            {
                "custody": MultiBinary([4, 2]),
            }
        )

    Example (with new_key) (M=2):
        env.observation_space = Dict(
            {
                "custody": MultiBinary(3),
                "bar": Box(0, 1)
            }

        wrapped_env = ConvertCustody2ActionMask(
            env,
            key = "custody",
            new_key = "foo"
            )

        wrapped_env.observation_space = Dict(
            {
                "custody": MultiBinary(3),
                "bar": Box(0, 1)
                "foo": MultiBinary([4, 2]),  # new item is at end of Dict
            }
        )


    """

    def __init__(
        self,
        env: Env,
        obs_info: str,
        key: str,
        new_key: str = None,
    ):
        """Wrap environment.

        Args:
            env (Env): Must have a Dict observation_space.
            obs_info (str): ["obs" | "info"] The wrapper modifies either the
                observation or the info (not both).
            key (str): Contained in env.observation_space or info, depending on
                value of `obs_info`. Must be a 1d MultiBinary space.
            new_key (str, optional): New key to be appended to observation or
                info, depending on value of `obs_info`. Overwrites unwrapped key
                if `new_key` already exists. If None, uses value of `key`. Defaults
                to None.
        """
        super().__init__(env=env, obs_info=obs_info)
        if new_key is None:
            new_key = key

        info_sample = getInfo(env)
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a gymnasium.spaces.MultiDiscrete."

        if obs_info == "obs":
            assert (
                key in env.observation_space.spaces
            ), f"{key} must be in observation space."
            assert isinstance(
                env.observation_space.spaces[key], MultiBinary
            ), f"env.observation_space[{key}] must be a MultiBinary."
            if new_key in env.observation_space.spaces:
                warn(
                    f"""{new_key} is already in env.observation_space. Values will
                be overwritten."""
                )
        elif obs_info == "info":
            assert key in info_sample, f"{key} must be in info."
            if new_key in info_sample:
                warn(
                    f"{new_key} is already in info. Values will be overwritten."
                )

        self.key = key
        self.new_key = new_key

        self.num_sensors = env.action_space.shape[0]
        if obs_info == "obs":
            self.num_targets = int(env.observation_space[key].n)
        elif obs_info == "info":
            self.num_targets = len(info_sample[key])

        # mask space used for debugging/development
        self.mask2d_space = MultiBinary(
            [self.num_targets + 1, self.num_sensors]
        )

        # convert num_targets from numpy dtype to Python int
        self.sdp = SelectiveDictProcessor(
            funcs=[partial(binary2ActionMask, num_sensors=self.num_sensors)],
            keys=[new_key],
        )

        if obs_info == "obs":
            # update obs space if necessary
            self.observation_space = deepcopy(env.observation_space)
            self.observation_space[new_key] = self.mask2d_space

    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Convert custody observation to action mask.

        Value is converted from custody array to action mask.
            See class description for details. Returned OrderedDict/dict has same
            keys as obs, unless self.new_key was set on instantiation. If
            self.new_key != self.key, then returned OrderedDict will not
            have self.key, but will have self.new_key. Order of returned
            OrderedDict is same as obs. "new_key", if used, is appended
            to end of OrderedDict.

        Args:
            obs (OrderedDict): Must have self.key in keys.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", same as input obs
                but with `self.new_key` item appended. Otherwise, same as input obs.
            new_info (dict): If self.obs_info == "info", same as input info but
                with `self.new_key` item appended. Otherwise, same as input info.
        """
        if self.obs_info == "obs":
            relevant_dict = deepcopy(obs)
        elif self.obs_info == "info":
            relevant_dict = deepcopy(info)

        # SDP overwrites a value of a dict. Copy unwrapped value to new key-value
        # pair (while keeping unwrapped kv pair). Copied value will be overwritten
        # when SDP applyies transformation function.
        relevant_dict.update({self.new_key: relevant_dict[self.key]})

        # sdp.applyFunc overwrites new_obs[new_key], leaves other keys untouched
        # new_obs = self.sdp.applyFunc(new_obs)
        transformed_dict = self.sdp.applyFunc(relevant_dict)

        if self.obs_info == "obs":
            new_obs = transformed_dict
            new_info = info
        elif self.obs_info == "info":
            new_obs = obs
            new_info = transformed_dict

        return new_obs, new_info


# %% VisMap2ActionMask
class VisMap2ActionMask(ModifyObsOrInfo):
    """Convert visibility map into a 2d action mask.

    Append a row of 1's to the bottom of a visibility map.

    Set action_mask_on == False to make the modified observation space item always
    an array of 1s.

    Example (B = 2):
        env.observation_space = {
            "vis_map": MultiBinary((A, B))
        }

        env.action_space = MultiDiscrete([A+1, A+1])

        wrapped_env = VisMap2ActionMask(env,
            obs_info="obs",
            vis_map_key="vis_map",
            new_key="action_mask")

        wrapped_env.observation_space = {
            "vis_map": MultiBinary((A, B))
            "action_mask": MultiBinary((A+1, B))
        }

    """

    def __init__(
        self,
        env: Env,
        obs_info: str,
        vis_map_key: str,
        new_key: str = None,
        action_mask_on: bool = True,
    ):
        """Wrap environment with VisMap2ActionMask.

        Args:
            env (Env): Must have:
                - Dict observation space
                - MultiDiscrete action space
                - vis_map_key must be in observation space or info (depending on
                    value of `obs_info`)
                - value associated with vis_map_key must be a 2d MultiBinary
                - Number of columns in value associated with vis_map_key must
                    be same as length of action space.
            obs_info (str): ["obs" | "info"] The wrapper modifies either the
                observation or the info (not both).
            vis_map_key (str): An item in observation space or info.
            new_key (str, optional): Name of key to be appended to obs or info.
                If None, value of `key` will be used, and associated values will
                be overwritten. Defaults to None.
            action_mask_on (bool, optional): If False, value associated with `new_key`
                will be an array of 1s. Otherwise, will be same as value from
                vis_map_key with row of 1s appended. Defaults to True.
        """
        super().__init__(env=env, obs_info=obs_info)

        assert isinstance(
            env.action_space, MultiDiscrete
        ), "env.action_space must be a MultiDiscrete."

        info_sample = getInfo(env)
        if obs_info == "obs":
            relevant_dict = env.observation_space.spaces
        elif obs_info == "info":
            relevant_dict = info_sample

        assert (
            vis_map_key in relevant_dict
        ), f"""vis_map_key must be in observation space or info (obs_info =
        {obs_info})."""

        vis_map_sample = relevant_dict[vis_map_key]

        if obs_info == "obs":
            # vis_map_sample is a Space
            assert isinstance(
                vis_map_sample, MultiBinary
            ), f"observation_space[{vis_map_key}] must be a gym.spaces.MultiBinary."
            assert (
                len(vis_map_sample.shape) == 2
            ), f"observation_space[{vis_map_key}] must be 2d."
        elif obs_info == "info":
            # vis_map_sample is a ndarray
            assert vis_map_sample.ndim == 2, f"info[{vis_map_key}] must be 2d."

        assert vis_map_sample.shape[1] == len(
            env.action_space.nvec
        ), """Shape mismatch between action space and selected item in observation
        space. The number of columns in observation_space[vis_map_key] must be equal
        to length of action_space.nvec."""

        if new_key is None:
            new_key = vis_map_key

        self.vis_map_key = vis_map_key
        self.new_key = new_key
        self.action_mask_on = action_mask_on

        # mask space used for dev/debugging
        num_rows, num_cols = vis_map_sample.shape
        self.mask_space = MultiBinary((num_rows + 1, num_cols))

        if obs_info == "obs":
            # redefine obs space if necessary
            self.observation_space = deepcopy(env.observation_space)
            self.observation_space[new_key] = self.mask_space

    def modifyOI(
        self, obs: OrderedDict, info: dict
    ) -> Tuple[OrderedDict, dict]:
        """Generate wrapped observation.

        Either obs or info (depending on value of self.obs_info) must contain
            `vis_map_key` as a key.

        Args:
            obs (OrderedDict): Unwrapped observation.
            info (dict): Unwrapped info.

        Returns:
            new_obs (OrderedDict): If self.obs_info == "obs", same as input obs
                but with `self.new_key` item appended. Otherwise, same as input obs.
            new_info (dict): If self.obs_info == "info", same as input info but
                with `self.new_key` item appended. Otherwise, same as input info.

        Example:
           vis_map = array([[1, 0],
                            [0, 0],
                            [0, 0]])

            action_mask = array([[1, 0],
                                 [0, 0],
                                 [0, 0],
                                 [1, 1]])  <- inaction always 1 (valid)
        """
        if self.obs_info == "obs":
            relevant_dict = deepcopy(obs)
        elif self.obs_info == "info":
            relevant_dict = deepcopy(info)

        mask = relevant_dict[self.vis_map_key]
        m = mask.shape[1]
        mask = append(mask, ones(shape=(1, m), dtype=int64), axis=0)

        if self.action_mask_on is False:
            # Get pass-thru action mask (no actions are masked)
            mask = ones(shape=mask.shape, dtype=int64)

        relevant_dict.update({self.new_key: mask})

        if self.obs_info == "obs":
            new_obs = relevant_dict
            new_info = info
        elif self.obs_info == "info":
            new_obs = obs
            new_info = relevant_dict

        return new_obs, new_info
