"""Custom policy builder and utils."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from typing import Tuple

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete, Space
from numpy import array, float32, int64, ndarray

# Punch Clock Imports
from punchclock.policies.greedy_cov_v2 import GreedyCovariance
from punchclock.policies.policy_base_class_v2 import CustomPolicy
from punchclock.policies.random_policy import RandomPolicy


# %% Gym Space Configs
class SpaceConfig:
    """Base class for Gym space config.

    All space configs have "space" as an attribute that is a str. SpaceConfigs are
    used to generate standardized dicts which can be saved as JSON files.
    """

    def __init__(self, space: str):
        """Initialize base class."""
        self.space = space

    def toDict(self) -> dict:
        """Output attributes as a dict."""
        # Need a dedicated method at the base class to allow DictConfig to have
        # a nested version.
        return vars(self)


class BoxConfig(SpaceConfig):
    """Config for gym.space.Box."""

    def __init__(
        self,
        low: ndarray,
        high: ndarray,
        dtype: str,
    ):
        """Initialize BoxConfig.

        Args:
            low (ndarray): See Gym.spaces.Box
            high (ndarray): See Gym.spaces.Box
            dtype (str): 'int' | 'float'
        """
        assert isinstance(low, ndarray), "low must be a numpy.ndarray."
        assert isinstance(high, ndarray), "high must be a numpy.ndarray."
        assert isinstance(dtype, str), "dtype must be a str."
        assert dtype in [
            "int",
            "float",
        ], f"dtype must be one of {['int', 'float']}"

        space = "Box"
        super().__init__(space)
        self.low = low.tolist()
        self.high = high.tolist()
        self.dtype = dtype

    def fromSpace(box_space: Box) -> BoxConfig:
        """Generate a BoxConfig from a gym.spaces.Box."""
        dtype_str = str(box_space.dtype)
        if "int" in dtype_str:
            dtype_arg = "int"
        elif "float" in dtype_str:
            dtype_arg = "float"

        space_config = BoxConfig(
            low=box_space.low,
            high=box_space.high,
            dtype=dtype_arg,
        )

        assert isinstance(space_config, SpaceConfig)

        return space_config


class MultiDiscreteConfig(SpaceConfig):
    """Config for gym.space.MultiDiscrete."""

    def __init__(self, nvec: ndarray[int] | list[int]):
        """Initialize MultiDiscreteConfig.

        Args:
            nvec (ndarray[int] | list[int]): See Gym.spaces.MultiDiscrete
        """
        space = "MultiDiscrete"
        super().__init__(space)
        assert isinstance(nvec, ndarray), "nvec must be a numpy.ndarray."
        assert all(
            isinstance(i, (int, int64)) for i in nvec
        ), "All entries of nvec must be ints."

        self.nvec = nvec.tolist()

    def fromSpace(md_space: MultiDiscrete) -> MultiDiscreteConfig:
        """Generate a MultiDiscreteConfig from a gym.spaces.MultiDiscrete."""
        space_config = MultiDiscreteConfig(nvec=md_space.nvec)
        assert isinstance(space_config, SpaceConfig)

        return space_config


class MultiBinaryConfig(SpaceConfig):
    """Config gym.space.MultiBinary."""

    def __init__(self, n: list[int] | ndarray[int] | Tuple):
        """Initialize MultiBinaryConfig.

        Args:
            n (list[int] | ndarray[int] | Tuple): See Gym.spaces.MultiBinary.
        """
        space = "MultiBinary"
        super().__init__(space)
        assert isinstance(
            n, (ndarray, list, Tuple)
        ), "n must be one of [ndarray, list, Tuple]."
        assert all(
            isinstance(i, int) for i in n
        ), "All entries of n must be ints."

        self.n = n

    def fromSpace(mb_space: MultiBinary) -> MultiBinaryConfig:
        """Generate a MultiBinaryConfig from a gym.spaces.MultiBinary."""
        space_config = MultiBinaryConfig(n=mb_space.n)
        assert isinstance(space_config, SpaceConfig)

        return space_config


class DictConfig(SpaceConfig):
    """Config for gym.space.Dict."""

    def __init__(self, spaces: dict[str, SpaceConfig]):
        """Initialize DictConfig.

        Args:
            spaces (dict): All values must be a SpaceConfig child class. Nested
                DictConfigs are allowed. Note that the argument itself must be
                a plain dict, but any nested dict-likes should be DictConfigs.
        """
        # NOTE: The arg type requirements mirror Gym Dict requirements.

        space = "Dict"
        super().__init__(space)
        # spaces must be a plain dict (NOT a gym.Dict)
        assert isinstance(spaces, dict), "spaces must be a dict"
        # Make sure "space" is not handed in as a key.
        assert all(
            [k != "space" for k in spaces.keys()]
        ), "'space' is a forbidden key in spaces."
        # Make sure all entries of dict (other than "space") are a SpaceConfig
        for v in spaces.values():
            assert isinstance(
                v, SpaceConfig
            ), "All values of spaces must be a SpaceConfig."

        self.spaces = spaces

    def fromSpace(dict_space: Dict) -> DictConfig:
        """Generate a DictConfig from a gym.spaces.Dict."""
        new_config = {}
        for k, v in dict_space.items():
            if isinstance(v, Dict):
                new_config[k] = DictConfig.fromSpace(v)
            else:
                new_config[k] = buildSpaceConfig(v)

        space_config = DictConfig(new_config)
        assert isinstance(space_config, SpaceConfig)
        return space_config

    def toDict(self) -> dict:
        """Convert DictConfig to a dict.

        Override base class method because this method needs to recurse through
        a dict.
        """
        dict_config = {"space": "Dict"}
        other_params = self.recurse2dict(self.spaces)
        dict_config.update(other_params)
        return dict_config

    def recurse2dict(self, space_config: dict | SpaceConfig) -> dict:
        """Recursively convert all SpaceConfigs in a dict to dicts."""
        if isinstance(space_config, SpaceConfig):
            config_dict = space_config.toDict()
        elif isinstance(space_config, dict):
            config_dict = {}
            for k, v in space_config.items():
                config_dict[k] = self.recurse2dict(v)

        return config_dict


# %% Maps for SpaceConfig <-> Gym Class
config_map = {
    "Box": BoxConfig,
    "Dict": DictConfig,
    "MultiBinary": MultiBinaryConfig,
    "MultiDiscrete": MultiDiscreteConfig,
}
gym_class_map = {
    "Box": Box,
    "Dict": Dict,
    "MultiBinary": MultiBinary,
    "MultiDiscrete": MultiDiscrete,
}


# %% Get SpaceConfig from Gym Space
def buildSpaceConfig(space: Space) -> SpaceConfig:
    """Build a SpaceConfig from a gym.spaces.Space.

    Args:
        space (Space): A Gym space.

    Returns:
        SpaceConfig: A JSON-able config to build a Gym space.
    """
    assert isinstance(space, Space)
    for k, v in gym_class_map.items():
        if type(space) == v:
            space_type = k
            break

    space_config = config_map[space_type].fromSpace(space)

    return space_config


# %% Build Gym space from config
def buildSpace(space_config: dict) -> Space:
    """Build a Gym space from a config dict.

    Intended for building spaces from a dict loaded from a JSON file.

    space_config (dict): Keys are specific to each type of SpaceConfig, but must
        include "space". See SpaceConfig and child classes for details.
    """
    assert isinstance(space_config, dict), "space_config must be a dict."
    assert (
        "space" in space_config.keys()
    ), "Key 'space' must be in space_config."
    assert (
        space_config["space"] in gym_class_map.keys()
    ), f"Value of space_config['space'] must be one of: {list(gym_class_map.keys())}"

    # Get the class to call when we instantiate it (later).
    space_class = gym_class_map[space_config["space"]]

    # Convert plain dtypes into ones required to instantiate Gym Spaces
    new_config = _prepForSpaceInstantiation(deepcopy(space_config))

    # Build Gym Space
    space = space_class(**new_config)

    return space


def _prepForSpaceInstantiation(space_config: dict) -> dict:
    """Recursively convert primitives to dtypes that are compatible with Gym Spaces."""
    dtype_map = {
        "int": int64,
        "float": float32,
    }

    new_config = {}
    if space_config["space"] != "Dict":
        # delete "space" entry so items can be **kwargs when building Gym space
        del space_config["space"]
        for k, v in space_config.items():
            if isinstance(v, list):
                new_config[k] = array(v)
            else:
                new_config[k] = v
            if k == "dtype":
                new_config[k] = dtype_map[v]
    else:
        del space_config["space"]
        for k, v in space_config.items():
            v_space = buildSpace(v)
            new_config[k] = v_space

    return new_config


# %% Build Custom Policy
def buildCustomPolicy(policy_config: dict) -> CustomPolicy:
    """Build a policy from a dict.

    Args:
        policy_config (dict): Contains the following items:
            {
                "policy" (str): The name of a recognized CustomPolicy.
                "observation_space" (dict): All values and sub-values must be
                    primitives. See buildSpace for details.
                "action_space" (dict): All values and sub-values must be primitives.
                    See buildSpace for details.
            }

    Returns:
        CustomPolicy: See CustomPolicy base class.
    """
    policy_map = {
        "GreedyCovariance": GreedyCovariance,
        "RandomPolicy": RandomPolicy,
    }

    assert (
        "policy" in policy_config
    ), "policy_config must contain 'policy' as a key"
    assert (
        policy_config["policy"] in policy_map
    ), "Value of 'policy' not recognized. Make sure it's on the list of supported \
        policies."

    policy_name = policy_config["policy"]
    del policy_config["policy"]

    # Observation and action spaces (Gym Spaces) need to be built from primitives.
    new_obs_space = buildSpace(policy_config["observation_space"])
    policy_config["observation_space"] = new_obs_space
    new_act_space = buildSpace(policy_config["action_space"])
    policy_config["action_space"] = new_act_space

    policy = policy_map[policy_name](**policy_config)

    return policy
