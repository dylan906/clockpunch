"""Tests for info_wrappers.py."""
# %% Imports
# Standard Library Imports
from copy import deepcopy

# Third Party Imports
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

# from gymnasium.utils.env_checker import check_env
from numpy import array, diag, eye, pi, stack, zeros
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.agents import buildRandomAgent
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.environment.info_wrappers import (
    ActionTypeCounter,
    CombineInfoItems,
    ConfigurableLogicGate,
    CovKLD,
    FilterInfo,
    GetNonZeroElements,
    LogisticTransformInfo,
    MaskViolationCounter,
    NumWindows,
    ThresholdInfo,
    TransformInfoWithNumpy,
)
from punchclock.environment.misc_wrappers import RandomInfo
from punchclock.ray.build_env import buildEnv

# %% Build env for NumWindows wrapper
print("\nBuild env for NumWindows test...")
# Dummy environment that has agents and propagates dynamics when .step() is called


class NumWindowsEnv(Env):
    """Dummy environment for NumWindows tests."""

    def __init__(self):  # noqa
        self.observation_space = Dict({"a": Box(low=0, high=1)})
        self.action_space = MultiDiscrete([1])
        agents = [buildRandomAgent(agent_type="sensor") for ag in range(2)]
        agents.extend(
            [buildRandomAgent(agent_type="target") for ag in range(3)]
        )
        self.agents = agents
        self.agents_backup = deepcopy(agents)
        self.horizon = 10
        self.time_step = 1000
        self.t = 0

    def reset(self, seed=None, options=None):  # noqa
        super().reset(seed=seed)
        self.agents = deepcopy(self.agents_backup)
        self.t = 0
        return self.observation_space.sample(), {}

    def step(self, action=None):  # noqa
        self.t += self.time_step
        for ag in self.agents:
            ag.propagate(self.t)

        return self.observation_space.sample(), 0, False, False, {}


dummy_env = NumWindowsEnv()
# %% Test NumWindows
print("\nTest NumWindows...")
print("  Case 1: Forecast every time step")
# test in default mode (forecast every step)
nw_env = NumWindows(env=deepcopy(dummy_env), use_estimates=False)

obs, info = nw_env.reset()
print("reset")
print(f"num windows = {info['num_windows_left']}")
print(f"vis_forcast.shape = {info['vis_forecast'].shape}")

for _ in range(3):
    obs, _, _, _, info = nw_env.step(nw_env.action_space.sample())
    print("step")
    print(f"num windows = {info['num_windows_left']}")
    print(f"vis_forecast.shape = {info['vis_forecast'].shape}")

obs, info = nw_env.reset()
print("reset")
print(f"num windows = {info['num_windows_left']}")
print(f"vis_forcast.shape = {info['vis_forecast'].shape}")

# Test with open_loop = True
print("\n")
print("  Case 2: Use lookup table for forecasts after initialization")

nwo_env = NumWindows(
    env=deepcopy(dummy_env), use_estimates=False, open_loop=True
)
obs, info = nwo_env.reset()
print("reset")
print(f"num windows = {info['num_windows_left']}")
print(f"vis_forcast.shape = {info['vis_forecast'].shape}")
for _ in range(3):
    obs, _, _, _, info = nwo_env.step(nwo_env.action_space.sample())
    print("step")
    print(f"num windows = {info['num_windows_left']}")
    print(f"vis_forecast.shape = {info['vis_forecast'].shape}")

obs, info = nwo_env.reset()
print("reset")
print(f"num windows = {info['num_windows_left']}")
print(f"vis_forcast.shape = {info['vis_forecast'].shape}")

# %% Test with SSAScheduler
print("\nTest NumWindows with SSAScheduler environment")
# Environment params
RE = getConstants()["earth_radius"]

horizon = 50
time_step = 100
# sensor locations (fixed)
x_sensors_lla = [
    array([0, 0, 0]),
    array([0, pi / 4, 0]),
]

num_sensors = len(x_sensors_lla)
num_targets = 3

x_sensors_ecef = [lla2ecef(x_lla=x) for x in x_sensors_lla]
x_sensors_eci = [ecef2eci(x_ecef=x, JD=0) for x in x_sensors_ecef]

# agent params
agent_params = {
    "num_sensors": num_sensors,
    "num_targets": num_targets,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": None,
    "target_dist": "uniform",
    "sensor_dist_frame": None,
    "target_dist_frame": "COE",
    "sensor_dist_params": None,
    "target_dist_params": [
        [RE + 300, RE + 800],
        [0, 0],
        [0, pi / 2],
        [0, 2 * pi],
        [0, 2 * pi],
        [0, 2 * pi],
    ],
    "fixed_sensors": x_sensors_eci,
    "fixed_targets": None,
}

# Set the UKF parameters. We are using the abbreviated interface for simplicity,
# see ezUKF for details.
temp_matrix = diag([1, 1, 1, 0.01, 0.01, 0.01])
filter_params = {
    "Q": 0.001 * temp_matrix,
    "R": 1 * 0.1 * temp_matrix,
    "p_init": 10 * temp_matrix,
}

reward_params = None

constructor_params = {
    "wrappers": [
        {
            "wrapper": "NumWindows",
            "wrapper_config": {
                "use_estimates": False,
                "new_keys": ["num_windows_alt", "vis_forecast"],
            },
        }
    ]
}

env_config = {
    "horizon": horizon,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "time_step": time_step,
    "constructor_params": constructor_params,
}
ssa_env = buildEnv(env_config)
ssa_env.reset()
# test in loop
for _ in range(10):
    obs, _, _, _, info = ssa_env.step(ssa_env.action_space.sample())
    print(f"num windows left = {info['num_windows_alt']}")
    print(f"vis forecast shape = {info['vis_forecast'].shape}")

# %% Test ActionTypeCounter
print("\nTest ActionTypeCounter...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((2, 2))}),
        "action_space": MultiDiscrete([3, 3]),
    }
)
nar_env = ActionTypeCounter(rand_env, new_key="count")

action = array([0, 2])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"info={info}")


nar_env = ActionTypeCounter(rand_env, new_key="count", count_null_actions=False)

action = array([0, 0])
(obs, reward, term, trunc, info) = nar_env.step(action)
print(f"action = {action}")
print(f"info={info}")

# %% Test MaskViolationCounter
print("\nTest MaskViolationCounter...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary((3, 4))}),
        "action_space": MultiDiscrete([3, 3, 3, 3]),
        "reward_space": Box(0, 0),
    }
)
maskvio_env = MaskViolationCounter(
    rand_env,
    new_key="count",
    action_mask_key="a",
)
action = array([0, 0, 0, 2])

(obs, reward, term, trunc, info) = maskvio_env.step(action)
print(f"obs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"info={info}")

# Test with counting valid actions instead
maskvio_env = MaskViolationCounter(
    rand_env,
    new_key="count",
    action_mask_key="a",
    count_valid_actions=True,
)

(obs, reward, term, trunc, info) = maskvio_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"info={info}")

# Test with accounting for null actions
maskvio_env = MaskViolationCounter(
    rand_env,
    new_key="count",
    action_mask_key="a",
    ignore_null_actions=False,
)

(obs, reward, term, trunc, info) = maskvio_env.step(action)
print(f"\nobs['a'] = \n{obs['a']}")
print(f"action = {action}")
print(f"info={info}")
# %% Test ThresholdInfo
print("\nTest ThresholdInfo...")
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({}),
            "action_space": MultiDiscrete([1]),
        }
    )
)

thresh_env = ThresholdInfo(
    rand_env,
    info_key=0,
    new_key="meets_threshold",
    threshold=0.5,
)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"threshold = {thresh_env.threshold}")
print(f"info = {info}")

# Test with MultiBinary space
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({}),
            "action_space": MultiDiscrete([2, 2]),
        }
    ),
    info_space=Dict({"a": MultiBinary(n=())}),
)
thresh_env = ThresholdInfo(
    rand_env, info_key="a", new_key="meets_threshold", threshold=0.5
)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"threshold = {thresh_env.threshold}")
print(f"info = {info}")

# Test with threshold_reward
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({}),
            "action_space": MultiDiscrete([1]),
        }
    )
)

thresh_env = ThresholdInfo(
    rand_env,
    info_key=0,
    new_key="meets_threshold",
    threshold=0.5,
    threshold_reward=-1.1,
    inequality=">",
)
(obs, reward, term, trunc, info) = thresh_env.step(
    thresh_env.action_space.sample()
)
print(f"threshold = {thresh_env.threshold}")
print(f"info = {info}")

# %% Test LogisticTransformInfo
print("\nTest LogisticTransformInfo...")

rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    )
)
log_env = LogisticTransformInfo(rand_env, key=0)

_, info_unwrapped = log_env.env.reset()
info_wrapped = log_env.updateInfo(None, None, None, None, info_unwrapped, None)
print(f"info (unwrapped) = {info_unwrapped}")
print(f"info (wrapped) = {info_wrapped}")

(_, _, _, _, info) = log_env.step(log_env.action_space.sample())
print(f"info (via step) = {info}")

# %% Test CovKLD
print("\nTest CovKLD...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict(
        {
            "sigma0": Box(
                low=zeros((2, 6, 6)),
                high=stack([eye(6), eye(6)]),
            ),
            "sigma1": Box(
                low=zeros((2, 6, 6)),
                high=stack([eye(6), eye(6)]),
            ),
        }
    ),
)

kld_env = CovKLD(
    rand_env,
    new_key="kld",
    pred_cov="sigma0",
    est_cov="sigma1",
)
(_, _, _, _, info) = kld_env.step(kld_env.action_space.sample())
print(f"info (via step) = {info}")
# %% TransformInfoWith Numpy
print("\nTest TransformInfoWithNumpy...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict({"a": Box(low=0, high=2, shape=(2, 2))}),
)

np_env = TransformInfoWithNumpy(env=rand_env, numpy_func_str="sum", key="a")
(_, _, _, _, info) = np_env.step(np_env.action_space.sample())
print(f"info (via step) = {info}")

np_env = TransformInfoWithNumpy(env=rand_env, numpy_func_str="nonzero", key="a")
(_, _, _, _, info) = np_env.step(np_env.action_space.sample())
print(f"info (via step) = {info}")

# %% Test CombineInfoItems
print("\nTest CombineInfoItems...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict(
        {
            "a": Box(low=0, high=2, shape=(2, 2)),
            "b": MultiBinary((2, 2)),
            "c": Box(low=-1, high=4, shape=(2, 2), dtype=int),
        }
    ),
)

cat_env = CombineInfoItems(rand_env, keys=["a", "b", "c"], new_key="new")
(_, _, _, _, info) = cat_env.step(cat_env.action_space.sample())
print(f"info (via step) = {info}")

# %% Test GetNonZeroElements
print("\nTest GetNonZeroElements...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict(
        {
            "a": Box(low=0, high=2, shape=(2, 2)),
            "b": MultiBinary((2, 2)),
            "c": Box(low=0, high=3, shape=(2,), dtype=int),
        }
    ),
)
for k in ["a", "b", "c"]:
    nz_env = GetNonZeroElements(rand_env, key=k, new_key="nonzeros")
    (_, _, _, _, info) = nz_env.step(nz_env.action_space.sample())
    print(f"info (via step) = {info}")

# %% Test ConfigrableLogicGate
print("\nTest ConfigrableLogicGate...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict(
        {
            "a": Box(low=0, high=2, shape=(2, 2)),
            "b": Discrete(2),
        }
    ),
)
clg_env = ConfigurableLogicGate(
    rand_env,
    key="b",
    return_if_false="a",
)

_, info = clg_env.reset()
print(f"info (via reset) = {info}")

(_, _, _, _, info) = clg_env.step(clg_env.action_space.sample())
print(f"info (via step) = {info}")

# %% FilterInfo
print("\nTest FilterInfo...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict(
        {
            "a": Discrete(2),
            "b": Discrete(2),
        }
    ),
)
fi_env = FilterInfo(env=rand_env, keys=["a"])

_, info = fi_env.reset()
print(f"info (via reset) = {info}")

(_, _, _, _, info) = fi_env.step(fi_env.action_space.sample())
print(f"info (via step) = {info}")

fi_env = FilterInfo(env=rand_env, keys=["a"], reverse=True)
(_, _, _, _, info) = fi_env.step(fi_env.action_space.sample())
print("Test with reverse filter:")
print(f"info (via step) = {info}")

# %% done
print("done")
