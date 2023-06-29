"""Test for threshold_v2.py."""
# %% Imports

# Third Party Imports
from numpy import array
from numpy.random import rand, randint

# Punch Clock Imports
from punchclock.reward_funcs.threshold_v2 import Threshold

# %% Test initialization
print("\nTest initialization...")
rwd_func = Threshold(
    obs_or_info="obs",
    metric="num_tasked",
    metric_value=1.0,
    penalty=2,
    reward=1,
    inequality=">",
    preprocessors=["max"],
)
print(f"metric = {rwd_func.metric}")
print(f"threshold = {rwd_func.metric_threshold}")
# %% Test initialization with dict preprocessor
print("\nTest initialization with dict preprocessor...")
# NOTE: Using only crop_array gives a non-functional reward function because it
# doesn't return a float.
rwd_func2 = Threshold(
    obs_or_info="obs",
    metric="a_matrix",
    metric_value=1.0,
    penalty=2,
    reward=1,
    inequality=">",
    preprocessors=[
        {
            "preprocessor": "crop_array",
            "config": {
                "indices_or_sections": 2,
                "section_to_keep": 0,
                "axis": 0,
            },
        },
    ],
)
print(f"preprocessor = {rwd_func2.preprocessors}")

# %% Test getMetrics
print("\nTest getMetrics()...")
# list_of_targets = [ag for ag in env.agents if type(ag) is Target]
obs = {"num_tasked": array([1, 5, 3, 6])}
metrics = rwd_func.getMetrics(obs=obs, info=None)
print(f"metrics = {metrics}")
# %% Test preprocess()
print("\nTest preprocess()...")
processed_metrics = rwd_func.preprocess(metrics=metrics)
print(f"processed_metrics = {processed_metrics}")

# %% Test getReward()
print("\nTest getReward()...")
threshold = rwd_func.metric_threshold
vals = [threshold - 1, threshold + 1, threshold]
for val in vals:
    reward = rwd_func.getReward(metric_value=val)
    print(f" val = {val}")
    print(f" reward = {reward}")

# %% Test calcReward()
print("\nTest calcReward()...")
reward = rwd_func.calcReward(obs=obs, info=None, actions=None)
print(f"reward = {reward}")

# %% Test with series of preprocessors
print("\nTest with series of preprocessors...")
# Observation is a 2d array. The preprocessor will first crop out the bottom half
# of array, then get the max of the resultant sub-array.
obs = {"a_matrix": rand(6, 3)}
# Test with 2 default preprocessors.
print("2 Default preprocessors:")
rwd_func = Threshold(
    obs_or_info="obs",
    metric="a_matrix",
    metric_value=1.0,
    penalty=2,
    inequality=">",
    preprocessors=[
        {
            "preprocessor": "crop_array",
            "config": {
                "indices_or_sections": 2,
                "section_to_keep": 0,
                "axis": 0,
            },
        },
        "max",
    ],
)
metrics = rwd_func.getMetrics(obs=obs, info=None)
processed_metrics = rwd_func.preprocess(metrics=metrics)
print(f"obs = \n{obs}")
print(f"processed metrics = {processed_metrics}")
reward = rwd_func.calcReward(obs=obs, info=None)

# Test with 2 default and 1 custom preprocessors
print("2 Default preprocessors and 1 custom:")


def customPreprocessor(x):
    """Add 1."""
    return x + 1


rwd_func = Threshold(
    obs_or_info="obs",
    metric="a_matrix",
    metric_value=1.0,
    penalty=2,
    inequality=">",
    preprocessors=[customPreprocessor, "sum_cols", "max"],
)
metrics = rwd_func.getMetrics(obs=obs, info=None)
processed_metrics = rwd_func.preprocess(metrics=metrics)
print(f"obs = \n{obs}")
print(f"processed metrics = {processed_metrics}")
reward = rwd_func.calcReward(obs=obs, info=None)

# %% Test with penalties and subsidies
print("\nTest with base penalties and subsidies...")
rwd_func = Threshold(
    obs_or_info="obs",
    metric="num_tasked",
    metric_value=1.0,
    penalty=2,
    reward=1,
    inequality=">",
    preprocessors=["max"],
    penalties={
        "multi_assignment": 1.5,
        "non_vis_assignment": 0.2,
    },
    subsidies={
        "inaction": 2.0,
        "active_action": 0.9,
    },
)
num_sensors = 2
num_targets = 3
obs = {"num_tasked": randint(0, 6, size=[num_targets])}
actions = randint(0, num_targets + 1, size=[num_sensors])
vis_map = randint(0, 2, size=[num_targets, num_sensors])
info = {
    "vis_map_truth": vis_map,
    "num_sensors": num_sensors,
    "num_targets": num_targets,
}
[pen, pen_report] = rwd_func.calcPenalties(info=info, actions=actions)
[sub, sub_report] = rwd_func.calcSubsidies(info=info, actions=actions)
print(f"actions = {actions}")
print(f"vis map = \n{vis_map}")
print(f"penalties = {pen}")
print(f"penalty report = {pen_report}")
print(f"subsidies = {sub}")
print(f"subsidy report = {sub_report}")

# %%
print("done")
