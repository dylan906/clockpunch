"""Retrain a pretrained model."""
# https://github.com/jlsvane/Ray-Tune-Tensorflow/blob/main/tune_reload_test.py
# %% Imports
# Standard Library Imports
from pathlib import Path

# Third Party Imports
import ray
import ray.rllib.algorithms.ppo as ppo
from ray import air, tune
from ray.air import Result
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# %% Load model
fpath = Path(__file__).parent
checkpoint_path = Path(
    # "issues/iss92/data/training_run_PRU/PPO_CartPole-v1_843ef_00000_0_2023-10-26_17-56-29/checkpoint_000001/policies/default_policy"
    # "issues/iss92/data/training_run_PRU/PPO_CartPole-v1_843ef_00000_0_2023-10-26_17-56-29"
    "issues/iss92/data/training_run_PRU"
)

# %% Script
ray.init()

# reload the previous tuner
restored_tuner = tune.Tuner.restore(path=str(checkpoint_path), trainable="PPO")
result_grid = restored_tuner.get_results()

# Check if there have been errors
if result_grid.errors:
    print("One of the trials failed!")
else:
    print("No errors!")

num_results = len(result_grid)
print("Number of results:", num_results)

results_df = result_grid.get_dataframe()
results_df[["training_iteration", "episode_reward_mean"]]

best_result_df = result_grid.get_dataframe(
    filter_metric="episode_reward_mean", filter_mode="max"
)
best_result_df[["training_iteration", "episode_reward_mean"]]

# Get the result with the maximum test set `episode_reward_mean`
best_result = result_grid.get_best_result(
    metric="episode_reward_mean", mode="max"
)

result_df = best_result.metrics_dataframe
result_df[["training_iteration", "episode_reward_mean", "time_total_s"]]

algo = (
    PPOConfig()
    .training(lr=best_result.config["lr"])
    .rollouts(num_rollout_workers=4)
    .environment("CartPole-v1")
    .framework(framework="torch")
    .resources(num_gpus=0)
    .build()
)
algo.restore(best_result.checkpoint)

policy = algo.get_policy()
fcn = (
    policy.model
)  # instance of ray.rllib.models.torch.fcnet.FullyConnectedNetwork
print(fcn)

print("done")
