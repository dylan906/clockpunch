"""Retrain a pretrained model."""
# https://github.com/jlsvane/Ray-Tune-Tensorflow/blob/main/tune_reload_test.py
# %% Imports
# Standard Library Imports
from pathlib import Path

# Third Party Imports
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from ray import air, tune
from ray.air import Result
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# %% Load model
fpath = Path(__file__).parent
checkpoint_path = Path("issues/iss92/data/training_run_JLU")

# %% Script
# ray.init(num_cpus=1)

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
    # .rollouts(num_rollout_workers=4)
    .environment("CartPole-v1")
    .framework(framework="torch")
    .resources(num_gpus=0)
    .build()
)
algo.restore(best_result.checkpoint)

policy = algo.get_policy()
# instance of ray.rllib.models.torch.fcnet.FullyConnectedNetwork
fcn = policy.model
print(fcn)

trained_weights = algo.get_policy().get_weights()
trained_state_dict = algo.get_policy().model.state_dict()

new_algo = (
    PPOConfig()
    .training(lr=best_result.config["lr"])
    .environment("CartPole-v1")
    .framework(framework="torch")
    .build()
)

untrained_weights = new_algo.get_policy().get_weights()

# check whether weights are equal

arrays_equal = all(
    [
        np.array_equal(t, u)
        for t, u in zip(trained_weights.values(), untrained_weights.values())
    ]
)

print(f"Trained and untrained weights equal? {arrays_equal}")  # False

new_algo.get_policy().model.load_state_dict(trained_state_dict)

loaded_weights = new_algo.get_policy().get_weights()

arrays_equal = all(
    [
        np.array_equal(t, l)
        for t, l in zip(trained_weights.values(), loaded_weights.values())
    ]
)

print(f"Trained and loaded weights equal? {arrays_equal}")  # True
#  %% Done
ray.shutdown()
print("done")
