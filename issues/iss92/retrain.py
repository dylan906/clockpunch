"""Retrain a pretrained model."""
# https://github.com/jlsvane/Ray-Tune-Tensorflow/blob/main/tune_reload_test.py
# %% Imports
# Standard Library Imports
from pathlib import Path

# Third Party Imports
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
import torch
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

# %% Load model
fpath = Path(__file__).parent
checkpoint_path = Path("issues/iss92/data/training_run_JLU")

# %% Script
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

# Restore policy (algo) from loaded config
algo = (
    PPOConfig()
    .training(lr=best_result.config["lr"])
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

# Get trained weights (state_dict is a PyTorch specified class)
trained_weights = algo.get_policy().get_weights()
trained_state_dict = algo.get_policy().model.state_dict()

# Save weights for later
model_weights_path = (
    fpath.joinpath("data").joinpath("model_weights").with_suffix(".pth")
)
torch.save(trained_state_dict, model_weights_path)
test_weights_loaded = torch.load(model_weights_path)

# Create new untraiend algo. Must be same structure as loaded algo.
new_algo = (
    PPOConfig()
    .training(lr=best_result.config["lr"])
    .environment("CartPole-v1")
    .framework(framework="torch")
    .build()
)

# check whether weights are equal
untrained_weights = new_algo.get_policy().get_weights()
arrays_equal = all(
    [
        np.array_equal(t, u)
        for t, u in zip(trained_weights.values(), untrained_weights.values())
    ]
)
print(f"Trained and untrained weights equal? {arrays_equal}")  # False

# Load trained weights into new model
new_algo.get_policy().model.load_state_dict(trained_state_dict)

# check that loaded weights are equal to trained weights
loaded_weights = new_algo.get_policy().get_weights()
arrays_equal = all(
    [
        np.array_equal(t, l)
        for t, l in zip(trained_weights.values(), loaded_weights.values())
    ]
)
print(f"Trained and loaded weights equal? {arrays_equal}")  # True

# Delete old algo
del algo


# New algo class that just loads the trained weights
class PPOalgo(ppo.PPO):  # noqa
    def __init__(self, config, **kwargs):  # noqa
        super(PPOalgo, self).__init__(config, **kwargs)
        """Needs full path here!"""
        weights_path = (
            Path(__file__)
            .parent.joinpath("data")
            .joinpath("model_weights")
            .with_suffix(".pth")
        )
        print(f"\nweight path = {weights_path} \n")
        loaded_weights = torch.load(weights_path)
        self.get_policy().model.load_state_dict(loaded_weights)
        self.workers.sync_weights()  # Important!!!

    def reset_config(self, new_config):
        """To enable reuse of actors."""
        self.config = new_config
        return True


# Create vanilla algo config
config = (
    PPOConfig()
    .training(lr=0.0)
    .framework(
        framework="torch",
    )
    .environment(env="CartPole-v1")
)

# Test class instantiation
PPOalgo(config=config)

# Make Tuner with custom algo class, but vanilla algo config
ray.shutdown()  # important!
storage_path = fpath.joinpath("data")

tuner = tune.Tuner(
    trainable=PPOalgo,
    param_space=config,
    run_config=air.RunConfig(
        name="PPOalgo",
        stop={"training_iteration": 1},
        storage_path=storage_path,
    ),
)
tuner.fit()

#  %% Done
ray.shutdown()
print("done")
