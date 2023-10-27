"""Workflow example of loading custom weights prior to training."""
# %% Imports
# Standard Library Imports
import random
import string
from pathlib import Path

# Third Party Imports
import ray
import ray.rllib.algorithms.ppo as ppo
import torch
from gymnasium import Env
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

# Punch Clock Imports
from issues.iss88.mask_repeat_after_me import MaskRepeatAfterMe

# %% Get paths
fpath = Path(__file__).parent
storage_path = fpath.joinpath("data")
trained_state_dict_path = storage_path.joinpath("model_weights").with_suffix(
    ".pth"
)

# Change experiment_name to desired checkpoint dir name
experiment_name = "training_run_ITP"
exp_path = storage_path.joinpath(experiment_name)


# %% Custom trainable
class PPOalgo(ppo.PPO):  # noqa
    def __init__(self, config, **kwargs):  # noqa
        """Needs full path here!"""
        assert "weights_path" in config
        weights_path = config.get("weights_path")
        print(f"\nweight path = {weights_path} \n")
        # delete "weights_path" form config before super
        del config["weights_path"]

        super(PPOalgo, self).__init__(config, **kwargs)
        loaded_weights = torch.load(weights_path)
        self.get_policy().model.load_state_dict(loaded_weights)
        self.workers.sync_weights()  # Important!!!

    def reset_config(self, new_config):
        """To enable reuse of actors."""
        self.config = new_config
        return True


# %% Function to get trained state dict
def getTrainedStateDict(
    path,
    trainable=None,
    algo_config=None,
    env: Env | str = None,
) -> dict:
    if trainable is None:
        trainable = "PPO"
    if algo_config is None:
        algo_config = PPOConfig()
    if env is None:
        env = "CartPole-v1"

    ray.init()

    restored_tuner = tune.Tuner.restore(path=str(path), trainable="PPO")
    best_result = restored_tuner.get_results().get_best_result(
        metric="episode_reward_mean",
        mode="max",
    )

    algo = (
        algo_config.training()
        .environment(env)
        .framework(framework="torch")
        .resources(num_gpus=0)
        .build()
    )
    algo.restore(best_result.checkpoint)

    trained_state_dict = algo.get_policy().model.state_dict()

    ray.shutdown()

    return trained_state_dict


# %% Get trained state dict
trained_state_dict = getTrainedStateDict(
    path=exp_path,
    # trainable="PPO",
    # algo_config=PPOConfig(),
    # env=MaskRepeatAfterMe,
)
torch.save(trained_state_dict, trained_state_dict_path)

# %% Build Tuner

trained_state_dict = torch.load(trained_state_dict_path)

param_space = {
    "framework": "torch",
    "env": "CartPole-v1",
    "model": {
        # "custom_model": "MaskedLSTM"
        # "state_dict": trained_state_dict
    },
    "weights_path": trained_state_dict_path,
}

rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
exp_name = "retraining_run_" + rand_str

tuner = tune.Tuner(
    trainable=PPOalgo,
    param_space=param_space,
    run_config=air.RunConfig(
        stop={
            "training_iteration": 1,
        },
        name=exp_name,
        storage_path=storage_path,
    ),
)
tuner.fit()
# %% Done
ray.shutdown()
print("done")
