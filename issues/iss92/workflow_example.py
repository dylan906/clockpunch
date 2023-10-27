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
from ray import air, tune

# %% Get paths
fpath = Path(__file__).parent
storage_path = fpath.joinpath("data")
trained_state_dict_path = storage_path.joinpath("model_weights").with_suffix(
    ".pth"
)


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
exp_name = "retrain_run_" + rand_str

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
