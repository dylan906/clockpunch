"""Custom RNN example from Ray."""
# Custom RNN Example: https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
# %% Imports
# Standard Library Imports
import argparse
import os

# Third Party Imports
import numpy as np
import ray
from ray import air, tune
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls, register_env

torch, nn = try_import_torch()


# %% Custom RNN Class
class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=256,
    ):
        nn.Module.__init__(self)
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True
        )
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


if __name__ == "__main__":
    ray.init()

    ModelCatalog.register_custom_model("rnn", TorchRNNModel)
    register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment("RepeatAfterMeEnv", env_config={"repeat_delay": 2})
        .framework("torch")
        .rollouts(num_rollout_workers=0, num_envs_per_worker=20)
        .training(
            model={
                "custom_model": "rnn",
                "max_seq_len": 20,
                "custom_model_config": {
                    "cell_size": 32,
                },
            },
            gamma=0.9,
            # TODO (Kourosh): Enable when LSTMs are supported.
            _enable_learner_api=False,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(_enable_rl_module_api=False)
    )

    config.training(entropy_coeff=0.001, num_sgd_iter=5, vf_loss_coeff=1e-5)

    stop = {
        "training_iteration": 1,
        "timesteps_total": 100000,
        "episode_reward_mean": 90.0,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
    )
    results = tuner.fit()

    ray.shutdown()
