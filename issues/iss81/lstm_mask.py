"""Custom LSTM + Action Mask model."""
# %% Imports
# Standard Library Imports
import argparse
import os
from typing import Dict, List, Tuple, Union

# Third Party Imports
import gymnasium as gym
import numpy as np
import ray
from numpy import array
from ray import air, tune
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.tune.registry import get_trainable_cls, register_env
from torch import tensor

torch, nn = try_import_torch()


# %% Class
class DerModel(TorchRNN, nn.Module):
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

        # Size of observations must include only "observations", not "action_mask"
        assert "observations" in obs_space.spaces
        assert "action_mask" in obs_space.spaces
        assert len(obs_space["action_mask"].shape) == 1
        assert obs_space["action_mask"].shape[0] == num_outputs

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.obs_size = obs_space["observations"].shape[0]
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

    # Override forward() to add an action mask step
    @override(TorchRNN)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass.
        """
        # Get observations from obs; not observations+action_mask
        # flat_inputs = input_dict["obs_flat"].float()
        flat_inputs = input_dict["observations"].float()
        action_mask = input_dict["action_mask"]

        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        # Mask raw logits here! Then return masked values
        output = self.maskLogits(logits=output, mask=action_mask)
        return output, new_state

    def maskLogits(self, logits: TensorType, mask: TensorType):
        """Apply mask over raw logits."""
        inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return masked_logits

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


# %% Env
env = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "observations": gym.spaces.Box(0, 1, shape=[4]),
                "action_mask": gym.spaces.MultiBinary([2]),
            }
        ),
        "action_space": gym.spaces.MultiBinary([2]),
    }
)


# %% Preprocess function for obs
def preprocessObs(obs: dict) -> dict:
    """Conert components of observation to Tensors."""
    for k, v in obs.items():
        obs[k] = tensor(v)
    return obs


# %% Build model
model = DerModel(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    model_config={},
    name=None,
)

# %% Test model
obs = env.observation_space.sample()
# override action mask to make sure we don't have all same values
obs["action_mask"][0] = 0
obs["action_mask"][1] = 1
obs = preprocessObs(obs)
seq_lens = tensor(array([0]))
init_state = model.get_initial_state()
[logits, state] = model.forward(
    input_dict=obs, state=init_state, seq_lens=seq_lens
)


print("done")
