"""Custom LSTM + Action Mask model."""
# %% Imports
# Standard Library Imports
from typing import Dict, List, Tuple

# Third Party Imports
import gymnasium as gym
import ray.rllib.algorithms.ppo as ppo
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import tensor

torch, nn = try_import_torch()


# %% Class
class MaskedLSTM(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: dict,
        name,
        fc_size: int = 64,
        lstm_state_size: int = 256,
    ):
        nn.Module.__init__(self)
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        # Convert space to proper gym space if handed is as a different type
        orig_space = getattr(obs_space, "original_space", obs_space)
        # Size of observations must include only "observations", not "action_mask"
        assert "observations" in orig_space.spaces
        assert "action_mask" in orig_space.spaces
        assert len(orig_space["action_mask"].shape) == 1
        assert orig_space["action_mask"].shape[0] == num_outputs

        self.obs_size = orig_space["observations"].shape[0]
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
        # When training, input_dict is handed in with an extra nested level from
        # the environment (input_dict["obs"]).
        # Get observations from obs; not observations+action_mask
        flat_inputs = input_dict["obs"]["observations"].float()
        action_mask = input_dict["obs"]["action_mask"]

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
env_config = {
    "observation_space": gym.spaces.Dict(
        {
            "observations": gym.spaces.Box(0, 1, shape=[4]),
            "action_mask": gym.spaces.Box(0, 1, shape=[2], dtype=int),
        }
    ),
    "action_space": gym.spaces.MultiDiscrete([2]),
}

env = RandomEnv(env_config)
print(f"observation_space = {env.observation_space}")
print(f"action_space = {env.action_space}")


# %% Preprocess function for obs
def preprocessObs(obs: dict) -> dict:
    """Convert components of observation to Tensors."""
    # Obs into .forward() are expected in a slightly different format than the
    # raw env's observation space at instantiation.
    for k, v in obs.items():
        obs[k] = tensor(v)
    obs = {"obs": obs}
    return obs


# %% Build model
model = MaskedLSTM(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    model_config={
        # "fcnet_hiddens": (5, 4),
        # "fcnet_activation": "relu",
    },
    fc_size=5,
    lstm_state_size=10,
    name=None,
)

# %% Test model (basic)
obs = env.observation_space.sample()
# override action mask to make sure we don't have all same values
obs["action_mask"][0] = 0
obs["action_mask"][1] = 1
print(f"obs (raw) = {obs}")
obs = preprocessObs(obs)
print(f"obs (preprocessed) = {obs}")

seq_lens = tensor(array([0]))
init_state = model.get_initial_state()
[logits, state] = model.forward(
    input_dict=obs, state=init_state, seq_lens=seq_lens
)
print(f"logits = {logits}")
print(f"state = {state}")

# %% Test training
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

config = (
    ppo.PPOConfig()
    .environment(RandomEnv, env_config=env_config)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedLSTM",
            # Extra kwargs to be passed to your model's c'tor.
            # "custom_model_config": {},
            # "fcnet_hiddens": [10],
            # "num_outputs": num_outputs,
        }
    )
)
algo = config.build()
algo.train()
algo.stop()

print("done")
