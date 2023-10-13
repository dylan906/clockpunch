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
from ray.rllib.models.torch.misc import SlimFC
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
    """Fully-connected layers feed into an LSTM layer."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: dict = None,
        name: str = None,
        **custom_model_kwargs,
    ):
        """Expected items in custom_model_kwargs:
        {
            "fc_size": int,
            "lstm_state_size: int,
        }
        """
        # Convert space to proper gym space if handed is as a different type
        orig_space = getattr(obs_space, "original_space", obs_space)
        # Size of observations must include only "observations", not "action_mask".
        # Action mask must be 1d and same len as num_outputs.
        # custom_model_kwargs must include "fc_size" and "lstm_state_size"
        assert "observations" in orig_space.spaces
        assert "action_mask" in orig_space.spaces
        assert len(orig_space.spaces) == 2
        assert len(orig_space["action_mask"].shape) == 1
        assert orig_space["action_mask"].shape[0] == num_outputs
        assert "fc_size" in custom_model_kwargs
        assert "lstm_state_size" in custom_model_kwargs

        print(f"obs_space = {obs_space}")
        print(f"action_space = {action_space}")
        print(f"num_outputs = {num_outputs}")
        print(f"model_config = {model_config}")
        print(f"name = {name}")
        print(f"custom_model_kwargs = {custom_model_kwargs}")
        fc_size = custom_model_kwargs.get("fc_size")
        lstm_state_size = custom_model_kwargs.get("lstm_state_size")
        print(f"fc_size = {fc_size}")
        print(f"lstm_state_size = {lstm_state_size}")

        # Defaults
        if model_config is None:
            model_config = {}
        if name is None:
            name = "MaskedLSTM"

        # Inheritance
        nn.Module.__init__(self)
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.obs_size = orig_space["observations"].shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.fc_layers = self.makeFCLayers(
            model_config=custom_model_kwargs,
            input_size=self.obs_size,
            output_size=self.fc_size,
        )

        # ---From Ray Example---
        # # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            input_size=self.fc_size,
            hidden_size=self.lstm_state_size,
            batch_first=True,
        )
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # ---End Ray Example---

    @override(ModelV2)
    def get_initial_state(self):
        """Initial states of hidden layers are initial states of final FC layer."""
        h = [
            self.fc_layers[-1]
            ._model[0]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
            self.fc_layers[-1]
            ._model[0]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
        ]
        # ---From Ray Example---
        # h = [
        #     self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        #     self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        # ]
        # ---End Ray Example
        return h

    @override(ModelV2)
    def value_function(self):  # noqa
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
        # x = nn.functional.relu(self.fc1(inputs))
        x = nn.functional.relu(self.fc_layers(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def makeFCLayers(
        self, model_config: dict, input_size: int, output_size: int
    ) -> nn.Sequential:
        """Make fully-connected layers.

        See Ray SlimFC for details.

        Args:
            model_config (dict): {
                "fcnet_hiddens": list[int] Numer of hidden layers is number of
                    entries; size of hidden layers is values of entries,
                "post_fcnet_hiddens": list[int],
                "fcnet_activation": str
            }
            input_size (int): Input layer size.
            output_size (int): Output layer size.

        Returns:
            nn.Sequential: Has N+1 layers, where N = len(model_config["fcnet_hiddens"]).
        """
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")

        self.fc_hiddens = hiddens
        self.fc_activation = activation

        print(f"self.fc_hiddens = {self.fc_hiddens}")
        print(f"self.fc_activation = {self.fc_activation}")

        layers = []
        prev_layer_size = input_size

        # Create hidden layers.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # Append output layer
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=output_size,
                activation_fn=activation,
            )
        )

        fc_layers = nn.Sequential(*layers)

        return fc_layers


if __name__ == "__main__":
    # %% Make test Env
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
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
        },
        fc_size=5,
        lstm_state_size=10,
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
                "custom_model_config": {
                    # "num_outputs": 2,
                    "fcnet_hiddens": [6, 6],
                    "fcnet_activation": "relu",
                    "fc_size": 5,
                    "lstm_state_size": 10,
                },
            }
        )
    )
    algo = config.build()
    algo.train()
    algo.stop()

    # %% Done
    print("done")
