"""Custom LSTM + Action Mask model."""

# %% Imports
# Standard Library Imports
from typing import Dict, List, Tuple
from warnings import warn

# Third Party Imports
import gymnasium as gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import all as tensorall

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
        """Initialize MaskedLSTM model.

        Args:
            obs_space (gym.spaces.Space): Environment observation space.
            action_space (gym.spaces.Space): Environment action space.
            num_outputs (int): Number of outputs of model. Should be equal to size
                of flattened action_space.
            model_config (dict, optional): Used for Ray defaults. Defaults to {}.
            name (str, optional): Used for inheritance. Defaults to "MaskedLSTM".
            custom_model_kwargs: Configure size of FC net and LSTM layer. Required.

        Expected items in custom_model_kwargs:
            fcnet_hiddens (list[int]): Number and size of FC layers.
            fcnet_activation (str): Activation function for FC layers. See Ray
                SlimFC documentation for recognized args.
            lstm_state_size (int): Size of LSTM layer.
        """
        # Convert space to proper gym space if handed is as a different type
        orig_space = getattr(obs_space, "original_space", obs_space)
        # Size of observations must include only "observations", not "action_mask".
        # Action mask must be 1d and same len as num_outputs.
        # custom_model_kwargs must include "lstm_state_size", "fcnet_hiddens",
        # and "fcnet_activation".
        assert "observations" in orig_space.spaces
        assert "action_mask" in orig_space.spaces
        assert len(orig_space.spaces) == 2
        assert len(orig_space["action_mask"].shape) == 1
        assert (
            orig_space["action_mask"].shape[0] == num_outputs
        ), f"""
        orig_space['action_mask'].shape[0] = {orig_space['action_mask'].shape[0]}\n
        num_outputs = {num_outputs}
        """
        assert "lstm_state_size" in custom_model_kwargs
        assert "fcnet_hiddens" in custom_model_kwargs
        assert "fcnet_activation" in custom_model_kwargs
        assert isinstance(custom_model_kwargs["fcnet_hiddens"], list)
        assert all([isinstance(i, int) for i in custom_model_kwargs["fcnet_hiddens"]])

        # print(f"obs_space = {obs_space}")
        # print(f"action_space = {action_space}")
        # print(f"num_outputs = {num_outputs}")
        # print(f"model_config = {model_config}")
        # print(f"name = {name}")
        # print(f"custom_model_kwargs = {custom_model_kwargs}")
        lstm_state_size = custom_model_kwargs.get("lstm_state_size")
        # print(f"lstm_state_size = {lstm_state_size}")

        # Defaults
        if model_config is None:
            model_config = {}
        if name is None:
            name = "MaskedLSTM"

        # Inheritance
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = orig_space["observations"].shape[0]
        # transition layer size: size of output of final hidden layer
        self.trans_layer_size = custom_model_kwargs["fcnet_hiddens"][-1]
        self.lstm_state_size = lstm_state_size

        self.fc_layers = self.makeFCLayers(
            model_config=custom_model_kwargs,
            input_size=self.obs_size,
        )

        # ---From Ray Example---
        # # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            input_size=self.trans_layer_size,
            hidden_size=self.lstm_state_size,
            batch_first=True,
        )
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # ---End Ray Example---
        print(f"MaskedLSTM built: \n{self}")

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
        # Resolve edge case where Policy.build() can pass in mask values <0 and
        # non-integers. Clamp values < 0  to 0, and values > 0 to 1.
        mask_binary = torch.clamp(mask, min=0, max=1)
        mask_binary[mask_binary > 0] = 1

        # check for binary action mask so error doesn't happen in action distribution
        # creation.
        assert all([i in [0, 1] for i in mask_binary.detach().numpy().flatten()])
        if tensorall(mask_binary == 0):
            warn("All actions masked")

        # Mask logits
        inf_mask = torch.clamp(torch.log(mask_binary), min=FLOAT_MIN)
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

    def makeFCLayers(self, model_config: dict, input_size: int) -> nn.Sequential:
        """Make fully-connected layers.

        See Ray SlimFC for details.

        Args:
            model_config (dict): {
                "fcnet_hiddens": (list[int]) Numer of hidden layers is number of
                    entries; size of hidden layers is values of entries,
                "fcnet_activation": (str) Recognized activation function
            }
            input_size (int): Input layer size.

        Returns:
            nn.Sequential: Has N layers, where N = len(model_config["fcnet_hiddens"]).
        """
        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")

        self.fc_hiddens = hiddens
        self.fc_activation = activation

        # print(f"self.fc_hiddens = {self.fc_hiddens}")
        # print(f"self.fc_activation = {self.fc_activation}")

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

        fc_layers = nn.Sequential(*layers)

        return fc_layers
