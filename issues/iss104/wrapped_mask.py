"""Test wrapping a basic action masking model with an attention net."""
# %% Imports
# Standard Library Imports
import inspect
import os
from typing import Dict, List, Optional, Tuple, Union

# Third Party Imports
import gymnasium as gym
import ray
import ray.rllib.algorithms.ppo as ppo
import torch.nn as nn
from gymnasium.spaces.utils import flatten
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelConfigDict, ModelV2
from ray.rllib.models.torch.attention_net import AttentionWrapper
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.tune import registry
from torch import TensorType

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.nets.utils import maskLogits


class CustomAttentionWrapper(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        orig_space = obs_space.original_space
        assert isinstance(orig_space, gym.spaces.Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        print(f"{type(obs_space)=}")
        print(f"{obs_space=}")
        print(f"{getattr(obs_space, 'original_space', obs_space)=}")
        wrapped_obs_space = orig_space.spaces["observations"]
        print(f"{wrapped_obs_space=}")

        nn.Module.__init__(self)
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=model_config,
            name=name,
        )

        self.internal_model = TorchFC(
            obs_space=wrapped_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name + "_internal",
        )

        self._value_out = None

    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: Optional[list[TensorType]],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, list[TensorType]]:
        print("\n")
        # print(f"{input_dict=}")
        # print(f"{state=}")
        # print(f"{input_dict['obs']=}")
        print(f"Line {inspect.currentframe().f_lineno}: {input_dict=}")
        print(f"Line {inspect.currentframe().f_lineno}: {state=}")
        print(f"Line {inspect.currentframe().f_lineno}: {input_dict['obs'].keys()=}")

        obs_only_dict = self.removeActionMask(input_dict)

        # Pass the observations and action mask to the internal model
        # and get the output and new state
        logits, new_state = self.internal_model(input_dict=obs_only_dict)
        # logits, new_state = self.internal_model.forward(
        #     input_dict=obs_only_dict, state=state, seq_lens=seq_lens
        # )

        # Mask the output
        action_mask = input_dict["obs"]["action_mask"]
        print(f"Line {inspect.currentframe().f_lineno}: {action_mask.shape=}")
        print(f"Line {inspect.currentframe().f_lineno}: {logits.shape=}")
        masked_logits = maskLogits(logits=logits, mask=action_mask)
        # self._value_out = self.internal_model._value_out

        # Return the masked output and the new state
        return masked_logits, new_state

    def removeActionMask(
        self, input_dict: dict[str, TensorType]
    ) -> dict[str, TensorType]:
        """Remove the action mask from the input dict."""
        modified_input_dict = input_dict.copy()
        modified_input_dict["obs"] = input_dict["obs"]["observations"]
        modified_input_dict["obs_flat"] = flatten(
            self.obs_space, modified_input_dict["obs"]
        )
        print(f"Line {inspect.currentframe().f_lineno}:" f" {modified_input_dict=}")
        print(
            f"Line {inspect.currentframe().f_lineno}: "
            f"{modified_input_dict['obs'].shape=}"
        )
        print(
            f"Line {inspect.currentframe().f_lineno}: {modified_input_dict['obs_flat'].shape=}"
        )
        print(
            f"Line {inspect.currentframe().f_lineno}: {getattr(modified_input_dict,'new_obs.shape', None)=}"
        )

        return modified_input_dict

    @override(ModelV2)
    def get_initial_state(self) -> list[TensorType]:
        state = self.internal_model.get_initial_state()
        for i in state:
            print(f"Line {inspect.currentframe().f_lineno}: {i.shape=}")
        return state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return self.internal_model.value_function()


if __name__ == "__main__":
    # %% Initialization stuff
    ray.init(local_mode=True)

    # register custom environments
    registry.register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    registry.register_env("MaskRepeatAfterMe", MaskRepeatAfterMe)
    ModelCatalog.register_custom_model("MyActionMaskModel", MyActionMaskModel)
    ModelCatalog.register_custom_model("CustomAttentionWrapper", CustomAttentionWrapper)

    # Make config
    config = (
        ppo.PPOConfig()
        .environment(
            "MaskRepeatAfterMe",
            env_config={"mask_config": "off"},
            # This env_config is only used for the RepeatAfterMeEnv env.
            # env_config={"repeat_delay": 2},
        )
        .training(
            gamma=0.99,
            entropy_coeff=0.001,
            num_sgd_iter=10,
            vf_loss_coeff=1e-5,
            model={
                "custom_model": "CustomAttentionWrapper",
                # "custom_model_config": {
                #     "no_masking": True,
                # },
                # "fcnet_hiddens": [32, 32],
                # Add any additional custom model parameters here
                "use_attention": True,
                # "max_seq_len": 10,
                # "attention_num_transformer_units": 1,
                # "attention_dim": 32,
                # "attention_memory_inference": 10,
                # "attention_memory_training": 10,
                # "attention_num_heads": 1,
                # "attention_head_dim": 32,
                # "attention_position_wise_mlp_dim": 32,
            },
        )
        .framework("torch")
        .rollouts(num_envs_per_worker=20)
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", 0))
        )
    )

    stop = {
        "training_iteration": 200,
        "timesteps_total": 500000,
        "episode_reward_mean": 80.0,
    }

    # %% Build an train
    algo = config.build()
    algo.train()
