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
from torch import TensorType, reshape

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.common.utilities import safeGetattr
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
        print(f"Line {inspect.currentframe().f_lineno}: {obs_space=}")
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, gym.spaces.Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{getattr(obs_space, 'original_space', obs_space)=}",
        )
        self.wrapped_obs_space = orig_space.spaces["observations"]

        print(f"Line {inspect.currentframe().f_lineno}: {self.wrapped_obs_space=}")
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{getattr(model_config,'fcnet_hiddens', None)=}",
        )

        nn.Module.__init__(self)
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )

        self.internal_model = TorchFC(
            obs_space=self.wrapped_obs_space,
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
        print(f"Line {inspect.currentframe().f_lineno}: {input_dict=}")
        print(f"Line {inspect.currentframe().f_lineno}: {state=}")
        print(
            f"Line {inspect.currentframe().f_lineno}: {safeGetattr(input_dict['obs'],'keys()', None)=}"
        )
        print(
            f"Line {inspect.currentframe().f_lineno}: {safeGetattr(input_dict['obs'],'shape', None)=}"
        )
        print(
            f"Line {inspect.currentframe().f_lineno}: {safeGetattr(input_dict,'new_obs.shape', None),=}"
        )

        # Remove action mask from: 'obs', 'new_obs', and 'obs_flat'.
        obs_only_dict = self.removeActionMask(input_dict)

        # Pass the observations and action mask to the internal model
        # and get the output and new state
        logits, new_state = self.internal_model(input_dict=obs_only_dict)
        # logits, new_state = self.internal_model.forward(
        #     input_dict=obs_only_dict, state=state, seq_lens=seq_lens
        # )

        # Mask the output
        action_mask = input_dict["obs"]["action_mask"]
        # action_mask = self.reshapeActionMask(action_mask, reference_tensor=logits)
        print(f"Line {inspect.currentframe().f_lineno}: {action_mask.shape=}")
        print(f"Line {inspect.currentframe().f_lineno}: {logits.shape=}")
        masked_logits = maskLogits(logits=logits, mask=action_mask)

        # Return the masked output and the new state
        return masked_logits, new_state

    def removeActionMask(
        self, input_dict: dict[str, TensorType]
    ) -> dict[str, TensorType]:
        """Remove the action mask from the input dict."""
        # Watch out for input_dict being a SampleBatch
        print(f"Line {inspect.currentframe().f_lineno}: {input_dict=}")
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{getattr(input_dict['obs'], 'shape', None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(input_dict['obs'],'observations.shape', None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(input_dict['obs'],'action_mask.shape', None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(input_dict['obs_flat'],'shape', None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(input_dict, 'new_obs.shape',None)=}",
        )

        modified_input_dict = input_dict.copy()
        modified_input_dict["obs"] = input_dict["obs"]["observations"]
        modified_input_dict["obs_flat"] = flatten(
            self.wrapped_obs_space, modified_input_dict["obs"]
        )
        if "new_obs" in modified_input_dict:
            # 'new_obs' is only present in the input dict when using attention wrapper
            modified_input_dict["new_obs"] = modified_input_dict["new_obs"][
                :, : modified_input_dict["obs"].shape[1]
            ]

        print(f"Line {inspect.currentframe().f_lineno}: {modified_input_dict=}")
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(modified_input_dict,'obs.shape', None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(modified_input_dict,'obs_flat.shape',None)=}",
        )
        print(
            f"Line {inspect.currentframe().f_lineno}:",
            f"{safeGetattr(modified_input_dict,'new_obs.shape', None)=}",
        )

        return modified_input_dict

    def reshapeActionMask(
        self, mask: TensorType, reference_tensor: TensorType
    ) -> TensorType:
        """Tile copy the action mask to match the output of the model."""
        mask_shape = mask.shape
        reference_shape = reference_tensor.shape
        mask = mask.repeat(1, reference_shape[1] // mask_shape[1])
        return mask

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
    env = MaskRepeatAfterMe()
    print(f"{env.observation_space=}")
    print(f"{env.action_space=}")
    # env.observation_space=Dict('action_mask': Box(0, 1, (2,), int64), 'observations': Box(0, 1, (2,), int64))
    # env.action_space=Discrete(2)
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
                "fcnet_hiddens": [32, 2],  # last layer must be size of action space
                "use_attention": True,
            },
        )
        .framework("torch")
        .rollouts(num_envs_per_worker=20)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", 0)))
    )

    stop = {
        "training_iteration": 200,
        "timesteps_total": 500000,
        "episode_reward_mean": 80.0,
    }

    # %% Build an train
    algo = config.build()
    algo.train()
