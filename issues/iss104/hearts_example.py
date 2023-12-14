"""Action mask with GTrXL example from hearts repo.

https://github.com/HelmholtzAI-FZJ/hearts-gym/tree/main
"""
# %% Imports
# Standard Library Imports
import inspect
import os
from copy import deepcopy
from typing import Any, Type

# Third Party Imports
import gymnasium as gym
import ray
import ray.rllib.algorithms.ppo as ppo
from gymnasium.spaces import Space
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import Preprocessor, get_preprocessor
from ray.rllib.models.torch.attention_net import (
    AttentionWrapper as TorchAttentionWrapper,
)
from ray.rllib.models.torch.attention_net import GTrXLNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.tune import registry
from ray.tune.registry import RLLIB_MODEL, _global_registry

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.common.utilities import safeGetattr

th, nn = try_import_torch()


def get_registered_model(name: str) -> type:
    """Return the model class registered in `ray` under the given name.

    Args:
        name (str): Name to query.

    Returns:
        type: A model class.
    """
    print(f"Line {inspect.currentframe().f_lineno}: {name=}")
    model = _global_registry.get(RLLIB_MODEL, name)
    print(f"Line {inspect.currentframe().f_lineno}: {model=}")
    return model


def to_preprocessed_obs_space(obs_space: Space) -> Space:
    """Return the given observation space in RLlib-preprocessed form.

    Args:
        obs_space (Space): Observation space to preprocess.

    Returns:
        Space: Preprocessed observation space.
    """
    prep = get_preprocessor(obs_space)(obs_space)
    return prep.observation_space


def preprocessed_get_default_model(
    obs_space: Space,
    model_config: ModelConfigDict,
    framework: str,
) -> type:
    """Return the default model class as specified by RLlib for an
    environment with the given preprocessed observation space.

    Args:
        obs_space (Space): An already preprocessed observation space.
        model_config (ModelConfigDict): Configuration for the model.
        framework (str): Deep learning framework used.

    Returns:
        type: The default model class for the given preprocessed
            observation space.

    """
    model_cls = ModelCatalog._get_v2_model_class(
        obs_space, model_config, framework=framework
    )
    return model_cls


def _split_input_dict(
    input_dict: dict[str, Any],
    action_mask_key: str = "action_mask",
) -> tuple[dict[str, Any], TensorType]:
    """Return a modified input dictionary and its removed action mask.

    `input_dict` is modified in-place so that the "obs_flat" key
    contains the flattened observations without the action mask.

    Args:
        input_dict (dict[str, Any]): Input dictionary containing
            observations with support for action masking.

    Returns:
        dict[str, Any]: Input dictionary with the action mask removed
            from its flattened observations.
        TensorType: The action mask removed from the input dictionary.
    """
    action_mask = input_dict["obs"][action_mask_key]
    # TODO allow ACTION_MASK_KEY to be placed anywhere, not just at
    #      start (get start index)
    action_mask_len = action_mask.shape[-1]

    # The action mask is at the front as the DictFlatteningProcessor
    # sorts its dictionary's items.
    sans_action_mask = input_dict["obs_flat"][:, action_mask_len:]
    input_dict["obs_flat"] = sans_action_mask
    return input_dict, action_mask


def _process_model_class_surrogate(
    obs_space: Space,
    model_config: ModelConfigDict,
    framework: str,
    model_cls: Type[ModelV2] | str | None,
) -> Type[ModelV2]:
    """Return the model class based on the provided model_cls.

    If model_cls is None, the default model class is returned based on
    the preprocessed observation space, model configuration, and
    framework. If model_cls is a string, the registered model class
    with that name is returned.

    Args:
        obs_space (Space): An already preprocessed observation space.
        model_config (ModelConfigDict): Configuration for the model.
        framework (str): Deep learning framework used.
        model_cls (Type[ModelV2] | str | None]): Class of the model to construct.

    Returns:
        Type[ModelV2]: The model class.

    """
    print(f"Line {inspect.currentframe().f_lineno}: {model_cls=}")
    if model_cls is None:
        model_cls = preprocessed_get_default_model(obs_space, model_config, framework)
    elif isinstance(model_cls, str):
        model_cls = get_registered_model(model_cls)

    print(f"Line {inspect.currentframe().f_lineno}: {model_cls=}")
    return model_cls


def _create_unmasked_action_space(
    obs_space: gym.spaces.Dict,
    model_config: ModelConfigDict,
) -> Space:
    """Generate unmasked action space by removing action mask from obs space.

    Args:
        obs_space (gym.spaces.Dict): The original observation space (including the action
            mask) of the environment.
        model_config (ModelConfigDict): The model configuration dictionary.

    Returns:
        Space: The unmasked action space.
    """
    action_mask_key = model_config["custom_model_config"]["action_mask_key"]
    obs_space_nomask = deepcopy(obs_space.original_space)
    del obs_space_nomask.spaces[action_mask_key]
    original_obs_space = to_preprocessed_obs_space(obs_space_nomask)
    return original_obs_space


def _create_with_adjusted_obs(
    obs_space: Space,
    action_space: Space,
    num_outputs: int,
    model_config: ModelConfigDict,
    name: str,
    model_cls: Type[ModelV2] | str | None,
    framework: str,
) -> ModelV2:
    """Return a model constructed with an observation space adjusted to _not_ include an action mask.

    See also `ModelV2.__init__`.

    Args:
        obs_space (Space): Original observation space (including the
            action mask) of the environment.
        action_space (Space): Action space of the environment.
        num_outputs (int): Number of output units of the model.
        model_config (ModelConfigDict): Model configuration dictionary.
        name (str): Name (scope) to give the model.
        model_cls (Union[Type[ModelV2], str, None]): Class of the model
            to construct.
        framework (str): Deep learning framework used.

    Returns:
        ModelV2: Model instance with an adjusted observation space.
    """
    model_cls = _process_model_class_surrogate(
        obs_space, model_config, framework, model_cls
    )
    print(f"Line {inspect.currentframe().f_lineno}: {model_cls=}")

    original_obs_space = _create_unmasked_action_space(obs_space, model_config)

    return model_cls(
        original_obs_space,
        action_space,
        num_outputs,
        model_config,
        name + "_wrapped",
    )


def _create_wrapped(
    obs_space: Space,
    action_space: Space,
    num_outputs: int,
    model_config: ModelConfigDict,
    name: str,
    model_cls: Type[ModelV2] | str | None,
    wrapper_cls: type,
    framework: str,
) -> ModelV2:
    """Return a model constructed with an observation space adjusted to
    _not_ include an action mask. Also wrap the model in the given
    wrapper class.

    See also `ModelV2.__init__`.

    Args:
        obs_space (Space): Original observation space (including the
            action mask) of the environment.
        action_space (Space): Action space of the environment.
        num_outputs (int): Number of output units of the model.
        model_config (ModelConfigDict): Model configuration dictionary.
        name (str): Name (scope) to give the model.
        model_cls (Type[ModelV2] | str | None): Class of the model
            to wrap.
        wrapper_cls (Union[type, str, None]): Class to wrap the model
            in. This is also used to construct the model.
        framework (str): Deep learning framework used.

    Returns:
        ModelV2: Wrapped model instance with an adjusted
            observation space.
    """
    model_cls = _process_model_class_surrogate(
        obs_space, model_config, framework, model_cls
    )
    original_obs_space = _create_unmasked_action_space(obs_space, model_config)

    wrapper_cls = ModelCatalog._wrap_if_needed(model_cls, wrapper_cls)
    wrapper_cls._wrapped_forward = model_cls.forward  # type: ignore[attr-defined]
    return wrapper_cls(
        original_obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    )


class TorchMaskedActionsWrapper(
    TorchModelV2,
    nn.Module,  # type: ignore[name-defined]
):
    """Wrapper class to support action masking for arbitrary non-recurrent PyTorch models."""

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        model_cls: Type[ModelV2] | str | None = None,
        framework: str = "torch",
    ) -> None:
        """Construct an action masking wrapper model around a given model class.

        See also `TorchModelV2.__init__`.

        Args:
            obs_space (Space): Observation space of the environment.
            action_space (Space): Action space of the environment.
            num_outputs (int): Number of output units of the model.
            model_config (ModelConfigDict): Model configuration dictionary.
            name (str): Name (scope) to give the model.
            model_cls (Union[Type[ModelV2], str, None]): Class of the model
                to construct.
            framework (str): Deep learning framework used.
        """
        nn.Module.__init__(self)
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        # print(f"Line {inspect.currentframe().f_lineno}: {model_config=}")
        assert "action_mask_key" in model_config["custom_model_config"]
        self.action_mask_key = model_config["custom_model_config"]["action_mask_key"]
        self._wrapped = _create_with_adjusted_obs(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            model_cls,
            framework,
        )
        self.view_requirements: dict[str, ViewRequirement] = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """Return the model applied to the given inputs. Apply the action mask
        to give prohibited actions a probability close to zero.

        See also `TorchModelV2.forward`.

        Args:
            input_dict (dict[str, TensorType]): Input tensors, including
                keys "obs", "obs_flat", "prev_action", "prev_reward",
                "is_training", "eps_id", "agent_id", "infos", and "t".
            state (List[TensorType]): List of RNN state tensors.
            seq_lens (TensorType): 1-D tensor holding input
                sequence lengths.

        Returns:
            TensorType: Output of the model with action masking applied.
            List[TensorType]: New RNN state.
        """
        _, action_mask = _split_input_dict(
            input_dict, action_mask_key=self.action_mask_key
        )
        print(f"Line {inspect.currentframe().f_lineno}: {self=}")
        # print(f"Line {inspect.currentframe().f_lineno}: {dir(self._wrapped)=}")
        # print(f"Line {inspect.currentframe().f_lineno}: {input_dict=}")
        # print(
        #     f"Line {inspect.currentframe().f_lineno}: {safeGetattr(state,'shape', None)=}"
        # )
        # print(f"Line {inspect.currentframe().f_lineno}: {seq_lens=}")

        model_out, state = self._wrapped.forward(input_dict, state, seq_lens)

        # We don't use -infinity for numerical stability.
        inf_mask = th.maximum(
            th.log(action_mask), th.tensor(th.finfo(model_out.dtype).min)
        )
        return model_out + inf_mask, state

    def value_function(self):
        return self._wrapped.value_function()


class TorchMaskedActionsAttentionWrapper(TorchMaskedActionsWrapper):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        model_cls: Type[ModelV2] | str | None = None,
        attn_cls: type = TorchAttentionWrapper,
        framework: str = "torch",
    ) -> None:
        """Attention wrapper for TorchMaskedActionsWrapper.

        Args:
            obs_space (Space): _description_
            action_space (Space): _description_
            num_outputs (int): _description_
            model_config (ModelConfigDict): _description_
            name (str): _description_
            *
            model_cls (Type[ModelV2] | str | None, optional): _description_. Defaults to None.
            attn_cls (type, optional): _description_. Defaults to TorchAttentionWrapper.
            framework (str, optional): _description_. Defaults to "torch".
        """
        # TorchAttentionWrapper includes a wrapped GTrXLNet.
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            framework=framework,
        )

        print(f"Line {inspect.currentframe().f_lineno}: {attn_cls=}")
        model_cls = model_config["custom_model_config"].get("model_cls", None)
        print(f"Line {inspect.currentframe().f_lineno}: {model_cls=}")

        self._wrapped = _create_wrapped(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_attn",
            model_cls,
            attn_cls,
            framework,
        )
        self.view_requirements = {
            **self._wrapped.view_requirements,
            SampleBatch.OBS: self.view_requirements[SampleBatch.OBS],
        }

    def get_initial_state(self) -> list[TensorType]:
        return self._wrapped.get_initial_state()


if __name__ == "__main__":
    env = MaskRepeatAfterMe()
    print(f"{env.observation_space=}")
    print(f"{env.action_space=}")

    ray.init(local_mode=True)

    # register custom environments
    registry.register_env("MaskRepeatAfterMe", MaskRepeatAfterMe)
    ModelCatalog.register_custom_model(
        "TorchMaskedActionsWrapper", TorchMaskedActionsWrapper
    )
    ModelCatalog.register_custom_model(
        "TorchMaskedActionsAttentionWrapper", TorchMaskedActionsAttentionWrapper
    )

    # Make config
    config = (
        ppo.PPOConfig()
        .environment(
            "MaskRepeatAfterMe",
            env_config={"mask_config": "off"},
        )
        .training(
            model={
                "custom_model": "TorchMaskedActionsAttentionWrapper",
                "custom_model_config": {
                    "action_mask_key": "action_mask",
                    # "attn_cls": "TorchAttentionWrapper",
                    # "attn_cls": der,
                },
            },
            # fcnet_hiddens=[51, 51],
            # attention_dim=39,
            # attention_num_heads=1,
            # attention_head_dim=17,
            # attention_memory_inference=50,
            # attention_memory_training=50,
            # attention_position_wise_mlp_dim=18,
            # attn_cls="der",
        )
        .framework("torch")
        .rollouts(num_envs_per_worker=20)
    )

    # %% Build an train
    algo = config.build()
    algo.train()

    # stop = {"training_iteration": 20}
    # tuner = tune.Tuner(
    #     "PPO",
    #     param_space=config.to_dict(),
    #     run_config=air.RunConfig(stop=stop, verbose=3),
    # )
    # tuner.fit()
    ray.shutdown()

    print("done")
