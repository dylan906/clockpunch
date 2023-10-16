"""Tests for lstm_mask.py."""
# %% Imports

# Third Party Imports
import gymnasium as gym
import ray
import ray.rllib.algorithms.ppo as ppo
from numpy import array, float32, zeros
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from torch import tensor

# Punch Clock Imports
from punchclock.nets.lstm_mask import MaskedLSTM

torch, nn = try_import_torch()

# %% Make test Env
print("\nBuild test env...")
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
env.reset()


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
print("\nBuild MaskedLSTM model...")

model = MaskedLSTM(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    # begin custom config kwargs
    fcnet_hiddens=[6, 6],
    fcnet_activation="relu",
    lstm_state_size=10,
)
print(f"model = {model}")
# %% Test model (basic)
print("\nTest model (basic)...")
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

# %% Test policy with model
print("\nTest policy with model...")
# See this Ray example: https://github.com/ray-project/ray/blob/c0ec20dc3a3f733fda85dcf9cc71f83d51132276/rllib/examples/custom_rnn_model.py#L101-L112
# Specifically, these lines:
# Example (use `config` from the above code):
# >> import numpy as np
# >> from ray.rllib.agents.ppo import PPOTrainer
# >>
# >> trainer = PPOTrainer(config)
# >> lstm_cell_size = config["model"]["custom_model_config"]["cell_size"]
# >> env = RepeatAfterMeEnv({})
# >> obs = env.reset()
# >>
# >> # range(2) b/c h- and c-states of the LSTM.
# >> init_state = state = [
# ..     np.zeros([lstm_cell_size], np.float32) for _ in range(2)
# .. ]
# >>
# >> while True:
# >>     a, state_out, _ = trainer.compute_single_action(obs, state)
# >>     obs, reward, done, _ = env.step(a)
# >>     if done:
# >>         obs = env.reset()
# >>         state = init_state
# >>     else:
# >>         state = state_out

ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

lstm_state_size = 10
config = (
    ppo.PPOConfig()
    .environment(RandomEnv, env_config=env_config)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedLSTM",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "fcnet_hiddens": [6, 6],
                "fcnet_activation": "relu",
                "lstm_state_size": lstm_state_size,
            },
        }
    )
)
algo = config.build()
policy = algo.get_policy()
print(f"policy = {policy}")

# 2 ways of getting initial state, both should give the same list of zeros.
state = [zeros([lstm_state_size], float32) for _ in range(2)]
# state = policy.get_initial_state()
print(f"state = {state}")

# obs = policy.observation_space.sample()
obs = env.observation_space.sample()
obs["action_mask"] = array([0, 1])
for i in range(3):
    print(f"i = {i}")
    action, state_out, _ = policy.compute_single_action(
        obs=policy.observation_space.sample(),
        state=state,
    )
    state = state_out
    print(f"state = {state}")
# %% Test training
print("\nTest training...")
algo.train()
algo.stop()

# %% Done
print("done")
