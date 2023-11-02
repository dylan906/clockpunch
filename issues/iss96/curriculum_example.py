"""Curriculum Training example."""
# https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
# Standard Library Imports
from pprint import pprint

# Third Party Imports
import numpy as np
import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
from ray.tune.registry import get_trainable_cls


def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    pprint(f"{train_results}")
    new_task = int(np.log10(train_results["episode_reward_mean"]) + 2.1)
    # Clamp between valid values, just in case:
    new_task = max(min(new_task, 5), 1)
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task


if __name__ == "__main__":
    # args = parser.parse_args()
    # ray.init(local_mode="store_true")
    ray.init(num_cpus=3)
    # os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(49)

    # Can also register the env creator function explicitly with:
    # register_env(
    #     "curriculum_env", lambda config: CurriculumCapableEnv(config))

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # or "curriculum_env" if registered above
        .environment(
            CurriculumCapableEnv,
            env_config={"start_level": 1},
            env_task_fn=curriculum_fn,
        )
        .framework("torch")
        .rollouts(num_rollout_workers=2, num_envs_per_worker=5)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=0)
    )

    stop = {
        # "training_iteration": 3,
        # "timesteps_total": 100,
        "episode_reward_mean": 100,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=2),
    )
    results = tuner.fit()

    ray.shutdown()
