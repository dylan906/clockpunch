"""Threshold reward function."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from operator import ge, gt, le, lt
from typing import Any, Callable

# Third Party Imports
from numpy import ndarray

# Punch Clock Imports
from punchclock.reward_funcs.reward_base_class import RewardFunc
from punchclock.reward_funcs.reward_utils import (
    lookupPreprocessor as lookupDefaultPreprocessor,
)


# %% Threshold reward function
class Threshold(RewardFunc):
    """Gives reward/penalty for metric meeting/violating an inequality condition."""

    def __init__(
        self,
        obs_or_info: str,
        metric: str,
        metric_value: float,
        reward: float = 0,
        penalty: float = 0,
        inequality: str = "<=",
        preprocessors: Callable | list[str | Callable | dict] = None,
        penalties: dict = None,
        subsidies: dict = None,
    ):
        """Initialize a Threshold reward function.

        Args:
            obs_or_info (`str`): ["obs", "info"] Whether get the metric from an
                environment observation or info frame. Assumes observation and
                info frames are dicts.
            metric (`str`): The name of the metric from an observation or info
                frame that is desired to be kept above/below a threshold.
            metric_value (`float`): The value to maintain metric below/above (inclusive).
            reward (`float`, optional): The reward given if the metric value meets
                the desired condition. Defaults to 0.
            penalty (`float`, optional): The penalty given if the metric value does not
                meet the desired condition. Defaults to 0.
            inequality (`str`, optional): Condition that metric must satisfy for
                function to grant reward. Can be one of ["<=", ">=", "<", ">"].
                Defaults to "<=".
            preprocessors (`Callable | list[str | Callable | dict]`, optional):
                Function(s) to apply to metric before checking if inequality is
                satisfied.If list is longer than 1 entry, the preprocessors will
                be applied in the order they are listed. The first entry in the
                list must accept metric as an argument; the last entry must return
                a float. Str entries are used for simple recognized preprocessors.
                Dict entries configure more complex recognized preprocessors. A
                dict entry takes the form:
                    {
                        "preprocessor": preprocessor name (str)
                        "config": kwargs of the preprocessor (dict)
                    }
                See reward_utils for recognized preprocessors. Defaults to None.
            penalties (`dict`, optional): See rewardFunc. Defaults to None.
            subsidies (`dict`, optional): See rewardFunc. Defaults to None.

            Example inputs for preprocessors:
                preprocessors = sum

                from numpy import sum as np_sum
                preprocessors = np_sum

                preprocessors = [sum]

                preprocessors = ["sum_cols", "mean"]

                def add1(x): return x+1
                preprocessors = [add1, "max"]

        """
        super().__init__(penalties=penalties, subsidies=subsidies)

        assert isinstance(
            preprocessors, (list, Callable)
        ), "preprocessors must be a list or a Callable"

        if obs_or_info == "obs":
            self.use_obs = True
        elif obs_or_info == "info":
            self.use_obs = False

        self.metric = metric
        self.metric_threshold = metric_value
        self.penalty = penalty
        self.reward = reward
        self.inequality_func = self._lookupInequality(inequality)

        if isinstance(preprocessors, list):
            preprocessors = [
                self._lookupPreprocessor(entry) for entry in preprocessors
            ]
        self.preprocessors = preprocessors

    def calcReward(
        self,
        obs: dict,
        info: dict,
        actions: ndarray[int] = None,
    ) -> float:
        """Give reward if metric is above/below threshold.

        Args:
            obs (`dict`): If metric is in observation, must contain self.metric
                as a key. If metric is in info, arg is not required.
            info (`dict`): If metric is in info, must contain self.metric
                as a key. If metric is in obs, arg is not required.
            actions (`ndarray[int]`, optional): Not used. Defaults to None.

        Returns:
            `float`: Reward.
        """
        metrics = self.getMetrics(obs, info)
        processed_metrics = self.preprocess(metrics)
        reward = self.getReward(metric_value=processed_metrics)
        return reward

    def getMetrics(self, obs: dict, info: dict) -> Any:
        """Get the desired value from a dict."""
        # Switch to pull from observation (o) or info frame (i).
        if self.use_obs:
            oi = obs
        else:
            oi = info

        metrics = oi[self.metric]

        return metrics

    def preprocess(self, metrics: Any) -> float:
        """Execute function or series of functions on metrics."""
        if self.preprocessors is None:
            out = metrics
        elif isinstance(self.preprocessors, Callable):
            out = self.preprocessors(metrics)
        elif isinstance(self.preprocessors, list):
            out = deepcopy(metrics)
            for func in self.preprocessors:
                out = func(out)

        return out

    def getReward(self, metric_value: float) -> float:
        """Determine if metric_value satisfies inequality and assign reward."""
        within_bound = self.inequality_func(
            metric_value,
            self.metric_threshold,
        )

        if within_bound:
            reward = self.reward
        else:
            reward = -self.penalty

        return reward

    def _lookupInequality(self, inequality_str: str) -> Callable:
        """Get inequality from string representation."""
        if inequality_str == "<=":
            inequality_func = le
        elif inequality_str == ">=":
            inequality_func = ge
        elif inequality_str == "<":
            inequality_func = lt
        elif inequality_str == ">":
            inequality_func = gt

        return inequality_func

    def _lookupPreprocessor(self, func: Callable | str | dict) -> Callable:
        if isinstance(func, Callable):
            func = func
        else:
            func = lookupDefaultPreprocessor(func_name=func)

        return func
