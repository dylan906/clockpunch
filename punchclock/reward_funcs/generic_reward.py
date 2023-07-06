"""GenericReward reward function."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from typing import Any, Callable

# Third Party Imports
from numpy import ndarray

# Punch Clock Imports
from punchclock.reward_funcs.reward_base_class import RewardFunc
from punchclock.reward_funcs.reward_utils import lookupPreprocessor


# %% NormalizedMetric reward function
class GenericReward(RewardFunc):
    """Pass a metric though a list of preprocessors.

    Preprocess is a list of functions. Metric is a value from either an observation
    or info frame.
    """

    def __init__(
        self,
        obs_or_info: str,
        metric: str,
        preprocessors: list[str | Callable | dict] = None,
        penalties: dict = None,
        subsidies: dict = None,
        **kwargs,
    ):
        """Initialize a NormalizedMetric reward function.

        Args:
            obs_or_info (`str`): ["obs", "info"] Whether get the metric from an
                environment observation or info frame. Assumes observation and
                info frames are dicts.
            metric (`str`): The name of the metric from an observation or info
                frame to be passed through preprocessors.
            preprocessors (`list[str  |  Callable  |  dict`], optional): Function(s)
                to apply to metric before dividing by norm_denominator. If list
                is longer than 1 entry, the preprocessors will be applied in the
                order they are listed. The first entry in the list must accept metric
                as an argument; the last entry must return a float. Str entries
                are used for simple recognized preprocessors. Dict entries configure
                more complex recognized preprocessors. A dict entry takes the form:
                    {
                        "preprocessor": preprocessor name (str)
                        "config": kwargs of the preprocessor (dict)
                    }
                See reward_utils for recognized preprocessors. Defaults to None.
            penalties (`dict`, optional): See rewardFunc. Defaults to None.
            subsidies (`dict`, optional): See rewardFunc. Defaults to None.
            kwargs: For backward compatibility, accepts "norm_denominator" as a
                kwarg. Append a division by norm_denominator to preprocessors.
        """
        super().__init__(penalties=penalties, subsidies=subsidies)
        assert isinstance(preprocessors, list), "preprocessors must be a list"

        assert isinstance(preprocessors, list), "preprocessors must be a list"

        preprocessors = self.backwardCompatibleArg(preprocessors, kwargs)

        if obs_or_info == "obs":
            self.use_obs = True
        elif obs_or_info == "info":
            self.use_obs = False

        self.metric = metric

        preprocessor_funcs = [
            lookupPreprocessor(entry) for entry in preprocessors
        ]
        self.preprocessors = preprocessor_funcs

    def calcReward(
        self,
        obs: dict,
        info: dict,
        actions: ndarray[int] = None,
    ) -> float:
        """Calculate reward.

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
        reward = processed_metrics

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

    def backwardCompatibleArg(self, preprocessors: list, kwargs: dict):
        """Append a division by a constant to preprocessors if old style arg used."""
        if "norm_denominator" in kwargs:
            preprocessors.extend(
                [
                    {
                        "preprocessor": "divide",
                        "config": {"x2": kwargs["norm_denominator"]},
                    }
                ]
            )
        return preprocessors
