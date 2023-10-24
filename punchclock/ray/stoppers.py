"""Custom Ray stoppers."""
# %% Imports
# Third Party Imports
from ray import train, tune
from ray.tune import Stopper


# %% Classes
class StopOnTrend(Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id: str, result: dict) -> bool:
        if not self.should_stop and result["mean_accuracy"] >= 0.8:
            self.should_stop = True
        return self.should_stop

    def stop_all(self) -> bool:
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop
