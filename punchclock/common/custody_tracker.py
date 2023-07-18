"""Target custody tracker."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from functools import partial
from typing import Any, Callable

# Third Party Imports
from numpy import diagonal, ndarray, ones, sqrt


# %% CustodyTracker class
class CustodyTracker:
    """Keeps track of target custody as a bool."""

    def __init__(
        self,
        num_targets: int,
        config: dict = None,
        target_names: list = None,
        initial_status: list[bool] = None,
    ):
        """Initialize CustodyTracker.

        Args:
            num_targets (int): Number of targets (>0).
            config (dict, optional): If using a supported function, structure as:
                {
                    "func": name (str),
                    "threshold": value (float),
                }.
                If using a custom function, structure as:
                {
                    "func": func (Callable),
                }.
                Defaults to:
                {
                    "func": "positional_std",
                    "threshold": 1000,
                }.
            target_names (list, optional): Target names for use in detailed return
                map. Defaults to sequential integers starting at 0.
            initial_status (list[bool], optional): Initial custody status for all
                targets. Defaults to all True.

        If input, target_names and initial_status must have length num_targets.
        """
        assert num_targets > 0, "num_targets must be >0."

        # Defaults
        if initial_status is None:
            initial_status = ones((num_targets), dtype=bool)
        if target_names is None:
            target_names = [i for i in range(num_targets)]
        if config is None:
            # Default config
            config = {
                "func": "positional_std",
                "threshold": 1000,
            }

        assert (
            len(target_names) == num_targets
        ), "len(target_names) must equal num_targets"
        assert (
            len(initial_status) == num_targets
        ), "len(initial_status) must equal num_targets"

        self.num_targets = num_targets
        self.target_names = target_names
        self.custody_status = initial_status
        self.custody_map = dict(zip(target_names, initial_status))
        self.config = config
        self.custodyFunc = self.getCustodyFunc(config)

    def getCustodyFunc(self, config: dict) -> Callable[..., list[bool]]:
        """Get the custody function to be used on self.update()."""
        custody_func_map = {
            "positional_std": CustodyPosStd,
        }

        config_func = config["func"]
        if isinstance(config_func, Callable):
            custodyFunc = config_func
        else:
            custodyFunc = partial(
                custody_func_map[config_func],
                threshold=config["threshold"],
            )

        return custodyFunc

    def update(self, obs: Any, return_map: bool = False) -> list[bool] | dict:
        """Update target custody status (see CustodyTracker.custody_status).

        Args:
            obs (Any): Must comply with previously-defined interface of
                CustodyTracker.custodyFunc.
            return_map (bool, optional): If True, returns a dict with custody status
                mapped to target name. Otherwise, returns a list of custody statuses
                (True/False). Defaults to False.

        Returns:
            list[bool] | dict: Custody status as a list or a mapping of target
                names to status.
        """
        custody_status = self.custodyFunc(obs)

        assert (
            len(custody_status) == self.num_targets
        ), "custodyFunc must return an N-long list."
        assert all(
            isinstance(x, bool) for x in custody_status
        ), "custodyFunc must return a list of bools."

        self.custody_status = custody_status
        self.custody_map = dict(zip(self.target_names, self.custody_status))

        if return_map is True:
            out = deepcopy(self.custody_map)
        else:
            out = deepcopy(self.custody_status)

        return out


# %% Supported custody functions
def CustodyPosStd(cov: ndarray, threshold: float) -> list[bool]:
    """Targets are in custody if the max positional principal STD < threshold.

    If the covariance for a single target is
        cov = array([
            [4, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [0, 0, 9, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 3],
        ])
    then the max positional principle STD is
        value = sqrt(max(diagonal(cov[:3, :3]))) = sqrt(9) = 3
    If value < threshold, then custody = True. Otherwise, custody = False.

    Args:
        cov (ndarray): (N, 6, 6) covariance matrix with positional elements in
            upper-left quadrant.
        threshold (float): Value to compare STD to with. If STD < threshold,
            custody = True.

    Returns:
        list[bool]: Custody status for all targets. Has length N.
    """
    assert cov.ndim == 3, "cov must be a 3-dimensional array."
    assert (
        cov.shape[1] == 6
    ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
    quadrant."""
    assert (
        cov.shape[2] == 6
    ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
    quadrant."""

    custody = [True for i in range(cov.shape[0])]
    for i in range(cov.shape[0]):
        c = cov[i, :, :]
        c_diags = diagonal(c)
        std_diags = sqrt(c_diags)
        pos_std_diags = std_diags[:3]
        if max(pos_std_diags) > threshold:
            custody[i] = False

    return custody
