"""Target custody tracker."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable

# Third Party Imports
from numpy import diagonal, ndarray, ones, sqrt, trace


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

        If using a custom custody function, recommend using partial() to fix as
        many arguments as possible. Keyword arguments can be used via
        CustodyTracker.update(..., `**kwargs`).
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
        # Custody function must output a list of bools, but can take any type input(s)

        config_func = config["func"]
        if isinstance(config_func, Callable):
            # If custom function provided
            custodyFunc = config_func
        else:
            # supported default functions
            custody_func_map = {
                "positional_std": CovarianceCustody(
                    "PosStd", threshold=config["threshold"]
                ).updateCustody,
                "tr_cov": CovarianceCustody(
                    "TrCov", threshold=config["threshold"]
                ).updateCustody,
                "tr_pos_cov": CovarianceCustody(
                    "TrPosCov", threshold=config["threshold"]
                ).updateCustody,
            }
            custodyFunc = custody_func_map[config_func]

        return custodyFunc

    def update(
        self,
        obs: Any,
        return_map: bool = False,
        **kwargs,
    ) -> list[bool] | dict:
        """Update target custody status (see CustodyTracker.custody_status).

        Args:
            obs (Any): Must comply with previously-defined interface of
                CustodyTracker.custodyFunc.
            return_map (bool, optional): If True, returns a dict with custody status
                mapped to target name. Otherwise, returns a list of custody statuses
                (True/False). Defaults to False.
            **kwargs: Used for custom custody functions.

        Returns:
            list[bool] | dict: Custody status as a list or a mapping of target
                names to status.
        """
        custody_status = self.custodyFunc(obs, **kwargs)

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

# Prototype Base class
# class BaseCovCustody(ABC):
#     def checkCovArgs(self, cov: Any):
#         """Generically check if covariance meets shape interface requirements."""
#         assert cov.ndim == 3, "cov must be a 3-dimensional array."
#         assert (
#             cov.shape[1] == 6
#         ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
#         quadrant."""
#         assert (
#             cov.shape[2] == 6
#         ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
#         quadrant."""

#     def checkReturns(self, y: Any):
#         assert isinstance(y, list), "Return value must be a list."
#         assert all(
#             isinstance(x, bool) for x in y
#         ), "All entries in return value must be bools."

#     def __call__(self, cov: ndarray, threshold: float):
#         self.checkCovArgs(cov)
#         retval = self._uniqueLogic(cov, threshold)
#         self.checkReturns(retval)
#         return retval

#     @abstractmethod
#     def _uniqueLogic(self, cov: ndarray, threshold: float):
#         raise NotImplementedError()


# class MaxPosStd(BaseCovCustody):
#     def _uniqueLogic(self, cov: ndarray, threshold: float):
#         custody = [True for i in range(cov.shape[0])]
#         for i in range(cov.shape[0]):
#             c = cov[i, :, :]
#             c_diags = diagonal(c)
#             std_diags = sqrt(c_diags)
#             pos_std_diags = std_diags[:3]
#             if max(pos_std_diags) > threshold:
#                 custody[i] = False

#         return custody


# calcMaxPosStd = MaxPosStd()


def checkCovArgs(cov: ndarray):
    """Generically check if covariance meets shape interface requirements."""
    assert cov.ndim == 3, "cov must be a 3-dimensional array."
    assert (
        cov.shape[1] == 6
    ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
    quadrant."""
    assert (
        cov.shape[2] == 6
    ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
    quadrant."""
    return


class CovarianceCustody:
    """A class of covariance matrix based custody rules."""

    def __init__(self, method: str, threshold: float):
        func_map = {
            "PosStd": self.checkPosStdCustody,
            "TrCov": self.checkTrCovCustody,
            "TrPosCov": self.checkTrPosCovCustody,
        }
        self.updateFunc = func_map[method]
        self.threshold = threshold
        return

    def updateCustody(self, cov: ndarray) -> list[bool]:
        assert cov.ndim == 3, "cov must be a 3-dimensional array."
        assert (
            cov.shape[1] == 6
        ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
        quadrant."""
        assert (
            cov.shape[2] == 6
        ), """cov must be a (N, 6, 6) array with positional elements in the upper-left
        quadrant."""
        custody = self.updateFunc(cov, self.threshold)
        return custody

    def checkPosStdCustody(self, cov: ndarray, threshold: float) -> list[bool]:
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
        custody = [True for i in range(cov.shape[0])]
        for i in range(cov.shape[0]):
            c = cov[i, :, :]
            c_diags = diagonal(c)
            std_diags = sqrt(c_diags)
            pos_std_diags = std_diags[:3]
            if max(pos_std_diags) > threshold:
                custody[i] = False

        return custody

    def checkTrCovCustody(self, cov: ndarray, threshold: float) -> list[bool]:
        custody = [True for i in range(cov.shape[0])]
        for i in range(cov.shape[0]):
            c = cov[i, :, :]
            if trace(c) > threshold:
                custody[i] = False
        return custody

    def checkTrPosCovCustody(
        self,
        cov: ndarray,
        threshold: float,
    ) -> list[bool]:
        custody = [True for i in range(cov.shape[0])]
        for i in range(cov.shape[0]):
            c = cov[i, :3, :3]
            if trace(c) > threshold:
                custody[i] = False
        return custody
