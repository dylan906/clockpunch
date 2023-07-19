"""Tests for custody_tracker.py."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from numpy import ndarray
from numpy.random import rand

# Punch Clock Imports
from punchclock.common.custody_tracker import (  # CovarianceCustody,; TrPosCov,
    CustodyTracker,
    MaxPosStd,
    TrCov,
)

# %% Test custody functions
print("\nTest checkPosStdCustody...")
funcMaxPosStd = MaxPosStd()
custody = funcMaxPosStd(rand(2, 6, 6), 0.5)
print(f"custody = {custody}")

test_cov = rand(2, 6, 6)
funcTrPosCov = TrCov(pos_vel="pos")
custody = funcTrPosCov(test_cov, 0.5)
print(f"custody = {custody}")

funcTrPosCov = TrCov(pos_vel="vel")
custody = funcTrPosCov(test_cov, 0.5)
print(f"custody = {custody}")

# %% Test Class
print("\nTest CustodyTracker...")
# test with defaults
ct = CustodyTracker(3)
custody = ct.update(obs=rand(3, 6, 6))
print(f"custody = {custody}")

# Test with custom config, supported func
ct = CustodyTracker(
    num_targets=3,
    config={
        "func": "max_pos_std",
        "threshold": 0.5,
    },
)
custody = ct.update(obs=rand(3, 6, 6))
print(f"custody = {custody}")

custody = ct.update(obs=rand(3, 6, 6), return_map=True)
print(f"custody = {custody}")


# Test with custom func
def customCustodyFunc(x: ndarray, b: bool) -> list[bool]:
    """Test function."""
    return [b for i in range(x.shape[0])]


ct = CustodyTracker(num_targets=3, config={"func": customCustodyFunc})
custody = ct.update(obs=rand(3, 6, 6), b=False)
print(f"custody = {custody}")


# %% done
print("done")
