"""Test for math.py module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from matplotlib import pyplot as plt
from numpy import array, dot, eye, linspace
from numpy.random import rand

# Punch Clock Imports
from punchclock.common.math import (
    entropyDiff,
    kldGaussian,
    logistic,
    normalVec,
    saturate,
)

# %% Test logistic function
print("\nTest logistic function...")

k = 1
log_in_long = linspace(-10, 10, 100)
log_out_long = logistic(log_in_long, k=k)

# test point values of logistic function
log_in1 = 1
log_out1 = logistic(log_in1, k=k)
print(f"log_out = {log_out1}")
log_in2 = 8
log_out2 = logistic(log_in2, k=k)
print(f"log_out = {log_out2}")

plt.style.use("default")
fig, ax = plt.subplots()
ax.plot(log_in_long, log_out_long)
ax.plot(log_in1, log_out1, marker="D")
ax.plot(log_in2, log_out2, marker="*")
ax.legend(["continuous", "point 1", "point 2"])

# %% Test Saturate
print("\nTest saturate...")
# Test with different combinations of min/max threshold
vals = [1, 0, 1, 0]
mins = [-100, None, -100]
maxs = [0, 0, None]
print(f"vals = {vals}")
for mint, maxt in zip(mins, maxs):
    print(f"min_t = {mint}")
    print(f"max_t = {maxt}")

    saturated_vals = saturate(
        vals,
        setpoint=0,
        min_threshold=mint,
        max_threshold=maxt,
    )
    print(f"saturated_vals = {saturated_vals}")

# %% Test normalVec
print("\nTest normalVec...")
r1 = array([1, 2, 3])
r2 = normalVec(r1)
print(f"dot(r1, r2) = {dot(r1, r2)}")

# %% Test kldGaussian
print("\nTest kldGaussian...")
kld = kldGaussian(
    mu0=rand(2, 1),
    mu1=rand(2, 1),
    sigma0=rand(1) * eye(2),
    sigma1=rand(1) * eye(2),
)
print(f"{kld=}")

# %% Test entropyDiff
print("\nTest entropyDiff...")
e = entropyDiff(sigma_den=0.1 * eye(2), sigma_num=eye(2))
print(f"{e=}")
# %% Done
plt.show()
print("done")
