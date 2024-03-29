"""Test for ez_ukf.py."""
# %% Imports
# Third Party Imports
from numpy import array, eye

# Punch Clock Imports
from punchclock.estimation.ez_ukf import ezUKF, getRandomParams

# %% Test getRandomParams
print("\nTest getRandomParams...")
params = getRandomParams(dist="uniform", params=[[1, 2], [2, 3]])
print(f"params =\n{params}")

params = []
for _ in range(20):
    params.append(getRandomParams(dist="uniform", params=[[1, 2], [2, 3]]))
params = array(params)
print(f"params =\n{params}")

# %% Test ezUKF
print("\nTest ezUKF...")
print("  Test with array inputs...")

# Q, R, p_init are arrays
ez_filter_params = {
    "x_init": array([8000, 0, 0, 0, 8, 0]),
    "p_init": 0.1 * eye(6),
    "dynamics_type": "satellite",
    "Q": 0.1 * eye(6),
    "R": 0.1 * eye(6),
}

ukf_test = ezUKF(ez_filter_params)
print(f"ukf_test.est_x =\n{ukf_test.est_x}")

ukf_test.predictAndUpdate(
    final_time=5,
    measurement=array([8000, 0, 0, 0, 8, 0]),
)
print(f"ukf_test.est_x after update = {ukf_test.est_x}")

print("  Test with float inputs...")

# Q, R, p_init are floats
ez_filter_params = {
    "dynamics_type": "satellite",
    "x_init": array([8000, 0, 0, 0, 8, 0]),
    "Q": 0.1,
    "R": 0.5,
    "p_init": 0.3,
}
ukf_test = ezUKF(ez_filter_params)
print(f"ukf_test.Q =\n{ukf_test.q_matrix}")

# %% Test with PDF inputs
ez_filter_params = {
    "dynamics_type": "satellite",
    "x_init": array([8000, 0, 0, 0, 8, 0]),
    "Q": 0.1,
    "R": 0.5,
    "p_init": {
        "dist": "uniform",
        "params": [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ],
    },
}
ukf_test = ezUKF(ez_filter_params)
print(f"ukf_test.est_p =\n{ukf_test.est_p}")

# Test in loop with PDF inputs
ukfs = []
for _ in range(20):
    ukfs.append(ezUKF(ez_filter_params))

p_inits = [f.est_p for f in ukfs]
p_inits = array(p_inits)

# %% Done
print("done")
