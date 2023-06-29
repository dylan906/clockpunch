"""Tests for ukf_new.py."""
# %% Imports
# Third Party Imports
from matplotlib import pyplot as plt
from numpy import (
    arange,
    array,
    concatenate,
    diag,
    eye,
    invert,
    ndarray,
    ones,
    where,
)
from numpy.random import normal, rand

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.transforms import ecef2eci
from punchclock.dynamics.dynamics_classes import (
    DynamicsModel,
    SatDynamicsModel,
    StaticTerrestrial,
)
from punchclock.estimation.ukf_v2 import UnscentedKalmanFilter


# %% utility function
def getOffDiags(a):
    inv_mask = eye(a.shape[0], dtype=bool)
    mask = invert(inv_mask)
    indices = where(mask)
    return a[indices]


# %% Simple Dynamics and measurement models
def passThruMeasurement(state: ndarray):
    """Fully observable measurement function."""
    meas = state
    return meas


def simpleDynamicsFunc(
    state: ndarray,
    start_time: float,
    stop_time: float,
):
    """Simple 2-state dynamics function."""
    next_state = state + (stop_time - start_time) * ones(state.shape)
    return next_state


class SimpleDynamicsModel(DynamicsModel):
    """Simple dynamics model class."""

    def __init__(self):
        dynamics_func = simpleDynamicsFunc
        super().__init__(dynamics_func)

    def propagate(
        self,
        start_time: float,
        end_time: float,
        start_state: ndarray,
        **kwargs,
    ) -> ndarray:
        return self._dynamics_func(
            start_state,
            start_time,
            end_time,
            **kwargs,
        )


simple_dynamics = SimpleDynamicsModel()

# %% Other initialization parameters
RE = getConstants()["earth_radius"]
# initial/final time
time = 0
time_final = 10

# initial state
truth_state = array([0, 0])

# initial state estimate and covariance
est_x = truth_state.copy()
est_p = eye(2)

# process and measurement covariance
q = eye(2)
r = eye(2)

# %% Initialize UKF

simp_ukf = UnscentedKalmanFilter(
    time=time,
    est_x=est_x,
    est_p=est_p,
    dynamics_model=simple_dynamics,
    measurement_model=passThruMeasurement,
    q_matrix=q,
    r_matrix=r,
)

# %% Check Attributes
print("\nTest attributes...")
print(f"x_dim = {simp_ukf.x_dim}")
print(f"est_p = \n{simp_ukf.est_p}")
print(f"num_sigmas = {simp_ukf.num_sigmas}")
print(f"sigma_points = {simp_ukf.sigma_points}")
print(f"gamma = {simp_ukf.gamma}")
print(f"mean_weight = \n{simp_ukf.mean_weight}")
print(f"cvr_weight = \n{simp_ukf.cvr_weight}")
print(f"cvr_weight.shape = \n{simp_ukf.cvr_weight.shape}")


# %% Test algorithm flow-- incrementally go through steps of UKF
print("\nTest algorithm flow...")

# STEP 1: Predict state estimate and covariance, and update time.
# .predict() calls 2 methods: .predictStateEstimate() and .predictCovariance().
# .predictStateEstimate() calls .generateSigmaPoints() and propagates dynamics.
# .predictCovariance() calculates residuals and predicted covariance.
simp_ukf.predict(final_time=time_final)
print(f"propagated sigma_points = \n{simp_ukf.sigma_points}")
print(f"pred_x = {simp_ukf.pred_x}")
print(f"sigma point residuals = \n{simp_ukf.sigma_x_res}")
print(f"pred_p = \n{simp_ukf.pred_p}")
print(f"updated time = {simp_ukf.time}")

# STEP 1A: Propagate truth model and generate measurement
truth_state = simple_dynamics.propagate(
    start_time=time,
    end_time=time_final,
    start_state=truth_state,
)
measurement = (
    passThruMeasurement(state=truth_state) + rand(truth_state.shape[0]) - 0.5
)
print(f"propagated truth state = {truth_state}")
print(f"measurement of truth state = {measurement}")

# STEP 2: Update state estimate with observations
# Generates NEW sigma points based on predicted state and covariance, whereas in STEP 1
#   the sigma points were calculated based on the state estimate at time t-1.
# The sigma matrix for observations is calculated internally, not assigned as an attribute
#   or return. The residuals of the observation sigma matrix (difference between observation
#   sigma points and mean of those points) are assigned as an attribute, and printed here.
# The innovation covariance (aka measurement residual cov) is the covariance of the residuals
#   between the measurement and predicted measurement.
# .update() calls 3 methods (assuming a measurement is taken): .forecast(),
#   .calculateInnovations(), and .updateStateEstimate().
# .forecast() calls 4 methods: .generateSigmaPoints(), .calculateMeasurementMatrix(),
# .calculateKalmanGain(), and .updateCovariance().
simp_ukf.update(measurement=measurement)
print(f"predicted sigma points = \n{simp_ukf.sigma_points}")
print(f"mean_pred_y = {simp_ukf.mean_pred_y}")
print(f"sigma_y_res = \n{simp_ukf.sigma_y_res}")
print(f"Innovation covariance = \n{simp_ukf.innov_cvr}")
print(f"cross covariance =\n{simp_ukf.cross_cvr}")
print(f"kalman_gain =\n{simp_ukf.kalman_gain}")
print(f"updated state covariance (est_p) =\n{simp_ukf.est_p}")
print(
    f"innovation (residual between y and predicted y) = {simp_ukf.innovation}"
)
print(f"normalized innovation squared (NIS) = {simp_ukf.nis}")
print(f"est_x = {simp_ukf.est_x}")


# %% Test reset()
# Some variables reset to initial values, others reset to empty lists/arrays
print("\nTest reset()...")

print(f"time before reset = {simp_ukf.time}")
print(f"pred_x before reset = {simp_ukf.pred_x}")
simp_ukf.reset()
print(f"time after reset = {simp_ukf.time}")
print(f"pred_x after reset = {simp_ukf.pred_x}")

# %% Test measurements of different sizes
print("\nTest measurements of different sizes...")
# 1-D meas
print("  1D measurement")
meas1 = array([1, 2])
print(f"measurement shape ={meas1.shape}")
simp_ukf.predict(simp_ukf.time + 10)
simp_ukf.update(meas1)
print(f"est_x shape = {simp_ukf.est_x.shape}")
print(f"x_cov shape = {simp_ukf.est_p.shape}")

# (M,1) singleton measurement
print("  (M, 1) singleton measurement")
meas1 = array([[1, 2]]).transpose()
print(f"measurement shape ={meas1.shape}")
simp_ukf.predict(simp_ukf.time + 10)
try:
    simp_ukf.update(meas1)
except ValueError as e:
    print(e)

# (1,M) singleton measurement
print("  (1,M) singleton measurement")
meas1 = array([[1, 2]])
simp_ukf.predict(simp_ukf.time + 10)
print(f"measurement shape ={meas1.shape}")
try:
    simp_ukf.update(meas1)
except ValueError as e:
    print(e)

# %% Test simple filter in loop
print("\nTest Simple filter in loop...")
error_hist = []
p_hist = []
time = simp_ukf.time
for i in range(time, time + 100):
    truth_state = simple_dynamics.propagate(
        start_time=i,
        end_time=i + 1,
        start_state=truth_state,
    )
    measurement = (
        passThruMeasurement(state=truth_state)
        + rand(truth_state.shape[0])
        - 0.5
    )
    simp_ukf.predict(i + 1)
    simp_ukf.update(measurement=measurement)
    error_hist.append(truth_state - simp_ukf.est_x)
    p_hist.append(simp_ukf.est_p)
    # print(f"error = {truth_state - simp_ukf.est_x}")
    # print(f"tr(est_p) = {trace(simp_ukf.est_p)}")

plt.figure()
plt.suptitle("Simple UKF Filter Results")
grid = plt.GridSpec(2, 2)
ax1 = plt.subplot(grid[0, :])
ax2 = plt.subplot(grid[1, 0])
ax3 = plt.subplot(grid[1, 1])

ax1.set_title("Error")
ax2.set_title("Cov Diagonals")
ax3.set_title("Cov Off-Diagonals")

ax1.plot(error_hist)
ax2.plot([diag(a) for a in p_hist])
ax3.plot([[a[0, 1], a[1, 0]] for a in p_hist])
plt.tight_layout()

# %% Test filter with orbit dynamics
# build filter and set initial conditions
sat_dynamics = SatDynamicsModel()
truth_state = array([7000, 0, 0, 0, 7.5, 0])
q = 2 * array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1e-3, 0, 0],
        [0, 0, 0, 0, 1e-3, 0],
        [0, 0, 0, 0, 0, 1e-3],
    ]
)
p_init = 10 * q
r = array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1e-3, 0, 0],
        [0, 0, 0, 0, 1e-3, 0],
        [0, 0, 0, 0, 0, 1e-3],
    ]
)

sat_ukf = UnscentedKalmanFilter(
    time=0,
    est_x=truth_state,
    est_p=p_init,
    dynamics_model=sat_dynamics,
    measurement_model=passThruMeasurement,
    q_matrix=q,
    r_matrix=r,
)

# initialize history variables for plotting
truth_state_hist = [truth_state]
error_hist = [truth_state - sat_ukf.est_x]
p_hist = [sat_ukf.est_p]
time = arange(0, 10000, 100)
# loop through time
for i, t in enumerate(time[:-1]):
    # propagate truth
    truth_state = sat_dynamics.propagate(
        start_state=truth_state,
        start_time=t,
        end_time=time[i + 1],
    )
    truth_state_hist.append(truth_state)

    # generate noisy measurement from truth state
    noise_pos = normal(0, 1, 3)
    noise_vel = normal(0, 1e-3, 3)
    noise = concatenate((noise_pos, noise_vel))
    measurement = passThruMeasurement(state=truth_state) + noise

    # call UKF and record error and covariance
    sat_ukf.predict(time[i + 1])
    sat_ukf.update(measurement=measurement)
    error_hist.append(truth_state - sat_ukf.est_x)
    p_hist.append(sat_ukf.est_p)

# Plots
fig, axs = plt.subplots(2, 2)
fig.suptitle("Orbit Dynamics UKF Filter Results")
axs[0, 0].set_title("Error (Pos)")
axs[0, 1].set_title("Error (Vel)")
axs[1, 0].set_title("Cov Diagonals")
axs[1, 1].set_title("Cov Off-Diagonals")
axs[1, 0].set_xlabel("Steps")
axs[1, 1].set_xlabel("Steps")

axs[0, 0].plot([err[:3] for err in error_hist])
axs[0, 1].plot([err[3:] for err in error_hist])
axs[1, 0].plot([diag(a) for a in p_hist])
off_diags = [getOffDiags(a) for a in p_hist]
axs[1, 1].plot(off_diags)
axs[1, 0].set_yscale("log")
plt.tight_layout()

fig, axs = plt.subplots(2)
fig.suptitle("Orbit Dynamics UKF True States")
axs[0].plot([a[:3] for a in truth_state_hist])
axs[0].set_title("truth position")
axs[1].plot([a[3:] for a in truth_state_hist])
axs[1].set_title("truth velocity")
axs[1].set_xlabel("Steps")
plt.tight_layout()
# %% Test with Terrestrial Dynamics
print("\nTest terrestrial dynamics...")
ground_dynamics = StaticTerrestrial()
truth_state_ecef = array([RE, 0, 0, 0, 0, 0])
truth_state = ecef2eci(truth_state, 0)
ground_ukf = UnscentedKalmanFilter(
    time=0,
    est_x=truth_state,
    est_p=0.1 * eye(6),
    dynamics_model=ground_dynamics,
    measurement_model=passThruMeasurement,
    q_matrix=0.1 * eye(6),
    r_matrix=0.1 * eye(6),
)

# initialize history variables for plotting
truth_state_hist = [truth_state]
error_hist = [truth_state - ground_ukf.est_x]
p_hist = [ground_ukf.est_p]
time = arange(0, 10000, 100)
# loop through time
for i, t in enumerate(time[:-1]):
    # propagate truth
    truth_state = ground_dynamics.propagate(
        start_state=truth_state,
        start_time=t,
        end_time=time[i + 1],
    )
    truth_state_hist.append(truth_state)

    # generate noisy measurement from truth state
    noise_pos = normal(0, 1, 3)
    noise_vel = normal(0, 1e-3, 3)
    noise = concatenate((noise_pos, noise_vel))
    measurement = passThruMeasurement(state=truth_state) + noise

    # call UKF and record error and covariance
    ground_ukf.predictAndUpdate(time[i + 1], measurement)
    error_hist.append(truth_state - ground_ukf.est_x)
    p_hist.append(ground_ukf.est_p)

fig, axs = plt.subplots(2)
fig.suptitle("Terrestrial Dynamics True States")
axs[0].plot([a[:3] for a in truth_state_hist])
axs[0].set_title("truth position")
axs[1].plot([a[3:] for a in truth_state_hist])
axs[1].set_title("truth velocity")
axs[1].set_xlabel("Steps")
plt.tight_layout()

fig, axs = plt.subplots(2, 2)
fig.suptitle("Terrestrial Dynamics Filter Results")
axs[0, 0].set_title("Error (Pos)")
axs[0, 1].set_title("Error (Vel)")
axs[1, 0].set_title("Cov Diagonals")
axs[1, 1].set_title("Cov Off-Diagonals")
axs[1, 0].set_xlabel("Steps")
axs[1, 1].set_xlabel("Steps")

axs[0, 0].plot([err[:3] for err in error_hist])
axs[0, 1].plot([err[3:] for err in error_hist])
axs[1, 0].plot([diag(a) for a in p_hist])
off_diags = [getOffDiags(a) for a in p_hist]
axs[1, 1].plot(off_diags)
plt.tight_layout()


# %% Test with no-measurement updates
# initialize new filter
sat_dynamics = SatDynamicsModel()
truth_state = array([7000, 0, 0, 0, 7.5, 0])
q = 2 * array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1e-3, 0, 0],
        [0, 0, 0, 0, 1e-3, 0],
        [0, 0, 0, 0, 0, 1e-3],
    ]
)
p_init = 10 * q
r = array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1e-3, 0, 0],
        [0, 0, 0, 0, 1e-3, 0],
        [0, 0, 0, 0, 0, 1e-3],
    ]
)

sat_ukf2 = UnscentedKalmanFilter(
    time=0,
    est_x=truth_state,
    est_p=p_init,
    dynamics_model=sat_dynamics,
    measurement_model=passThruMeasurement,
    q_matrix=q,
    r_matrix=r,
)

# initialize history variables for plotting
truth_state_hist = [truth_state]
error_hist = [truth_state - sat_ukf2.est_x]
p_hist = [sat_ukf2.est_p]
meas_times = []
filt_times = []
time = arange(0, 10000, 100)
# loop through time
for i, t in enumerate(time[:-1]):
    # propagate truth
    truth_state = sat_dynamics.propagate(
        start_state=truth_state,
        start_time=t,
        end_time=time[i + 1],
    )
    truth_state_hist.append(truth_state)

    # get measurement every 5 iterations
    if i % 5 == 0:
        # generate noisy measurement from truth state
        noise_pos = normal(0, 1, 3)
        noise_vel = normal(0, 1e-3, 3)
        noise = concatenate((noise_pos, noise_vel))
        measurement = passThruMeasurement(state=truth_state) + noise
    else:
        # otherwise, no measurement
        measurement = None

    # call UKF and record error and covariance
    sat_ukf2.predict(time[i + 1])
    sat_ukf2.update(measurement=measurement)
    error_hist.append(truth_state - sat_ukf2.est_x)
    p_hist.append(sat_ukf2.est_p)
    meas_times.append(sat_ukf2.last_measurement_time)
    filt_times.append(sat_ukf2.time)

# Plots
fig, axs = plt.subplots()
fig.suptitle("Orbit Dynamics, Intermittent Measurements")
axs.plot(time[1:], meas_times, label="meas time")
axs.plot(time[1:], filt_times, label="filter time")
axs.set_title("Last Measurement Time")
axs.set_xlabel("Time")
axs.legend()

fig, axs = plt.subplots(2, 2)
fig.suptitle("Orbit Dynamics, Intermittent Measurements, Filter Results")
axs[0, 0].set_title("Error (Pos)")
axs[0, 1].set_title("Error (Vel)")
axs[1, 0].set_title("Cov Diagonals")
axs[1, 1].set_title("Cov Off-Diagonals")
axs[1, 0].set_xlabel("Steps")
axs[1, 1].set_xlabel("Steps")

axs[0, 0].plot([err[:3] for err in error_hist])
axs[0, 1].plot([err[3:] for err in error_hist])
axs[1, 0].plot([diag(a) for a in p_hist])
off_diags = [getOffDiags(a) for a in p_hist]
axs[1, 1].plot(off_diags)
axs[1, 0].set_yscale("log")
plt.tight_layout()

# %%
plt.show()
print("done")
