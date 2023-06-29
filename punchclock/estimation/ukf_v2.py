"""Defines the :class:`.UnscentedKalmanFilter` class.

Based on implementation by Dylan Thomas (dylan.thomas@vt.edu).

References:
        Eric A. Wan, Rudolph can der Merwe, "The Unscented Kalman Filter for Nonlinear Estimation",
            https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf, 2000, IEEE Adaptive
            Systems for Signal Processing, Communications, and Control Symposium
        John Crassidis, John Junkins, "Optimal Estimation of Dynamic Systems, 2d Edition", 2012
"""
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy
from typing import Callable

# Third Party Imports
from numpy import array, concatenate, diagflat, full, ndarray, ones, sqrt, zeros
from scipy.linalg import LinAlgError, cholesky, inv

# Punch Clock Imports
from punchclock.dynamics.dynamics_classes import DynamicsModel
from punchclock.estimation.filter_base_class import Filter
from punchclock.estimation.ukf_utils import findNearestPositiveDefiniteMatrix


class UnscentedKalmanFilter(Filter):
    R"""Describes necessary equations for state estimation using the Unscented Transform.

    Attributes:
        time: Current filter time (sec)
        last_measurement_time: Last time filter received a measurement
        dynamics: Dynamics model used to propagate motion
        est_x: State estimate
        est_p: State estimate covariance
        q_matrix: Process noise
        r_matrix: Measurement noise
        x_dim: Number of dimensions in state
        y_dim: Number of dimensions in measurement
        kalman_gain: Kalman gain
        cross_cvr: Covariance between state residuals and measurement residuals
        innovation: Residual of predicted measurement and measurement
        innov_cvr: Covariance of innovation residuals
        nis: Normalized innovation squared
        pred_x: Predicted state
        pred_p: Predicted state covariance
        mean_pred_y: Mean predicted measurement
        num_sigmas: Number of sigma points used in filter prediction and update
        sigma_points: Sigma points (used for both state and measurement prediction)
        sigma_x_res: State sigma point residuals
        sigma_y_res: Measurement sigma point residuals
        mean_weight: Mean weight applied to sigma points
        cvr_weight: Covariance weight applied to sigma points
        gamma: Factor used to calculate sigma point weights (gamma = sqrt(x_dim + lambda_kf))
        reset_params: Parameters used to reset filter to original state

    """

    def __init__(
        self,
        time: float,
        est_x: ndarray,
        est_p: ndarray,
        dynamics_model: DynamicsModel,
        measurement_model: Callable,
        q_matrix: ndarray,
        r_matrix: ndarray,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float | None = None,
    ):
        R"""Initialize a UKF instance.

        Args:
            time (``float``): value for the initial time (sec)
            est_x (``ndarray``): :math:`N\times 1` initial state estimate
            est_p (``ndarray``): :math:`N\times N` initial covariance
            dynamics_model (`Dynamics`): dynamics object associated with the filter's target
            measurement_model (`Measurements`): measurement object associated with the filter's
                sensor.
            q_matrix (``ndarray``): dynamics error covariance matrix
            r_matrix (``ndarray``): measurement error covariance matrix
            alpha (``float``, optional): sigma point spread. Defaults to 0.001. This should be a
                small positive value: :math:`\alpha <= 1`.
            beta (``float``, optional): Gaussian pdf parameter. Defaults to 2.0. This parameter
                defines prior knowledge of the distribution, and the default value of 2 is optimal
                for Gaussian distributions.
            kappa (``float``, optional): scaling parameter. Defaults to :math:`3 - N`. This
                parameter defines knowledge of the higher order moments. The equation used by
                default minimizes the mean squared error to the fourth degree. However, when this
                value is negative, the predicted error covariance can become positive semi-
                definite.
        """
        super().__init__(time, est_x, est_p)
        # Store reset parameters and reset filter to assign initial values
        self.reset_params = {
            "time": time,
            "est_x": est_x,
            "est_p": est_p,
        }
        self.reset()

        self.dynamics = dynamics_model
        self.measurement_model = measurement_model

        # Initialize key variables used in filter process
        self.x_dim = len(est_x)
        self.q_matrix = q_matrix
        self.r_matrix = r_matrix
        dummy_measurement = self.measurement_model(est_x)
        self.y_dim = len(dummy_measurement)

        # Calculate scaling parameters lambda & gamma.
        if not kappa:
            kappa = 3 - self.x_dim
        lambda_kf = (alpha**2) * (self.x_dim + kappa) - self.x_dim

        self.gamma = sqrt(self.x_dim + lambda_kf)

        # Calculate weight values
        first_weight = lambda_kf / (self.x_dim + lambda_kf)
        weight = 1 / (2.0 * (lambda_kf + self.x_dim))

        # Weights for the mean
        self.mean_weight = full(self.num_sigmas, weight)
        self.mean_weight[0] = first_weight
        # Weights for the covariance
        self.cvr_weight = diagflat(self.mean_weight)
        self.cvr_weight[0, 0] += 1 - alpha**2.0 + beta

    def reset(self):
        """Reset UKF initial values to initial inputs or empty, where applicable."""
        self.time = deepcopy(self.reset_params["time"])
        self.last_measurement_time = None

        # Main estimation products, used as outputs of the filter class
        self.est_x = deepcopy(self.reset_params["est_x"])
        self.est_p = deepcopy(self.reset_params["est_p"])
        self.pred_x = array([])
        self.pred_p = array([])

        # Intermediate values, used for checking statistical consistency & simplifying equations
        self.nis = array([])
        self.innov_cvr = array([])
        self.cross_cvr = array([])
        self.kalman_gain = array([])
        self.mean_pred_y = array([])
        self.innovation = array([])

        self.sigma_points = array([])
        self.sigma_x_res = array([])
        self.sigma_y_res = array([])

    @property
    def num_sigmas(self):
        R"""``int``: Returns the number of sigma points to use in this UKF."""
        return 2 * self.x_dim + 1

    def generateSigmaPoints(self, mean: ndarray, cov: ndarray) -> ndarray:
        R"""Generate sigma points according to the Unscented Transform.

        Args:
            mean (``ndarray``): :math:`N\times 1` estimate mean to sample around.
            cov (``ndarray``): :math:`N\times N` covariance used to sample sigma points.

        Returns:
            ``ndarray``: :math:`N\times S` sampled sigma points around the given mean and
                covariance.
        """
        # Find the square root of the error covariance
        try:
            sqrt_cov = cholesky(cov)
        except LinAlgError:
            msg = f"`nearestPD()` function was used on RSO"
            print(msg)
            print(f"cov ={cov}")
            sqrt_cov = findNearestPositiveDefiniteMatrix(cov)

        # Calculate the sigma points based on the current state estimate
        return mean.reshape((self.x_dim, 1)).dot(
            ones((1, self.num_sigmas))
        ) + self.gamma * concatenate(
            (zeros((self.x_dim, 1)), sqrt_cov, -sqrt_cov), axis=1
        )

    def predictAndUpdate(self, final_time: float, measurement: ndarray | None):
        """Shortcut method that runs predict and update steps of UKF.

        Args:
            final_time (``float``): time to propagate to
            measurement (``ndarray`` | ``None``): (M,) array of measurements or
                `None` to signify no measurement but update filter based
                on prediction.
        """

        self.predict(final_time)
        self.update(measurement)

    def predict(self, final_time: float):
        R"""Propagate the state estimate and error covariance with uncertainty.

        Args:
            final_time (``float``): time to propagate to
        """
        # STEP 1: Calculate the predicted state estimate at t(k) (X(k + 1|k))
        self.predictStateEstimate(final_time)
        # STEP 2: Calculate the predicted covariance at t(k) (P(k + 1|k))
        self.predictCovariance(final_time)
        # STEP 3: Update the time step
        self.time = final_time

    def forecast(self, measurement: ndarray | None):
        R"""Update the error covariance with observations.

        Args:
           measurement (`ndarray` or `None`): (M,) array of measurements or
                `None` to signify no measurement, but to update filter based
                on prediction.
        """
        # STEP 1: Re-sample the sigma points around predicted (sampled) state estimate
        self.sigma_points = self.generateSigmaPoints(self.pred_x, self.pred_p)
        # STEP 2: Calculate the Measurement Matrix (H)
        self.calculateMeasurementMatrix(measurement)
        # STEP 3: Compile the Observation Noise Covariance (R)
        # STEP 4: Calculate the Cross Covariance (C), the Innovations Covariance (S), & the Kalman
        #   Gain (K)
        self.calculateKalmanGain()
        # STEP 5: Update the Covariance for the state (P(k + 1|k + 1))
        self.updateCovariance()

    def update(self, measurement: ndarray | None):
        R"""Update the state estimate with observations.

        Args:
            measurement (`ndarray` or `None`): (M,) array of measurements or
                `None` to signify no measurement, but to update filter based
                on prediction.
        """
        if type(measurement) == type(None):
            # Save the 0th sigma point because it is the non-sampled propagated state. This means
            #   that we don't inject any noise into the state estimate when no measurements occur.
            self.est_x = self.sigma_points[:, 0].copy()
            # If there are no observations, there is no update information and the predicted error
            #   covariance is stored as the updated error covariance
            self.est_p = self.pred_p.copy()
        else:
            if measurement.ndim > 1:
                raise ValueError(
                    "Measurement must be a (M,) array, but has ndim > 1"
                )

            # If there are observations, the predicted state estimate and covariance are updated
            self.forecast(measurement)
            # STEP 1: Compile the true measurement state vector (Yt)
            # STEP 2: Calculate the Innovations vector (nu)
            self.calculateInnovations(measurement)
            # STEP 3: Update the State EstimateAgent (X(k + 1|k + 1))
            self.updateStateEstimate()
            # if there are observations, update last_measurement_time
            self.last_measurement_time = self.time

    def predictStateEstimate(self, final_time: float):
        R"""Propagate the previous state estimate from :math:`k` to :math:`k+1`.

        Args:
            final_time (``float``): time to propagate to
        """
        # Sample sigma points and propagate through dynamics
        sigma_points_k = self.generateSigmaPoints(self.est_x, self.est_p)
        self.sigma_points = self.dynamics.propagate(
            start_time=self.time,
            end_time=final_time,
            start_state=sigma_points_k,
        )

        # Calculate the predicted state estimate by combining the sigma points
        self.pred_x = self.sigma_points.dot(self.mean_weight)

    def predictCovariance(self, final_time: float):
        R"""Propagate the previous covariance estimate from :math:`k` to :math:`k+1`.

        Args:
            final_time (``float``): time to propagate to
        """
        # pylint: disable=unused-argument
        # Calculate the predicted covariance estimate using the sigma points
        self.sigma_x_res = self.sigma_points - self.pred_x.reshape(
            (self.x_dim, 1)
        ).dot(ones((1, self.num_sigmas)))
        self.pred_p = (
            self.sigma_x_res.dot(self.cvr_weight.dot(self.sigma_x_res.T))
            + self.q_matrix
        )

    def calculateMeasurementMatrix(self, measurement: ndarray):
        R"""Calculate the stacked observation/measurement matrix for a set of observations.

        The UKF doesn't use an :math:`H` Matrix. Instead, the differences between the predicted
        state or observations, and the associated sigma values are calculated. These are used to
        determine the cross and innovations covariances.

        Args:
            measurement (``ndarray``): (M,) array of measurements
        """
        sigma_obs = zeros([self.y_dim, self.num_sigmas])
        for sigma_idx in range(self.num_sigmas):
            sigma_obs[:, sigma_idx] = self.measurement_model(
                self.sigma_points[:, sigma_idx]
            ).squeeze()

        # Save mean predicted measurement vector
        self.mean_pred_y = self.calcMeasurementMean(sigma_obs)

        # Determine the difference between the sigma pt observations and the mean observation
        self.sigma_y_res = zeros(sigma_obs.shape)
        for item in range(sigma_obs.shape[1]):
            self.sigma_y_res[:, item] = self.calcMeasurementResiduals(
                sigma_obs[:, item], self.mean_pred_y
            )

    def calculateKalmanGain(self):
        R"""Calculate the Kalman gain matrix.

        Compiles the stacked measurement noise covariance matrix, calculates the innovations
        covariance matrix, and calculates the cross covariance matrix.
        """
        self.innov_cvr = (
            self.sigma_y_res.dot(self.cvr_weight.dot(self.sigma_y_res.T))
            + self.r_matrix
        )
        self.cross_cvr = self.sigma_x_res.dot(
            self.cvr_weight.dot(self.sigma_y_res.T)
        )
        self.kalman_gain = self.cross_cvr.dot(inv(self.innov_cvr))

    def calculateInnovations(self, measurement: ndarray):
        R"""Calculate the innovations (residuals) vector and normalized innovations squared.

        Args:
            measurement (``ndarray``): (M,) array of measurements
        """
        self.innovation = self.calcMeasurementResiduals(
            measurement, self.mean_pred_y
        )
        self.nis = self.innovation.T.dot(
            inv(self.innov_cvr).dot(self.innovation)
        )

    def calcMeasurementResiduals(
        self, meas_set_a: ndarray, meas_set_b: ndarray
    ) -> ndarray:
        R"""Determine the measurement residuals.

        This is done generically which allows for measurements to be ordered in any fashion, but
        requires an associated boolean vector to flag for angle measurements. This special
        treatment is required because angles are nonlinear (modular), so subtraction is not a
        linear operation.

        Args:
            measurement_set_a (``ndarray``): :math:`M\times 1` compiled measurement array.
            measurement_set_b (``ndarray``): :math:`M\times 1` compiled measurement array.

        Returns:
            ``ndarray``: :math:`M\times 1` measurement residual
        """
        return array(meas_set_a - meas_set_b)

    def updateCovariance(self):
        R"""Update the covariance estimate at :math:`k+1`."""
        self.est_p = self.pred_p - self.kalman_gain.dot(
            self.innov_cvr.dot(self.kalman_gain.T)
        )

    def updateStateEstimate(self):
        R"""Update the state estimate estimate at :math:`k+1`."""
        self.est_x = self.pred_x + self.kalman_gain.dot(self.innovation)

    def calcMeasurementMean(self, measurement_sigma_pts: ndarray) -> ndarray:
        R"""Determine the mean of the predicted measurements.

        This is done generically which allows for measurements to be ordered in any fashion, but
        requires an associated boolean vector to flag for angle measurements. This special
        treatment is required because angles are nonlinear (modular), so calculating the mean is
        not a linear operation.

        Args:
            measurement_sigma_pts (``ndarray``): :math:`M\times S` array of predicted measurements,
                where :math:`M` is the compiled measurement space, and :math:`S` is the number of sigma points.

        Returns:
            ``ndarray``: :math:`M\times 1` predicted measurement mean
        """
        meas_mean = zeros((measurement_sigma_pts.shape[0],))
        for idx, meas in enumerate(measurement_sigma_pts):
            meas_mean[idx] = meas.dot(self.mean_weight)

        return meas_mean
