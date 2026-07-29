"""3-DOF attitude EKF: gyro predicts, the accelerometer's gravity corrects.

State ``[roll, pitch, yaw]`` (rad, body frame x forward / y left / z up).

The two Kalman halves map directly onto the classic Trial 2D estimators:

  * **predict** -- gyro integration (``angle += omega * dt``). Fast and smooth
    but drifts, so its covariance GROWS by ``Q * dt`` every step.
  * **update** -- accelerometer trigonometry (``roll = atan2(ay, az)``,
    ``pitch = atan2(-ax, hypot(ay, az))``). Absolute but noisy, so it is fused
    with a measurement variance ``R`` instead of being trusted outright.

The Kalman gain ``K = P / (P + R)`` then does automatically what a
complementary filter hard-codes: with the default noise values the
steady-state gain at 200 Hz works out to ~0.03 per step, an effective
time constant of ~0.2 s -- the same blend the reference complementary
filter used (TAU = 0.20), but now derived from noise models, and with an
honest covariance to show for it.

Yaw has no absolute reference here (no calibrated magnetometer), so its
variance grows without bound -- deliberately honest. The startup gyro-bias
capture in the preprocessor is what keeps the actual drift to a few degrees
per minute. ``update_yaw`` exists for a calibrated magnetometer if one is
ever available (``use_mag``).

Pure numpy: no ROS imports, pytest-able on any laptop.
"""

import math

import numpy as np

from .math_utils import wrap_angle

GRAVITY = 9.80665
# The accelerometer only measures the gravity direction while |a| is close to
# g. During braking/cornering/bumps it measures motion, so we widen R with the
# deviation and skip the update entirely outside this band (trust gyro alone).
ACCEL_TRUST_BAND = 0.2


class AttitudeEkf:
    """Estimate ``[roll, pitch, yaw]`` from gyro + accelerometer."""

    ROLL = 0
    PITCH = 1
    YAW = 2

    def __init__(self, q_roll_pitch=3.0e-3, q_yaw=3.0e-4, r_accel=2.0e-2):
        # q_*: continuous process-noise density (rad^2/s) — how fast pure gyro
        # integration drifts. r_accel: variance (rad^2) of an accel-derived
        # angle (~8 deg std: car vibration makes these trig angles noisy).
        self.state = np.zeros(3)
        self.covariance = np.diag([0.03, 0.03, 0.03])
        self.process_variance = np.array([q_roll_pitch, q_roll_pitch, q_yaw])
        self.r_accel = float(r_accel)

    @property
    def roll(self):
        return float(self.state[self.ROLL])

    @property
    def pitch(self):
        return float(self.state[self.PITCH])

    @property
    def yaw(self):
        return float(self.state[self.YAW])

    def reset(self):
        self.state.fill(0.0)
        self.covariance = np.diag([0.03, 0.03, 0.03])

    def predict(self, dt, gyro):
        """Gyro integration with covariance growth.

        Body rates are integrated directly (roll += gx*dt, ...). The exact
        Euler-rate kinematics add sin/tan coupling terms, but for a ground
        vehicle that stays within a few degrees of level they are second-order
        small — and the accel update absorbs the residual. The Jacobian of
        this model is the identity, so the covariance step is simply
        ``P += Q * dt``: uncertainty grows whenever we only integrate.
        """
        dt = float(np.clip(dt, 5.0e-4, 5.0e-2))
        gyro = np.asarray(gyro, dtype=float)
        self.state = self.state + gyro * dt
        self.state = np.array([wrap_angle(angle) for angle in self.state])
        self.covariance = self.covariance + np.diag(self.process_variance * dt)

    def update_accel(self, accel):
        """Fuse accel-derived roll/pitch; returns True if the update ran.

        Outside the trust band (|a| far from g: braking, cornering, bumps) the
        accelerometer is measuring motion, not gravity — skip entirely.
        Inside the band, R is inflated quadratically with the deviation, so a
        sample right at the edge is trusted an order of magnitude less than a
        clean 1-g sample. This adaptive R replaces the reference
        complementary filter's binary trust gate with the standard Kalman way
        of saying "less sure".
        """
        ax, ay, az = (float(v) for v in accel)
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        deviation = abs(magnitude - GRAVITY)
        if deviation > ACCEL_TRUST_BAND * GRAVITY:
            return False

        variance = self.r_accel * (
            1.0 + 8.0 * (deviation / (ACCEL_TRUST_BAND * GRAVITY)) ** 2
        )
        roll_measured = math.atan2(ay, az)
        pitch_measured = math.atan2(-ax, math.hypot(ay, az))

        observation = np.zeros((1, 3))
        observation[0, self.ROLL] = 1.0
        self._row_update(roll_measured, observation, variance)
        observation = np.zeros((1, 3))
        observation[0, self.PITCH] = 1.0
        self._row_update(pitch_measured, observation, variance)
        return True

    def update_yaw(self, yaw_measured, variance):
        """Fuse an absolute yaw fix (tilt-compensated calibrated magnetometer)."""
        observation = np.zeros((1, 3))
        observation[0, self.YAW] = 1.0
        self._row_update(yaw_measured, observation, variance)

    def _row_update(self, measurement, observation, variance):
        """Joseph-form EKF update for one scalar angle measurement.

        All states here are angles, so the innovation is always wrapped —
        an accel roll of -179 deg against a state of +179 deg must read as a
        2 deg disagreement, not 358.
        """
        innovation = wrap_angle(float(measurement) - (observation @ self.state).item())
        innovation_variance = (
            observation @ self.covariance @ observation.T
        ).item() + max(float(variance), 1.0e-9)
        gain = self.covariance @ observation.T / innovation_variance
        self.state = self.state + gain[:, 0] * innovation
        self.state = np.array([wrap_angle(angle) for angle in self.state])
        identity = np.eye(3)
        # Joseph form preserves positive semi-definiteness under round-off.
        residual = identity - gain @ observation
        noise = np.array([[max(float(variance), 1.0e-9)]])
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
