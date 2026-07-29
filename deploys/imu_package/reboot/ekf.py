"""Ackermann-model planar EKF: one filter for velocity AND pose (Trial 2D 2+3).

State ``[x, y, yaw, forward_speed]``. The bicycle (Ackermann) motion model
drives the predict step:

  * heading advances by the measured gyro yaw rate, which the caller first
    clamps to the range the (physically limited) steering can actually
    produce — ``|yaw_rate| <= margin * |v| * tan(delta_max) / L``;
  * position advances by the current speed along the heading;
  * speed itself is a MEASURED quantity (wheel encoder, or the commanded
    speed through its calibration), NOT the integral of a
    gravity-contaminated accelerometer.

That last choice is the whole anti-"integration accumulator error" story:
integrating accel gives speed drift that then integrates AGAIN into position.
Observing speed every cycle means it can never run away — stop the wheels and
the estimate returns to zero.

Pure numpy: no ROS imports, pytest-able on any laptop.
(Ported from the QUEST_LOG imu_investigation package, minus the scan-to-map
ICP correction — this package runs IMU + encoder dead reckoning only.)
"""

import math

import numpy as np

from .math_utils import wrap_angle

# Physical limits of the racecar platform. Canonical here so estimation and
# control share one set of numbers.
WHEELBASE_M = 0.32    # axle-to-axle distance
MAX_STEER_RAD = 0.35  # ~20 deg mechanical steering limit


def steering_yaw_rate_bound(speed, margin=1.5, speed_floor=1.0):
    """Largest ``|yaw rate|`` the steering geometry can produce at ``speed``.

    Ackermann: ``|yaw_rate| <= |v| * tan(delta_max) / L``. ``margin`` absorbs
    slides and calibration error; ``speed_floor`` keeps the bound meaningful
    near standstill. The floor must stay ABOVE hand-rotation speeds: with no
    absolute yaw correction, any clipping of a real rotation (e.g. picking the
    car up and turning it during a bench test) becomes a permanent heading
    error — measured 40 deg of false pose yaw with the old 0.3 floor. The
    clamp is an outlier gate against gyro glitches, not a hard kinematic
    constraint.
    """
    return margin * math.tan(MAX_STEER_RAD) / WHEELBASE_M * max(abs(speed), speed_floor)


class AckermannEkf:
    """Estimate ``[x, y, yaw, forward_speed]`` from gyro + measured speed."""

    X = 0
    Y = 1
    YAW = 2
    SPEED = 3

    def __init__(self):
        self.state = np.zeros(4)
        self.covariance = np.diag([0.01, 0.01, 0.03, 0.25])
        # Continuous-time process-noise density [x, y, yaw, speed]: how fast
        # pure dead-reckoning drifts. Position drifts from speed/heading error,
        # so its density dominates; heading (gyro) drifts slower; speed is a
        # random walk corrected by the encoder every cycle.
        self.process_variance = np.array([0.3, 0.3, 0.05, 0.5])

    def reset(self, yaw=0.0):
        self.state.fill(0.0)
        self.state[self.YAW] = wrap_angle(yaw)
        self.covariance = np.diag([0.01, 0.01, 0.03, 0.25])

    def predict(self, dt, yaw_rate):
        """Ackermann predict from a (steering-bounded) yaw rate.

        The midpoint heading ``yaw + 0.5 * yaw_rate * dt`` integrates the arc
        more accurately than either endpoint. Speed is propagated unchanged (a
        random-walk prior); the encoder / commanded-speed update pulls it
        toward the true value.
        """
        dt = float(np.clip(dt, 1.0e-4, 0.1))
        x, y, yaw, speed = self.state
        yaw_rate = float(yaw_rate)
        yaw_mid = yaw + 0.5 * yaw_rate * dt

        self.state[self.X] = x + speed * math.cos(yaw_mid) * dt
        self.state[self.Y] = y + speed * math.sin(yaw_mid) * dt
        self.state[self.YAW] = wrap_angle(yaw + yaw_rate * dt)
        # speed unchanged: it is a measured quantity, not integrated here.

        # Jacobian of the motion model with respect to the state.
        transition = np.eye(4)
        transition[self.X, self.YAW] = -speed * math.sin(yaw_mid) * dt
        transition[self.X, self.SPEED] = math.cos(yaw_mid) * dt
        transition[self.Y, self.YAW] = speed * math.cos(yaw_mid) * dt
        transition[self.Y, self.SPEED] = math.sin(yaw_mid) * dt

        process_noise = np.diag(self.process_variance * dt)
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _row_update(self, measurement, observation, variance):
        """Joseph-form EKF update for one scalar measurement of
        ``observation @ state``."""
        innovation = float(measurement) - (observation @ self.state).item()
        innovation_variance = (
            observation @ self.covariance @ observation.T
        ).item() + max(float(variance), 1.0e-9)
        gain = self.covariance @ observation.T / innovation_variance
        self.state += gain[:, 0] * innovation
        self.state[self.YAW] = wrap_angle(self.state[self.YAW])
        identity = np.eye(4)
        # Joseph form preserves positive semi-definiteness under round-off.
        residual = identity - gain @ observation
        noise = np.array([[max(float(variance), 1.0e-9)]])
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update_speed(self, speed, variance):
        """Fuse a forward-speed measurement (wheel encoder or commanded speed)."""
        observation = np.zeros((1, 4))
        observation[0, self.SPEED] = 1.0
        self._row_update(speed, observation, variance)
