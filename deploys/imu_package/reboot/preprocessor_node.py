"""Rotate the raw RealSense IMU into the body frame and remove the gyro bias.

This is a deliberately simplified version of a full IMU calibration front-end:
no scale/non-orthogonality matrices and no magnetometer ellipsoid fit. It does
the two things the downstream Kalman filters cannot live without:

  1. **Axis remap.** The camera publishes in its optical frame (x right,
     y down, z forward); every estimator here reasons in the REP-103 body
     frame (x forward, y left, z up). One proper rotation fixes that.
  2. **Stationary gyro-bias capture.** A raw MEMS gyro reads a few tenths of a
     deg/s even when perfectly still; integrating that turns into unbounded
     yaw drift ("integration accumulator error"). At startup, while the car
     sits still, we average a couple of seconds of gyro samples and subtract
     that bias from every later reading. This one step is the difference
     between a yaw that drifts degrees per minute and one that holds.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, MagneticField

from .gyro_bias import GyroBiasEstimator
from .math_utils import stamp_to_seconds, validate_rotation


# Standard ROS optical frame (x right, y down, z forward) to vehicle body
# frame (x forward, y left, z up). A proper rotation, det = +1.
OPTICAL_TO_BODY = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
IDENTITY = np.eye(3).reshape(-1).tolist()


def vector3_to_array(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def assign_vector3(vector, values):
    vector.x, vector.y, vector.z = (float(values[0]), float(values[1]), float(values[2]))


def transform_covariance(values, rotation, default_variance):
    """Rotate a 3x3 covariance into the body frame (R C R^T); substitute an
    isotropic default when the driver left it unknown (first element -1) or
    all-zero."""
    covariance = np.asarray(values, dtype=float).reshape(3, 3)
    if covariance.reshape(-1)[0] < 0.0 or not np.any(covariance):
        covariance = np.eye(3) * float(default_variance)
    return (rotation @ covariance @ rotation.T).reshape(-1).tolist()


class ImuPreprocessor(Node):
    def __init__(self):
        super().__init__("imu_preprocessor")
        self.declare_parameter("imu_input", "/imu/realsense")
        self.declare_parameter("imu_output", "/imu/primary")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("imu_rotation_body_from_sensor", OPTICAL_TO_BODY)
        self.declare_parameter("bias_capture_sec", 2.0)
        self.declare_parameter("bias_capture_still_std", 0.03)
        # Gyro yaw-axis scale. QUEST_LOG fitted 0.9186 on this platform against
        # scan-matched ground truth: the RealSense yaw gyro reads ~8% high, so
        # without this every full turn adds ~29 deg heading error and position
        # error grows with distance. Verify per car: spin exactly 2 turns, yaw
        # must read 720 deg; set scale *= 720/reading. Set 1.0 to disable.
        self.declare_parameter("gyro_scale_z", 0.9186)
        self.declare_parameter("accel_default_variance", 0.04)
        self.declare_parameter("gyro_default_variance", 0.0004)
        # Magnetometer pass-through, kept for the optional attitude yaw update.
        # OFF by default: an uncalibrated magnetometer indoors is worse than
        # a bias-corrected gyro.
        self.declare_parameter("use_mag", False)
        self.declare_parameter("mag_input", "/mag")
        self.declare_parameter("mag_output", "/mag/calibrated")
        self.declare_parameter("mag_rotation_body_from_sensor", IDENTITY)
        self.declare_parameter("mag_default_variance", 1.0e-10)

        def param(name):
            return self.get_parameter(name).value

        self._frame = str(param("frame_id"))
        self._rotation = validate_rotation(
            param("imu_rotation_body_from_sensor"), "imu_rotation_body_from_sensor"
        )
        self._accel_var = float(param("accel_default_variance"))
        self._gyro_var = float(param("gyro_default_variance"))
        self._gyro_scale_z = float(param("gyro_scale_z"))
        self._bias = GyroBiasEstimator(
            capture_sec=float(param("bias_capture_sec")),
            still_std=float(param("bias_capture_still_std")),
        )

        # Adaptive decimation to ~100 Hz output: the RealSense driver has been
        # seen publishing anywhere from 200 Hz to 2 kHz depending on its
        # config. Downstream estimators only need ~100 Hz; the divisor is
        # re-derived from the measured input rate every couple of seconds.
        # Integration accuracy is unaffected — consumers take dt from stamps.
        self._target_hz = 100.0
        self._decimate = 2
        self._decimate_counter = 0
        self._rate_count = 0
        self._rate_t0 = None
        # Window accumulators: each ~100 Hz output sample is the MEAN of the
        # raw samples in its window (anti-aliasing). Averaging also divides
        # the per-sample noise by sqrt(window), which keeps the stationary
        # bias capture's stillness test working at any input rate.
        self._accum_gyro = np.zeros(3)
        self._accum_accel = np.zeros(3)
        self._accum_n = 0

        qos = qos_profile_sensor_data
        imu_input = str(param("imu_input"))
        self._imu_pub = self.create_publisher(Imu, str(param("imu_output")), qos)
        self.create_subscription(Imu, imu_input, self._imu_callback, qos)

        self._mag_pub = None
        if bool(param("use_mag")):
            self._mag_rotation = validate_rotation(
                param("mag_rotation_body_from_sensor"), "mag_rotation_body_from_sensor"
            )
            self._mag_var = float(param("mag_default_variance"))
            self._mag_pub = self.create_publisher(
                MagneticField, str(param("mag_output")), qos
            )
            self.create_subscription(
                MagneticField, str(param("mag_input")), self._mag_callback, qos
            )

        self.get_logger().info(
            f"remapping {imu_input} into {self._frame}; hold the car still "
            f"{self._bias.capture_sec:.1f}s for gyro-bias capture"
        )

    def _imu_callback(self, message):
        stamp_sec = stamp_to_seconds(message.header.stamp)
        self._rate_count += 1
        if self._rate_t0 is None:
            self._rate_t0 = stamp_sec
        elif stamp_sec - self._rate_t0 >= 2.0:
            rate = self._rate_count / (stamp_sec - self._rate_t0)
            divisor = max(1, int(round(rate / self._target_hz)))
            if divisor != self._decimate:
                self._decimate = divisor
                self.get_logger().info(
                    f"imu input {rate:.0f} Hz -> decimate /{divisor} "
                    f"(~{rate / divisor:.0f} Hz out)")
            self._rate_count = 0
            self._rate_t0 = stamp_sec

        self._accum_gyro += self._rotation @ vector3_to_array(message.angular_velocity)
        self._accum_accel += self._rotation @ vector3_to_array(message.linear_acceleration)
        self._accum_n += 1
        self._decimate_counter += 1
        if self._decimate_counter < self._decimate:
            return
        self._decimate_counter = 0
        angular_velocity = self._accum_gyro / self._accum_n
        acceleration = self._accum_accel / self._accum_n
        self._accum_gyro = np.zeros(3)
        self._accum_accel = np.zeros(3)
        self._accum_n = 0

        if self._bias.submit(stamp_sec, angular_velocity):
            bias_dps = np.degrees(self._bias.bias)
            self.get_logger().info(
                "gyro bias locked: "
                f"[{bias_dps[0]:+.3f}, {bias_dps[1]:+.3f}, {bias_dps[2]:+.3f}] deg/s"
            )
        if not self._bias.locked:
            # Publish NOTHING until the bias is known. If downstream estimators
            # were fed the raw stream during the capture window they would
            # integrate the full bias (~10 deg/s of fake yaw rate on this IMU)
            # into an initial heading error that gyro-only yaw can never undo.
            return
        angular_velocity = angular_velocity - self._bias.bias
        angular_velocity[2] *= self._gyro_scale_z

        output = Imu()
        output.header = message.header
        output.header.frame_id = self._frame
        assign_vector3(output.linear_acceleration, acceleration)
        assign_vector3(output.angular_velocity, angular_velocity)
        output.linear_acceleration_covariance = transform_covariance(
            message.linear_acceleration_covariance, self._rotation, self._accel_var
        )
        output.angular_velocity_covariance = transform_covariance(
            message.angular_velocity_covariance, self._rotation, self._gyro_var
        )
        output.orientation_covariance[0] = -1.0  # no orientation estimate here
        self._imu_pub.publish(output)

    def _mag_callback(self, message):
        output = MagneticField()
        output.header = message.header
        output.header.frame_id = self._frame
        magnetic_field = self._mag_rotation @ vector3_to_array(message.magnetic_field)
        assign_vector3(output.magnetic_field, magnetic_field)
        output.magnetic_field_covariance = transform_covariance(
            message.magnetic_field_covariance, self._mag_rotation, self._mag_var
        )
        self._mag_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPreprocessor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
