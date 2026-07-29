"""Publish a Kalman-filtered 3-DOF attitude on /attitude (Trial 2D part 1).

Subscribes the body-frame 9-DOF IMU stream (/imu/primary) and runs an EKF:
gyro predict at the IMU rate, accelerometer gravity update when trustworthy
(see attitude_ekf.py). Publishes:

  * ``/attitude``      sensor_msgs/Imu — orientation quaternion + covariance,
                       with the input angular velocity copied through so a
                       consumer (the wall follower) gets attitude AND yaw rate
                       from this one topic.
  * ``/attitude/rpy``  geometry_msgs/Vector3Stamped — roll/pitch/yaw in rad,
                       the human-readable/grading view (see state_view.py).
"""

import math

from geometry_msgs.msg import Vector3Stamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, MagneticField

from .attitude_ekf import AttitudeEkf
from .math_utils import quat_from_rpy, stamp_to_seconds

SAMPLE_RATE = 200.0  # used only for the very first time step
MAG_YAW_VARIANCE = 0.05  # (~13 deg)^2: even calibrated, indoor mag yaw is rough


def vec3(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


class AttitudeNode(Node):
    def __init__(self):
        super().__init__("attitude_node")
        # Yaw from the magnetometer is OFF by default: without a hard/soft-iron
        # calibration the motor and building steel bend it by tens of degrees,
        # which is far worse than the bias-corrected gyro's few deg/min drift.
        self.declare_parameter("use_mag", False)
        self._use_mag = bool(self.get_parameter("use_mag").value)

        self._filter = AttitudeEkf()
        self._last_time = None
        # /imu/primary already arrives decimated (~95 Hz); publish 1:1.
        self._publish_divisor = 1
        self._publish_counter = 0

        qos = qos_profile_sensor_data
        self._attitude_pub = self.create_publisher(Imu, "/attitude", qos)
        self._rpy_pub = self.create_publisher(Vector3Stamped, "/attitude/rpy", qos)
        self.create_subscription(Imu, "/imu/primary", self._on_imu, qos)
        if self._use_mag:
            self.create_subscription(
                MagneticField, "/mag/calibrated", self._on_mag, qos
            )
        self.get_logger().info(
            "attitude EKF: gyro predict + accel gravity update"
            + (" + mag yaw" if self._use_mag else " (yaw = gyro only)")
        )

    def _on_mag(self, msg):
        """Tilt-compensated magnetometer yaw, fused as one absolute yaw fix."""
        mx, my, mz = vec3(msg.magnetic_field)
        sr, cr = math.sin(self._filter.roll), math.cos(self._filter.roll)
        sp, cp = math.sin(self._filter.pitch), math.cos(self._filter.pitch)
        mx_level = mx * cp + my * sr * sp + mz * cr * sp
        my_level = my * cr - mz * sr
        self._filter.update_yaw(math.atan2(-my_level, mx_level), MAG_YAW_VARIANCE)

    def _on_imu(self, msg):
        now = stamp_to_seconds(msg.header.stamp)
        if self._last_time is None:
            dt = 1.0 / SAMPLE_RATE
        else:
            dt = now - self._last_time
        self._last_time = now

        self._filter.predict(dt, vec3(msg.angular_velocity))
        self._filter.update_accel(vec3(msg.linear_acceleration))
        self._publish_counter += 1
        if self._publish_counter < self._publish_divisor:
            return
        self._publish_counter = 0
        q = quat_from_rpy(self._filter.roll, self._filter.pitch, self._filter.yaw)

        att = Imu()
        att.header = msg.header
        att.orientation.x, att.orientation.y, att.orientation.z, att.orientation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3])
        )
        # The EKF's own covariance: watch yaw (index 8) grow while roll/pitch
        # (0, 4) stay bounded by the gravity updates.
        covariance = self._filter.covariance
        att.orientation_covariance = covariance.reshape(-1).tolist()
        # Pass the gyro through so /attitude alone gives consumers a yaw rate.
        att.angular_velocity = msg.angular_velocity
        att.angular_velocity_covariance = msg.angular_velocity_covariance
        self._attitude_pub.publish(att)

        rpy = Vector3Stamped()
        rpy.header = msg.header
        rpy.vector.x = self._filter.roll
        rpy.vector.y = self._filter.pitch
        rpy.vector.z = self._filter.yaw
        self._rpy_pub.publish(rpy)


def main(args=None):
    rclpy.init(args=args)
    node = AttitudeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
