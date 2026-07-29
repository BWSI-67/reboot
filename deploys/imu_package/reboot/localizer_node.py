"""Localizer: publish /velocity and /pose_estimate from one Ackermann EKF.

Trial 2D deliverables 2 and 3 come from a single shared filter — velocity and
pose are not estimated twice:

  * ``/imu/primary``   — the gyro gives the yaw rate. The predict step turns
                         the heading by it (clamped to what the limited
                         steering angle can physically produce) and slides the
                         position forward at the current speed.
  * ``/encoder/speed`` — wheel speed is the forward-speed measurement. When it
    ``/drive``           is absent (dead encoder) the commanded speed through
                         its calibration table stands in; with neither source
                         the filter is told "stopped".

There is deliberately NO accelerometer integration, and therefore no
zero-velocity / speed-limit patches: speed is measured every cycle and can
never run away. This is dead reckoning — position error grows slowly with
distance — but over a 30 m / 20 s course it stays small enough to race on.
"""

import math

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    TransformStamped,
    TwistWithCovarianceStamped,
)
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, Float32
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from .ekf import AckermannEkf, steering_yaw_rate_bound
from .math_utils import quat_from_rpy, rpy_from_quat, stamp_to_seconds

# --- speed sources ---
ENCODER_VARIANCE = 0.01       # wheel-speed measurement variance (trusted)
ENCODER_TIMEOUT = 0.20        # seconds before a wheel-speed reading is stale
ENCODER_ALIVE_SPEED = 0.05    # one reading above this proves the encoder works
CMD_SPEED_VARIANCE = 0.04     # commanded-speed fallback variance (less trusted)
CMD_TIMEOUT = 0.50            # seconds before a /drive command is stale
STOP_SPEED_VARIANCE = 0.04    # variance of the "no live source -> stopped" prior

# Steering-limited yaw-rate clamp margin. The gyro is the true yaw-rate
# sensor; the clamp is only an outlier gate against glitches — at racing slip
# the real yaw rate can exceed the no-sideslip steering geometry, so the
# bound is kept loose.
YAW_BOUND_MARGIN = 3.0

FRAME_PARENT = "map"
FRAME_CHILD = "base_link"
LASER_FRAME = "laser"         # the lidar publishes /scan here; treated as
                              # coincident with base_link so viewers can place it.


class LocalizerNode(Node):
    def __init__(self):
        super().__init__("localizer_node")
        # Per-track configuration (not tuning): where the car starts, the
        # normalized-command -> m/s calibration for the encoder fallback, and
        # the along-track encoder scale (drive a taped >3 m straight and match
        # /pose_estimate to the tape to set it; the wheel typically over-reads
        # a few percent).
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_yaw", 0.0)
        self.declare_parameter("speed_cmd_points", [0.0, 0.42, 0.8, 1.0])
        self.declare_parameter("speed_mps_points", [0.0, 0.0, 1.686, 2.5])
        self.declare_parameter("encoder_scale", 0.96)
        # The commanded-speed fallback is OFF by default: /drive commands are
        # gated by the mux deadman (right bumper). With the bumper released
        # the car does NOT move while /drive still streams speed — fusing the
        # command would integrate phantom motion into the pose. The encoder
        # is the first-class speed source; with no fresh encoder data the
        # filter assumes "stopped". Enable only on a car with a dead encoder.
        self.declare_parameter("use_cmd_speed_fallback", False)

        self._start_pose = (
            float(self.get_parameter("start_x").value),
            float(self.get_parameter("start_y").value),
            float(self.get_parameter("start_yaw").value),
        )
        self._cal_cmd = np.asarray(self.get_parameter("speed_cmd_points").value, dtype=float)
        self._cal_mps = np.asarray(self.get_parameter("speed_mps_points").value, dtype=float)
        self._encoder_scale = float(self.get_parameter("encoder_scale").value)
        self._use_cmd_fallback = bool(self.get_parameter("use_cmd_speed_fallback").value)

        self._filter = AckermannEkf()
        self._reset_to_start()
        self._last_imu_time = None
        # /imu/primary arrives decimated (~95 Hz); publish Odometry+Twist+TF
        # every 2nd predict (~47 Hz) — plenty for control, half the
        # serialization cost.
        self._publish_divisor = 2
        self._publish_counter = 0
        self._latest_encoder_time = None
        self._encoder_alive = False   # a broken encoder reads constant zero forever
        self._cmd_speed = None        # latest /drive speed, normalized [-1, 1]
        self._cmd_time = None

        qos = qos_profile_sensor_data
        self._velocity_pub = self.create_publisher(TwistWithCovarianceStamped, "/velocity", qos)
        self._pose_pub = self.create_publisher(Odometry, "/pose_estimate", qos)
        self.create_subscription(Imu, "/imu/primary", self._imu_callback, qos)
        self.create_subscription(Float32, "/encoder/speed", self._encoder_callback, qos)
        self.create_subscription(AckermannDriveStamped, "/drive", self._drive_callback, qos)
        # Re-anchor tools: simulator reset, RViz/Foxglove "2D Pose Estimate",
        # and a service for repeated course runs.
        self.create_subscription(Empty, "/sim/reset_cmd", self._external_reset, qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._initialpose_callback, 10
        )
        self.create_service(Trigger, "/pose_estimate/reset", self._reset_callback)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info(
            "localizer: Ackermann EKF, gyro predict + encoder/cmd speed (odometry only)"
        )

    # ------------------------------------------------------------------ predict
    def _imu_callback(self, message):
        """High-rate predict: turn by the (steering-bounded) gyro, slide by speed."""
        current_time = stamp_to_seconds(message.header.stamp)
        if current_time <= 0.0:
            current_time = self.get_clock().now().nanoseconds * 1.0e-9
        if self._last_imu_time is None:
            self._last_imu_time = current_time
            return
        dt = current_time - self._last_imu_time
        self._last_imu_time = current_time
        if dt <= 0.0 or dt > 0.1:
            return

        gyro_z = float(message.angular_velocity.z)
        bound = steering_yaw_rate_bound(
            self._filter.state[AckermannEkf.SPEED], margin=YAW_BOUND_MARGIN
        )
        yaw_rate = float(np.clip(gyro_z, -bound, bound))
        self._filter.predict(dt, yaw_rate)
        self._feed_speed(current_time)
        self._publish_counter += 1
        if self._publish_counter >= self._publish_divisor:
            self._publish_counter = 0
            self._publish(message.header.stamp)

    def _feed_speed(self, now):
        """Fuse a forward-speed measurement: encoder first, command, then stop.

        The encoder (when alive and fresh) is fused in its own callback at the
        wheel-speed rate; here we only fill in when it is absent. A live
        /drive command drives speed through the calibration table. With
        neither source we assume the car is stopped — a safe prior."""
        encoder_fresh = (
            self._encoder_alive
            and self._latest_encoder_time is not None
            and now - self._latest_encoder_time <= ENCODER_TIMEOUT
        )
        if encoder_fresh:
            return
        if (self._use_cmd_fallback and self._cmd_speed is not None
                and now - self._cmd_time <= CMD_TIMEOUT):
            self._filter.update_speed(self._cmd_to_speed(self._cmd_speed), CMD_SPEED_VARIANCE)
        else:
            # encoder silent -> assume stopped. NEVER trust /drive here: with
            # the mux deadman released the command streams but the car stands
            # still, and fusing it would fake motion into the pose.
            self._filter.update_speed(0.0, STOP_SPEED_VARIANCE)

    def _cmd_to_speed(self, cmd_speed):
        """Normalized [-1, 1] speed command -> m/s via the calibration table."""
        magnitude = float(np.interp(abs(float(cmd_speed)), self._cal_cmd, self._cal_mps))
        return math.copysign(magnitude, cmd_speed)

    # ------------------------------------------------------------ speed inputs
    def _encoder_callback(self, message):
        value = float(message.data)
        self._latest_encoder_time = self.get_clock().now().nanoseconds * 1.0e-9
        if not self._encoder_alive and abs(value) > ENCODER_ALIVE_SPEED:
            self._encoder_alive = True
            self.get_logger().info("encoder alive: wheel-speed updates enabled")
        if self._encoder_alive:
            self._filter.update_speed(value * self._encoder_scale, ENCODER_VARIANCE)

    def _drive_callback(self, message):
        self._cmd_speed = float(message.drive.speed)
        self._cmd_time = self.get_clock().now().nanoseconds * 1.0e-9

    # ------------------------------------------------------------------- output
    def _publish(self, stamp):
        state = self._filter.state
        covariance = self._filter.covariance
        q = quat_from_rpy(0.0, 0.0, state[AckermannEkf.YAW])
        speed = float(state[AckermannEkf.SPEED])
        speed_variance = float(covariance[AckermannEkf.SPEED, AckermannEkf.SPEED])

        pose = Odometry()
        pose.header.stamp = stamp
        pose.header.frame_id = FRAME_PARENT
        pose.child_frame_id = FRAME_CHILD
        pose.pose.pose.position.x = float(state[AckermannEkf.X])
        pose.pose.pose.position.y = float(state[AckermannEkf.Y])
        pose.pose.pose.orientation.x = float(q[0])
        pose.pose.pose.orientation.y = float(q[1])
        pose.pose.pose.orientation.z = float(q[2])
        pose.pose.pose.orientation.w = float(q[3])
        pose.twist.twist.linear.x = speed

        pose_covariance = np.zeros((6, 6))
        xyw = [AckermannEkf.X, AckermannEkf.Y, AckermannEkf.YAW]
        pose_covariance[np.ix_([0, 1, 5], [0, 1, 5])] = covariance[np.ix_(xyw, xyw)]
        pose_covariance[2, 2] = pose_covariance[3, 3] = pose_covariance[4, 4] = 1.0e3
        pose.pose.covariance = pose_covariance.reshape(-1).tolist()
        pose.twist.covariance[0] = speed_variance
        self._pose_pub.publish(pose)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = FRAME_PARENT
        transform.child_frame_id = FRAME_CHILD
        transform.transform.translation.x = float(state[AckermannEkf.X])
        transform.transform.translation.y = float(state[AckermannEkf.Y])
        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])
        # base_link -> laser (identity): without it the scan has no TF path to
        # "map" and viewers cannot render it over the trajectory.
        laser_transform = TransformStamped()
        laser_transform.header.stamp = stamp
        laser_transform.header.frame_id = FRAME_CHILD
        laser_transform.child_frame_id = LASER_FRAME
        laser_transform.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform([transform, laser_transform])

        velocity = TwistWithCovarianceStamped()
        velocity.header = pose.header
        velocity.twist.twist.linear.x = speed
        velocity.twist.covariance[0] = speed_variance
        self._velocity_pub.publish(velocity)

    # ------------------------------------------------------------ (re)anchoring
    def _reset_to_start(self):
        """Initialize/reset the filter at the configured start pose."""
        self._filter.reset(yaw=self._start_pose[2])
        self._filter.state[AckermannEkf.X] = self._start_pose[0]
        self._filter.state[AckermannEkf.Y] = self._start_pose[1]

    def _external_reset(self, _message):
        self._reset_to_start()
        self.get_logger().info("external reset: filter re-initialized at start pose")

    def _initialpose_callback(self, message):
        """Re-anchor the filter to a pose set by hand (RViz/Foxglove tool)."""
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        quaternion = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w], dtype=float
        )
        norm = float(np.linalg.norm(quaternion))
        yaw = rpy_from_quat(quaternion / norm)[2] if norm > 0.5 else 0.0
        self._filter.reset(yaw=yaw)
        self._filter.state[AckermannEkf.X] = float(position.x)
        self._filter.state[AckermannEkf.Y] = float(position.y)
        self.get_logger().info(
            f"initialpose: re-anchored at ({position.x:.2f}, {position.y:.2f}, "
            f"{np.degrees(yaw):.0f} deg)"
        )

    def _reset_callback(self, _request, response):
        self._reset_to_start()
        response.success = True
        response.message = "filter reset to the configured start pose"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LocalizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
