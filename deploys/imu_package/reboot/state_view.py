"""Live full-state dashboard in the terminal — the readable alternative to
`ros2 topic echo`.

    ros2 run reboot state_view

One screen, refreshed 10x per second, showing every estimator output plus the
raw inputs that feed them (encoder, drive command) with per-topic freshness.
A topic that stops arriving turns to "STALE" — that alone diagnoses most
"estimate looks wrong" situations (e.g. the encoder only publishes while the
motor drives, so hand-pushing the car shows a stale encoder and the speed
falls back to the commanded-speed prior).
"""

import math
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TwistWithCovarianceStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

REFRESH = 0.1
STALE = 0.5  # seconds without a message -> mark the row STALE


def bar(value, full_scale, width=21):
    """Signed horizontal bar:  [<<<<<<|      ]  /  [      |>>>>>> ]"""
    half = width // 2
    n = int(round(min(abs(value) / full_scale, 1.0) * half))
    left = "<" * n if value < 0 else " " * n
    right = ">" * n if value > 0 else " " * n
    return "[" + left.rjust(half) + "|" + right.ljust(half) + "]"


class Row:
    def __init__(self):
        self.value = None
        self.stamp = 0.0
        self.count = 0
        self.rate = 0.0
        self._win_t = time.monotonic()
        self._win_n = 0

    def put(self, value):
        self.value = value
        self.stamp = time.monotonic()
        self.count += 1
        window = self.stamp - self._win_t
        if window >= 1.0:  # true average rate over the last ~1 s window
            self.rate = (self.count - self._win_n) / window
            self._win_t, self._win_n = self.stamp, self.count

    @property
    def age(self):
        return time.monotonic() - self.stamp if self.count else math.inf


class StateView(Node):
    def __init__(self):
        super().__init__("state_view")
        self.rpy = Row()
        self.att = Row()      # yaw rate via /attitude passthrough
        self.vel = Row()
        self.pose = Row()
        self.enc = Row()
        self.drive = Row()
        self._t0 = time.monotonic()

        qos = qos_profile_sensor_data
        self.create_subscription(Vector3Stamped, "/attitude/rpy",
                                 lambda m: self.rpy.put(m.vector), qos)
        self.create_subscription(Imu, "/attitude",
                                 lambda m: self.att.put(m.angular_velocity.z), qos)
        self.create_subscription(TwistWithCovarianceStamped, "/velocity",
                                 lambda m: self.vel.put((m.twist.twist.linear.x,
                                                        m.twist.covariance[0])), qos)
        self.create_subscription(Odometry, "/pose_estimate", self._on_pose, qos)
        self.create_subscription(Float32, "/encoder/speed",
                                 lambda m: self.enc.put(m.data), qos)
        self.create_subscription(AckermannDriveStamped, "/drive",
                                 lambda m: self.drive.put((m.drive.speed,
                                                          m.drive.steering_angle)), qos)
        self.create_timer(REFRESH, self._draw)
        print("\x1b[2J", end="")  # clear once

    def _on_pose(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        self.pose.put((p.x, p.y, yaw))

    @staticmethod
    def _age_tag(row):
        if not row.count:
            return "\x1b[31m  never\x1b[0m"
        if row.age > STALE:
            return f"\x1b[31mSTALE {row.age:4.1f}s\x1b[0m"
        return f"\x1b[32m{row.rate:5.0f}Hz\x1b[0m" if row.rate > 0.0 \
            else "\x1b[32m ok\x1b[0m"

    def _draw(self):
        lines = ["\x1b[H\x1b[0m"]
        lines.append(f"  reboot state   t+{time.monotonic() - self._t0:6.1f}s"
                     "        (Ctrl-C to quit)\x1b[K")
        lines.append("\x1b[K")

        if self.rpy.count:
            v = self.rpy.value
            r, p, y = (math.degrees(x) for x in (v.x, v.y, v.z))
            lines.append(f"  ATTITUDE  roll {r:+7.1f}°  pitch {p:+7.1f}°"
                         f"   {self._age_tag(self.rpy)}\x1b[K")
            lines.append(f"            yaw  {y:+7.1f}°  {bar(y, 180.0)}\x1b[K")
        else:
            lines.append(f"  ATTITUDE  --        {self._age_tag(self.rpy)}\x1b[K")
            lines.append("\x1b[K")
        yaw_rate = math.degrees(self.att.value) if self.att.count else 0.0
        lines.append(f"  YAW RATE  {yaw_rate:+7.1f}°/s {bar(yaw_rate, 200.0)}\x1b[K")
        lines.append("\x1b[K")

        if self.vel.count:
            speed, var = self.vel.value
            lines.append(f"  SPEED     {speed:+6.2f} m/s ±{math.sqrt(max(var, 0)):4.2f}"
                         f"  {bar(speed, 2.5)}  {self._age_tag(self.vel)}\x1b[K")
        else:
            lines.append(f"  SPEED     --        {self._age_tag(self.vel)}\x1b[K")
        enc = f"{self.enc.value:+6.2f} m/s" if self.enc.count else "  --  "
        lines.append(f"  ENCODER   {enc}  (raw wheel)          {self._age_tag(self.enc)}\x1b[K")
        lines.append("\x1b[K")

        if self.pose.count:
            x, y, yaw = self.pose.value
            lines.append(f"  POSE      x {x:+7.2f} m   y {y:+7.2f} m   "
                         f"θ {math.degrees(yaw):+6.1f}°   {self._age_tag(self.pose)}\x1b[K")
        else:
            lines.append(f"  POSE      --        {self._age_tag(self.pose)}\x1b[K")
        if self.drive.count:
            s, a = self.drive.value
            lines.append(f"  DRIVE cmd speed {s:+5.2f}  steer {a:+5.2f}"
                         f"   {self._age_tag(self.drive)}\x1b[K")
        else:
            lines.append(f"  DRIVE cmd --        {self._age_tag(self.drive)}\x1b[K")
        lines.append("\x1b[K")
        print("\n".join(lines), flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = StateView()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\x1b[0m")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
