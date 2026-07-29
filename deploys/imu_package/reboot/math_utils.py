"""Small angle/quaternion utilities shared by every estimator in this package.

Quaternions follow the ROS convention ``[x, y, z, w]`` and rotate vectors from
the vehicle body frame into the world frame.
"""

import math

import numpy as np


EPS = 1.0e-12


def wrap_angle(angle):
    """Wrap any angle into [-pi, pi). Every heading difference must go through
    this, otherwise an innovation across the +/-pi seam (e.g. 179 deg vs
    -179 deg) looks like a 358 deg error instead of the true 2 deg."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def stamp_to_seconds(stamp):
    """ROS builtin_interfaces/Time -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quat_normalize(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < EPS or not np.isfinite(norm):
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quaternion / norm


def quat_from_rpy(roll, pitch, yaw):
    """Intrinsic ZYX (yaw-pitch-roll) Euler angles -> quaternion [x, y, z, w]."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return quat_normalize(
        np.array(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ]
        )
    )


def rpy_from_quat(quaternion):
    """Quaternion [x, y, z, w] -> (roll, pitch, yaw), gimbal-guarded at pitch
    +/-90 deg via copysign."""
    x, y, z, w = quat_normalize(quaternion)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def validate_rotation(matrix, name="rotation"):
    """Assert a 3x3 matrix is a proper rotation (orthonormal, det = +1).

    Catches the classic axis-remap mistake: a hand-typed matrix that mirrors an
    axis (det = -1) silently flips a gyro sign and makes every turn integrate
    backwards."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-4):
        raise ValueError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=1.0e-4):
        raise ValueError(f"{name} must be right-handed (det=+1), got {determinant:.3f}")
    return matrix
