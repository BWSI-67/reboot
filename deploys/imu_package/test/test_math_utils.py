"""Unit tests for math_utils (pure math, no ROS)."""

import math

import numpy as np
import pytest

from reboot.math_utils import (
    quat_from_rpy,
    quat_normalize,
    rpy_from_quat,
    validate_rotation,
    wrap_angle,
)


def test_wrap_angle_is_identity_inside_range():
    for angle in (-3.0, -1.0, 0.0, 1.0, 3.0):
        assert abs(wrap_angle(angle) - angle) < 1.0e-12


def test_wrap_angle_wraps_across_pi():
    assert abs(wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1.0e-12
    assert abs(wrap_angle(-math.pi - 0.1) - (math.pi - 0.1)) < 1.0e-12
    assert abs(wrap_angle(3.0 * math.pi) - (-math.pi)) < 1.0e-9


def test_quat_rpy_round_trip():
    cases = [
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 1.0),
        (-1.2, 0.5, -2.8),
        (0.1, 1.2, 3.0),
    ]
    for roll, pitch, yaw in cases:
        r, p, y = rpy_from_quat(quat_from_rpy(roll, pitch, yaw))
        assert abs(wrap_angle(r - roll)) < 1.0e-9
        assert abs(wrap_angle(p - pitch)) < 1.0e-9
        assert abs(wrap_angle(y - yaw)) < 1.0e-9


def test_quat_from_rpy_yaw_only_matches_half_angle_form():
    q = quat_from_rpy(0.0, 0.0, math.pi / 2.0)
    assert np.allclose(q, [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])


def test_quat_normalize_handles_degenerate_input():
    assert np.allclose(quat_normalize([0.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(quat_normalize([0.0, 0.0, 0.0, 2.0]), [0.0, 0.0, 0.0, 1.0])


def test_validate_rotation_accepts_proper_rotation():
    # The optical->body remap used by the preprocessor: a proper rotation.
    optical_to_body = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
    validate_rotation(optical_to_body, "optical_to_body")


def test_validate_rotation_rejects_reflection():
    mirror = np.diag([1.0, 1.0, -1.0])  # det = -1: flips a gyro axis
    with pytest.raises(ValueError):
        validate_rotation(mirror, "mirror")


def test_validate_rotation_rejects_non_orthonormal():
    with pytest.raises(ValueError):
        validate_rotation(np.eye(3) * 2.0, "scaled")
