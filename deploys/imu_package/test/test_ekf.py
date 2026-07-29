"""Unit tests for the Ackermann-model EKF (pure math, no ROS)."""

import math

import numpy as np

from reboot.ekf import AckermannEkf, steering_yaw_rate_bound


def test_predict_straight_line_advances_position_by_speed():
    estimator = AckermannEkf()
    estimator.state[AckermannEkf.SPEED] = 1.0
    for _ in range(100):
        estimator.predict(0.01, yaw_rate=0.0)  # 1 s at 1 m/s along +x
    assert abs(estimator.state[AckermannEkf.X] - 1.0) < 0.02
    assert abs(estimator.state[AckermannEkf.Y]) < 1.0e-6
    assert abs(estimator.state[AckermannEkf.YAW]) < 1.0e-9


def test_predict_turns_heading_by_yaw_rate():
    estimator = AckermannEkf()
    estimator.state[AckermannEkf.SPEED] = 1.0
    for _ in range(100):
        estimator.predict(0.01, yaw_rate=0.5)  # 1 s at 0.5 rad/s
    assert abs(estimator.state[AckermannEkf.YAW] - 0.5) < 0.02
    assert estimator.state[AckermannEkf.X] > 0.0  # arc curves into quadrant I
    assert estimator.state[AckermannEkf.Y] > 0.0


def test_predict_never_changes_speed():
    estimator = AckermannEkf()
    estimator.state[AckermannEkf.SPEED] = 1.7
    for _ in range(500):
        estimator.predict(0.01, yaw_rate=0.2)
    # No accelerometer integration: speed only ever changes by a measurement.
    assert estimator.state[AckermannEkf.SPEED] == 1.7


def test_steering_bound_widens_with_speed_and_floors_near_rest():
    assert steering_yaw_rate_bound(0.0) == steering_yaw_rate_bound(0.1)  # speed floor
    assert steering_yaw_rate_bound(2.0) > steering_yaw_rate_bound(0.3)   # faster -> wider
    assert 2.5 < steering_yaw_rate_bound(2.0) < 4.5  # ~1.5*tan(0.35)/0.32*2


def test_update_speed_converges_to_measurement():
    estimator = AckermannEkf()
    for _ in range(50):
        estimator.update_speed(1.5, variance=0.01)
    assert abs(estimator.state[AckermannEkf.SPEED] - 1.5) < 0.05


def test_speed_returns_to_zero_when_wheels_stop():
    # The anti-accumulator demo: drive at 2 m/s, then feed "stopped" encoder
    # readings — the speed estimate must collapse to zero, never run away.
    estimator = AckermannEkf()
    for _ in range(50):
        estimator.update_speed(2.0, variance=0.01)
        estimator.predict(0.01, yaw_rate=0.0)
    for _ in range(50):
        estimator.update_speed(0.0, variance=0.01)
        estimator.predict(0.01, yaw_rate=0.0)
    assert abs(estimator.state[AckermannEkf.SPEED]) < 0.05


def test_dead_reckoning_straight_then_quarter_turn():
    # 200 Hz scenario: 10 m straight at 2 m/s, then a 90-degree arc at a
    # constant yaw rate. The endpoint must land within a few percent of the
    # analytic Ackermann path.
    dt = 0.005
    estimator = AckermannEkf()
    for _ in range(20):
        estimator.update_speed(2.0, variance=0.01)  # spin up the speed state

    for _ in range(1000):  # 5 s straight at 2 m/s = 10 m
        estimator.predict(dt, yaw_rate=0.0)
        estimator.update_speed(2.0, variance=0.01)

    yaw_rate = 1.0  # rad/s at 2 m/s -> radius 2 m, quarter turn in pi/2 s
    steps = int((math.pi / 2.0) / yaw_rate / dt)
    for _ in range(steps):
        estimator.predict(dt, yaw_rate=yaw_rate)
        estimator.update_speed(2.0, variance=0.01)

    # Analytic endpoint: straight (10, 0) then arc of radius 2 turning left:
    # (10 + 2*sin(90deg), 2*(1 - cos(90deg))) = (12, 2), yaw = pi/2.
    assert abs(estimator.state[AckermannEkf.X] - 12.0) < 0.1
    assert abs(estimator.state[AckermannEkf.Y] - 2.0) < 0.1
    assert abs(estimator.state[AckermannEkf.YAW] - math.pi / 2.0) < 0.02


def test_covariance_stays_symmetric_positive_definite():
    estimator = AckermannEkf()
    for _ in range(500):
        estimator.predict(0.005, yaw_rate=0.3)
        estimator.update_speed(1.0, variance=0.01)
    assert np.allclose(estimator.covariance, estimator.covariance.T)
    assert np.all(np.linalg.eigvalsh(estimator.covariance) > 0.0)


def test_reset_restores_start_state():
    estimator = AckermannEkf()
    estimator.state[:] = [3.0, -1.0, 2.0, 1.5]
    estimator.reset(yaw=0.5)
    assert np.allclose(estimator.state, [0.0, 0.0, 0.5, 0.0])
