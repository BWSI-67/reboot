"""Unit tests for the attitude EKF (pure math, no ROS)."""

import math

import numpy as np

from reboot.attitude_ekf import ACCEL_TRUST_BAND, GRAVITY, AttitudeEkf

DT = 0.005  # 200 Hz
LEVEL_ACCEL = (0.0, 0.0, GRAVITY)


def run(estimator, steps, gyro=(0.0, 0.0, 0.0), accel=LEVEL_ACCEL):
    for _ in range(steps):
        estimator.predict(DT, gyro)
        estimator.update_accel(accel)


def test_level_and_still_stays_level():
    estimator = AttitudeEkf()
    run(estimator, 500)
    assert abs(estimator.roll) < 1.0e-6
    assert abs(estimator.pitch) < 1.0e-6
    assert abs(estimator.yaw) < 1.0e-6


def test_gyro_integration_tracks_yaw():
    estimator = AttitudeEkf()
    run(estimator, 400, gyro=(0.0, 0.0, 0.5))  # 2 s at 0.5 rad/s
    assert abs(estimator.yaw - 1.0) < 1.0e-6


def test_wrong_initial_roll_converges_to_gravity():
    estimator = AttitudeEkf()
    estimator.state[AttitudeEkf.ROLL] = 0.5
    initial_variance = estimator.covariance[0, 0]
    run(estimator, 400)  # 2 s of clean gravity
    assert abs(estimator.roll) < 0.05  # pulled back to level
    assert estimator.covariance[0, 0] < initial_variance  # and more certain


def test_steady_state_time_constant_matches_reference_tau():
    # The Q/R defaults are chosen so that at steady state the accel correction
    # acts like the reference complementary filter's TAU~0.2 s. (While the
    # covariance is still large right after startup the gain is bigger — fast
    # initial convergence is correct Kalman behavior — so we let it settle
    # first, then inject an error and watch the decay rate.)
    estimator = AttitudeEkf()
    run(estimator, 500)  # settle to steady-state covariance
    estimator.state[AttitudeEkf.ROLL] = 0.5
    run(estimator, 10)  # 0.05 s
    assert estimator.roll > 0.3  # not an instant snap to the noisy accel angle
    run(estimator, 290)  # ~1.5 s total
    assert abs(estimator.roll) < 0.05  # but gone within ~5 time constants


def test_dynamic_accel_is_ignored():
    estimator = AttitudeEkf()
    estimator.state[AttitudeEkf.ROLL] = 0.3
    # |a| = 1.5 g: hard cornering. The gravity update must not run at all.
    dynamic = (0.0, 0.0, 1.5 * GRAVITY)
    assert estimator.update_accel(dynamic) is False
    assert abs(estimator.roll - 0.3) < 1.0e-9


def test_adaptive_r_trusts_edge_of_band_less():
    edge_deviation = 0.99 * ACCEL_TRUST_BAND * GRAVITY
    clean, edgy = AttitudeEkf(), AttitudeEkf()
    clean.state[AttitudeEkf.ROLL] = edgy.state[AttitudeEkf.ROLL] = 0.4
    clean.update_accel(LEVEL_ACCEL)
    edgy.update_accel((0.0, 0.0, GRAVITY + edge_deviation))
    # Both updates pull roll toward the measured value, but the edge-of-band
    # sample must move the state noticeably less.
    assert abs(edgy.roll - 0.4) < abs(clean.roll - 0.4)


def test_yaw_variance_grows_and_roll_pitch_variance_stays_bounded():
    estimator = AttitudeEkf()
    yaw_var_early = estimator.covariance[2, 2]
    run(estimator, 2000)  # 10 s
    yaw_var_late = estimator.covariance[2, 2]
    assert yaw_var_late > yaw_var_early  # no absolute yaw reference: honest growth
    assert estimator.covariance[0, 0] < 0.01  # gravity keeps roll/pitch certain
    assert estimator.covariance[1, 1] < 0.01


def test_yaw_update_wraps_across_pi():
    estimator = AttitudeEkf()
    estimator.state[AttitudeEkf.YAW] = math.radians(179.0)
    estimator.update_yaw(math.radians(-179.0), variance=1.0e-4)
    # The fix must move yaw the short way (through +/-180), not through zero.
    assert abs(abs(estimator.yaw) - math.pi) < math.radians(3.0)


def test_covariance_stays_symmetric_positive_definite():
    rng = np.random.default_rng(2)
    estimator = AttitudeEkf()
    for _ in range(500):
        estimator.predict(DT, rng.normal(0.0, 0.2, size=3))
        estimator.update_accel(
            np.array(LEVEL_ACCEL) + rng.normal(0.0, 0.3, size=3)
        )
    assert np.allclose(estimator.covariance, estimator.covariance.T)
    assert np.all(np.linalg.eigvalsh(estimator.covariance) > 0.0)
