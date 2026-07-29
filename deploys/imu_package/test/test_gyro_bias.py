"""Unit tests for the stationary gyro-bias capture (pure math, no ROS)."""

import numpy as np

from reboot.gyro_bias import GyroBiasEstimator


def feed(estimator, times, samples):
    locked_at = None
    for t, sample in zip(times, samples):
        if estimator.submit(t, sample):
            locked_at = t
    return locked_at


def test_locks_on_quiet_window_and_recovers_true_bias():
    rng = np.random.default_rng(0)
    bias = np.array([0.002, -0.005, 0.007])  # ~0.4 deg/s on z, realistic
    times = np.arange(0.0, 2.5, 0.005)  # 200 Hz
    samples = bias + rng.normal(0.0, 0.005, size=(times.size, 3))

    estimator = GyroBiasEstimator(capture_sec=2.0, still_std=0.03)
    locked_at = feed(estimator, times, samples)

    assert estimator.locked
    assert locked_at is not None and locked_at >= 2.0
    assert np.allclose(estimator.bias, bias, atol=1.0e-3)


def test_discards_window_with_motion_then_locks_on_quiet_one():
    rng = np.random.default_rng(1)
    bias = np.array([0.0, 0.0, 0.006])
    times = np.arange(0.0, 5.0, 0.005)
    samples = bias + rng.normal(0.0, 0.005, size=(times.size, 3))
    # Someone rotates the car during the first window.
    moving = times < 1.5
    samples[moving, 2] += 0.5

    estimator = GyroBiasEstimator(capture_sec=2.0, still_std=0.03)
    feed(estimator, times, samples)

    assert estimator.locked
    # The motion must not contaminate the locked bias.
    assert abs(estimator.bias[2] - bias[2]) < 2.0e-3


def test_never_locks_while_continuously_moving():
    times = np.arange(0.0, 10.0, 0.01)
    samples = np.column_stack(
        [np.zeros_like(times), np.zeros_like(times), 0.3 * np.sin(times * 3.0)]
    )
    estimator = GyroBiasEstimator(capture_sec=2.0, still_std=0.03)
    feed(estimator, times, samples)
    assert not estimator.locked
    assert np.allclose(estimator.bias, 0.0)  # bias stays at the safe zero default


def test_capture_disabled_locks_immediately():
    estimator = GyroBiasEstimator(capture_sec=0.0)
    assert estimator.locked
    assert not estimator.submit(0.0, [0.1, 0.0, 0.0])  # never re-captures
    assert np.allclose(estimator.bias, 0.0)
