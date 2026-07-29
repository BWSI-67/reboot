"""Stationary gyro-bias capture (pure numpy, no ROS — pytest-able anywhere).

A raw MEMS gyro reads a few tenths of a deg/s even when perfectly still;
integrating that becomes unbounded yaw drift ("integration accumulator
error"). At startup, while the car sits still, average a window of gyro
samples and subtract that bias from every later reading.
"""

import numpy as np


class GyroBiasEstimator:
    """Average the gyro over a stationary startup window to get its bias.

    Collects body-frame samples until ``capture_sec`` of them are buffered;
    locks the bias only if the buffer is quiet (per-axis std below
    ``still_std`` rad/s, i.e. the car really was still). If someone bumps the
    car during capture, the noisy window is discarded and capture restarts.
    """

    def __init__(self, capture_sec=2.0, still_std=0.03):
        self.capture_sec = float(capture_sec)
        self.still_std = float(still_std)
        self.bias = np.zeros(3)
        self.locked = capture_sec <= 0.0
        self._times = []
        self._samples = []

    def submit(self, time_sec, gyro):
        """Feed one body-frame gyro sample; returns True the moment the bias
        locks so the caller can log it once."""
        if self.locked:
            return False
        self._times.append(float(time_sec))
        self._samples.append(np.asarray(gyro, dtype=float))
        if self._times[-1] - self._times[0] < self.capture_sec:
            return False
        samples = np.asarray(self._samples)
        if np.all(samples.std(axis=0) <= self.still_std):
            self.bias = samples.mean(axis=0)
            self.locked = True
            return True
        # Motion during capture: drop the window and try again.
        self._times.clear()
        self._samples.clear()
        return False
