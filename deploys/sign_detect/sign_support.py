"""Helpers for labs/sign_detect_GF.py.

The camera publishes at ~30 Hz while update() runs at 60 Hz, so racecar_core
hands out every frame roughly twice. Letting a repeat through would burn TPU
time and, worse, cast two votes from one observation — halving the K-of-N
filter's ability to reject a single-frame false positive.
"""

from collections import deque

import numpy as np

from sign_detector import CLASS_NAMES, SignDetector

__all__ = ["CLASS_NAMES", "SignDetector", "SignWatcher", "findMaxW"]


def findMaxW(scan, minA, maxA, window):
    length = scan.shape[0]

    originA = minA
    minA %= 360
    maxA %= 360

    start = int(minA * length / 360)
    end = int(maxA * length / 360)

    if start > end:
        sScan = np.append(scan[start:], scan[:end + 1])
    else:
        sScan = scan[start:end + 1]

    wSize = int(window * length / 360)
    sums = np.convolve(sScan, np.ones(wSize), "valid")
    center = int(sums.argmax()) + wSize // 2

    return originA + center * 360 / length


class SignWatcher:
    def __init__(self, model, classes, conf, trigger_h, vote_n, core=3, nice=10):
        self.detector = SignDetector(model, classes=classes, conf=conf,
                                     core=core, niceness=nice)
        self.trigger_h = trigger_h
        self.votes = deque(maxlen=vote_n)
        self.new = self.dup = 0
        self._seen = None

    def poll(self, image):
        """One vote per NEW frame: the closest sign in it, or None."""
        if image is None:
            return
        fingerprint = int(image[::40, ::40, 0].sum())   # cheaper than a hash
        if fingerprint == self._seen:
            self.dup += 1
            return
        self._seen = fingerprint
        self.new += 1
        near = [c for c, _s, box in self.detector.detect(image)
                if (box[3] - box[1]) / image.shape[0] >= self.trigger_h]
        self.votes.append(near[0] if near else None)

    def count(self, class_id):
        return sum(v == class_id for v in self.votes)

    def winner(self, k):
        """The class with at least k votes, or None. k > vote_n / 2, so at
        most one class can ever reach it."""
        for class_id in {v for v in self.votes if v is not None}:
            if self.count(class_id) >= k:
                return class_id
        return None

    def clear(self):
        self.votes.clear()

    def summary(self):
        names = [CLASS_NAMES[v][:4] if v is not None else "-" for v in self.votes]
        return f"votes={names} frames new/dup={self.new}/{self.dup}"
