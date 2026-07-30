"""Helpers for labs/sign_detect_GF.py.

One SignWatcher.poll() per new camera frame drives both perceptions:

  * signs, on the Edge TPU, scored into a sliding evidence window
  * the coloured wall strips, in HSV on the CPU, with the sign boxes painted
    out first -- a red DO_NOT_ENTER at the trackside reads as a red strip
    otherwise, which is why the detector is opened to ALL nine classes even
    though only three of them ever trigger an action

The camera publishes at ~30 Hz while update() runs at 60 Hz, so racecar_core
hands out every frame roughly twice. Letting a repeat through would burn TPU
time and, worse, count one observation twice in both windows.
"""

from collections import deque

import numpy as np

from ground_lights import SideLights
from sign_detector import CLASS_NAMES, SignDetector

__all__ = ["CLASS_NAMES", "SideLights", "SignDetector", "SignWatcher", "findMaxW"]


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


MIN_H = 0.10      # box shorter than this fraction of the frame: too far to be
                  # evidence at all, and small boxes are where the model's false
                  # positives live
MARGIN = 1.5      # the winner must also beat the runner-up by this factor


class SignWatcher:
    """Accumulates weighted evidence per sign class over a sliding window.

    A plain K-of-N vote over "is the box taller than trigger_h" throws away the
    two things that say most about whether a detection is real: how confident
    the model was, and how big the sign is. A 0.51 detection counted the same as
    a 0.95 one, and a sign one frame short of the height cut counted as nothing.

    Here each new frame contributes, per class,

        confidence * min(1, height / trigger_h)

    and a class fires once its sum over the window reaches `need`. That is
    simultaneously more sensitive (a sign further away still accumulates, just
    slower, instead of being discarded) and more robust (a single weak or tiny
    false positive contributes ~0.2 and dies out of the window long before
    anything reaches the threshold).
    """

    def __init__(self, model, classes, conf, trigger_h, vote_n, core=3, nice=10,
                 light_n=7, light_k=5, light_min_px=150, red_max_s=20.0):
        # ALL nine classes: filtering happens after inference in numpy, so it is
        # free, and the extra boxes are what keeps a red sign out of the strip
        # ROI. Actions still only fire for `classes`.
        self.detector = SignDetector(model, classes=tuple(range(9)), conf=conf,
                                     core=core, niceness=nice)
        self.act_on = tuple(classes)
        self.trigger_h = trigger_h
        self.window = deque(maxlen=vote_n)   # per frame: {class_id: weight}
        self.new = self.dup = 0
        self._seen = None

        self.lights = SideLights(colors=("RED", "GREEN"), min_px=light_min_px)
        self.light_votes = deque(maxlen=light_n)
        self.light_k, self.red_max_s = light_k, red_max_s
        self.red_since = None          # not None => a red strip is latching us

    def poll(self, image, now):
        """Score one NEW frame. Repeats are skipped (see the module docstring)."""
        if image is None:
            return
        fingerprint = int(image[::40, ::40, 0].sum())   # cheaper than a hash
        if fingerprint == self._seen:
            self.dup += 1
            return
        self._seen = fingerprint
        self.new += 1
        detections = self.detector.detect(image)

        height = image.shape[0]
        frame = {}
        for class_id, confidence, box in detections:
            ratio = (box[3] - box[1]) / height
            if class_id not in self.act_on or ratio < MIN_H:
                continue
            weight = confidence * min(1.0, ratio / self.trigger_h)
            # keep the strongest instance when a class appears twice in a frame
            frame[class_id] = max(frame.get(class_id, 0.0), weight)
        self.window.append(frame)

        self.light_votes.append(
            self.lights.detect(image, exclude=[b for _c, _s, b in detections]))
        self._latch(now)

    def _latch(self, now):
        """Red latches until green. A strip stays in view while the car waits in
        front of it, so 'red is gone' is not a usable release condition — green,
        or the timeout, is. Without the timeout one false red parks the car."""
        def voted(color):
            return sum(v == color for v in self.light_votes) >= self.light_k

        if self.red_since is None:
            if voted("RED"):
                self.red_since = now
                print("[light] RED -> hold")
        elif voted("GREEN"):
            print("[light] GREEN -> go")
            self.red_since = None
            self.light_votes.clear()
        elif now - self.red_since > self.red_max_s:
            print(f"[light] no green after {self.red_max_s:.0f}s -> assume false")
            self.red_since = None
            self.light_votes.clear()

    def totals(self):
        out = {}
        for frame in self.window:
            for class_id, weight in frame.items():
                out[class_id] = out.get(class_id, 0.0) + weight
        return out

    def count(self, class_id):
        """Frames in the window that saw this class — the 'still in view' test."""
        return sum(class_id in frame for frame in self.window)

    def winner(self, need):
        """Class whose accumulated evidence reaches `need` and clearly leads."""
        totals = self.totals()
        if not totals:
            return None
        ranked = sorted(totals.values(), reverse=True)
        best = max(totals, key=totals.get)
        runner_up = ranked[1] if len(ranked) > 1 else 0.0
        if totals[best] < need or totals[best] < MARGIN * runner_up:
            return None
        return best

    def clear(self):
        self.window.clear()

    def summary(self):
        totals = self.totals()
        top = " ".join(f"{CLASS_NAMES[c][:4]}:{v:.1f}"
                       for c, v in sorted(totals.items(), key=lambda kv: -kv[1])[:3])
        light = "RED-HOLD" if self.red_since else (self.light_votes[-1]
                                                   if self.light_votes else "-")
        return (f"[{top or '-'}] light={light} px={self.lights.counts} "
                f"frames new/dup={self.new}/{self.dup}")
