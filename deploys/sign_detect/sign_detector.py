#!/usr/bin/env python3
"""Edge TPU sign detector for RACECAR — deliberately CPU-cheap.

The car already runs a full ROS stack (~3 of 4 cores busy), so this module
avoids PyTorch/Ultralytics entirely: only tflite_runtime + libedgetpu + numpy.
All convolutions run on the Coral chip; the CPU does one resize and a NMS over
a handful of candidates.

Cost-control measures, in order of impact:
  1. No torch, no ultralytics  -> no heavy imports, no hidden thread pools.
  2. Class whitelist applied BEFORE dequantisation, in int8 space, so the
     score threshold prunes 2100 candidates down to a few with one compare.
  3. num_threads=1 on the interpreter — the TPU does the work, extra CPU
     threads would only steal cycles from the control loop.
  4. Optional `every_n` frame skipping and `nice`/`taskset` helpers.

Usage:
    from sign_detector import SignDetector
    det = SignDetector("best_v5_edgetpu.tflite", classes=(3, 7, 8))
    for cls_id, conf, (x1, y1, x2, y2) in det.detect(frame_bgr):
        ...
"""

import os
import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

EDGETPU_LIB = "libedgetpu.so.1"

CLASS_NAMES = [
    "DO_NOT_ENTER", "DO_NOT_GO_AROUND", "FAKE_STOP", "GO_AROUND", "NOT_YIELD",
    "ONE_WAY_(left)", "ONE_WAY_(right)", "STOP", "YIELD",
]


def limit_cpu(core: int | None = None, niceness: int = 10) -> None:
    """Keep inference off the control loop's back.

    `niceness` > 0 means the scheduler prefers ROS nodes whenever both are
    runnable. Pinning to a single core (`core`) additionally guarantees the
    detector can never spread across the cores the control stack needs.
    """
    try:
        os.nice(niceness)
    except OSError:
        pass
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
        except OSError:
            pass


class SignDetector:
    def __init__(self, model_path, classes=(3, 7, 8), conf=0.5, iou=0.45,
                 every_n=1, core=None, niceness=10):
        limit_cpu(core, niceness)
        self.interp = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(EDGETPU_LIB)],
            num_threads=1,
        )
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]
        self.in_idx, self.out_idx = inp["index"], out["index"]
        self.in_dtype = inp["dtype"]
        self.size = int(inp["shape"][1])
        self.in_scale, self.in_zero = inp["quantization"]
        self.out_scale, self.out_zero = out["quantization"]

        self.classes = tuple(classes)
        self.conf = conf
        self.iou = iou
        self.every_n = max(1, int(every_n))
        self._frame = 0
        self._last = []

        # score threshold pushed into int8 space: filtering happens before any
        # float math, so the common "nothing in view" frame costs almost nothing
        self._q_thresh = (conf / self.out_scale + self.out_zero
                          if self.out_scale else conf)

    # -- pre/post ---------------------------------------------------------
    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        idx = scores.argsort()[::-1]
        keep = []
        while idx.size:
            i = idx[0]
            keep.append(i)
            if idx.size == 1:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = ((boxes[rest, 2] - boxes[rest, 0])
                      * (boxes[rest, 3] - boxes[rest, 1]))
            idx = rest[inter / (area_i + area_r - inter + 1e-9) <= iou_thr]
        return keep

    # -- public API -------------------------------------------------------
    def detect(self, frame_bgr):
        """Return [(class_id, confidence, (x1, y1, x2, y2)), ...] in frame px."""
        self._frame += 1
        if self._frame % self.every_n:
            return self._last

        img, ratio, pad_x, pad_y = self._letterbox(frame_bgr)
        img = img[:, :, ::-1]                       # BGR -> RGB
        if self.in_dtype == np.int8:
            x = (img.astype(np.int16) - 128).astype(np.int8)
        elif self.in_dtype == np.uint8:
            x = img
        else:                                        # float model fallback
            x = (img.astype(np.float32) / 255.0)
        self.interp.set_tensor(self.in_idx, x[None])
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out_idx)[0]      # (13, 2100)

        cls_rows = raw[4 + np.asarray(self.classes, np.int32)]  # (n_cls, 2100)
        best = cls_rows.max(axis=0)
        hit = np.flatnonzero(best >= self._q_thresh)           # int8-space cut
        if hit.size == 0:
            self._last = []
            return self._last

        scores = ((best[hit].astype(np.float32) - self.out_zero) * self.out_scale
                  if self.out_scale else best[hit].astype(np.float32))
        cls_ids = np.asarray(self.classes)[cls_rows[:, hit].argmax(axis=0)]

        box = raw[:4, hit].astype(np.float32)
        if self.out_scale:
            box = (box - self.out_zero) * self.out_scale
        if box.max() <= 2.0:                        # normalised -> input px
            box *= self.size
        cx, cy, bw, bh = box
        xyxy = np.stack([cx - bw / 2, cy - bh / 2,
                         cx + bw / 2, cy + bh / 2], axis=1)

        h, w = frame_bgr.shape[:2]
        xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - pad_x) / ratio, 0, w)
        xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - pad_y) / ratio, 0, h)

        out = []
        for c in np.unique(cls_ids):                # per-class NMS
            m = np.flatnonzero(cls_ids == c)
            for k in self._nms(xyxy[m], scores[m], self.iou):
                j = m[k]
                out.append((int(c), float(scores[j]),
                            tuple(xyxy[j].round().astype(int))))
        out.sort(key=lambda d: -d[1])
        self._last = out
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="benchmark the Coral detector")
    ap.add_argument("--model", default="best_v5_edgetpu.tflite")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    det = SignDetector(args.model, core=args.core)
    frame = (cv2.imread(args.image) if args.image
             else np.random.randint(0, 255, (480, 640, 3), np.uint8))

    det.detect(frame)                                # warm up
    t_cpu0, t0 = time.process_time(), time.perf_counter()
    for _ in range(args.n):
        res = det.detect(frame)
    wall = (time.perf_counter() - t0) / args.n
    cpu = (time.process_time() - t_cpu0) / args.n
    print(f"input {det.size}px | {wall * 1000:.1f} ms/frame ({1 / wall:.0f} FPS)")
    print(f"CPU time {cpu * 1000:.1f} ms/frame -> {cpu / wall * 100:.0f}% of one core")
    print(f"last detections: {[(CLASS_NAMES[c], round(s, 2)) for c, s, _ in res]}")
