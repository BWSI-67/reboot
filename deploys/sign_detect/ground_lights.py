#!/usr/bin/env python3
"""Read the coloured LED strips running along both track walls.

Deliberately shape-blind. The strips get occluded, split by shadows, cut off at
the frame edge, and on a tight turn only one of the two is visible at all --
any contour/aspect/left-right test built on top of that is one more thing that
can be wrong on the day. So this just counts coloured pixels in the lower half
of the frame and lets the loudest colour win.

What makes that safe here is what is *not* in the scene:

  * The track and walls are YELLOW, and yellow is absent from COLOR_BANDS, so
    the dominant surface in view is invisible to this detector by construction.
  * Red and green are the only colours searched, and they sit far apart in hue.
  * Signs are painted out via `detect(exclude=...)`, so a red sign at the
    trackside cannot be counted as red strip.

Two more guards, both cheap:

  * `min_px`  -- the winner needs a real amount of pixels, not a few specks.
  * `margin`  -- the winner must beat the runner-up by this factor, so a frame
                 that is genuinely ambiguous reports nothing instead of
                 flickering red/green into the voting window.

Frame-to-frame stability is the caller's job (K-of-N voting + latching in
sign_drive.py); this returns a per-frame opinion only.

Calibrate on the real track -- LED hue depends on the strip and the exposure:

    python3 ground_lights.py --racecar --probe      # numbers in the terminal
    python3 ground_lights.py --racecar --http       # http://10.42.0.1:8001
"""

import cv2
import numpy as np

# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
# An LED blows out to near-white in its core, so the saturation floor stays low
# enough to catch the coloured halo; V_min rejects paint and dull reflections.
COLOR_BANDS = {
    "RED":   (((0, 90, 110), (8, 255, 255)), ((170, 90, 110), (180, 255, 255))),
    "GREEN": (((45, 70, 100), (85, 255, 255)),),
    "BLUE":  (((95, 90, 90), (128, 255, 255)),),
    # No YELLOW entry, on purpose: the track surface and walls are yellow.
}

DRAW = {"RED": (0, 0, 255), "GREEN": (0, 220, 0), "BLUE": (255, 120, 0)}


class SideLights:
    def __init__(self, colors=("RED", "GREEN"), roi_top=0.5, min_px=150,
                 margin=2.0, scale=0.25):
        """
        roi_top  where the crop starts, as a fraction of frame height
        min_px   pixels the winning colour needs, counted at `scale`
        margin   how far ahead of the runner-up the winner must be
        scale    the crop is shrunk this much first; at 0.25 the whole thing
                 is ~10k pixels and the noise averages itself away
        """
        self.colors = tuple(colors)
        self.roi_top, self.scale = roi_top, scale
        self.min_px, self.margin = min_px, margin
        self.counts = {}                     # last frame's tally, for printing
        self.masks = {}                      # kept for the tuner's --mask view

    def detect(self, frame_bgr, exclude=()):
        """Return "RED" / "GREEN" / ... for this frame, or None."""
        y0 = int(frame_bgr.shape[0] * self.roi_top)
        s = self.scale
        roi = cv2.resize(frame_bgr[y0:], None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA)
        for x1, y1, x2, y2 in exclude:       # paint out the signs
            cv2.rectangle(roi, (int(x1 * s), int((y1 - y0) * s)),
                          (int(x2 * s), int((y2 - y0) * s)), (0, 0, 0), -1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        for name in self.colors:
            mask = None
            for lo, hi in COLOR_BANDS[name]:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            self.masks[name] = mask
            self.counts[name] = cv2.countNonZero(mask)

        ranked = sorted(self.counts.items(), key=lambda kv: -kv[1])
        top, n = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        if n < self.min_px or n < runner * self.margin:
            return None
        return top

    # -- tuning helpers ---------------------------------------------------
    def probe(self, frame_bgr):
        """What hue is actually down there? Brightest 5% of the ROI."""
        y0 = int(frame_bgr.shape[0] * self.roi_top)
        hsv = cv2.cvtColor(cv2.resize(frame_bgr[y0:], None, fx=self.scale,
                                      fy=self.scale, interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        m = v >= np.percentile(v, 95)
        return {"H": int(np.median(hsv[:, :, 0][m])),
                "S": int(np.median(hsv[:, :, 1][m])),
                "V": int(np.median(hsv[:, :, 2][m]))}

    def draw(self, frame_bgr, color):
        h, w = frame_bgr.shape[:2]
        y0 = int(h * self.roi_top)
        cv2.line(frame_bgr, (0, y0), (w, y0), (90, 90, 90), 1)
        # tint the ROI where the winning colour was found, so a bad threshold
        # is obvious at a glance instead of hiding behind a number
        if color:
            m = cv2.resize(self.masks[color], (w, h - y0),
                           interpolation=cv2.INTER_NEAREST) > 0
            frame_bgr[y0:][m] = DRAW.get(color, (255, 255, 255))
        txt = f"{color or '-'}   " + "  ".join(f"{k}:{v}" for k, v
                                               in self.counts.items())
        cv2.putText(frame_bgr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    DRAW.get(color, (200, 200, 200)), 2)
        return frame_bgr


# -- standalone tuner ------------------------------------------------------
def _main():
    import argparse
    import sys
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    ap = argparse.ArgumentParser(description="tune the trackside-strip reader")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--racecar", action="store_true",
                    help="pull frames through racecar_core")
    ap.add_argument("--probe", action="store_true",
                    help="print the HSV of the brightest pixels in the ROI")
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--roi-top", type=float, default=0.5)
    ap.add_argument("--min-px", type=int, default=150)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--mask", default=None,
                    help="stream this colour's mask instead of the image")
    args = ap.parse_args()

    det = SideLights(roi_top=args.roi_top, min_px=args.min_px, margin=args.margin)
    state = {"jpeg": None}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        buf = state["jpeg"]
                    if buf is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode()
                                     + b"\r\n\r\n" + buf + b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    if args.http:
        threading.Thread(target=HTTPServer(("0.0.0.0", args.port), H).serve_forever,
                         daemon=True).start()
        print(f"watch it at  http://10.42.0.1:{args.port}")

    if args.racecar:
        sys.path.insert(0, "../../library")
        import racecar_core
        rc = racecar_core.create_racecar(isSimulation=False)

        def frames():
            while True:
                img = rc.camera.get_color_image()
                if img is not None:
                    yield img
                time.sleep(0.01)
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit()
                               else args.source)
        if not cap.isOpened():
            sys.exit(f"cannot open source {args.source!r}")

        def frames():
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f

    last, t = "", 0.0
    for frame in frames():
        color = det.detect(frame)
        line = f"{color or '-':6} " + "  ".join(f"{k}:{v}" for k, v
                                                in det.counts.items())
        if args.probe:
            line += f"   brightest: {det.probe(frame)}"
        if line != last:
            print(line)
            last = line
        if args.http:
            if args.mask:
                view = cv2.cvtColor(det.masks.get(args.mask.upper(),
                                                  np.zeros((8, 8), np.uint8)),
                                    cv2.COLOR_GRAY2BGR)
            else:
                view = det.draw(frame.copy(), color)
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with lock:
                    state["jpeg"] = buf.tobytes()
        if args.probe:
            time.sleep(max(0.0, 0.2 - (time.perf_counter() - t)))
            t = time.perf_counter()


if __name__ == "__main__":
    _main()
