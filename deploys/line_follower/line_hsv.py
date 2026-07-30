#!/usr/bin/env python3
"""Calibrate the line's HSV thresholds, and see whether both bands see it.

    python3 line_hsv.py                 # terminal only, ~0 extra CPU
    python3 line_hsv.py --http          # browser: http://10.42.0.1:8002
    python3 line_hsv.py --probe         # print the real H/S/V to copy in
    python3 line_hsv.py --mask          # stream the raw mask instead

Read it like this:

  * The two bands are drawn as boxes. A band that never turns green is a band
    that never sees the line -- for the FAR band that means the geometry is
    wrong (it is pointed at the wall or the horizon), and no amount of gain
    tuning will make the lookahead work. Move FAR_TOP down.
  * Tinted pixels are what the threshold currently accepts. They must cover the
    tape and nothing else. Tinting on the floor or the walls means the bands are
    catching background, and the contour centre will wander.
  * --probe prints the median H/S/V of the most saturated pixels inside each
    band, which is what the tape actually reads as under this lighting. Widen
    HSV_MIN/HSV_MAX around those numbers, then re-check the tint.

Hue is the number worth trusting; S and V move with exposure and with how much
of the strip is blown out, so keep their floors generous.
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# keep these identical to the follower being tuned
HSV_MIN, HSV_MAX = (90, 80, 60), (120, 255, 255)
NEAR_TOP, FAR_TOP, BAND = 400, 300, 80
MIN_AREA = 150

sys.path.insert(0, "../../library")


def band_of(image, top):
    return image[top:top + BAND]


def mask_of(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(HSV_MIN, np.uint8),
                       np.array(HSV_MAX, np.uint8))


def probe(bgr):
    """Median H/S/V of the most saturated 5% of a band -- i.e. of the tape."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    hot = s >= np.percentile(s, 95)
    return {k: int(np.median(hsv[:, :, i][hot]))
            for i, k in enumerate("HSV")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = forever")
    args = ap.parse_args()

    import racecar_core
    import racecar_utils as rc_utils
    import rclpy

    rc = racecar_core.create_racecar(isSimulation=False)
    # racecar_core only spins its executor inside rc.go(), which this tool never
    # calls, so pump it here or no camera callback ever fires.
    executor = rclpy.get_global_executor()

    frame_state, lock = {"jpeg": None}, threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        buf = frame_state["jpeg"]
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
        threading.Thread(
            target=HTTPServer(("0.0.0.0", args.port), Handler).serve_forever,
            daemon=True).start()
        print(f"watch it at  http://10.42.0.1:{args.port}")

    print(f"HSV_MIN={HSV_MIN}  HSV_MAX={HSV_MAX}  "
          f"NEAR_TOP={NEAR_TOP} FAR_TOP={FAR_TOP} BAND={BAND}")
    seen = {"near": 0, "far": 0, "n": 0}
    last_id, last_line, t0 = None, "", time.time()

    while args.seconds <= 0 or time.time() - t0 < args.seconds:
        executor.spin_once(timeout_sec=0.05)
        image = rc.camera.get_color_image_async()
        if image is None or id(image) == last_id:
            continue
        last_id = id(image)
        seen["n"] += 1

        view = image.copy() if (args.http and not args.mask) else None
        report, masks = [], {}
        for name, top in (("near", NEAR_TOP), ("far", FAR_TOP)):
            crop = band_of(image, top)
            mask = mask_of(crop)
            masks[name] = mask
            contour = rc_utils.get_largest_contour(
                cv2.findContours(mask, cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_SIMPLE)[0], MIN_AREA)
            hit = contour is not None
            seen[name] += hit
            if hit:
                cx = int(cv2.moments(contour)["m10"]
                         / max(cv2.moments(contour)["m00"], 1))
                area = int(cv2.contourArea(contour))
                report.append(f"{name} col {cx:4d} area {area:5d}")
            else:
                report.append(f"{name}  --- no line ---")
            if view is not None:
                view[top:top + BAND][mask > 0] = (0, 0, 255)
                cv2.rectangle(view, (0, top), (view.shape[1] - 1, top + BAND),
                              (0, 220, 0) if hit else (120, 120, 120), 2)
                if hit:
                    cv2.circle(view, (cx, top + BAND // 2), 6, (255, 255, 0), -1)

        line = "  |  ".join(report)
        if args.probe:
            line += ("   near " + str(probe(band_of(image, NEAR_TOP)))
                     + "  far " + str(probe(band_of(image, FAR_TOP))))
        if line != last_line:
            print(line)
            last_line = line

        if args.http:
            if args.mask:
                stack = np.vstack([masks["far"], masks["near"]])
                view = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with lock:
                    frame_state["jpeg"] = buf.tobytes()
        if args.probe:
            time.sleep(0.2)

    n = max(seen["n"], 1)
    print(f"\n{seen['n']} frames:  near hit {100 * seen['near'] / n:.0f}%   "
          f"far hit {100 * seen['far'] / n:.0f}%")
    print("far below ~30% => move FAR_TOP down; the lookahead is blind there")


if __name__ == "__main__":
    main()
