#!/usr/bin/env python3
"""Measure what colours are actually in front of the camera, then test bands.

    python3 cone_hsv.py                  # hue histogram + per-colour report
    python3 cone_hsv.py --http           # browser: http://10.42.0.1:8003
    python3 cone_hsv.py --only ORANGE    # tint just one colour, for tuning

Point the camera at ONE cone at a time and read the row for that colour. What
you want to see is a large `px` count and a single contour whose `area` is the
cone, with `H` sitting comfortably inside the band rather than at its edge.

Hue is the number to trust. S and V drift with exposure and with how much of
the cone is blown out or in shadow, so their floors are deliberately loose --
tightening them is how you lose a cone the moment it moves into shade.

BANDS below are starting points measured on this car; copy the corrected ones
straight into the lab. Red/orange wraps around H=0, so ORANGE and RED each need
two ranges -- a single (low, high) pair cannot express a wrap.
"""

import argparse
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# OpenCV HSV: H 0-179, S 0-255, V 0-255
BANDS = {
    # S floor 80, not 120: the lit face of a cone washes toward white and its
    # saturation drops to ~95 while the hue holds. At 120 only the shaded middle
    # of the cone was accepted -- roughly half of it -- and the contour centre
    # sat on that strip instead of on the cone. A grey floor reads S < 40, so 80
    # still clears the background by a wide margin.
    "ORANGE": (((3, 80, 60), (18, 255, 255)),),      # measured H~8 S~255 V~203
    "GREEN":  (((40, 70, 60), (85, 255, 255)),),     # measured H=53 S=193 V=128
    "BLUE":   (((90, 80, 60), (120, 255, 255)),),
    "RED":    (((0, 80, 60), (3, 255, 255)), ((170, 80, 60), (180, 255, 255))),
}
DRAW = {"ORANGE": (0, 140, 255), "GREEN": (0, 220, 0),
        "BLUE": (255, 120, 0), "RED": (0, 0, 255)}

CROP_TOP = 200      # ignore the ceiling and anything above the cones
MIN_AREA = 300

sys.path.insert(0, "../../library")


def masks_for(hsv, name):
    mask = None
    for lo, hi in BANDS[name]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def hue_histogram(hsv, bins=18):
    """Which hues are present at all, among pixels saturated enough to be paint
    rather than floor or glare."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    keep = (s > 90) & (v > 70)
    if not np.any(keep):
        return []
    counts = Counter((h[keep] // (180 // bins)).tolist())
    total = sum(counts.values())
    return [(b * (180 // bins), (b + 1) * (180 // bins) - 1,
             100 * c // total, c)
            for b, c in counts.most_common(4)]


def sweep(image, name):
    """How much of the cone each S/V floor buys, and what it costs in bleed.

    `blob` is the largest connected region -- the cone. `other` is every
    accepted pixel outside it. A floor is too low the moment `other` starts
    climbing faster than `blob`: that is the floor, the walls, or a shoe.
    """
    hsv = cv2.cvtColor(image[CROP_TOP:], cv2.COLOR_BGR2HSV)
    lo, hi = BANDS[name][0]
    print(f"sweeping {name}, hue band H{lo[0]}-{hi[0]} held fixed\n")
    print(f"{'S floor':>8}{'V floor':>8}{'blob px':>10}{'other px':>10}"
          f"{'median HSV of blob':>22}")
    for s_floor in (60, 80, 100, 120, 150):
        for v_floor in (40, 60, 90):
            mask = cv2.inRange(hsv, np.array((lo[0], s_floor, v_floor), np.uint8),
                               np.array(hi, np.uint8))
            contours = cv2.findContours(mask, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
            if not contours:
                print(f"{s_floor:>8}{v_floor:>8}{0:>10}{0:>10}")
                continue
            big = max(contours, key=cv2.contourArea)
            inside = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(inside, [big], -1, 255, -1)
            blob = int(cv2.countNonZero(cv2.bitwise_and(mask, inside)))
            other = int(cv2.countNonZero(mask)) - blob
            med = [int(np.median(hsv[:, :, i][inside > 0])) for i in range(3)]
            print(f"{s_floor:>8}{v_floor:>8}{blob:>10}{other:>10}"
                  f"{f'H{med[0]} S{med[1]} V{med[2]}':>22}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--only", default=None, help="tint only this colour")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--sweep", default=None, metavar="COLOUR",
                    help="table of S/V floors vs coverage, then exit")
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="write one frame to PATH and exit")
    args = ap.parse_args()
    names = [args.only.upper()] if args.only else list(BANDS)

    import racecar_core
    import rclpy

    rc = racecar_core.create_racecar(isSimulation=False)
    # racecar_core spins its executor only inside rc.go(), which this never
    # calls, so pump it here or the camera callback never fires.
    executor = rclpy.get_global_executor()

    shared, lock = {"jpeg": None}, threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        buf = shared["jpeg"]
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

    if args.sweep or args.save:
        image = None
        while image is None:
            executor.spin_once(timeout_sec=0.05)
            image = rc.camera.get_color_image_async()
        if args.save:
            cv2.imwrite(args.save, image)
            print(f"wrote {args.save}  {image.shape}")
            return
        sweep(image, args.sweep.upper())
        return

    last_id, t0 = None, time.time()
    while args.seconds <= 0 or time.time() - t0 < args.seconds:
        executor.spin_once(timeout_sec=0.05)
        image = rc.camera.get_color_image_async()
        if image is None or id(image) == last_id:
            continue
        last_id = id(image)

        crop = image[CROP_TOP:]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        view = crop.copy() if args.http else None

        rows = []
        for name in names:
            mask = masks_for(hsv, name)
            px = int(cv2.countNonZero(mask))
            contours = cv2.findContours(mask, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
            big = max(contours, key=cv2.contourArea) if contours else None
            area = int(cv2.contourArea(big)) if big is not None else 0
            if area >= MIN_AREA:
                inside = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(inside, [big], -1, 255, -1)
                med = [int(np.median(hsv[:, :, i][inside > 0])) for i in range(3)]
                rows.append(f"{name:<7} px {px:6d} area {area:6d} "
                            f"H{med[0]:3d} S{med[1]:3d} V{med[2]:3d}")
            else:
                rows.append(f"{name:<7} px {px:6d} area {area:6d} (too small)")
            if view is not None:
                view[mask > 0] = DRAW[name]

        hist = "  ".join(f"H{lo}-{hi}:{pct}%" for lo, hi, pct, _ in
                         hue_histogram(hsv))
        print(" | ".join(rows) + ("   hues " + hist if hist else "   (no colour)"))

        if args.http:
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with lock:
                    shared["jpeg"] = buf.tobytes()
        time.sleep(0.25)


if __name__ == "__main__":
    main()
