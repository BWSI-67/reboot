"""HSV probe: point the CENTER of the camera at a cone and read the range to use.

Run on the car (or sim), aim the center box at the cone body, and read the
printed 5th-95th percentile H/S/V. Set your threshold as roughly:
    H: lo-3 .. hi+3      S: lo-30 .. 255      V: lo-30 .. 255
Move the car / change lighting angle and confirm the H band stays stable.
"""

import sys

import cv2 as cv
import numpy as np

sys.path.insert(1, '../../library')
import racecar_core

rc = racecar_core.create_racecar()
BOX = 25            # half-size (px) of the sampling box at frame center
_t = [0.0]


def start():
    print("hsv_probe: aim the frame center at the cone body")


def update():
    _t[0] += rc.get_delta_time()
    if _t[0] < 0.5:                        # print at 2 Hz
        return
    _t[0] = 0.0
    img = rc.camera.get_color_image()
    if img is None:
        print("no image")
        return
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    patch = img[cy - BOX:cy + BOX, cx - BOX:cx + BOX]
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    lo = np.percentile(hsv, 5, axis=0).astype(int)
    hi = np.percentile(hsv, 95, axis=0).astype(int)
    med = np.median(hsv, axis=0).astype(int)
    print(f"H {lo[0]:3d}-{hi[0]:3d} (med {med[0]:3d}) | "
          f"S {lo[1]:3d}-{hi[1]:3d} (med {med[1]:3d}) | "
          f"V {lo[2]:3d}-{hi[2]:3d} (med {med[2]:3d})   "
          f"-> try ((({max(lo[0]-3,0)}, {max(lo[1]-30,40)}, {max(lo[2]-30,40)}), "
          f"({min(hi[0]+3,179)}, 255, 255)))")
    # draw the sampling box so you can see what you are measuring
    vis = img.copy()
    cv.rectangle(vis, (cx - BOX, cy - BOX), (cx + BOX, cy + BOX), (0, 255, 0), 2)
    rc.display.show_color_image(vis)


rc.set_start_update(start, update)
rc.go()
