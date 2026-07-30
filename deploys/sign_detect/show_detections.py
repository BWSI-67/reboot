#!/usr/bin/env python3
"""Live view of what the Coral detector sees.

Three ways to watch, cheapest first:

    python3 show_detections.py                     # terminal only, ~0 extra CPU
    python3 show_detections.py --http              # browser: http://10.42.0.1:8000
    python3 show_detections.py --show              # local cv2 window (needs a display)

Frame source:
    --source 0            USB camera index (default)
    --source clip.mp4     video file
    --racecar             pull frames through racecar_core (use when the ROS
                          stack owns the RealSense)

Drawing and JPEG encoding are the only costs on top of inference, so --http
streams at a reduced rate/quality by default and stays out of the control
loop's way (pinned core + positive niceness, same as the detector).
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from sign_detector import CLASS_NAMES, SignDetector

TRIAL3A = (3, 7, 8)          # GO_AROUND / STOP / YIELD
COLORS = {3: (0, 200, 255), 7: (0, 0, 255), 8: (0, 220, 220)}

latest_jpeg = None           # shared with the HTTP thread
lock = threading.Lock()


def draw(frame, dets, fps, trigger_h):
    h = frame.shape[0]
    for cid, conf, (x1, y1, x2, y2) in dets:
        near = (y2 - y1) / h >= trigger_h
        color = COLORS.get(cid, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if near else 1)
        tag = f"{CLASS_NAMES[cid]} {conf:.2f}{' *' if near else ''}"
        cv2.putText(frame, tag, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"{fps:4.1f} FPS   * = close enough to trigger",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while True:
                with lock:
                    buf = latest_jpeg
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
        pass                                  # keep the console readable


def frames_from_racecar():
    sys.path.insert(0, "../../library")
    import racecar_core
    import rclpy

    rc = racecar_core.create_racecar(isSimulation=False)
    # racecar_core only spins its executor inside rc.go(), and this tool never
    # calls it — so pump the executor here or no ROS callback ever fires and the
    # camera stays None forever. Same reason for the _async getter: the plain
    # one returns the frame that rc's update loop promotes, and there is no
    # update loop here.
    executor = rclpy.get_global_executor()
    while True:
        executor.spin_once(timeout_sec=0.05)
        img = rc.camera.get_color_image_async()
        if img is not None:
            yield img


def frames_from_capture(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"cannot open source {source!r}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_v5_edgetpu.tflite")
    ap.add_argument("--source", default="0")
    ap.add_argument("--racecar", action="store_true")
    ap.add_argument("--http", action="store_true", help="serve MJPEG on --port")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--show", action="store_true", help="local cv2 window")
    ap.add_argument("--all-classes", action="store_true",
                    help="show all 9 classes instead of the Trial 3A three")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--trigger-h", type=float, default=0.22)
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--core", type=int, default=3)
    ap.add_argument("--jpeg-quality", type=int, default=70)
    args = ap.parse_args()

    classes = tuple(range(9)) if args.all_classes else TRIAL3A
    det = SignDetector(args.model, classes=classes, conf=args.conf,
                       every_n=args.every_n, core=args.core)

    if args.http:
        srv = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"watch it at  http://10.42.0.1:{args.port}")

    frames = (frames_from_racecar() if args.racecar
              else frames_from_capture(args.source))
    print(f"classes: {[CLASS_NAMES[c] for c in classes]}   (ctrl-c to stop)")

    fps, last_line = 0.0, ""
    try:
        for frame in frames:
            t0 = time.perf_counter()
            dets = det.detect(frame)

            if args.http or args.show:
                view = draw(frame.copy(), dets, fps, args.trigger_h)
                if args.http:
                    ok, buf = cv2.imencode(
                        ".jpg", view,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    if ok:
                        globals().__setitem__("latest_jpeg", buf.tobytes())
                if args.show:
                    cv2.imshow("signs", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            line = "  ".join(
                f"{CLASS_NAMES[c]} {s:.2f}"
                f"{'*' if (b[3] - b[1]) / frame.shape[0] >= args.trigger_h else ''}"
                for c, s, b in dets) or "-"
            if line != last_line:
                print(f"[{fps:4.1f} FPS] {line}")
                last_line = line

            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
