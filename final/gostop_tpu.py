"""Coral GO/STOP detector toolkit -- extracted verbatim from the team's
field-tested WORKING ELEVATOR (commit c9d22e7). Model search, TPU
delegate loading with busy-holder release and diagnosis, letterboxed
YOLO decode, class-map auto-selection. No racecar dependencies.
"""


import os
import sys

# pycoral is installed system-wide on the Pi but not inside the racecar-venv.
# Prepend the Pi's dist-packages so the venv Python can still find it.
for _p in ("/usr/lib/python3/dist-packages",
           "/usr/local/lib/python3.9/dist-packages",
           "/usr/local/lib/python3.11/dist-packages"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2 as cv
import numpy as np


# ---------------- TUNING MODEL ----------------#

# Try models in order — first one found wins.
#
# gostop_v1 is FIRST on purpose. The elevator indicator is an LED TEXT PANEL
# reading "Stop" / "Go", and gostop_v1 is the only model trained on it
# (99-100% detection, median confidence ~0.89, measured on the IMG_3640-3642
# clips of this exact rig).
#
# best_v5 / sign_detector are the 9-class ROAD SIGN models -- stop, yield,
# do_not_enter, one_way_left/right, go_around and friends. Those are printed
# sign graphics. Pointed at an LED text panel they output pure noise (every
# score 1-2 quantisation steps above zero), because the panel is nothing like
# anything in their training set. They belong to the Trial 3A sign task, NOT
# to the elevator.
MODEL_FILENAMES      = ["gostop_v1_edgetpu.tflite", "best_v5_edgetpu.tflite",
                        "sign_detector_edgetpu.tflite", "sign_detector3_edgetpu.tflite"]
MODEL_FILENAME       = MODEL_FILENAMES[0]   # used in error messages only
MODEL_SEARCH_PATHS   = [".", "models", "~/gostop", "/home/racecar/models",
                        "/home/racecar/Desktop/TPU_Signs_Training"]

# Label maps, chosen AUTOMATICALLY from the loaded model's output shape. Upload
# either model without editing this file: a YOLO head emits (4 + n_classes)
# rows, so n_classes falls straight out of the tensor and picks the map.
#
# 2-class gostop_v1:
#   0=GO  1=STOP
# 9-class best_v5 / sign_detector (objects.txt order):
#   0=stop  1=fake_stop  2=yield  3=not_yield  4=do_not_enter
#   5=one_way_left  6=one_way_right  7=go_around  8=do_not_go_around
#   -> go_around(7) is the elevator GO; stop(0)/fake_stop(1) are STOP.
# Classes absent from a map are IGNORED: a yield sign in frame must not become
# the pick and stall the car at the door.
CLASS_MAPS = {
    2: {0: "GO", 1: "STOP"},
    9: {0: "STOP", 1: "STOP", 7: "GO"},
}
CLASS_NAMES = CLASS_MAPS[2]        # replaced at load time by _resolve_classes()

OBJECT_LABELS_BY_N = {
    2: ["GO", "STOP"],
    9: ["stop", "fake_stop", "yield", "not_yield", "do_not_enter",
        "one_way_left", "one_way_right", "go_around", "do_not_go_around"],
}
OBJECT_LABELS = OBJECT_LABELS_BY_N[9]


def _resolve_classes(n_classes):
    """Pick the label map for a model with this many classes.

    Returns (map, labels, note). An unknown class count is not fatal: fall back
    to the 2-class meaning so the script still runs and says what it assumed.
    """
    if n_classes in CLASS_MAPS:
        return (CLASS_MAPS[n_classes],
                OBJECT_LABELS_BY_N.get(n_classes,
                                       [str(i) for i in range(n_classes)]),
                None)
    return (CLASS_MAPS[2], [str(i) for i in range(n_classes)],
            "unknown class count {}; assuming 0=GO 1=STOP".format(n_classes))

INPUT_SIZE           = 320
SAVE_DEBUG_FRAMES    = False    # write the camera frame and the exact letterboxed
                               # tensor the model is fed, ~1/sec, beside this
                               # script. Set False once detection is confirmed.
MIN_CONF             = 0.25    # start low for diagnosis; raise to 0.55 once detections appear
MIN_BOX_AREA         = 400     # start low for diagnosis; raise to 1500 once detections appear
IOU_THRESH           = 0.45
STABLE_FRAMES        = 3


# ---------------- TUNING ELEVATOR ----------------#

# --- how close is "at the door" -------------------------------------------------
# Absolute px^2 is resolution-dependent, so this is a FRACTION OF THE FRAME and
# transfers across camera resolutions. 0.039 == the old 12000 px^2 on 640x480.
# Raise it to make the car come closer before it stops to read.
AT_DOOR_FRAC         = 0.039
AT_DOOR_AREA         = 12000   # fallback only, used if the frame size is unknown

# --- how far in is "inside" -----------------------------------------------------
# NOT an absolute area. The area measured the moment the car stopped at the door
# is recorded, and the car is inside once the sign has grown by this factor.
#
# Absolute INSIDE_AREA was the bug: if the sign's real area at the door already
# exceeded it, the very first Entering frame promoted straight to inElevator and
# the car sat still for the whole ride without moving. A ratio cannot do that --
# it is measured from wherever the door actually turned out to be.
INSIDE_AREA_RATIO    = 10     # sign must roughly double in area from the door
INSIDE_AREA          = 30000   # fallback only, if no door area was captured

# The car must actually DRIVE for this long before it is allowed to believe it
# is inside. Second guard against declaring arrival without moving.
ENTER_MIN_TIME       = 0.8     # sec

SIGN_KP              = 0.90
SIGN_KI              = 0.00
SIGN_KD              = 0.12
SIGN_MAX_STEER       = 0.85
APPROACH_SPEED       = 0.20
SIGN_LOST_TIME       = 1.0    # sec sign may vanish before Approach gives up and idles

ENTERING_SPEED       = 0.20
SLOW_ENTERING_SPEED  = 0.10
SLOW_ENTER_AREA      = 22000
ENTER_TIMEOUT        = 5.0
STOP_ON_STOP         = True
ELEVATOR_RIDE_TIME   = 10.5
EXITING_SPEED        = 0.20
EXITING_ANGLE        = -0.05
EXIT_DRIVE_TIME      = 2.2

# --- exit wait ---------------------------------------------------------------
# The LIDAR gate was dropped -- it was not reliably switching from inElevator
# to Exiting on the rig. Replaced with a plain hardcoded wait: after the ride
# countdown, sit still for EXIT_WAIT_TIME seconds, then drive out. No sensor
# in the loop, so nothing to debug except this one number.
EXIT_WAIT_TIME       = 0.5    # sec to sit still after the ride, before driving out


# ---------------- STATE ----------------#

elevator_state  = "Idle"
elevator_countdown = ELEVATOR_RIDE_TIME
ENTERING_TIMER  = 0.0
EXITING_TIMER   = 0.0
_clock          = 0.0
_sign_angle     = 0.0
_last_seen      = -999.0
_err_prev       = 0.0
_err_sum        = 0.0
_door_area      = 0.0    # sign area at the instant the car stopped at the door;
                         # INSIDE_AREA_RATIO is measured against this
_peak_area      = 0.0    # largest area seen this run, for the tuning printout

_exit_wait_timer = 0.0   # counts up to EXIT_WAIT_TIME once inElevator's ride ends

_detector       = None
_detector_warned = False


# =====================================================================
#  MODEL LOADING  (identical to TIME_TRIAL -- never raises)
# =====================================================================

def _find_model():
    env = os.environ.get("GOSTOP_MODEL")
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.expanduser(env)

    roots = []
    try:
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    roots.append(os.getcwd())

    for fname in MODEL_FILENAMES:
        for root in roots:
            for rel in MODEL_SEARCH_PATHS:
                cand = os.path.join(root, os.path.expanduser(rel), fname)
                if os.path.isfile(cand):
                    return cand
        for rel in MODEL_SEARCH_PATHS:
            cand = os.path.join(os.path.expanduser(rel), fname)
            if os.path.isfile(cand):
                return cand
    return None


# Shared-library names for the Edge TPU runtime, most-specific first.
_EDGETPU_LIBS = ("libedgetpu.so.1", "libedgetpu.so.1.0", "libedgetpu.so",
                 "libedgetpu.dylib", "edgetpu.dll")


def list_tpus():
    """Edge TPUs the system can see, or None if pycoral is unavailable."""
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        return list_edge_tpus()
    except Exception:
        return None


def _accept(interp):
    """An interpreter is only usable if allocate_tensors() succeeds.

    Constructing an Interpreter on an edgetpu model SUCCEEDS even with no
    delegate -- it only blows up at allocate. Validating here instead of at the
    call site is what stops a dead CPU interpreter being reported as a working
    backend.
    """
    interp.allocate_tensors()
    return interp


def _have_edgetpu_lib():
    """True if libedgetpu looks present. Checked before asking for a delegate
    because a failed load_delegate leaves a half-built object whose destructor
    raises, spraying tracebacks that hide the real error."""
    import ctypes.util
    if ctypes.util.find_library("edgetpu"):
        return True
    for d in ("/usr/lib/aarch64-linux-gnu", "/usr/lib/arm-linux-gnueabihf",
              "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/usr/lib",
              "/opt/homebrew/lib"):
        for lib in _EDGETPU_LIBS:
            if os.path.isfile(os.path.join(d, lib)):
                return True
    return False


# Device selector strings passed to the delegate. The default (None) asks
# libedgetpu to pick, which does NOT always find a PCIe/M.2 Coral (/dev/apex_0)
# -- naming the bus explicitly often succeeds where the default fails.
_EDGETPU_DEVICE_OPTS = (None,
                        {"device": "pci"},
                        {"device": "pci:0"},
                        {"device": "usb"},
                        {"device": "usb:0"})


AUTO_RELEASE_TPU     = True   # If the driver's own edgetpu_node is sitting on
                              # /dev/apex_0, SIGTERM it and retry the load. It
                              # respawns on every boot and cannot run this model
                              # (it needs an SSD 4-output head), so it would
                              # otherwise block every run forever. It runs as the
                              # same user as this script, so no sudo is needed.
                              # Set False to be asked to do it by hand instead.
RELEASE_ONLY_NAMES   = ("edgetpu_node",)   # never signal anything else


def _tpu_holders():
    """[(pid, command_name)] holding /dev/apex_*, excluding this process.

    Walks /proc rather than shelling out to fuser: no sudo, no dependency, and
    it yields the process NAME so nothing outside RELEASE_ONLY_NAMES is ever
    signalled.
    """
    found = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except Exception:
        return found
    me = os.getpid()
    for pid in pids:
        if int(pid) == me:
            continue
        fd_dir = "/proc/{}/fd".format(pid)
        try:
            fds = os.listdir(fd_dir)
        except Exception:
            continue                      # not ours / already gone
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except Exception:
                continue
            if target.startswith("/dev/apex"):
                try:
                    name = open("/proc/{}/comm".format(pid)).read().strip()
                except Exception:
                    name = "?"
                found.append((int(pid), name))
                break
    return found


def release_tpu(timeout=6.0):
    """Ask any known-safe holder of the Edge TPU to exit. Returns a report.

    Only signals processes whose name is in RELEASE_ONLY_NAMES, so an unknown
    process holding the device is reported and left alone.
    """
    import signal
    import time as _time

    holders = _tpu_holders()
    if not holders:
        return "no process is holding the TPU"

    report = []
    for pid, name in holders:
        if name not in RELEASE_ONLY_NAMES:
            report.append("{} (pid {}) holds the TPU but is not in "
                          "RELEASE_ONLY_NAMES -- left alone".format(name, pid))
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            report.append("sent SIGTERM to {} (pid {})".format(name, pid))
        except PermissionError:
            report.append("cannot signal {} (pid {}): not owned by this user, "
                          "run: sudo kill {}".format(name, pid, pid))
            continue
        except ProcessLookupError:
            report.append("{} (pid {}) already gone".format(name, pid))
            continue

    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if not _tpu_holders():
            report.append("TPU released")
            return "; ".join(report)
        _time.sleep(0.25)
    still = _tpu_holders()
    report.append("still held by {} after {:.0f}s".format(still, timeout))
    return "; ".join(report)


def tpu_diagnosis():
    """Why the delegate could not attach. Returns a list of human-readable
    findings, most likely cause first. Pure inspection, changes nothing.

    'Failed to load delegate' from libedgetpu is a single opaque message that
    covers permissions, a busy device, and a driver mismatch -- three problems
    with three different fixes -- so it has to be narrowed down here.
    """
    out = []
    apex = [p for p in ("/dev/apex_0", "/dev/apex_1") if os.path.exists(p)]
    if apex:
        for dev in apex:
            readable = os.access(dev, os.R_OK)
            writable = os.access(dev, os.W_OK)
            try:
                st = os.stat(dev)
                import grp
                try:
                    group = grp.getgrgid(st.st_gid).gr_name
                except Exception:
                    group = str(st.st_gid)
                mode = oct(st.st_mode & 0o777)
            except Exception:
                group, mode = "?", "?"
            out.append("{} exists (mode {}, group {}), readable={} writable={}"
                       .format(dev, mode, group, readable, writable))
            if not (readable and writable):
                out.append("  --> NO ACCESS. This is almost certainly it. Fix:")
                out.append("      sudo usermod -aG apex $USER   # then log out and back in")
                out.append("      (or, to test right now:  sudo chmod 666 {} )".format(dev))
        # Anything else holding the TPU? Only one process may open it.
        try:
            import subprocess
            r = subprocess.run(["fuser", "-v"] + apex, capture_output=True,
                               text=True, timeout=5)
            holder = (r.stdout + r.stderr).strip()
            if holder and any(ch.isdigit() for ch in holder):
                out.append("  --> ANOTHER PROCESS HOLDS THE TPU. Only one may "
                           "open it at a time:")
                out.append("      " + holder.replace("\n", "\n      "))

                # Pull the PIDs out so the fix can be an exact command.
                pids = []
                for tok in holder.replace(":", " ").split():
                    if tok.isdigit():
                        pids.append(tok)
                if "edgetpu_node" in holder:
                    out.append("      This is the RACECAR driver's own ROS2 "
                               "edgetpu_node. It cannot run this model anyway --")
                    out.append("      it requires an SSD 4-output head and this "
                               "is a single-tensor YOLO head -- so it is holding")
                    out.append("      hardware it cannot use for you. Safe to stop:")
                if pids:
                    out.append("      sudo kill {}".format(" ".join(sorted(set(pids)))))
                    out.append("      sudo fuser -v /dev/apex_0   # should print "
                               "nothing, then re-run")
                if "edgetpu_node" in holder:
                    out.append("      If it respawns it is under a launch file or "
                               "service:")
                    out.append("      systemctl list-units --type=service | grep -i racecar")
        except Exception:
            out.append("  (could not run fuser to check whether the TPU is busy;")
            out.append("   if a Jupyter kernel ran this before, restart it -- a")
            out.append("   stale kernel keeps /dev/apex_0 open and blocks this one)")
    else:
        out.append("no /dev/apex_* node: PCIe Coral driver (gasket/apex) not loaded.")
        out.append("  Fix: sudo apt install gasket-dkms, then reboot.")
    return out


def _delegate_interpreter(mod_name, path, errors):
    """Build an Interpreter with the Edge TPU delegate attached.

    THIS is what actually runs an edgetpu-compiled model, and it does NOT need
    pycoral -- only tflite_runtime (or LiteRT) plus libedgetpu. pycoral is
    pinned to Python <= 3.9, so on a newer Pi image it simply will not import,
    and without this path there is no way to load the model at all.

    Each library name is tried against each device selector, because the
    default selector does not reliably find a PCIe Coral.
    """
    if not _have_edgetpu_lib():
        return None, None
    try:
        m = __import__(mod_name, fromlist=["Interpreter", "load_delegate"])
    except ImportError:
        return None, None
    Interpreter = getattr(m, "Interpreter", None)
    load_delegate = getattr(m, "load_delegate", None)
    if Interpreter is None or load_delegate is None:
        return None, None
    short = mod_name.split(".")[0]
    for lib in _EDGETPU_LIBS:
        for opts in _EDGETPU_DEVICE_OPTS:
            try:
                delegate = load_delegate(lib) if opts is None else load_delegate(lib, opts)
            except Exception as exc:
                errors.append("{} {} {}: {}".format(
                    short, lib, opts or "default", exc))
                continue
            try:
                interp = _accept(Interpreter(model_path=path,
                                             experimental_delegates=[delegate]))
                return interp, "{}+libedgetpu ({} {})".format(
                    short, lib, opts or "default")
            except Exception as exc:
                errors.append("{} {} {} allocate: {}".format(
                    short, lib, opts or "default", exc))
        # Only the first library name that actually exists is worth cycling
        # device options over; the rest are missing-file errors.
        if any("cannot open shared object" in e for e in errors[-len(_EDGETPU_DEVICE_OPTS):]):
            continue
    return None, None


def _legacy_delegate_body(mod_name, path, errors):
    """Unused: kept only so the old single-attempt path is easy to diff."""
    if not _have_edgetpu_lib():
        return None, None
    try:
        m = __import__(mod_name, fromlist=["Interpreter", "load_delegate"])
    except ImportError:
        return None, None
    Interpreter = getattr(m, "Interpreter", None)
    load_delegate = getattr(m, "load_delegate", None)
    if Interpreter is None or load_delegate is None:
        return None, None
    for lib in _EDGETPU_LIBS:
        try:
            delegate = load_delegate(lib)
        except Exception as exc:
            errors.append("{} delegate {}: {}".format(mod_name.split(".")[0], lib, exc))
            continue
        try:
            interp = _accept(Interpreter(model_path=path,
                                         experimental_delegates=[delegate]))
            return interp, "{}+libedgetpu ({})".format(mod_name.split(".")[0], lib)
        except Exception as exc:
            errors.append("{} with {}: {}".format(mod_name.split(".")[0], lib, exc))
    return None, None


def _make_interpreter(path):
    """(interpreter, backend_name, error_text). Never raises.

    Order matters. An edgetpu-compiled .tflite holds one opaque
    `edgetpu-custom-op`, so a plain CPU interpreter CANNOT run it -- it fails
    with "unresolved custom op" no matter which runtime is installed. Every TPU
    path is therefore tried before any CPU path, and the CPU path exists only
    for a plain (non-compiled) int8 model.
    """
    runtimes_found = []
    load_errors = []

    # 1. pycoral -- the classic Coral path.
    try:
        from pycoral.utils.edgetpu import make_interpreter
        runtimes_found.append("pycoral")
        try:
            return make_interpreter(path), "pycoral/edgetpu", None
        except Exception as exc:
            load_errors.append("pycoral: {}".format(exc))
    except ImportError:
        pass
    except Exception as exc:
        load_errors.append("pycoral: {}".format(exc))

    # 2. tflite_runtime / LiteRT WITH the Edge TPU delegate. Works on Python
    #    versions pycoral does not support, and is the path that matters most.
    for mod_name in ("tflite_runtime.interpreter",
                     "ai_edge_litert.interpreter",
                     "tensorflow.lite.python.interpreter"):
        interp, backend = _delegate_interpreter(mod_name, path, load_errors)
        if interp is not None:
            return interp, backend, None

    # 3. Plain CPU. SKIPPED for an edgetpu model: constructing the Interpreter
    #    SUCCEEDS and only allocate_tensors() fails, so returning it would
    #    report a healthy-looking "cpu-tflite" backend that cannot run a single
    #    frame. Failing here instead lets the error below name the real cause.
    if "edgetpu" not in os.path.basename(path).lower():
        for mod, attr in (("tflite_runtime.interpreter", "Interpreter"),
                          ("ai_edge_litert.interpreter", "Interpreter"),
                          ("tensorflow.lite.python.interpreter", "Interpreter")):
            short = mod.split(".")[0]
            try:
                m = __import__(mod, fromlist=[attr])
            except ImportError:
                continue
            if short not in runtimes_found:
                runtimes_found.append(short)
            try:
                return (_accept(getattr(m, attr)(model_path=path)),
                        "cpu-tflite (" + short + ")", None)
            except Exception as exc:
                load_errors.append("{} cpu: {}".format(short, exc))

    # The edgetpu case is checked FIRST. For such a model the CPU loop is
    # skipped, so runtimes_found stays empty and the generic "no runtime
    # installed" message would fire even when LiteRT is installed and fine --
    # sending you to fix the wrong thing. The real cause is libedgetpu.
    if "edgetpu" in os.path.basename(path).lower():
        return None, None, (
            "{} is an EDGE TPU model: its whole graph is one edgetpu-custom-op, "
            "so no CPU interpreter can run it, and libedgetpu was not loadable "
            "(found={}). Fix: apt install libedgetpu1-std, confirm the Coral is "
            "plugged in (lsusb | grep -i 'Global Unichip\\|Google'), then retry. "
            "Attempts: {}".format(
                os.path.basename(path), _have_edgetpu_lib(),
                "; ".join(load_errors) or "none"))
    if not runtimes_found and not load_errors:
        return None, None, ("no TFLite runtime installed. On the Pi: "
                            "pip install tflite-runtime, and apt install "
                            "libedgetpu1-std for the Coral.")
    return None, None, ("runtime(s) {} present but none could load {}: {}"
                        .format("/".join(runtimes_found) or "?", path,
                                "; ".join(load_errors) or "unknown"))


# =====================================================================
#  DETECTOR  (identical to TIME_TRIAL)
# =====================================================================

class GoStopDetector(object):
    def __init__(self, model_path, stable_frames=STABLE_FRAMES):
        self.model_path = model_path
        self.stable_frames = stable_frames
        self.ok = False
        self.error = None
        self.backend = None
        self._history = []
        self._stable_class = None
        try:
            self.interpreter, self.backend, err = _make_interpreter(model_path)
            if self.interpreter is None:
                self.error = err
                return
            self.interpreter.allocate_tensors()
            self._in = self.interpreter.get_input_details()[0]
            self._out = self.interpreter.get_output_details()[0]
            self._nchw = list(self._in["shape"])[1] == 3

            # Pick the label map from the model itself so either model can be
            # uploaded with no edit to this file. A YOLO head is
            # (4 + n_classes) x n_anchors, and the feature axis is always the
            # SHORTER of the two -- 6x2100 for gostop_v1, 13x2100 for the
            # 9-class model, either orientation.
            oshape = [int(v) for v in self._out["shape"]]
            dims = [d for d in oshape if d > 1] or [6]
            n_feat = min(dims) if len(dims) > 1 else dims[0]
            self.n_classes = max(1, n_feat - 4)
            global CLASS_NAMES, OBJECT_LABELS
            CLASS_NAMES, OBJECT_LABELS, note = _resolve_classes(self.n_classes)
            self.class_note = note
            self.ok = True
        except Exception as exc:
            self.error = "{}: {}".format(type(exc).__name__, exc)

    @staticmethod
    def letterbox(image, size=INPUT_SIZE):
        h, w = image.shape[:2]
        scale = min(float(size) / w, float(size) / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        out = np.full((size, size, 3), 114, np.uint8)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        out[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv.resize(
            image, (nw, nh), interpolation=cv.INTER_LINEAR)
        return out, scale, pad_x, pad_y

    @staticmethod
    def _nms(boxes, scores, iou_thresh=IOU_THRESH):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thresh]
        return keep

    @classmethod
    def decode(cls, raw, scale, pad_x, pad_y, orig_w, orig_h, min_conf=MIN_CONF):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        # Ensure shape is (num_features, num_anchors) — features first, anchors second.
        # The old check `!= 6` breaks for 9-class models (13 features); use shape compare.
        if raw.shape[0] > raw.shape[1]:
            raw = raw.T

        cx, cy, w, h = raw[0], raw[1], raw[2], raw[3]
        if cx.size and max(float(cx.max()), float(cy.max())) <= 2.0:
            cx, cy, w, h = (cx * INPUT_SIZE, cy * INPUT_SIZE,
                            w * INPUT_SIZE, h * INPUT_SIZE)

        scores = raw[4:]   # all class scores — works for 2-class AND 9-class models
        cls_ids = scores.argmax(axis=0)
        confs = scores.max(axis=0)

        keep = confs >= min_conf
        if not keep.any():
            return []
        cx, cy, w, h = cx[keep], cy[keep], w[keep], h[keep]
        cls_ids, confs = cls_ids[keep], confs[keep]

        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, orig_w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, orig_h)

        out = []
        for c in np.unique(cls_ids):
            m = cls_ids == c
            idx = np.flatnonzero(m)
            for k in cls._nms(boxes[m], confs[m]):
                i = idx[k]
                out.append((int(cls_ids[i]), float(confs[i]), tuple(boxes[i].tolist())))
        out.sort(key=lambda t: -t[1])
        return out

    @staticmethod
    def select(detections, min_area=MIN_BOX_AREA):
        survivors = [d for d in detections
                     if (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]) >= min_area
                     and d[0] in CLASS_NAMES]   # only elevator-relevant classes
        if not survivors:
            return None
        return min(survivors, key=lambda d: (d[2][1] + d[2][3]) / 2.0)

    def _invoke_raw(self, letterboxed_rgb):
        d = self._in
        if d["dtype"] == np.int8:
            s, z = d["quantization"]
            x = np.clip(np.round(letterboxed_rgb.astype(np.float32) / 255.0 / s) + z,
                        -128, 127).astype(np.int8)
        elif d["dtype"] == np.uint8:
            x = letterboxed_rgb.astype(np.uint8)
        else:
            x = letterboxed_rgb.astype(np.float32) / 255.0
        x = x[None]
        if self._nchw:
            x = np.transpose(x, (0, 3, 1, 2))
        self.interpreter.set_tensor(d["index"], np.ascontiguousarray(x))
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self._out["index"])
        if self._out["dtype"] in (np.int8, np.uint8):
            s, z = self._out["quantization"]
            raw = (raw.astype(np.float32) - z) * s
        return raw

    def detect(self, bgr_image):
        if not self.ok:
            return None, 0.0, None, 0.0
        try:
            h, w = bgr_image.shape[:2]
            lb, scale, pad_x, pad_y = self.letterbox(bgr_image)
            raw = self._invoke_raw(cv.cvtColor(lb, cv.COLOR_BGR2RGB))

            # --- raw diagnostic (printed ~once per second at 30fps) ---
            if not hasattr(self, '_diag_frame'):
                self._diag_frame = 0
            self._diag_frame += 1

            # Dump exactly what the model is fed. An all-scores-near-zero
            # output means either nothing is in view or the input is wrong,
            # and those look identical from the numbers alone -- so look at
            # the picture. Writes to the script's own folder.
            if SAVE_DEBUG_FRAMES and self._diag_frame % 30 == 1:
                try:
                    here = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    here = os.getcwd()
                cv.imwrite(os.path.join(here, "debug_camera_raw.jpg"), bgr_image)
                cv.imwrite(os.path.join(here, "debug_model_input.jpg"), lb)
                print("[diag] wrote debug_camera_raw.jpg ({}x{}) and "
                      "debug_model_input.jpg ({}x{}) -- open them: if the sign "
                      "is not clearly visible in debug_model_input.jpg, the "
                      "model is not being shown the sign."
                      .format(w, h, lb.shape[1], lb.shape[0]))
                print("[diag] input tensor: dtype={} quant={} nchw={}".format(
                    self._in["dtype"].__name__, self._in["quantization"],
                    self._nchw))
                print("[diag] output quant step = {:.5f} (scores are multiples "
                      "of this; 1-2 steps above zero IS the noise floor)"
                      .format(self._out["quantization"][0]
                              if self._out["quantization"][0] else float("nan")))

            if self._diag_frame % 30 == 1:
                r = np.asarray(raw, dtype=np.float32)
                if r.ndim == 3:
                    r = r[0]
                if r.shape[0] > r.shape[1]:
                    r = r.T
                class_scores = r[4:]            # shape: (num_classes, num_anchors)
                per_class_max = class_scores.max(axis=1)
                print("[diag] raw shape={} num_classes={} overall_max={:.3f}".format(
                    raw.shape, len(per_class_max), float(per_class_max.max())))
                # Per-class top score — tells you exactly which sign the model sees
                for ci, sc in enumerate(per_class_max):
                    if sc >= 0.05:
                        lbl = OBJECT_LABELS[ci] if ci < len(OBJECT_LABELS) else str(ci)
                        mapped = CLASS_NAMES.get(ci, "(ignored)")
                        print("[diag]   class{} {:<16} max={:.3f}  -> {}".format(
                            ci, lbl, float(sc), mapped))
                all_dets = self.decode(raw, scale, pad_x, pad_y, w, h, min_conf=0.10)
                if all_dets:
                    areas = [(d[2][2]-d[2][0])*(d[2][3]-d[2][1]) for d in all_dets]
                    print("[diag] dets@0.10conf={} max_area={:.0f} area_gate={}".format(
                        len(all_dets), max(areas), MIN_BOX_AREA))
                else:
                    print("[diag] no detections above 0.10 conf")
            # ------------------------------------------------

            pick = self.select(self.decode(raw, scale, pad_x, pad_y, w, h))
        except Exception as exc:
            self.error = "{}: {}".format(type(exc).__name__, exc)
            return None, 0.0, None, 0.0

        cls_id = None if pick is None else pick[0]
        self._history.append(cls_id)
        if len(self._history) > self.stable_frames:
            self._history.pop(0)
        if (len(self._history) == self.stable_frames
                and all(c == self._history[0] for c in self._history)):
            self._stable_class = self._history[0]

        if pick is None or self._stable_class is None:
            return None, 0.0, None, 0.0
        label = CLASS_NAMES.get(self._stable_class)
        if label is None:
            return None, 0.0, None, 0.0
        _, conf, (x1, y1, x2, y2) = pick
        return (label, (x2 - x1) * (y2 - y1), (x1 + x2) / 2.0, conf)

    def reset(self):
        self._history = []
        self._stable_class = None


def _get_detector():
    global _detector, _detector_warned
    if _detector is None:
        path = _find_model()
        if path is None:
            _detector = GoStopDetector.__new__(GoStopDetector)
            _detector.ok = False
            _detector.backend = None
            _detector.error = ("{} not found. Looked in {} relative to this file "
                               "and cwd, and at GOSTOP_MODEL."
                               .format(MODEL_FILENAME, MODEL_SEARCH_PATHS))
            _detector.reset = lambda: None
            _detector.detect = lambda img: (None, 0.0, None, 0.0)
        else:
            _detector = GoStopDetector(path, stable_frames=STABLE_FRAMES)

            # The commonest failure by far is the driver's own edgetpu_node
            # holding /dev/apex_0. It respawns every boot, so retrying after
            # releasing it is worth doing automatically rather than making
            # this a manual step before every single run.
            if (not _detector.ok and AUTO_RELEASE_TPU
                    and _tpu_holders()):
                print("[elevator test] TPU is busy -- attempting to release it")
                print("[elevator test]   {}".format(release_tpu()))
                if not _tpu_holders():
                    print("[elevator test] retrying model load")
                    _detector = GoStopDetector(path, stable_frames=STABLE_FRAMES)

        if not _detector_warned:
            _detector_warned = True
            if _detector.ok:
                print("[elevator test] model OK: {} via {}".format(path, _detector.backend))
                if "cpu" in (_detector.backend or ""):
                    print("[elevator test] WARNING: running on CPU, not Edge TPU.")
            else:
                print("[elevator test] ERROR: model not loaded -- {}".format(_detector.error))
                print("[elevator test] Sign detection is DISABLED. Car will stay idle.")
    return _detector


# =====================================================================
#  STATE MACHINE HELPERS
# =====================================================================

