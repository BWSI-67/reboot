"""wf_lite — lidar-only robust racer: model twin + scan carry-forward + ets steering.

The ablation-trimmed wf_distill. Every part that an 8-scenario ablation grid
(ideal / realism / unseen maps / perturbed vehicles) showed to buy nothing was
removed; every survivor is load-bearing with measured evidence:

  KEPT    wedge recovery   (removing it: 5/8 -> 3/8 finishes)
  KEPT    brake feedforward (removing it: 5/8 -> 3/8)
  KEPT    blind-travel governor (3 lines; a mistuned trust once cost a DNF)
  OPPORTUNISTIC sensors (2026-07-31): gyro and wheel speed are used as
          CORRECTIONS to the twin when present and sane, and silently skipped
          otherwise — the lidar remains the only REQUIRED sensor. Rationale:
          a measured omega bypasses the steering-map/wheelbase/understeer
          model, a measured v bypasses the throttle map — the two model-error
          axes the vehicle-perturbation sweep actually found DNFs on.
  CUT     understeer-feasibility cap (the lateral cap covers it)
  KEPT    anti-stall creep floor — cutting it looked safe under tuned params,
          then default params parked the car mid-wall forever ("stuck"): the
          brake law can zero the speed somewhere the wedge never fires
  FIXED   tau_pred = 0.15 s, blind_trust = 1.8 m (constants, not knobs: the
          first tunes flat, the second is invisible to an IDEAL tuner and
          must never be left for one to random-walk)

Three ideas, one sentence each:
  1. SIMULATE YOUR OWN CAR — commands take 0.10-0.12 s to land; a replica
     (delay buffers -> measured static maps -> first-order lags -> bicycle
     with understeer) always knows the pose the next command executes in.
  2. CARRY THE OLD SCAN FORWARD — the 7 Hz zero-order-held scan is detected
     byte-identically; the last real point cloud is rigid-body warped into
     the predicted pose, so steering never looks at a stale frame.
  3. BRAKE FOR THE ARC YOU CHOSE — v = min(stop-within-measured-free-length,
     lateral-accel cap), decayed once blind dead-reckoning exceeds trust.

Steering is verbatim rb_ets: windowed openness per side, blend the two most
open bearings by openness difference, angle = blend * kp.

7 knobs, every one observable by a tuner and meaningful by hand:
    window_deg / ray_width_deg / weight_range_m / kp     the ets steering four
    half_width                                    swath half-width for braking
    a_brake / a_lat                               how late to brake / how fast to corner
"""

import math

import numpy as np

# ----------------------------------------------------------------- geometry
N = 720
ARES = 2.0 * math.pi / N
IDX = np.arange(N)
PHI = -(((IDX + N // 2) % N) - N // 2) * ARES     # CCW-positive bearing
COS_P = np.cos(PHI)
SIN_P = np.sin(PHI)
FRONT_M = np.abs(PHI) < math.radians(22.0)        # wedge window
CONE_M = np.abs(PHI) < math.radians(15.0)         # anti-stall cone

# ----------------------------------------------------------- plant constants
L = 0.225
K_US = 0.77         # measured on-car 2026-08-01 (venue, re-provisioned car,
                     # auto_calib PHASE=steer, gyro /imu/primary). Jointly fit
                     # with the steer table below: low-speed steering is ~1.8x
                     # the July-31 numbers, but at 2.5 m/s the effective yaw
                     # per command is nearly unchanged (bigger delta offset by
                     # bigger 1+K v^2).
SPD_U = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
SPD_V = np.array([0.0, 1.94, 2.51, 3.0, 4.12, 4.5])
STE_S = np.array([0.0, 0.25, 0.5, 1.0])
STE_D = np.array([0.0, 0.43, 0.55, 0.79])   # 2026-08-01 venue calib; the 0.5
                                             # cell failed its fit gate and is
                                             # interpolated between neighbours
SPEED_DELAY = 0.10
STEER_DELAY = 0.184   # venue calib 2026-08-01 (was 0.12)
TAU_UP = 0.373
TAU_DOWN = 0.247
TAU_STEER = 0.10
LIDAR_LATENCY = 0.06

# ------------------------------------------------------------- fixed tunings
RMAX = 10.0          # m, range clip
V_MAX = 4.0          # m/s, user cap (plant ceiling is 4.5)
V_CREEP = 1.40       # m/s anti-stall floor while the front cone is open
CREEP_D0 = 0.30      # m of cone clearance where the floor starts
CREEP_SPAN = 0.45    # m over which the floor ramps to V_CREEP
TAU_PRED = 0.15      # s, prediction horizon (ablation: tunes flat)
BLIND_TRUST = 1.8    # m of blind travel at full speed (NOT a knob on purpose)
BLIND_STOP = 3.0     # m: beyond this the scan is not stale, it is DEAD. Measured
                     # on-car 2026-07-31: a mid-run lidar freeze left blind
                     # climbing to 31 m while the 0.9 m/s crawl floor kept the
                     # car creeping into the unknown. Stale data degrades;
                     # dead sensors stop.
T_REACT = 0.15       # s, reaction time inside the brake law
NOSE = 0.26          # m, stop margin ahead of the lidar
BRAKE_W = 0.52       # brake swath width, fraction of half_width. 0.62 phantom-
                     # braked on-car: transient wall-pointing during small
                     # corrections kept clipping the swath at speed (2026-07-31)
BRAKE_KAP = 0.40     # fraction of commanded curvature the swath sweeps
S_SKIP = 0.25        # m of arc the swath ignores (apex-graze guard)
ANG_CAP = 2.0        # rad of arc beyond which the plan is stale
SLEW = 0.13          # per-tick steering slew (normalized)
U_SLEW = 0.30        # per-tick throttle slew
BRAKE_FF = 1.2       # over-speed feedforward (ablation: load-bearing)
U_MIN = -0.40        # deepest braking throttle
WEDGE_DIST = 0.35    # m; the wedge recovery is load-bearing (5/8 -> 3/8)
WEDGE_HOLD = 0.5     # s blocked before reversing
REV_T = 0.34         # s of reverse
REV_U = -0.32        # reverse throttle
RECOV_COOL = 1.30    # s after a reverse ...
RECOV_WIN = 70.0     # ... with the steering window capped (no U-turns)
GYRO_W = 0.25        # gyro weight in the yaw complementary filter
GYRO_KB = 0.004      # gyro bias tracker gain per tick
BIAS0 = 0.16         # rad/s, sim-injected mean yaw bias (real /attitude: ~0)
V_BLEND = 0.20       # per-tick blend toward a sane measured wheel speed
V_MEAS_LO = 0.30     # m/s, below this a sensorless encoder reads garbage
V_MEAS_HI = 6.0

PARAM_SPEC = {
    "window_deg":     {"default": 110.0, "min": 60.0,  "max": 120.0},
    "ray_width_deg":  {"default": 25.0,  "min": 8.0,   "max": 40.0},
    "weight_range_m": {"default": 1.50,  "min": 0.40,  "max": 4.00},
    "kp":             {"default": 0.02,  "min": 0.004, "max": 0.06},
    "half_width":     {"default": 0.21,  "min": 0.13,  "max": 0.36},
    "a_brake":        {"default": 13.0,  "min": 2.0,   "max": 20.0},
    "a_lat":          {"default": 21.0,  "min": 3.0,   "max": 32.0},
    "wall_rep":       {"default": 0.0,   "min": 0.0,   "max": 2.5},
    "u_floor":        {"default": 0.0,   "min": 0.0,   "max": 0.9},
    "spd_scale":      {"default": 1.0,   "min": 0.30,  "max": 1.0},
}
ALGO_FAMILY = "wf_lite"


def create_controller(params, seed):
    return WfLite(params)


# mirrors rc.drive.set_max_speed(spd_scale): the plant sees u * spd_scale.
# Keeping it in the maps keeps the twin honest while the COMMAND (what the
# student HUD shows) stays in the high band the driver asked for.
SPD_SCALE = 1.0
REP_LP = 0.35        # wall-repulsion low-pass blend per tick

def _speed_map(u):
    return math.copysign(float(np.interp(abs(u) * SPD_SCALE, SPD_U, SPD_V)), u)


def _speed_inv(v):
    return math.copysign(
        float(np.interp(abs(v), SPD_V, SPD_U)) / max(SPD_SCALE, 1e-6), v)


def _steer_map(s):
    return math.copysign(float(np.interp(abs(s), STE_S, STE_D)), s)


class WfLite:
    def __init__(self, p):
        self.p = p
        global SPD_SCALE
        SPD_SCALE = float(p.get("spd_scale", 1.0))
        self.reset()

    def reset(self):
        self.uh = [0.0]
        self.ah = [0.0]
        self.v = 0.0
        self.delta = 0.0
        self.x = self.y = self.th = 0.0
        self.pose_hist = [(0.0, 0.0, 0.0)] * 16
        self.bias = BIAS0
        self.prev_raw = None
        self.scan_m = None
        self.cap_pose = (0.0, 0.0, 0.0)
        self.blind = 0.0
        self.prev_angle = 0.0
        self.rep_f = 0.0
        self.prev_u = 0.0
        self.wedge_t = 0.0
        self.rev_t = 0.0
        self.rev_dir = 1.0
        self.cool = 0.0

    # -------------------------------------------------- module 1: plant twin
    @staticmethod
    def _step(v, delta, x, y, th, u, s_cmd, dt):
        v_ss = _speed_map(u)
        tau = TAU_UP if abs(v_ss) > abs(v) else TAU_DOWN
        v = v + (v_ss - v) * (1.0 - math.exp(-dt / tau))
        d_ss = _steer_map(-s_cmd)            # sim negates the student angle
        delta = delta + (d_ss - delta) * (1.0 - math.exp(-dt / TAU_STEER))
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        w = v * math.tan(delta) / (L * (1.0 + K_US * v * v))
        th += w * dt
        return v, delta, x, y, th, w

    @staticmethod
    def _hist(h, back):
        i = len(h) - 1 - back
        return h[i] if i >= 0 else h[0]

    # ------------------------------------------------------------- main tick
    def update(self, scan_cm, dt, imu):
        p = self.p
        if dt <= 0.0:
            dt = 1.0 / 60.0
        ks = max(1, int(round(SPEED_DELAY / dt)))
        ka = max(1, int(round(STEER_DELAY / dt)))
        kl = max(1, int(round(LIDAR_LATENCY / dt)))

        # advance the twin to "now" with the commands that are landing now
        self.v, self.delta, self.x, self.y, self.th, w_mod = self._step(
            self.v, self.delta, self.x, self.y, self.th,
            self._hist(self.uh, ks), self._hist(self.ah, ka), dt)

        # opportunistic corrections: measured omega bypasses the steering
        # model, measured v bypasses the throttle map. Absent or insane
        # readings fall through to the pure model (the required-sensor set
        # stays: lidar only).
        try:
            gy = float(imu["gyro"][1])           # sim convention: -yaw_ccw + bias
            self.bias += GYRO_KB * (gy - (self.bias - w_mod))
            w_fus = (1.0 - GYRO_W) * w_mod + GYRO_W * (self.bias - gy)
            self.th += (w_fus - w_mod) * dt
        except Exception:
            pass
        try:
            vm = float(imu["speed"])
            if V_MEAS_LO < vm < V_MEAS_HI:
                self.v += V_BLEND * (vm - self.v)
        except Exception:
            pass

        self.pose_hist.append((self.x, self.y, self.th))
        if len(self.pose_hist) > 16:
            del self.pose_hist[0]

        # ------------------------------------------- module 2: carry forward
        raw = np.asarray(scan_cm, dtype=np.float64)
        fresh = self.prev_raw is None or not np.array_equal(raw, self.prev_raw)
        if fresh:
            self.prev_raw = raw.copy()
            m = raw * 0.01
            self.scan_m = np.where((m > 0.05) & (m < RMAX + 2.0), m, 0.0)
            self.cap_pose = self.pose_hist[max(0, len(self.pose_hist) - 1 - kl)]
            self.blind = 0.0
        else:
            self.blind += abs(self.v) * dt

        if self.scan_m is None or not np.any(self.scan_m > 0.05):
            self.uh.append(0.25)
            self.ah.append(0.0)
            self._trim()
            return 0.25, 0.0

        # predict the pose the next command will execute in; warp into it
        v_f, d_f, x_f, y_f, th_f = self.v, self.delta, self.x, self.y, self.th
        for j in range(int(TAU_PRED / dt)):
            v_f, d_f, x_f, y_f, th_f, _ = self._step(
                v_f, d_f, x_f, y_f, th_f,
                self._hist(self.uh, max(0, ks - 1 - j)),
                self._hist(self.ah, max(0, ka - 1 - j)), dt)
        cx, cy, cth = self.cap_pose
        ct0, st0 = math.cos(cth), math.sin(cth)
        gx, gy = x_f - cx, y_f - cy
        rr, qx, qy = self._warp(self.scan_m,
                                ct0 * gx + st0 * gy,
                                -st0 * gx + ct0 * gy,
                                th_f - cth)

        # ---------------------------------------- module 3a: rb_ets steering
        win_deg = float(p["window_deg"])
        if self.cool > 0.0:
            self.cool -= dt
            win_deg = min(win_deg, RECOV_WIN)   # no U-turn hunting after a reverse
        w = max(int(float(p["ray_width_deg"]) * N / 360), 2)
        vals = np.where(rr > 0.05, rr, 0.0)
        mask = (vals > 0).astype(float)
        k = np.ones(w)
        s = np.convolve(np.r_[vals, vals[:w]], k, "valid")[:N]
        c = np.convolve(np.r_[mask, mask[:w]], k, "valid")[:N]
        avg = np.roll(np.where(c > 0, s / np.maximum(c, 1), 0.0), w // 2)

        win = int(win_deg * N / 360)
        right = avg[:win]
        left = avg[N - win:]
        ri = int(right.argmax())
        li = int(left.argmax())
        shift = float(np.clip((float(right[ri]) - float(left[li]))
                              / float(p["weight_range_m"]), -0.5, 0.5))
        blend = (ri * 360.0 / N * (0.5 + shift)
                 - (win - li) * 360.0 / N * (0.5 - shift)) / 2
        angle = float(np.clip(blend * float(p["kp"]), -1.0, 1.0))
        # optional lateral wall repulsion (ported from R165): 1/d centring pulls
        # the car off the deep outside sweeps that pure openness-following
        # loves in wide sections. ~0 in a centred corridor, grows only when a
        # flank gets close. wall_rep=0 disables (exactly the old behaviour).
        wrep = float(p.get("wall_rep", 0.0))
        if wrep > 0.0:
            beside = (qx > -0.30) & (qx < 0.85)
            by = qy[beside]
            pl = by[by > 0.0]
            pr = by[by < 0.0]
            dl = float(pl.min()) if pl.size else 2.0
            dr = float(-pr.max()) if pr.size else 2.0
            hwd0 = float(p["half_width"])
            # student convention: angle + = RIGHT. Left wall close (dl small)
            # must push RIGHT (+), right wall close must push LEFT (-).
            rep = wrep * hwd0 * (1.0 / max(dl, 0.5 * hwd0)
                                 - 1.0 / max(dr, 0.5 * hwd0))
            # width gate: centring only helps in WIDE sections (the pockets it
            # was added for). In a narrow corridor the nearest wall at turn-in
            # is the apex wall, so ungated repulsion fights the ets turn — late,
            # twisty entries. Fade in from 4x to 6x half_width local width.
            gate = min(max((dl + dr - 4.0 * hwd0) / (2.0 * hwd0), 0.0), 1.0)
            # low-pass: dl/dr are box minima, single-point noise otherwise
            # lands straight on the servo
            self.rep_f += REP_LP * (gate * rep - self.rep_f)
            angle = float(np.clip(angle + self.rep_f, -1.0, 1.0))
        angle = self.prev_angle + max(-SLEW, min(SLEW, angle - self.prev_angle))

        # -------------------------------------- module 3b: physics speed law
        kappa = math.tan(_steer_map(-angle)) / (L * (1.0 + K_US * v_f * v_f))
        d_free = max(self._swath(qx, qy, BRAKE_KAP * kappa,
                                 BRAKE_W * float(p["half_width"])) - NOSE, 0.0)
        ab = float(p["a_brake"])
        v_brake = -ab * T_REACT + math.sqrt((ab * T_REACT) ** 2 + 2.0 * ab * d_free)
        v_lat = math.sqrt(float(p["a_lat"]) / max(abs(kappa), 1e-3))
        v_tgt = min(V_MAX, v_brake, v_lat)
        # anti-stall: at v = 0 the bicycle model cannot yaw at all
        cone = rr[CONE_M]
        cone = cone[cone > 0.05]
        d_cone = float(np.percentile(cone, 25)) if cone.size else RMAX
        floor = V_CREEP * min(max((d_cone - CREEP_D0) / CREEP_SPAN, 0.0), 1.0)
        v_tgt = max(v_tgt, min(floor, V_MAX))
        if self.blind > BLIND_TRUST:            # dead-reckoning trust decay
            # reference the ACHIEVABLE top speed, not the absolute V_MAX:
            # under spd_scale the governor was never binding (4.0-based cap
            # sat above the whole reachable band) and blind corners crashed.
            v_tgt = min(v_tgt, max(0.9, _speed_map(1.0) * BLIND_TRUST / self.blind))
        if self.blind > BLIND_STOP:             # sensor is dead, not stale
            v_tgt = 0.0

        # ------------------------------------------------------ emit + wedge
        over = self.v - v_tgt
        u = _speed_inv(v_tgt - BRAKE_FF * over) if over > 0.35 else _speed_inv(v_tgt)
        u = max(U_MIN, min(1.0, u))
        # race-day floor: the throttle command never drops below u_floor.
        # Deliberately overrides the brake law (steering must carry the
        # corners); the blind-stop governor and wedge reverse still win.
        # Single escape valve: when the swept arc says a wall arrives within
        # ~0.8 s, physics outranks the floor for a beat (capped at 0.45 so
        # the car keeps rolling and re-floors the moment the arc clears).
        # (floor also yields while a wedge recovery cools down — re-approaching
        # the wall you just kissed at full throttle is how cars get stuck — and
        # while the scan has gone stale beyond the dead-reckoning trust.)
        uf = float(p.get("u_floor", 0.0))
        if uf > 0.0 and self.blind <= BLIND_STOP and self.cool <= 0.0:
            u_lo = 0.45 if (d_free < v_f * 0.8
                            or self.blind > BLIND_TRUST) else uf
            u = max(u, u_lo)
        u = self.prev_u + max(-U_SLEW, min(U_SLEW, u - self.prev_u))
        angle, u = self._wedge(float(np.clip(angle, -1, 1)),
                               float(np.clip(u, -1, 1)), dt)

        self.uh.append(u)
        self.ah.append(angle)
        self._trim()
        self.prev_angle, self.prev_u = angle, u
        return u, angle

    def _trim(self):
        if len(self.uh) > 40:
            del self.uh[:20]
            del self.ah[:20]

    # ----------------------------------------------------------- scan warping
    @staticmethod
    def _warp(scan_m, dx, dy, dth):
        """Rigid-body project the capture-frame scan into the predicted frame.

        Empty bins after re-binning stay 0.0 ("no return") — the steering's
        zero-ignoring window average is the occlusion repair."""
        vmask = scan_m > 0.05
        px = scan_m[vmask] * COS_P[vmask]
        py = scan_m[vmask] * SIN_P[vmask]
        ct, st = math.cos(-dth), math.sin(-dth)
        ox, oy = px - dx, py - dy
        qx = ct * ox - st * oy
        qy = st * ox + ct * oy
        d = np.hypot(qx, qy)
        idx = np.rint(-np.arctan2(qy, qx) / ARES).astype(np.int64) % N
        out = np.full(N, np.inf)
        np.minimum.at(out, idx, d)
        out[~np.isfinite(out)] = 0.0
        return np.minimum(out, RMAX), qx, qy

    # ------------------------------------------------- swept-arc free length
    @staticmethod
    def _swath(qx, qy, kappa, half_w):
        """Arc length until the planned circular path first sweeps a point."""
        if qx.size == 0:
            return RMAX
        ahead = qx > -0.05
        if not ahead.any():
            return RMAX
        ax, ay = qx[ahead], qy[ahead]
        if abs(kappa) < 1e-3:
            m = (np.abs(ay) < half_w) & (ax > S_SKIP)
            return float(ax[m].min()) if m.any() else RMAX
        r = 1.0 / kappa
        sg = 1.0 if r > 0 else -1.0
        ra = abs(r)
        yy = ay * sg
        e = np.abs(np.hypot(ax, yy - ra) - ra)
        ang = np.arctan2(ax, ra - yy)
        ang = np.where(ang < 0.0, ang + 2.0 * math.pi, ang)
        s = ra * ang
        m = (e < half_w) & (s > S_SKIP) & (ang < ANG_CAP)
        if not m.any():
            return RMAX
        return float(min(s[m].min(), RMAX))

    # -------------------------------------------------------- wedge recovery
    def _wedge(self, angle, u, dt):
        if self.rev_t > 0.0:
            self.rev_t -= dt
            return self.rev_dir, REV_U
        fr = self.scan_m[FRONT_M]
        fr = fr[fr > 0.05]
        front = float(np.percentile(fr, 20)) if fr.size >= 4 else RMAX
        if front < WEDGE_DIST:
            self.wedge_t += dt
        elif front > 1.6 * WEDGE_DIST:
            self.wedge_t = 0.0
        if self.wedge_t > WEDGE_HOLD:
            self.wedge_t = 0.0
            self.rev_t = REV_T
            self.cool = RECOV_COOL
            self.rev_dir = -1.0 if self.prev_angle >= 0.0 else 1.0
            return self.rev_dir, REV_U
        return angle, u
