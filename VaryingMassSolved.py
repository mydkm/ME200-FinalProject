#!/usr/bin/env python3
"""
Falcon 9-ish 1D vertical "suicide burn" model (upright rocket, no rotation).

Sign convention:
  +z is up, so descending means v < 0.
  A "good landing" means v_touchdown close to 0 (from below).

State: y = [z, v, m]
  z: height above "ground floor" (m). We enforce z >= z_floor > 0.
  v: vertical velocity (m/s), +up
  m: mass (kg)

EOM:
  dz/dt = v
  dv/dt = (T(t,m) + F_D(v,z) - m*g)/m
  dm/dt = -T(t,m)/(Isp*g0)   (when burning & prop available)

Phases:
  1) Coast (T=0) until z hits z_burn (or z_floor)
  2) Burn (constant thrust) until z hits z_hover_start (or z_floor)
  3) Hover controller (PD) for t_hover (or until z_floor)
  4) Final approach controller (PD) until z_floor

We choose burn-start height z_burn such that v_touchdown ~ v_target (~0).

Key fixes implemented:
  - FIX 1: treat p.t_max as *duration per phase* (t_span=(t0, t0+p.t_max) for phases 2-4)
  - FIX 2: z_burn search uses bracketing/bisection when possible, otherwise falls back to
           minimizing |v_touchdown - v_target| (prevents "nonsense" when no sign change exists).
"""

from dataclasses import dataclass
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# -------------------- Parameters --------------------

@dataclass
class Params:
    # Gravity
    g: float = 9.80665
    g0: float = 9.80665

    # Vehicle / mass
    diameter_m: float = 3.7
    m_dry: float = 25_600.0
    prop_total: float = 395_700.0
    landing_prop_frac: float = 0.06

    # Engine / burn (constant thrust during burn)
    Isp_s: float = 283.0
    T_engine_max_N: float = 845_000.0
    throttle: float = 0.70
    n_engines: int = 10

    # Drag
    rho0: float = 1.225
    H: float = 8500.0
    Cd: float = 0.47

    # Integration
    # Interpreted as "max integration duration per phase" (not absolute wall-clock time).
    t_max: float = 400.0
    max_step: float = 0.05

    # Hard ground floor
    z_floor: float = 1e-3

    # Hover control
    enable_hover: bool = True
    z_hover_start: float = 30.0
    t_hover: float = 20.0
    z_ref: float = 30.0

    # PD gains
    Kp: float = 0.8
    Kd: float = 1.6

    # Engine limits for controller
    throttle_min: float = 0.0

    # Final descent controller target (after hover)
    v_land_ref: float = -0.5

    @property
    def area_m2(self) -> float:
        return np.pi * (self.diameter_m / 2.0) ** 2

    @property
    def T_const(self) -> float:
        return self.n_engines * self.throttle * self.T_engine_max_N

    @property
    def T_max(self) -> float:
        return self.n_engines * self.T_engine_max_N

    @property
    def T_min(self) -> float:
        return self.throttle_min * self.T_max

    @property
    def m0(self) -> float:
        landing_prop = self.landing_prop_frac * self.prop_total
        return self.m_dry + landing_prop


# -------------------- Atmos/Drag --------------------

def rho_air(z: float, p: Params) -> float:
    return p.rho0 * np.exp(-max(0.0, float(z)) / p.H)


def drag_force(v: float, z: float, p: Params) -> float:
    if p.Cd <= 0.0:
        return 0.0
    return -0.5 * rho_air(z, p) * p.Cd * p.area_m2 * v * abs(v)


# -------------------- Events --------------------

def touchdown_event(z_floor: float):
    def ev(t, y):
        return y[0] - z_floor
    ev.terminal = True
    ev.direction = 0.0
    return ev


def burn_start_event(z_burn: float):
    def ev(t, y):
        return y[0] - z_burn
    ev.terminal = True
    ev.direction = -1.0
    return ev


def hover_start_event(z_hover_start: float):
    def ev(t, y):
        return y[0] - z_hover_start
    ev.terminal = True
    ev.direction = -1.0
    return ev


def time_event(t_end: float):
    def ev(t, y):
        return t - t_end
    ev.terminal = True
    ev.direction = 1.0
    return ev


# -------------------- RHS builders --------------------

def make_rhs(p: Params, burning: bool):
    def rhs(t, y):
        z, v, m = y
        Fd = drag_force(v, z, p)

        if burning and (m > p.m_dry):
            T = p.T_const
            mdot = -T / (p.Isp_s * p.g0)
        else:
            T = 0.0
            mdot = 0.0

        dz = v
        dv = (T + Fd - m * p.g) / m
        dm = mdot
        return [dz, dv, dm]
    return rhs


def thrust_pd(z: float, v: float, m: float, p: Params, z_ref: float, v_ref: float) -> float:
    Fd = drag_force(v, z, p)
    a_cmd = p.Kp * (z_ref - z) + p.Kd * (v_ref - v)
    T_raw = m * (p.g + a_cmd) - Fd
    return float(np.clip(T_raw, p.T_min, p.T_max))


def make_ctrl_rhs(p: Params, z_ref_fn, v_ref_fn):
    def rhs(t, y):
        z, v, m = y
        Fd = drag_force(v, z, p)

        if m <= p.m_dry:
            T = 0.0
            mdot = 0.0
        else:
            z_ref = float(z_ref_fn(t))
            v_ref = float(v_ref_fn(t))
            T = thrust_pd(z, v, m, p, z_ref=z_ref, v_ref=v_ref)
            mdot = -T / (p.Isp_s * p.g0) if T > 0.0 else 0.0

        dz = v
        dv = (T + Fd - m * p.g) / m
        dm = mdot
        return [dz, dv, dm]
    return rhs


# -------------------- Utilities --------------------

def quick_feasibility_check(z0: float, v0: float, p: Params):
    if v0 >= 0:
        return None

    a_net = p.T_const / p.m0 - p.g
    if a_net <= 0:
        return {"ok": False, "reason": "T <= weight (no net upward decel)", "a_net": a_net}

    d_stop = (v0 * v0) / (2.0 * a_net)
    return {"ok": d_stop <= z0, "a_net": a_net, "d_stop": d_stop}


def stitch(tA, yA, tB, yB, eps=1e-12):
    if tA.size == 0:
        return tB, yB
    if tB.size == 0:
        return tA, yA
    if abs(tB[0] - tA[-1]) < eps:
        tB = tB[1:]
        yB = yB[:, 1:]
    return np.concatenate([tA, tB]), np.concatenate([yA, yB], axis=1)


def enforce_floor_on_out(out: dict, p: Params) -> dict:
    zf = float(p.z_floor)
    for k in ("y", "y_coast", "y_burn_seg", "y_hover_seg", "y_land_seg"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].size > 0:
            out[k][0] = np.maximum(out[k][0], zf)
    return out


# -------------------- Simulation --------------------

def simulate_for_zburn(z0: float, v0: float, p: Params, z_burn: float, eps: float = 1e-6):
    y0 = np.array([z0, v0, p.m0], dtype=float)
    td = touchdown_event(p.z_floor)

    # ---- Phase 1: Coast ----
    if z_burn >= z0 - eps:
        t_burn = 0.0
        y_burn0 = y0
        t_coast = np.array([])
        y_coast = np.empty((3, 0))
    else:
        bs = burn_start_event(z_burn)
        sol_coast = solve_ivp(
            fun=make_rhs(p, burning=False),
            t_span=(0.0, p.t_max),
            y0=y0,
            events=[bs, td],
            max_step=p.max_step,
            rtol=1e-8,
            atol=1e-10,
        )

        if len(sol_coast.t_events[1]) > 0:
            t_td = float(sol_coast.t_events[1][0])
            z_td, v_td, m_td = sol_coast.y_events[1][0]
            return enforce_floor_on_out({
                "status": "crash_before_burn",
                "z_burn": z_burn,
                "t_burn": None,
                "t_hover_start": None,
                "t_hover_end": None,
                "t_touchdown": t_td,
                "v_touchdown": float(v_td),
                "m_touchdown": float(m_td),
                "t": sol_coast.t,
                "y": sol_coast.y,
                "t_coast": sol_coast.t,
                "y_coast": sol_coast.y,
                "t_burn_seg": np.array([]),
                "y_burn_seg": np.empty((3, 0)),
                "t_hover_seg": np.array([]),
                "y_hover_seg": np.empty((3, 0)),
                "t_land_seg": np.array([]),
                "y_land_seg": np.empty((3, 0)),
            }, p)

        if len(sol_coast.t_events[0]) == 0:
            return None

        t_burn = float(sol_coast.t_events[0][0])
        y_burn0 = sol_coast.y_events[0][0]
        t_coast = sol_coast.t
        y_coast = sol_coast.y

    # ---- Phase 2: Burn ----
    do_hover = bool(p.enable_hover and (p.t_hover > 0.0) and (p.z_hover_start > 0.0))

    z_at_burn0 = float(y_burn0[0])
    if do_hover and (z_at_burn0 <= p.z_hover_start + eps):
        # Already below hover start: skip burn and enter hover (this is why we constrain search).
        t_hover_start = float(t_burn)
        y_hover0 = y_burn0
        t_burn_seg = np.array([])
        y_burn_seg = np.empty((3, 0))
    else:
        events_burn = [td]
        if do_hover:
            hs = hover_start_event(p.z_hover_start)
            events_burn = [hs, td]

        # FIX 1: p.t_max is a duration from t_burn
        sol_burn = solve_ivp(
            fun=make_rhs(p, burning=True),
            t_span=(t_burn, t_burn + p.t_max),
            y0=y_burn0,
            events=events_burn,
            max_step=p.max_step,
            rtol=1e-8,
            atol=1e-10,
        )

        t_burn_seg = sol_burn.t
        y_burn_seg = sol_burn.y

        if (not do_hover) and (len(sol_burn.t_events[0]) > 0):
            t_td = float(sol_burn.t_events[0][0])
            z_td, v_td, m_td = sol_burn.y_events[0][0]
            t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
            return enforce_floor_on_out({
                "status": "ok",
                "z_burn": z_burn,
                "t_burn": float(t_burn),
                "t_hover_start": None,
                "t_hover_end": None,
                "t_touchdown": t_td,
                "v_touchdown": float(v_td),
                "m_touchdown": float(m_td),
                "t": t_all,
                "y": y_all,
                "t_coast": t_coast,
                "y_coast": y_coast,
                "t_burn_seg": t_burn_seg,
                "y_burn_seg": y_burn_seg,
                "t_hover_seg": np.array([]),
                "y_hover_seg": np.empty((3, 0)),
                "t_land_seg": np.array([]),
                "y_land_seg": np.empty((3, 0)),
            }, p)

        if do_hover:
            if len(sol_burn.t_events[1]) > 0:
                t_td = float(sol_burn.t_events[1][0])
                z_td, v_td, m_td = sol_burn.y_events[1][0]
                t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
                return enforce_floor_on_out({
                    "status": "ok",
                    "z_burn": z_burn,
                    "t_burn": float(t_burn),
                    "t_hover_start": None,
                    "t_hover_end": None,
                    "t_touchdown": t_td,
                    "v_touchdown": float(v_td),
                    "m_touchdown": float(m_td),
                    "t": t_all,
                    "y": y_all,
                    "t_coast": t_coast,
                    "y_coast": y_coast,
                    "t_burn_seg": t_burn_seg,
                    "y_burn_seg": y_burn_seg,
                    "t_hover_seg": np.array([]),
                    "y_hover_seg": np.empty((3, 0)),
                    "t_land_seg": np.array([]),
                    "y_land_seg": np.empty((3, 0)),
                }, p)

            if len(sol_burn.t_events[0]) == 0:
                t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
                return enforce_floor_on_out({
                    "status": "no_touchdown",
                    "z_burn": z_burn,
                    "t_burn": float(t_burn),
                    "t_hover_start": None,
                    "t_hover_end": None,
                    "t_touchdown": None,
                    "v_touchdown": None,
                    "m_touchdown": float(y_burn_seg[2, -1]),
                    "t": t_all,
                    "y": y_all,
                    "t_coast": t_coast,
                    "y_coast": y_coast,
                    "t_burn_seg": t_burn_seg,
                    "y_burn_seg": y_burn_seg,
                    "t_hover_seg": np.array([]),
                    "y_hover_seg": np.empty((3, 0)),
                    "t_land_seg": np.array([]),
                    "y_land_seg": np.empty((3, 0)),
                }, p)

            t_hover_start = float(sol_burn.t_events[0][0])
            y_hover0 = sol_burn.y_events[0][0]

    # ---- Phase 3: Hover for t_hover ----
    t_hover_end = t_hover_start + float(p.t_hover)
    rhs_hover = make_ctrl_rhs(p, z_ref_fn=lambda t: p.z_ref, v_ref_fn=lambda t: 0.0)

    # FIX 1: p.t_max is a duration from t_hover_start
    sol_hover = solve_ivp(
        fun=rhs_hover,
        t_span=(t_hover_start, t_hover_start + p.t_max),
        y0=y_hover0,
        events=[td, time_event(t_hover_end)],
        max_step=p.max_step,
        rtol=1e-8,
        atol=1e-10,
    )

    t_hover_seg = sol_hover.t
    y_hover_seg = sol_hover.y

    if len(sol_hover.t_events[0]) > 0:
        t_td = float(sol_hover.t_events[0][0])
        z_td, v_td, m_td = sol_hover.y_events[0][0]
        t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
        t_all, y_all = stitch(t_all, y_all, t_hover_seg, y_hover_seg)
        return enforce_floor_on_out({
            "status": "ok",
            "z_burn": z_burn,
            "t_burn": float(t_burn),
            "t_hover_start": t_hover_start,
            "t_hover_end": None,
            "t_touchdown": t_td,
            "v_touchdown": float(v_td),
            "m_touchdown": float(m_td),
            "t": t_all,
            "y": y_all,
            "t_coast": t_coast,
            "y_coast": y_coast,
            "t_burn_seg": t_burn_seg,
            "y_burn_seg": y_burn_seg,
            "t_hover_seg": t_hover_seg,
            "y_hover_seg": y_hover_seg,
            "t_land_seg": np.array([]),
            "y_land_seg": np.empty((3, 0)),
        }, p)

    if len(sol_hover.t_events[1]) == 0:
        t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
        t_all, y_all = stitch(t_all, y_all, t_hover_seg, y_hover_seg)
        return enforce_floor_on_out({
            "status": "no_touchdown",
            "z_burn": z_burn,
            "t_burn": float(t_burn),
            "t_hover_start": t_hover_start,
            "t_hover_end": None,
            "t_touchdown": None,
            "v_touchdown": None,
            "m_touchdown": float(y_hover_seg[2, -1]),
            "t": t_all,
            "y": y_all,
            "t_coast": t_coast,
            "y_coast": y_coast,
            "t_burn_seg": t_burn_seg,
            "y_burn_seg": y_burn_seg,
            "t_hover_seg": t_hover_seg,
            "y_hover_seg": y_hover_seg,
            "t_land_seg": np.array([]),
            "y_land_seg": np.empty((3, 0)),
        }, p)

    # ---- Phase 4: Final approach ----
    t_land_start = float(sol_hover.t_events[1][0])
    y_land0 = sol_hover.y_events[1][0]

    rhs_land = make_ctrl_rhs(
        p,
        z_ref_fn=lambda t: p.z_floor,
        v_ref_fn=lambda t: p.v_land_ref,
    )

    # FIX 1: p.t_max is a duration from t_land_start
    sol_land = solve_ivp(
        fun=rhs_land,
        t_span=(t_land_start, t_land_start + p.t_max),
        y0=y_land0,
        events=[td],
        max_step=p.max_step,
        rtol=1e-8,
        atol=1e-10,
    )

    t_land_seg = sol_land.t
    y_land_seg = sol_land.y

    t_all, y_all = stitch(t_coast, y_coast, t_burn_seg, y_burn_seg)
    t_all, y_all = stitch(t_all, y_all, t_hover_seg, y_hover_seg)
    t_all, y_all = stitch(t_all, y_all, t_land_seg, y_land_seg)

    if len(sol_land.t_events[0]) == 0:
        return enforce_floor_on_out({
            "status": "no_touchdown",
            "z_burn": z_burn,
            "t_burn": float(t_burn),
            "t_hover_start": t_hover_start,
            "t_hover_end": t_land_start,
            "t_touchdown": None,
            "v_touchdown": None,
            "m_touchdown": float(y_land_seg[2, -1]),
            "t": t_all,
            "y": y_all,
            "t_coast": t_coast,
            "y_coast": y_coast,
            "t_burn_seg": t_burn_seg,
            "y_burn_seg": y_burn_seg,
            "t_hover_seg": t_hover_seg,
            "y_hover_seg": y_hover_seg,
            "t_land_seg": t_land_seg,
            "y_land_seg": y_land_seg,
        }, p)

    t_td = float(sol_land.t_events[0][0])
    z_td, v_td, m_td = sol_land.y_events[0][0]
    return enforce_floor_on_out({
        "status": "ok",
        "z_burn": z_burn,
        "t_burn": float(t_burn),
        "t_hover_start": t_hover_start,
        "t_hover_end": t_land_start,
        "t_touchdown": t_td,
        "v_touchdown": float(v_td),
        "m_touchdown": float(m_td),
        "t": t_all,
        "y": y_all,
        "t_coast": t_coast,
        "y_coast": y_coast,
        "t_burn_seg": t_burn_seg,
        "y_burn_seg": y_burn_seg,
        "t_hover_seg": t_hover_seg,
        "y_hover_seg": y_hover_seg,
        "t_land_seg": t_land_seg,
        "y_land_seg": y_land_seg,
    }, p)


# -------------------- Search for z_burn --------------------

def find_zburn_for_target_touchdown_speed(
    z0: float,
    v0: float,
    p: Params,
    v_target: float = 0.0,
    tol_v: float = 0.25,
    max_iter: int = 60,
):
    """
    Find z_burn in [z_lo, z0] so v_touchdown ~ v_target.

    Behavior:
      - If hover is enabled, enforce z_burn > z_hover_start (small margin) so we preserve:
            coast -> burn -> hover
        instead of:
            coast -> (already below hover_start) -> hover   [burn skipped]
      - Try sign-bracket + bisection when possible.
      - If no sign change exists (common when v_target=0), fall back to minimizing |error|.
    """
    z_lo = float(p.z_floor)

    do_hover = bool(p.enable_hover and (p.t_hover > 0.0) and (p.z_hover_start > 0.0))
    if do_hover and (z0 > p.z_hover_start):
        margin = max(1.0, 10.0 * float(p.z_floor))
        z_lo2 = max(z_lo, float(p.z_hover_start + margin))
        if z_lo2 > z_lo:
            print(
                f"[INFO] Hover enabled: constraining z_burn search to z_burn >= {z_lo2:.3f} m "
                f"(> z_hover_start={p.z_hover_start:.3f} m) to preserve coast->burn->hover."
            )
        z_lo = z_lo2

    if z0 <= z_lo:
        raise ValueError(f"z0 must be > search lower bound ({z_lo}).")

    cache = {}

    def sim(zb: float):
        k = round(float(zb), 6)
        if k in cache:
            return cache[k]
        out = simulate_for_zburn(z0, v0, p, float(zb))
        cache[k] = out
        return out

    # ----- Phase A: coarse scan -----
    samples = np.linspace(z_lo, z0, 41)
    vals = []   # [(zb, f, out), ...] where f = v_touchdown - v_target
    best = None # (err, zb, out)

    for zb in samples:
        out = sim(float(zb))
        if out is None or out.get("v_touchdown") is None:
            continue

        vt = float(out["v_touchdown"])
        f = vt - float(v_target)
        err = abs(f)

        vals.append((float(zb), f, out))
        if best is None or err < best[0]:
            best = (err, float(zb), out)

    if not vals:
        raise RuntimeError("No valid simulations in z_burn scan (check t_max / parameters).")

    if best[0] <= tol_v:
        return best[1], {"feasible": True, "reason": "tol_hit", "final_f": float(best[2]["v_touchdown"]) - v_target}

    # ----- Phase B: sign-bracket bisection if possible -----
    vals.sort(key=lambda x: x[0])
    fs = np.array([f for _, f, _ in vals], dtype=float)
    have_pos = np.any(fs >= 0.0)
    have_neg = np.any(fs <= 0.0)

    if have_pos and have_neg:
        bracket = None
        for (z1, f1, _), (z2, f2, _) in zip(vals[:-1], vals[1:]):
            if f1 == 0.0:
                return z1, {"feasible": True, "reason": "exact_hit"}
            if f1 * f2 < 0.0:
                bracket = (z1, z2)
                break

        if bracket is not None:
            lo, hi = bracket
            out_lo = sim(lo)
            out_hi = sim(hi)
            f_lo = float(out_lo["v_touchdown"]) - v_target
            f_hi = float(out_hi["v_touchdown"]) - v_target

            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                out_mid = sim(mid)
                f_mid = float(out_mid["v_touchdown"]) - v_target

                if abs(f_mid) <= tol_v:
                    return mid, {"feasible": True, "reason": "tol_hit", "final_f": f_mid}

                if f_lo * f_mid < 0.0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid

            mid = 0.5 * (lo + hi)
            out_mid = sim(mid)
            return mid, {"feasible": True, "reason": "max_iter", "final_f": float(out_mid["v_touchdown"]) - v_target}

    # ----- Phase C: no sign change -> minimize |error| locally -----
    zs = np.array([z for z, _, _ in vals], dtype=float)
    k = int(np.argmin([abs(f) for _, f, _ in vals]))

    lo = zs[max(0, k - 1)]
    hi = zs[min(len(zs) - 1, k + 1)]
    if hi <= lo:
        lo = max(z_lo, best[1] - 0.5 * (z0 - z_lo) / 40.0)
        hi = min(z0, best[1] + 0.5 * (z0 - z_lo) / 40.0)

    refine = np.linspace(lo, hi, 31)
    for zb in refine:
        out = sim(float(zb))
        if out is None or out.get("v_touchdown") is None:
            continue
        err = abs(float(out["v_touchdown"]) - v_target)
        if err < best[0]:
            best = (err, float(zb), out)

    return best[1], {
        "feasible": (best[0] <= tol_v),
        "reason": "min_abs_error_no_bracket",
        "best_err": best[0],
        "best_v_touchdown": float(best[2]["v_touchdown"]),
    }


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    d = Params()

    p = argparse.ArgumentParser(
        description="1D vertical rocket suicide-burn + hover model (with z_floor constraint).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Initial conditions / search
    p.add_argument("--z0", type=float, default=1500.0, help="Initial height (m).")
    p.add_argument("--v0", type=float, default=-250.0, help="Initial vertical velocity (m/s); descending is negative.")
    p.add_argument("--v-target", type=float, default=0.0, help="Target touchdown velocity (m/s).")
    p.add_argument("--tol-v", type=float, default=0.25, help="Touchdown velocity tolerance (m/s).")
    p.add_argument("--max-iter", type=int, default=60, help="Max bisection iterations for z_burn search.")

    # Plot options
    p.add_argument(
        "--plot-final-approach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include the final-approach (phase 4) curve in the plot.",
    )

    # Params fields (every field in Params becomes a CLI arg)
    p.add_argument("--g", type=float, default=d.g,
                   help="Gravity magnitude (m/s^2). Acceleration term is -g in dv/dt.")
    p.add_argument("--g0", type=float, default=d.g0,
                   help="Standard gravity used for Isp mass-flow conversion (m/s^2).")

    p.add_argument("--diameter-m", type=float, default=d.diameter_m, dest="diameter_m",
                   help="Rocket diameter (m) used to compute reference area for drag.")
    p.add_argument("--m-dry", type=float, default=d.m_dry, dest="m_dry",
                   help="Dry mass (kg). If m <= m_dry, engines cannot burn propellant.")
    p.add_argument("--prop-total", type=float, default=d.prop_total, dest="prop_total",
                   help="Total propellant mass (kg) for the full vehicle.")
    p.add_argument("--landing-prop-frac", type=float, default=d.landing_prop_frac, dest="landing_prop_frac",
                   help="Fraction of total prop reserved for landing (dimensionless).")

    p.add_argument("--isp-s", type=float, default=d.Isp_s, dest="Isp_s",
                   help="Specific impulse (s) used for mdot = -T/(Isp*g0).")
    p.add_argument("--t-engine-max-n", type=float, default=d.T_engine_max_N, dest="T_engine_max_N",
                   help="Maximum thrust per engine at full throttle (N).")
    p.add_argument("--throttle", type=float, default=d.throttle,
                   help="Throttle fraction (0..1) used ONLY during the constant-thrust burn phase.")
    p.add_argument("--n-engines", type=int, default=d.n_engines, dest="n_engines",
                   help="Number of engines (scales total thrust).")

    p.add_argument("--rho0", type=float, default=d.rho0,
                   help="Sea-level air density (kg/m^3) for the exponential atmosphere model.")
    p.add_argument("--H", type=float, default=d.H,
                   help="Atmospheric scale height (m) in rho(z) = rho0 * exp(-z/H).")
    p.add_argument("--Cd", type=float, default=d.Cd,
                   help="Drag coefficient (dimensionless). Set 0 to disable drag.")

    p.add_argument("--t-max", type=float, default=d.t_max, dest="t_max",
                   help="Max integration duration (s) allowed per phase (burn/hover/final).")
    p.add_argument("--max-step", type=float, default=d.max_step, dest="max_step",
                   help="Max internal step size (s) for solve_ivp.")

    p.add_argument("--z-floor", type=float, default=d.z_floor, dest="z_floor",
                   help="Hard floor height (m). Simulation terminates when z reaches z_floor.")

    p.add_argument(
        "--enable-hover",
        action=argparse.BooleanOptionalAction,
        default=d.enable_hover,
        dest="enable_hover",
        help="Enable hover phase (PD controller) once z <= z_hover_start.",
    )
    p.add_argument("--z-hover-start", type=float, default=d.z_hover_start, dest="z_hover_start",
                   help="When z decreases to this height (m), switch from burn to hover controller.")
    p.add_argument("--t-hover", type=float, default=d.t_hover, dest="t_hover",
                   help="Hover duration (s) before switching to final approach.")
    p.add_argument("--z-ref", type=float, default=d.z_ref, dest="z_ref",
                   help="Hover altitude setpoint (m) used by the PD controller during hover.")

    p.add_argument("--Kp", type=float, default=d.Kp,
                   help="Hover/landing PD proportional gain (accel per meter).")
    p.add_argument("--Kd", type=float, default=d.Kd,
                   help="Hover/landing PD derivative gain (accel per (m/s)).")

    p.add_argument("--throttle-min", type=float, default=d.throttle_min, dest="throttle_min",
                   help="Minimum throttle fraction (0..1) for controller clamp.")
    p.add_argument("--v-land-ref", type=float, default=d.v_land_ref, dest="v_land_ref",
                   help="Final approach velocity reference (m/s). Negative means descending.")

    return p.parse_args()


# -------------------- Main --------------------

def main():
    args = parse_args()

    p = Params(
        g=args.g,
        g0=args.g0,
        diameter_m=args.diameter_m,
        m_dry=args.m_dry,
        prop_total=args.prop_total,
        landing_prop_frac=args.landing_prop_frac,
        Isp_s=args.Isp_s,
        T_engine_max_N=args.T_engine_max_N,
        throttle=args.throttle,
        n_engines=args.n_engines,
        rho0=args.rho0,
        H=args.H,
        Cd=args.Cd,
        t_max=args.t_max,
        max_step=args.max_step,
        z_floor=args.z_floor,
        enable_hover=args.enable_hover,
        z_hover_start=args.z_hover_start,
        t_hover=args.t_hover,
        z_ref=args.z_ref,
        Kp=args.Kp,
        Kd=args.Kd,
        throttle_min=args.throttle_min,
        v_land_ref=args.v_land_ref,
    )

    z0 = float(args.z0)
    v0 = float(args.v0)
    v_target = float(args.v_target)

    chk = quick_feasibility_check(z0, v0, p)
    if chk is not None and (not chk["ok"]):
        print("\n[WARN] Back-of-envelope says target is NOT reachable with current constant thrust.")
        if "a_net" in chk:
            print(f"       a_net ≈ {chk['a_net']:.3f} m/s^2 (upward)")
        if "d_stop" in chk:
            print(f"       stopping distance ≈ {chk['d_stop']:.1f} m (need <= z0={z0:.1f} m)")
        if "reason" in chk:
            print(f"       reason: {chk['reason']}")

    z_burn, info = find_zburn_for_target_touchdown_speed(
        z0, v0, p,
        v_target=v_target,
        tol_v=float(args.tol_v),
        max_iter=int(args.max_iter),
    )
    out = simulate_for_zburn(z0, v0, p, z_burn)

    print("\n=== Result ===")
    print(f"z0                = {z0:.2f} m")
    print(f"v0                = {v0:.2f} m/s")
    print(f"v_target          = {v_target:.2f} m/s")
    print(f"z_floor           = {p.z_floor:.6f} m")
    print(f"T_const           = {p.T_const/1000:.1f} kN  (burn throttle={p.throttle:.2f}, engines={p.n_engines})")
    print(f"T_max(ctrl)       = {p.T_max/1000:.1f} kN  (controller limit)")
    print(f"m0                = {p.m0:.1f} kg   (m_dry={p.m_dry:.1f} kg)")
    print(f"z_burn_start      = {z_burn:.2f} m")
    print(f"t_burn_start      = {out.get('t_burn')}")
    print(f"t_hover_start     = {out.get('t_hover_start')}")
    print(f"t_hover_end       = {out.get('t_hover_end')}")
    print(f"t_touchdown       = {out.get('t_touchdown')}")
    print(f"v_touchdown       = {out.get('v_touchdown')}")
    print(f"m_touchdown       = {out.get('m_touchdown'):.1f} kg")
    print(f"status            = {out['status']}")
    print(f"search_feasible   = {info.get('feasible')} ({info.get('reason')})")
    if info.get("reason") == "min_abs_error_no_bracket" and not info.get("feasible"):
        print(f"[WARN] Could not reach tol_v={float(args.tol_v):.3f} m/s; best_err={info.get('best_err'):.3f} m/s "
              f"(best_v_touchdown={info.get('best_v_touchdown'):.3f} m/s).")

    # ---- Plot ----
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    if out["t_coast"].size > 0:
        ax[0].plot(out["t_coast"], out["y_coast"][0], label="coast")
        ax[1].plot(out["t_coast"], out["y_coast"][1], label="coast")
        ax[2].plot(out["t_coast"], out["y_coast"][2], label="coast")

    if out["t_burn_seg"].size > 0:
        ax[0].plot(out["t_burn_seg"], out["y_burn_seg"][0], label="burn (const)")
        ax[1].plot(out["t_burn_seg"], out["y_burn_seg"][1], label="burn (const)")
        ax[2].plot(out["t_burn_seg"], out["y_burn_seg"][2], label="burn (const)")

    if out["t_hover_seg"].size > 0:
        ax[0].plot(out["t_hover_seg"], out["y_hover_seg"][0], label="hover (PD)")
        ax[1].plot(out["t_hover_seg"], out["y_hover_seg"][1], label="hover (PD)")
        ax[2].plot(out["t_hover_seg"], out["y_hover_seg"][2], label="hover (PD)")

    if bool(args.plot_final_approach) and out["t_land_seg"].size > 0:
        ax[0].plot(out["t_land_seg"], out["y_land_seg"][0], label="final approach (PD)")
        ax[1].plot(out["t_land_seg"], out["y_land_seg"][1], label="final approach (PD)")
        ax[2].plot(out["t_land_seg"], out["y_land_seg"][2], label="final approach (PD)")

    ax[0].axhline(p.z_floor, linestyle="--", linewidth=1.0, alpha=0.4, color="gray")
    ax[1].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.4, color="gray")

    if out.get("t_burn") is not None:
        for a in ax:
            a.axvline(out["t_burn"], linestyle=":", linewidth=1.2, alpha=0.7, color="gray")
    if out.get("t_hover_start") is not None:
        for a in ax:
            a.axvline(out["t_hover_start"], linestyle=":", linewidth=1.2, alpha=0.7, color="gray")
    if bool(args.plot_final_approach) and out.get("t_hover_end") is not None:
        for a in ax:
            a.axvline(out["t_hover_end"], linestyle=":", linewidth=1.2, alpha=0.7, color="gray")

    ax[0].set_ylabel("z (m)")
    ax[1].set_ylabel("v (m/s)")
    ax[2].set_ylabel("m (kg)")
    ax[2].set_xlabel("t (s)")
    ax[0].set_title("Vertical suicide-burn simulation with hover phase (1D)")

    for a in ax:
        a.grid(True, alpha=0.2)
        a.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
