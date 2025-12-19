import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Falcon 9-ish 1D vertical "suicide burn" model (upright rocket, no rotation).

# Sign convention:
#   +z is up, so descending means v < 0.
#   A "good landing" means v_touchdown close to 0 (from below).

# State: y = [z, v, m]
#   z: height above "ground floor" (m). We enforce z >= z_floor > 0.
#   v: vertical velocity (m/s), +up
#   m: mass (kg)

# EOM:
#   dz/dt = v
#   dv/dt = (T(t,m) + F_D(v,z) - m*g)/m
#   dm/dt = -T(t,m)/(Isp*g0)   (when burning & prop available)

# Boundary Conditions:
#   z(t0) = z0
#   v(t0) = v0
#   m(t0) = m0
#   z(t_burn) = z_burn 
#   z(t_hover_start) = z_floor + z_error
#   z(t) >= z_floor for all t>=0
#   0 <= T(t) <= T_max for all t>=0
#   v(t) approx 0 when z(t) approx z_floor
#   t_hover_start + t_hover = t_f

# Phases:
#   1) Coast (T=0) until z hits z_burn.
#   2) Burn until z hits z_hover_start (or z_floor)
#   3) Hover controller (PD) for t_hover (or until z_floor)

# Constants
g = 9.80665  # m/s^2

diameter_m = 3.7
area_ref = np.pi * (diameter_m / 2.0) ** 2

# Initial conditions
z0 = 15000.0    # m
v0 = 0.0    # m/s
z_floor = 10.0   # m
z_error = 90.0   # m
z_safety = (2.0 / 30.0) * z0 # m
v_safety = 0.001 * v0 # m/s
z_hover_start = z_floor + z_error

prop_total = 395700.0  # kg (full)
m_dry = 25600.0        # kg
landing_prop_frac = 0.03 
m0 = m_dry + landing_prop_frac * prop_total # IMPORTANT: start with *landing* mass

# Engine
Isp = 283.0
T_max = 981000.0

# Atmos/Drag
rho_0 = 1.225   # kg/m^3
H = 8500.0      # m
Cd = 0.47

# Atmos / drag / mass flow
def rho_air(z):
    return rho_0 * np.exp(-max(0.0, float(z)) / H)

def drag_force(v, z):
    # Opposes velocity: downward v<0 -> drag upward (+)
    return -0.5 * rho_air(z) * Cd * area_ref * v * abs(v)

def mdot_from_thrust(T):
    # dm/dt = -T/(Isp*g)
    return -T / (Isp * g)

# Thrust laws (return "commanded" thrust)
def thrust_zero(t, z, v, m):
    return 0.0

def thrust_pd(z_target, v_target, kp, kd):
    """
    PD with drag feedforward:
      a_cmd = kp(z_ref - z) + kd(v_ref - v)
      T_raw = m(g + a_cmd) - Fd
    so dv ≈ a_cmd after substitution into dv equation.
    """
    def law(t, z, v, m):
        Fd = drag_force(v, z)
        a_cmd = kp * (z_target - z) + kd * (v_target - v)
        T_raw = m * (g + a_cmd) - Fd
        return T_raw
    return law

def thrust_profiled_descent(z_target, a_des=7.0, kd_v=1.2, v_cap=180.0):
    """
    Big-decel controller that avoids the 'altitude PD to far-below setpoint' problem.

    Choose a velocity profile:
        v_ref(z) = -sqrt(2 a_des (z - z_target))   for z > z_target
    so v_ref -> 0 as z -> z_target from above.

    Then do (mostly) velocity control with drag feedforward:
        a_cmd = kd_v (v_ref - v)
        T_raw = m(g + a_cmd) - Fd
    """
    def law(t, z, v, m):
        dz = max(z - z_target, 0.0)
        v_ref = -np.sqrt(2.0 * a_des * dz)
        v_ref = max(v_ref, -v_cap)

        Fd = drag_force(v, z)
        a_cmd = kd_v * (v_ref - v)
        T_raw = m * (g + a_cmd) - Fd
        return T_raw
    return law

# RHS builder (enforces 0 <= T <= T_max and no thrust after dry mass)
def make_rhs(thrust_law):
    def rhs(t, y):
        z, v, m = y

        if m <= m_dry:
            m = m_dry # clamp mass

        Fd = drag_force(v, z)

        if m <= m_dry:
            T = 0.0 # If no prop remains, engines cannot produce thrust
        else:
            T = float(thrust_law(t, z, v, m))
            T = np.clip(T, 0.0, T_max)  # thrust nonnegative + capped

        dz = v
        dv = (T + Fd - m * g) / m
        dm = 0.0 if (m <= m_dry or T <= 0.0) else mdot_from_thrust(T)
        
        return [dz, dv, dm]
    return rhs

# Events (helper functions for system sim's)
def event_hit(z_target):
    def f(t, y):
        return y[0] - z_target
    f.terminal = True
    f.direction = -1
    return f

def event_touchdown():
    def f(t, y):
        return y[0] - z_floor
    f.terminal = True
    f.direction = -1  # descending through the floor
    return f

def event_reach(z_target):
    # reach from either direction (robust if controller overshoots slightly)
    def f(t, y):
        return y[0] - z_target
    f.terminal = True
    f.direction = 0
    return f

def touchdown_info(segments):
    for seg in segments:
        sol = seg["sol"]
        if sol.t_events and len(sol.t_events) > 0 and len(sol.t_events[-1]) > 0:
            t_td = float(sol.t_events[-1][0])
            z_td, v_td, m_td = sol.y_events[-1][0]
            return True, t_td, float(z_td), float(v_td), float(m_td)
    return False, None, None, None, None

# Simulation (t_max is DURATION PER PHASE; guarded phase transitions)
def simulate_system(
    z_burn,
    t_hover=20.0,
    # burn/profiled-descent knobs
    a_des=12.0,
    kd_v=1.2,
    # hover/final PD gains
    kp_hover=0.25, kd_hover=0.9,
    kp_final=0.5,  kd_final=1.4,
    # integration
    t_max=300.0,
    max_step=0.05
):
    segments = []

    def run_segment(label, thrust_law, t_start, y_start, events, duration=None):
        if duration is None:
            duration = t_max
        sol = solve_ivp(
            fun=make_rhs(thrust_law),
            t_span=(t_start, t_start + float(duration)),  # FIX: duration per phase
            y0=y_start,
            events=events,
            max_step=max_step,
            rtol=1e-7,
            atol=1e-9
        )
        segments.append({"label": label, "t": sol.t, "Y": sol.y.T, "sol": sol, "thrust_law": thrust_law})
        return sol

    # Phase 1: coast to z_burn (or touchdown)
    t0 = 0.0
    y0 = np.array([z0, v0, m0], dtype=float)

    sol1 = run_segment(
        "coast",
        thrust_zero,
        t_start=t0,
        y_start=y0,
        events=[event_hit(z_burn), event_touchdown()],
        duration=t_max
    )

    # touched down during coast?
    if sol1.status == 1 and sol1.t_events[1].size > 0:
        return segments

    # Did we actually hit z_burn?
    if sol1.t_events[0].size == 0:
        return segments # couldn't reach z_burn before timing out -> stop (guarded)

    # Phase 2: ignite at z_burn with profiled descent controller, descend to z_hover_start (or touchdown)
    t1 = sol1.t[-1]
    y1 = sol1.y[:, -1]

    burn_law = thrust_profiled_descent(z_target=z_hover_start, a_des=a_des, kd_v=kd_v)

    sol2 = run_segment(
        "burn (profiled)",
        burn_law,
        t_start=t1,
        y_start=y1,
        events=[event_reach(z_hover_start), event_touchdown()],
        duration=t_max
    )

    # touchdown during burn/controller?
    if sol2.status == 1 and sol2.t_events[1].size > 0:
        return segments

    # If we didn't reach hover-start, do NOT blindly continue
    if sol2.t_events[0].size == 0:
        return segments

    # Phase 3: hover for t_hover (or touchdown)
    t2 = sol2.t[-1]
    y2 = sol2.y[:, -1]
    hover_law = thrust_pd(z_target=z_hover_start, v_target=0.0, kp=kp_hover, kd=kd_hover)

    sol3 = run_segment(
        "hover (PD)",
        hover_law,
        t_start=t2,
        y_start=y2,
        events=[event_touchdown()],
        duration=min(t_hover, t_max)
    )

    if sol3.status == 1 and sol3.t_events[0].size > 0:
        return segments

    # Phase 4: final approach to touchdown
    # NOT NECESSARY FOR OUR USECASE
    # t3 = sol3.t[-1]
    # y3 = sol3.y[:, -1]
    # final_law = thrust_pd(z_target=z_floor, v_target=0.0, kp=kp_final, kd=kd_final)

    # _ = run_segment(
    #     "final approach (PD)",
    #     final_law,
    #     t_start=t3,
    #     y_start=y3,
    #     events=[event_touchdown()],
    #     duration=t_max
    # )

    return segments

def concat_segments(segments):
    t_all, Y_all = [], []
    for i, seg in enumerate(segments):
        t = seg["t"]
        Y = seg["Y"]
        if i > 0:
            t = t[1:]
            Y = Y[1:, :]
        t_all.append(t)
        Y_all.append(Y)
    if not t_all:
        return np.array([]), np.empty((0, 3))
    return np.concatenate(t_all), np.vstack(Y_all)

def get_touchdown_v_from_segments(segments):
    """
    Returns v_touchdown from the first segment that triggered the touchdown event.
    Assumes touchdown event is the LAST event in each segment's events list.
    """
    for seg in segments:
        sol = seg["sol"]
        if sol.t_events and len(sol.t_events[-1]) > 0:
            # y_events[-1][0] is the state at touchdown: [z, v, m]
            z_td, v_td, m_td = sol.y_events[-1][0]
            return float(v_td)
    return None

# z_burn tuning (this is used to find an optimal z_burn)
def touchdown_speed_for_zburn(z_burn, v_target=-0.25):
    segs = simulate_system(z_burn)
    v_td = get_touchdown_v_from_segments(segs)
    return v_td

def find_zburn(v_target=-0.25, tol=0.05, n_scan=41, max_iter=60):
    # If hover is part of the mission, constrain search to start burns above hover start
    z_lo = max(z_floor + 1e-3, z_hover_start + 1.0)
    z_hi = z0

    zs = np.linspace(z_lo, z_hi, n_scan)

    vals = []  # (zb, f) where f = vtd - v_target
    for zb in zs:
        vtd = touchdown_speed_for_zburn(zb, v_target=v_target)
        if vtd is None:
            continue
        vals.append((float(zb), float(vtd - v_target)))

    if len(vals) < 2:
        # fallback if too many sims fail
        return float(0.5 * (z_lo + z_hi))

    # best by absolute error
    best_z, best_f = min(vals, key=lambda p: abs(p[1]))
    if abs(best_f) <= tol:
        return best_z

    # Try to find a sign-change bracket (sorted by z)
    vals.sort(key=lambda p: p[0])
    bracket = None
    for (z1, f1), (z2, f2) in zip(vals[:-1], vals[1:]):
        if f1 == 0.0:
            return z1
        if f1 * f2 < 0.0:
            bracket = (z1, z2, f1, f2)
            break

    # If we have a bracket -> bisection
    if bracket is not None:
        a, b, fa, fb = bracket
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            vtd = touchdown_speed_for_zburn(m, v_target=v_target)
            if vtd is None:
                # shrink toward earlier burn as a conservative move
                b = m
                continue
            fm = vtd - v_target
            if abs(fm) <= tol:
                return float(m)
            if fa * fm < 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return float(0.5 * (a + b))

    # No sign change -> LOCAL REFINEMENT around best (improvement E)
    zs_sorted = np.array([z for z, _ in vals])
    idx = int(np.argmin(np.abs(zs_sorted - best_z)))
    lo = zs_sorted[max(0, idx - 1)]
    hi = zs_sorted[min(len(zs_sorted) - 1, idx + 1)]
    if hi <= lo:
        lo = max(z_lo, best_z - (z_hi - z_lo) / n_scan)
        hi = min(z_hi, best_z + (z_hi - z_lo) / n_scan)

    refine = np.linspace(lo, hi, 31)
    for zb in refine:
        vtd = touchdown_speed_for_zburn(float(zb), v_target=v_target)
        if vtd is None:
            continue
        f = float(vtd - v_target)
        if abs(f) < abs(best_f):
            best_z, best_f = float(zb), f

    return best_z

# Plotting (YIPPEE!)
def plot_segments(segments):
    import numpy as np
    import matplotlib.pyplot as plt

    # Turn on LaTeX-style rendering for text (mathtext; no external LaTeX needed)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # --- FILTER: only plot phases 1-3 ---
    segments_to_plot = [
        seg for seg in segments
        if seg["label"] in ("coast", "burn (profiled)", "hover (PD)")
    ]

    # vertical lines at phase boundaries (within the plotted subset)
    switch_times = [seg["t"][0] for seg in segments_to_plot[1:]]

    for seg in segments_to_plot:
        t = seg["t"]
        Y = seg["Y"]
        z = Y[:, 0]
        v = Y[:, 1]
        m = Y[:, 2]
        label = seg["label"]

        ax[0].plot(t, m, label=label)
        ax[1].plot(t, z, label=label)
        ax[2].plot(t, v, label=label)

    for a in ax:
        for tsw in switch_times:
            a.axvline(tsw, linestyle="--", linewidth=1)
        a.grid(True)

    # ---- LaTeX-style axis labels ----
    ax[0].set_ylabel(r"$m\;(\mathrm{kg})$")
    ax[1].set_ylabel(r"$z\;(\mathrm{m})$")
    ax[2].set_ylabel(r"$v\;(\mathrm{m\,s^{-1}})$")
    ax[2].set_xlabel(r"$t\;(\mathrm{s})$")

    ax[1].axhline(z_floor, linestyle="--", linewidth=1)

    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    for seg in segments_to_plot:
        t = seg["t"]
        Y = seg["Y"]
        z = Y[:, 0]
        v = Y[:, 1]
        m = Y[:, 2]
        label = seg["label"]
        law = seg.get("thrust_law", thrust_zero)

        T = np.empty_like(t, dtype=float)
        TW = np.empty_like(t, dtype=float)

        for i, (ti, zi, vi, mi) in enumerate(zip(t, z, v, m)):
            mi_eff = m_dry if mi <= m_dry else mi

            if mi_eff <= m_dry:
                Ti = 0.0
            else:
                Ti = float(law(ti, zi, vi, mi_eff))
                Ti = float(np.clip(Ti, 0.0, T_max))

            T[i] = Ti
            TW[i] = Ti / (mi_eff * g) if mi_eff > 0 else 0.0

        ax2[0].plot(t, T, label=label)
        ax2[1].plot(t, TW, label=label)

    # optional reference lines
    ax2[0].axhline(T_max, linestyle="--", linewidth=1)
    ax2[1].axhline(1.0, linestyle="--", linewidth=1)

    for a in ax2:
        for tsw in switch_times:
            a.axvline(tsw, linestyle="--", linewidth=1)
        a.grid(True)

    # ---- LaTeX-style axis labels ----
    ax2[0].set_ylabel(r"$T\;(\mathrm{N})$")
    ax2[1].set_ylabel(r"$T/W\;(-)$")
    ax2[1].set_xlabel(r"$t\;(\mathrm{s})$")

    ax2[0].legend(loc="best")
    fig2.suptitle("Thrust and thrust-to-weight ratio (1D)")
    plt.tight_layout()

    ax[0].legend(loc="best")
    fig.suptitle("Vertical suicide-burn simulation with hover phase (1D)")
    plt.tight_layout()
    plt.show()

# Hi Professor!
if __name__ == "__main__":
    print(f"Initial mass m0 = {m0:.1f} kg")
    print(f"T/W at start = {T_max / (m0*g):.3f}")

    z_burn = find_zburn(v_target= 0.00 + v_safety, tol = 0.05)
    z_burn += z_safety  # start (z_safety)m earlier than “minimum” solution
    print(f"Chosen z_burn = {z_burn:.3f} m")

    segments = simulate_system(z_burn)
    t, Y = concat_segments(segments)
    
    hit, t_td, z_td, v_td, m_td = touchdown_info(segments)
    if hit:
        print(f"Touchdown detected: True at t={t_td:.3f} s (z={z_td:.3f} m, v={v_td:.3f} m/s, m={m_td:.1f} kg)")
    else:
        print("Touchdown detected: False (no touchdown event triggered in any phase)")

    if Y.shape[0] > 0:
        print(f"Final state: z={Y[-1,0]:.3f} m, v={Y[-1,1]:.3f} m/s, m={Y[-1,2]:.1f} kg")
    else:
        print("Simulation produced no samples (check parameters).")

    plot_segments(segments)
