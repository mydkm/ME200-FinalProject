import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# Falcon-9-ish 2D planar rigid-body model with inputs [u1, tau]
# (u1 = total thrust along body axis, tau = pitch torque about COM)

# Sign convention:
#   +z is up, so descending means v_z < 0.
#   +x is horizontal (right).
#   theta = 0 means perfectly vertical/upright.
#   theta > 0 means the rocket body axis points slightly toward +x.
#   omega = d(theta)/dt.
#   A "good landing" means touchdown at z = z_floor with v_z ~ 0 (from below),
#   plus v_x ~ 0, theta ~ 0, omega ~ 0.

# State: y = [x, z, v_x, v_z, theta, omega, m]
#   x     : horizontal position (m)
#   z     : height above ground floor (m), enforce z >= z_floor
#   v_x   : horizontal velocity (m/s)
#   v_z   : vertical velocity (m/s)
#   theta : pitch angle from vertical (rad)
#   omega : pitch rate (rad/s)
#   m     : mass (kg)

# Inputs:
#   u1(t)   : total thrust magnitude along rocket body axis (N)
#   tau(t)  : pitch torque about COM (N·m)
#
# (If you still imagine two thrusters, u1 and tau are the "virtual controls"
#  that you later allocate back to per-engine thrusts.)

# Parameters:
#   I     : pitch moment of inertia about COM (kg·m^2)
#           (optional) uniform solid cylinder about transverse axis:
#           I = (1/12) * m * (3*R^2 + L^2)   (or treat I constant for simplicity)
#   g     : gravity (m/s^2)
#   Isp   : specific impulse (s)
#   g    : standard gravity (m/s^2)
#   z_floor : ground height (m)
#   m_dry   : dry mass (kg)

# Body-axis unit vector in inertial coordinates:
#   b_hat(theta) = [sin(theta), cos(theta)]

# Aerodynamics (optional):
#   D_x, D_z = drag components in inertial frame (N)
#   Super-simple: D_x = D_z = 0
#   Simple quadratic drag example:
#     v = [v_x, v_z], speed = ||v||
#     D = -0.5 * rho(z) * C_D * A * speed * v

# EOM:
#   dx/dt      = v_x
#   dz/dt      = v_z
#   dv_x/dt    = (u1/m) * sin(theta) + D_x/m
#   dv_z/dt    = (u1/m) * cos(theta) - g + D_z/m
#   dtheta/dt  = omega
#   domega/dt  = tau / I
#   dm/dt      = -u1 / (Isp * g)      (while burning and prop available)

# Boundary / path constraints:
#   Initial:
#     x(t0)=x0, z(t0)=z0, v_x(t0)=v_x0, v_z(t0)=v_z0,
#     theta(t0)=theta0, omega(t0)=omega0, m(t0)=m0
#
#   Ground + fuel:
#     z(t) >= z_floor     for all t >= 0
#     m(t) >= m_dry       for all t >= 0
#
#   Actuation limits (virtual):
#     0 <= u1(t) <= u1_max
#     |tau(t)| <= tau_max
#   (If you ultimately have two thrusters, tau_max is not independent of u1;
#    it depends on per-engine limits—see "allocation constraints" below.)

# Touchdown event:
#   touchdown when z(t) crosses z_floor while descending (v_z < 0)

# Terminal ("good landing") targets (tolerances / objectives):
#   at touchdown:
#     v_z ~ v_target (near 0 from below)
#     v_x ~ 0
#     theta ~ 0
#     omega ~ 0

# Two-thruster allocation constraints (if applicable):
#   If two engines at lever arm d with per-engine limits 0 <= T_L,T_R <= T_max:
#     u1 = T_L + T_R
#     tau = d * (T_R - T_L)
#
#   Then feasible (u1, tau) must satisfy:
#     0 <= u1 <= 2*T_max
#     -d*u1 <= tau <= d*u1                  (because T_L,T_R >= 0)
#     d*(u1 - 2*T_max) <= tau <= d*(2*T_max - u1)  (because T_L,T_R <= T_max)
#
#   A compact way to implement: compute (T_L,T_R) from desired (u1,tau) and clip:
#     ΔT = tau/d
#     T_R = 0.5*(u1 + ΔT)
#     T_L = 0.5*(u1 - ΔT)
#     clip each to [0, T_max]
#     recompute achieved u1, tau from clipped thrusts (for the dynamics step)

# Phases (simple but realistic: attitude control runs throughout):
#   1) Coast:        u1 = 0, tau = 0 until z hits z_burn
#   2) Burn:         choose u1(z, v_z, ...) for descent; choose tau(theta,omega,...) to reorient
#   3) Hover/land:   choose u1 to hold/bleed vertical speed; choose tau to keep theta ~ 0

# Constants / Parameters
g  = 9.80665   # m/s^2

# Rocket geometry (rough Falcon-9-ish)
diameter_m = 3.7
R = diameter_m / 2.0
L = 40.0
area_ref = np.pi * R**2

# Two-thruster geometry (lever arm)
d_thruster = R

# Engine / mass
Isp = 283.0
T_max = 981_000.0            # per engine
u1_max = 2.0 * T_max         # total

prop_total = 395_700.0
m_dry = 25_600.0
landing_prop_frac = 0.03
m0 = m_dry + landing_prop_frac * prop_total

# Atmos/Drag (optional)
USE_DRAG = True
rho_0 = 1.225
H = 8500.0
Cd = 0.47

# Allocation (hardware realism)
USE_THRUSTER_ALLOCATION = True
tau_max_independent = d_thruster * T_max

# Initial conditions
x0 = 0.0
z0 = 3000.0
vx0 = 0.0
vz0 = 0.0

theta0 = np.deg2rad(30.0)
omega0 = 0.0

z_floor = 10.0
z_error = 90.0
z_hover_start = z_floor + z_error

t_hover = 20.0
t_max_phase = 400.0

# Attitude LQR settle tolerances (STRONG: omega near 0)
theta_tol_deg = 0.1     # deg
omega_tol = 0.001       # rad/s  (near 0)

# Safety margins (like 1D's v_safety, z_safety)
USE_SAFETY_MARGINS = True
z_safety = (2.0 / 30.0) * z0      # meters
v_safety = 0.001 * vz0            # m/s (will be 0 if vz0=0; set manually if you want)

# Helper functions
def rho_air(z):
    return rho_0 * np.exp(-max(0.0, float(z)) / H)

def drag_force_2d(vx, vz, z):
    """Quadratic drag opposite velocity. Returns (Dx, Dz)."""
    if not USE_DRAG:
        return 0.0, 0.0
    v = np.array([vx, vz], dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < 1e-9:
        return 0.0, 0.0
    D = -0.5 * rho_air(z) * Cd * area_ref * speed * v
    return float(D[0]), float(D[1])

def I_pitch(m):
    """Pitch inertia about COM for a uniform solid cylinder about transverse axis."""
    return (1.0 / 12.0) * float(m) * (3.0 * R**2 + L**2)

def allocate_u1_tau_to_thrusters(u1_cmd, tau_cmd, d=d_thruster, Tcap=T_max):
    """
    Map desired (u1,tau) to per-engine thrusts (T_L, T_R), clip to [0,Tcap],
    then return achieved (u1_act, tau_act).

    Assumption: thrusters are at ±d in body-x at COM height, thrust along body axis.
    Then: tau = d*(T_R - T_L)  (independent of theta)
    """
    u1_cmd = float(np.clip(u1_cmd, 0.0, 2.0 * Tcap))

    dT = float(tau_cmd) / max(1e-9, d)
    TR = 0.5 * (u1_cmd + dT)
    TL = 0.5 * (u1_cmd - dT)

    TLc = float(np.clip(TL, 0.0, Tcap))
    TRc = float(np.clip(TR, 0.0, Tcap))

    u1_act = TLc + TRc
    tau_act = d * (TRc - TLc)

    return TLc, TRc, float(u1_act), float(tau_act)

# Attitude LQR helpers
def lqr_gain_attitude(I0, q_theta=800.0, q_omega=120.0, r_tau=1.0):
    """
    LQR for subsystem:
        theta_dot = omega
        omega_dot = (1/I0) * tau
    """
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [1.0 / float(I0)]])
    Q = np.diag([q_theta, q_omega])
    Rm = np.array([[r_tau]])

    P = solve_continuous_are(A, B, Q, Rm)
    K = np.linalg.solve(Rm, B.T @ P)   # shape (1,2)
    return K

def make_ctrl_attitude_lqr(K, theta_ref=0.0):
    """
    Attitude-only controller that starts at t=0.
    With two positive-only thrusters you can't create torque without some total thrust,
    so choose u1 just large enough to make the requested tau feasible: |tau| <= d*u1.
    """
    K = np.asarray(K).reshape(1, 2)

    def ctrl(t, y):
        x, z, vx, vz, theta, omega, m = y
        e = np.array([float(theta - theta_ref), float(omega)], dtype=float)
        tau_cmd = float(-(K @ e)[0])

        u1_min = abs(tau_cmd) / max(1e-9, d_thruster)
        u1_cmd = float(np.clip(u1_min, 0.0, u1_max))
        return u1_cmd, tau_cmd

    return ctrl

# Controllers (return desired u1, tau)
def ctrl_zero(t, y):
    return 0.0, 0.0

def make_ctrl_profiled_descent(z_target, a_des=12.0, kd_v=1.2,
                               kp_theta=8.0, kd_theta=4.0,
                               theta_ref=0.0):
    def ctrl(t, y):
        x, z, vx, vz, theta, omega, m = y
        m_eff = max(m_dry, float(m))

        Dx, Dz = drag_force_2d(vx, vz, z)

        dz = max(float(z) - float(z_target), 0.0)
        v_ref = -np.sqrt(2.0 * a_des * dz) if dz > 0 else 0.0

        a_cmd_z = kd_v * (v_ref - float(vz))
        cos_th = max(1e-3, float(np.cos(theta)))

        u1_raw = (m_eff * (g + a_cmd_z) - Dz) / cos_th
        u1_cmd = float(np.clip(u1_raw, 0.0, u1_max))

        Ieff = I_pitch(m_eff)
        alpha_cmd = -kp_theta * (float(theta) - float(theta_ref)) - kd_theta * float(omega)
        tau_cmd = float(Ieff * alpha_cmd)

        return u1_cmd, tau_cmd
    return ctrl

def make_ctrl_hover(z_ref, v_ref=0.0,
                    kp_z=0.25, kd_z=0.9,
                    kp_theta=10.0, kd_theta=5.0,
                    theta_ref=0.0):
    def ctrl(t, y):
        x, z, vx, vz, theta, omega, m = y
        m_eff = max(m_dry, float(m))

        Dx, Dz = drag_force_2d(vx, vz, z)

        a_cmd_z = kp_z * (float(z_ref) - float(z)) + kd_z * (float(v_ref) - float(vz))
        cos_th = max(1e-3, float(np.cos(theta)))

        u1_raw = (m_eff * (g + a_cmd_z) - Dz) / cos_th
        u1_cmd = float(np.clip(u1_raw, 0.0, u1_max))

        Ieff = I_pitch(m_eff)
        alpha_cmd = -kp_theta * (float(theta) - float(theta_ref)) - kd_theta * float(omega)
        tau_cmd = float(Ieff * alpha_cmd)

        return u1_cmd, tau_cmd
    return ctrl

# RHS builder
def make_rhs(ctrl_law):
    def rhs(t, y):
        x, z, vx, vz, theta, omega, m = y
        m_eff = max(m_dry, float(m))

        Dx, Dz = drag_force_2d(vx, vz, z)

        if m_eff <= m_dry + 1e-9:
            u1_des, tau_des = 0.0, 0.0
        else:
            u1_des, tau_des = ctrl_law(t, y)

        if USE_THRUSTER_ALLOCATION:
            _, _, u1, tau = allocate_u1_tau_to_thrusters(u1_des, tau_des)
        else:
            u1 = float(np.clip(u1_des, 0.0, u1_max))
            tau = float(np.clip(tau_des, -tau_max_independent, tau_max_independent))

        sin_th = float(np.sin(theta))
        cos_th = float(np.cos(theta))

        dx      = float(vx)
        dz      = float(vz)
        dvx     = (u1 / m_eff) * sin_th + (Dx / m_eff)
        dvz     = (u1 / m_eff) * cos_th - g + (Dz / m_eff)
        dtheta  = float(omega)

        Ieff = I_pitch(m_eff)
        domega = tau / Ieff

        if (m_eff <= m_dry + 1e-9) or (u1 <= 0.0):
            dm = 0.0
        else:
            dm = -u1 / (Isp * g)

        return [dx, dz, dvx, dvz, dtheta, domega, dm]
    return rhs

# Events
def event_hit_z(z_target):
    def f(t, y):
        return float(y[1] - z_target)
    f.terminal = True
    f.direction = -1
    return f

def event_reach_z(z_target):
    def f(t, y):
        return float(y[1] - z_target)
    f.terminal = True
    f.direction = 0
    return f

def event_touchdown():
    def f(t, y):
        return float(y[1] - z_floor)
    f.terminal = True
    f.direction = -1
    return f

def event_attitude_settled(theta_tol_deg=1.0, omega_tol=0.005):
    """
    STRONG rule: stop LQR only when BOTH |theta| <= theta_tol and |omega| <= omega_tol.
    """
    th_tol = np.deg2rad(theta_tol_deg)
    om_tol = float(omega_tol)

    def f(t, y):
        theta = float(y[4])
        omega = float(y[5])
        # <= 0 means inside BOTH bounds (box constraint)
        return max(abs(theta) / th_tol, abs(omega) / om_tol) - 1.0

    f.terminal = True
    f.direction = -1
    return f

# Simulation
def simulate_system(
    z_burn,
    t_hover=t_hover,
    theta_tol_deg=theta_tol_deg,
    omega_tol=omega_tol,
    a_des=12.0,
    kd_v=1.2,
    kp_z=0.25, kd_z=0.9,
    kp_theta_burn=8.0,  kd_theta_burn=4.0,
    kp_theta_hover=10.0, kd_theta_hover=5.0,
    q_theta=800.0, q_omega=120.0, r_tau=3e-10,
    max_step=0.05,
    rtol=1e-7,
    atol=1e-9
):
    segments = []

    def run_segment(label, ctrl_law, t_start, y_start, events, duration):
        sol = solve_ivp(
            fun=make_rhs(ctrl_law),
            t_span=(t_start, t_start + float(duration)),
            y0=y_start,
            events=events,
            max_step=max_step,
            rtol=rtol,
            atol=atol
        )
        segments.append({
            "label": label,
            "t": sol.t,
            "Y": sol.y.T,
            "sol": sol,
            "ctrl_law": ctrl_law,
        })
        return sol

    t0 = 0.0
    y0 = np.array([x0, z0, vx0, vz0, theta0, omega0, m0], dtype=float)

    # Phase 0: attitude LQR from t=0
    # It only stops when attitude is settled (theta ~ 0 AND omega ~ 0) OR touchdown OR timeout.
    K_att = lqr_gain_attitude(I_pitch(m0), q_theta=q_theta, q_omega=q_omega, r_tau=r_tau)
    ctrl_att = make_ctrl_attitude_lqr(K_att, theta_ref=0.0)

    sol0 = run_segment(
        "attitude LQR (t=0)",
        ctrl_att,
        t_start=t0,
        y_start=y0,
        events=[
            event_attitude_settled(theta_tol_deg=theta_tol_deg, omega_tol=omega_tol),
            event_touchdown()
        ],
        duration=t_max_phase
    )

    # touchdown during attitude phase?
    if sol0.status == 1 and sol0.t_events[1].size > 0:
        return segments

    # If not settled (timeout), stop the simulation (LQR "can't stop" rule).
    settled = (sol0.status == 1 and sol0.t_events[0].size > 0)
    if not settled:
        return segments

    # After LQR settles, decide whether to coast to z_burn or go straight to burn
    tA = sol0.t[-1]
    yA = sol0.y[:, -1]
    zA = float(yA[1])

    # Phase 1: coast to z_burn (engines off), only if we're still above z_burn
    if zA > z_burn:
        sol1 = run_segment(
            "coast",
            ctrl_zero,
            t_start=tA,
            y_start=yA,
            events=[event_hit_z(z_burn), event_touchdown()],
            duration=t_max_phase
        )

        if sol1.status == 1 and sol1.t_events[1].size > 0:
            return segments
        if sol1.t_events[0].size == 0:
            return segments

        t1 = sol1.t[-1]
        y1 = sol1.y[:, -1]
    else:
        # Already below z_burn when we settled attitude: skip coast
        t1 = tA
        y1 = yA

    # Phase 2: Burn (profiled descent + attitude PD) to hover start (or touchdown)
    ctrl_burn = make_ctrl_profiled_descent(
        z_target=z_hover_start,
        a_des=a_des,
        kd_v=kd_v,
        kp_theta=kp_theta_burn,
        kd_theta=kd_theta_burn,
        theta_ref=0.0
    )

    sol2 = run_segment(
        "burn (profiled + attitude PD)",
        ctrl_burn,
        t_start=t1,
        y_start=y1,
        events=[event_reach_z(z_hover_start), event_touchdown()],
        duration=t_max_phase
    )

    if sol2.status == 1 and sol2.t_events[1].size > 0:
        return segments
    if sol2.t_events[0].size == 0:
        return segments

    # Phase 3: Hover (z PD + attitude PD) for t_hover (or touchdown)
    t2 = sol2.t[-1]
    y2 = sol2.y[:, -1]

    ctrl_hover = make_ctrl_hover(
        z_ref=z_hover_start,
        v_ref=0.0,
        kp_z=kp_z,
        kd_z=kd_z,
        kp_theta=kp_theta_hover,
        kd_theta=kd_theta_hover,
        theta_ref=0.0
    )

    _ = run_segment(
        "hover (z PD + attitude PD)",
        ctrl_hover,
        t_start=t2,
        y_start=y2,
        events=[event_touchdown()],
        duration=min(t_hover, t_max_phase)
    )

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
        return np.array([]), np.empty((0, 7))
    return np.concatenate(t_all), np.vstack(Y_all)

def touchdown_info(segments):
    for seg in segments:
        sol = seg["sol"]
        if sol.t_events and len(sol.t_events[-1]) > 0:
            t_td = float(sol.t_events[-1][0])
            y_td = sol.y_events[-1][0]
            return True, t_td, [float(v) for v in y_td]
    return False, None, None

def get_touchdown_vz(segments):
    hit, _, y_td = touchdown_info(segments)
    if not hit:
        return None
    return float(y_td[3])

def touchdown_vz_for_zburn(z_burn, v_target=-0.25):
    segs = simulate_system(z_burn)
    return get_touchdown_vz(segs)

def find_zburn(v_target=-0.25, tol=0.05, n_scan=31, max_iter=50):
    z_lo = max(z_floor + 1e-3, z_hover_start + 1.0)
    z_hi = z0
    zs = np.linspace(z_lo, z_hi, n_scan)

    vals = []
    for zb in zs:
        vz_td = touchdown_vz_for_zburn(float(zb), v_target=v_target)
        if vz_td is None:
            continue
        vals.append((float(zb), float(vz_td - v_target)))

    if len(vals) < 2:
        return float(0.5 * (z_lo + z_hi))

    best_z, best_f = min(vals, key=lambda p: abs(p[1]))
    if abs(best_f) <= tol:
        return best_z

    vals.sort(key=lambda p: p[0])
    bracket = None
    for (z1, f1), (z2, f2) in zip(vals[:-1], vals[1:]):
        if f1 == 0.0:
            return z1
        if f1 * f2 < 0.0:
            bracket = (z1, z2, f1, f2)
            break

    if bracket is None:
        return best_z

    a, b, fa, fb = bracket
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        vz_td = touchdown_vz_for_zburn(float(m), v_target=v_target)
        if vz_td is None:
            b = m
            continue
        fm = float(vz_td - v_target)
        if abs(fm) <= tol:
            return float(m)
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))

def choose_zburn_with_safety(v_target=-0.10, tol=0.05):
    z_lo = max(z_floor + 1e-3, z_hover_start + 1.0)
    z_hi = z0 - 1e-6

    zb_nom = find_zburn(v_target=float(v_target + v_safety), tol=tol)
    zb_safe = zb_nom + (z_safety if USE_SAFETY_MARGINS else 0.0)
    zb_safe = float(np.clip(zb_safe, z_lo, z_hi))
    return float(zb_nom), float(zb_safe)

# Plotting
def compute_u1_tau_timeseries(seg):
    t = seg["t"]
    Y = seg["Y"]
    ctrl = seg["ctrl_law"]

    u1 = np.zeros_like(t, dtype=float)
    tau = np.zeros_like(t, dtype=float)

    for i, (ti, yi) in enumerate(zip(t, Y)):
        m_eff = max(m_dry, float(yi[6]))
        if m_eff <= m_dry + 1e-9:
            u1_des, tau_des = 0.0, 0.0
        else:
            u1_des, tau_des = ctrl(ti, yi)

        if USE_THRUSTER_ALLOCATION:
            _, _, u1_i, tau_i = allocate_u1_tau_to_thrusters(u1_des, tau_des)
        else:
            u1_i = float(np.clip(u1_des, 0.0, u1_max))
            tau_i = float(np.clip(tau_des, -tau_max_independent, tau_max_independent))
        u1[i], tau[i] = u1_i, tau_i

    return u1, tau

def plot_segments(segments):
    import numpy as np
    import matplotlib.pyplot as plt

    # LaTeX-style rendering for labels (mathtext; no external LaTeX install needed)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    # states figure (colored by segment)
    if not segments:
        print("Nothing to plot.")
        return

    # Simple label -> color mapping (matches your example)
    def seg_color(label: str) -> str:
        s = label.lower()
        if "attitude lqr" in s:
            return "C0"  # blue
        if s.strip() == "coast":
            return "C1"  # orange
        if s.startswith("burn"):
            return "C2"  # green
        if s.startswith("hover"):
            return "C3"  # red
        return "C7"      # fallback gray

    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(11, 11))

    for k, seg in enumerate(segments):
        t = np.asarray(seg["t"], dtype=float)
        Y = np.asarray(seg["Y"], dtype=float)
        c = seg_color(seg["label"])
        lbl = seg["label"]

        # avoid duplicating the boundary point between segments
        if k > 0 and t.size > 1:
            t = t[1:]
            Y = Y[1:, :]

        x     = Y[:, 0]
        z     = Y[:, 1]
        vx    = Y[:, 2]
        vz    = Y[:, 3]
        theta = Y[:, 4]
        m     = Y[:, 6]

        # Put labels ONLY on the first subplot so legend is not cluttered
        ax[0].plot(t, m, color=c, label=lbl)
        ax[1].plot(t, z, color=c)
        ax[2].plot(t, vz, color=c)
        ax[3].plot(t, x, color=c)
        ax[4].plot(t, vx, color=c)
        ax[5].plot(t, np.rad2deg(theta), color=c)

    ax[1].axhline(z_floor, linestyle="--", linewidth=1)

    # LaTeX-style axis labels
    ax[0].set_ylabel(r"$m\;(\mathrm{kg})$")
    ax[1].set_ylabel(r"$z\;(\mathrm{m})$")
    ax[2].set_ylabel(r"$v_z\;(\mathrm{m\,s^{-1}})$")
    ax[3].set_ylabel(r"$x\;(\mathrm{m})$")
    ax[4].set_ylabel(r"$v_x\;(\mathrm{m\,s^{-1}})$")
    ax[5].set_ylabel(r"$\theta\;(\mathrm{deg})$")
    ax[5].set_xlabel(r"$t\;(\mathrm{s})$")

    for a in ax:
        a.grid(True)

    # Phase boundaries
    for seg in segments[1:]:
        tsw = float(seg["t"][0])
        for a in ax:
            a.axvline(tsw, linestyle="--", linewidth=1)

    # Legend only once (main/top subplot)
    ax[0].legend(loc="best")

    fig.suptitle("Suicide-burn simulation with reorientation (2D)")
    plt.tight_layout()

    # controls figure (keep your existing behavior) 
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for seg in segments:
        t = seg["t"]
        u1, tau = compute_u1_tau_timeseries(seg)
        ax2[0].plot(t, u1, label=seg["label"])
        ax2[1].plot(t, tau, label=seg["label"])

    ax2[0].axhline(u1_max, linestyle="--", linewidth=1)

    # LaTeX-style axis labels
    ax2[0].set_ylabel(r"$u_1\;(\mathrm{N})$")
    ax2[1].set_ylabel(r"$\tau\;(\mathrm{N\,m})$")
    ax2[1].set_xlabel(r"$t\;(\mathrm{s})$")

    for a in ax2:
        a.grid(True)
        for seg in segments[1:]:
            a.axvline(float(seg["t"][0]), linestyle="--", linewidth=1)

    ax2[0].legend(loc="best")
    fig2.suptitle("Virtual controls (2D)")
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    print(f"Initial mass m0 = {m0:.1f} kg")
    print(f"Total max thrust u1_max = {u1_max:.1f} N")
    print(f"Initial T/W (total) = {u1_max / (m0*g):.3f}")
    print(f"Initial theta0 = {np.rad2deg(theta0):.2f} deg")
    print(f"Attitude settle: |theta|<={theta_tol_deg} deg AND |omega|<={omega_tol} rad/s")

    # Base touchdown vertical-speed target (same meaning as your existing find_zburn)
    v_target_vz = -0.10

    zb_nom, zb_safe = choose_zburn_with_safety(v_target=v_target_vz, tol=0.05)
    if USE_SAFETY_MARGINS:
        print(f"Safety margins enabled:")
        print(f"  z_safety = {z_safety:.3f} m   (burn starts earlier)")
        print(f"  v_safety = {v_safety:.6f} m/s (target vz shifted)")
        print(f"Nominal z_burn (tuned) = {zb_nom:.3f} m")
        print(f"Chosen  z_burn (+safety)= {zb_safe:.3f} m")
        z_burn = zb_safe
    else:
        print(f"Chosen z_burn = {zb_nom:.3f} m")
        z_burn = zb_nom

    segments = simulate_system(z_burn)

    hit, t_td, y_td = touchdown_info(segments)
    if hit:
        x_td, z_td, vx_td, vz_td, th_td, om_td, m_td = y_td
        print(f"Touchdown: True at t={t_td:.3f} s")
        print(f"  z={z_td:.3f} m, vz={vz_td:.3f} m/s, vx={vx_td:.3f} m/s")
        print(f"  theta={np.rad2deg(th_td):.3f} deg, omega={om_td:.6f} rad/s, m={m_td:.1f} kg")
    else:
        print("Touchdown: False (no touchdown event triggered)")

    plot_segments(segments)