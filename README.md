# ME200-FinalProject

This repository contains **two “Falcon‑9‑ish” suicide‑burn descent simulators**:

- **1D vertical model** (upright rocket, no rotation): state $[z, v, m]$
- **2D planar rigid‑body model** (translation + pitch): state $[x, z, v_x, v_z, \theta, \omega, m]$

Both use:

- `scipy.integrate.solve_ivp` with **event‑driven phase transitions**
- A simple **exponential atmosphere + quadratic drag** option
- **Thrust saturation** and **propellant limits** ($m \ge m_{\mathrm{dry}}$)
- A **$z_{\text{burn}}$** search routine (scan → bracket → bisection) to hit a target touchdown vertical speed
- Plotters (with **phase coloring**, **phase boundary markers**, and **LaTeX‑style axis labels**)

---

## 0. Description of the Final

The “final” deliverable is a working simulation that, given an initial condition at altitude $z_0$, computes a physically constrained descent and selects a burn start altitude $z_{\text{burn}}$ such that touchdown vertical speed is near a target value (typically close to $0$ from below), with optional safety margins.

---

## 1. Models (as implemented)

### 1.1 1D Vertical Suicide‑Burn Model

**Sign convention**

- $+z$ is up
- descending means $v<0$

**State**

$$
y = [z,\ v,\ m]
$$

**Equations of motion**

$$
\begin{aligned}
\dot z &= v, \\
\dot v &= \frac{T(t,z,v,m) + F_D(v,z) - mg}{m}, \\
\dot m &= -\frac{T}{I_{sp}g}\quad \text{(only when burning and } m>m_{\mathrm{dry}}\text{)}.
\end{aligned}
$$

**Atmosphere + drag**

Atmosphere:

$$
\rho(z) = \rho_0 e^{-z/H}.
$$

Drag force (opposes velocity):

$$
F_D = -\tfrac12\,\rho(z)\,C_D\,A_{\mathrm{ref}}\,v\,|v|.
$$

**Thrust constraints**

- $0 \le T \le T_{\max}$
- If $m \le m_{\mathrm{dry}}$, thrust is forced to $T=0$

**Phases**

1. **Coast:** $T=0$ until $z=z_{\text{burn}}$ (or touchdown)
2. **Burn (profiled):** track the velocity reference $v_z = -\sqrt{2a_{des}(z-z_{target})}$
   (with an optional speed cap), then apply mostly velocity feedback + drag feedforward to compute thrust.
3. **Hover (PD):** hold $z \approx z_{\text{hover start}} = z_{\text{floor}} + z_{\text{error}}$ for $t_{\text{hover}}$ seconds (or until touchdown).

**$z_{\text{burn}}$ selection**

- `find_zburn(...)` searches for a burn altitude that makes touchdown velocity $v_{\text{td}}$ near `v_target`
- A safety margin is optionally added:
  - `z_burn += z_safety`
  - `v_target += v_safety`

---

### 1.2 2D Planar Rigid‑Body Model (Translation + Pitch)

**Sign convention**

- $+z$ is up
- $+x$ is horizontal
- $\theta=0$ is perfectly upright (body axis aligned with $+z$)
- $\omega=\dot\theta$

**State**

$$
y = [x,\ z,\ v_x,\ v_z,\ \theta,\ \omega,\ m]
$$

**Virtual inputs**

- $u_1(t)$: **total thrust magnitude** along the rocket body axis (N)
- $\tau(t)$: **pitch torque** about COM (N·m)

Body‑axis unit vector in inertial coordinates:

$$
\hat b(\theta) = [\sin\theta,\ \cos\theta].
$$

**Equations of motion**

$$
\begin{aligned}
\dot x &= v_x, \\
\dot z &= v_z, \\
\dot v_x &= \frac{u_1}{m}\sin\theta + \frac{D_x}{m}, \\
\dot v_z &= \frac{u_1}{m}\cos\theta - g + \frac{D_z}{m}, \\
\dot\theta &= \omega, \\
\dot\omega &= \frac{\tau}{I(\dot m)} \\
\dot m &= -\frac{u_1}{I_{sp}g}\quad \text{(only when burning and } m>m_{\mathrm{dry}}\text{)}.
\end{aligned}
$$

**Pitch inertia**

A simple varying‑mass cylinder model:

$$
I(m)=\frac{1}{12}m(3R^2 + L^2).
$$

**Drag (optional)**

Quadratic drag opposing the velocity vector:

$$
\mathbf{D} = -\tfrac12\,\rho(z)\,C_D\,A_{\mathrm{ref}}\,\|\mathbf v\|\,\mathbf v.
$$

**Two‑thruster allocation (realism)**

If enabled, $(u_1,\tau)$ is allocated to two nonnegative thrusters $(T_L,T_R)$ with $0\le T_{L,R}\le T_{\max}$:

$$
\begin{aligned}
u_1 &= T_L + T_R, \\
\tau &= d\,(T_R - T_L).
\end{aligned}
$$

Requested $(u_1,\tau)$ may be infeasible; allocation clips the thrusters and returns the **achieved** $(u_1,\tau)$.

**Phases (as coded)**

0. **Attitude LQR (from $t=0$):** reorient until $|\theta|$ and $|\omega|$ satisfy tolerances (or timeout / touchdown)
1. **Coast:** $u_1=0,\ \tau=0$ until $z=z_{\text{burn}}$
2. **Burn (profiled + attitude PD):** profiled vertical descent + attitude stabilization
3. **Hover (z PD + attitude PD):** for $t_{\text{hover}}$ seconds (or touchdown)

**Safety margins (2D)**

The 2D script includes a helper that mirrors the 1D safety behavior:

- shift the tuning target: `v_target + v_safety`
- start earlier: `z_burn_safe = z_burn_nom + z_safety`

---

## 2. Project Setup

### 2.1 Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib

### 2.2 Install

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install numpy scipy matplotlib
```

---

## 3. Running the Simulations

> The scripts are configured via constants near the top (no CLI yet).

### 3.1 Run the 1D model

```bash
python <1d_script_name>.py
```

Outputs include:

- chosen $z_{\text{burn}}$
- touchdown detection and touchdown state
- plots for $m(t)$, $z(t)$, $v(t)$, plus thrust and $T/W$

### 3.2 Run the 2D model

```bash
python <2d_script_name>.py
```

Outputs include:

- chosen $z_{\text{burn}}$ (nominal + safety if enabled)
- touchdown state $[x,z,v_x,v_z,\theta,\omega,m]$
- plots for states + controls:
  - $u_1(t)$ (total thrust)
  - $\tau(t)$ (pitch torque)

---

## 4. Running the Code & Available Input Parameters

> **Note:** Both scripts are currently configured primarily through **constants near the top of the file** and a few **function arguments** (no CLI yet).

### 4.1 Common “Scenario” Inputs (shared concepts)

| Parameter                          | Meaning                                                          | Where it appears                                          |
| ---------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| `z0`                               | Initial altitude (m)                                             | constants                                                 |
| `z_floor`                          | Ground “floor” altitude (m) used for touchdown event             | constants + `event_touchdown()`                           |
| `z_error`                          | Hover‑start offset above the floor (m)                           | constants → `z_hover_start = z_floor + z_error`           |
| `t_hover`                          | Hover duration (s)                                               | `simulate_system(..., t_hover=...)`                       |
| `m_dry`                            | Dry mass (kg)                                                    | constants + RHS mass clamp                                |
| `prop_total` / `landing_prop_frac` | Landing prop allocation (kg fraction)                            | constants → `m0 = m_dry + landing_prop_frac * prop_total` |
| `Isp`                              | Specific impulse (s)                                             | `mdot` / `dm/dt` equations                                |
| `Cd`, `rho_0`, `H`, `diameter_m`   | Drag model parameters                                            | `rho_air(...)`, drag functions                            |
| `USE_DRAG`                         | Toggle drag on/off (2D only; 1D always uses drag in your script) | 2D constants + `drag_force_2d(...)`                       |

---

### 4.2 1D Script Parameters (Vertical)

| Parameter                  | Meaning                                              | Where it appears                                   |
| -------------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| `v0`                       | Initial vertical velocity (m/s)                      | constants                                          |
| `T_max`                    | Maximum thrust (N)                                   | constants + `make_rhs()` thrust clamp              |
| `area_ref`                 | Reference area for drag                              | constants                                          |
| `a_des`                    | Aggressiveness of velocity profile (m/s²)            | `thrust_profiled_descent(..., a_des=...)`          |
| `kd_v`                     | Velocity tracking gain for profiled burn             | `thrust_profiled_descent(..., kd_v=...)`           |
| `v_cap`                    | Cap on downward reference speed magnitude            | `thrust_profiled_descent(..., v_cap=...)`          |
| `kp_hover`, `kd_hover`     | Hover PD gains (altitude + velocity feedback)        | `simulate_system(..., kp_hover=..., kd_hover=...)` |
| `max_step`, `rtol`, `atol` | Integrator resolution / accuracy                     | each `solve_ivp(...)` call                         |
| `v_target`                 | Target touchdown vertical speed (m/s)                | `find_zburn(v_target=...)`                         |
| `tol`                      | Acceptable touchdown‑speed error band                | `find_zburn(tol=...)`                              |
| `n_scan`                   | Number of coarse scan samples for `z_burn` search    | `find_zburn(n_scan=...)`                           |
| `max_iter`                 | Max bisection iterations                             | `find_zburn(max_iter=...)`                         |
| `z_safety`                 | “Start burn earlier” margin added after tuning (m)   | `z_burn += z_safety` in `__main__`                 |
| `v_safety`                 | Safety shift applied to target touchdown speed (m/s) | `find_zburn(v_target=... + v_safety)`              |

---

### 4.3 2D Script Parameters (Planar Rigid‑Body)

#### Geometry / actuation

| Parameter                 | Meaning                                            | Where it appears                |
| ------------------------- | -------------------------------------------------- | ------------------------------- |
| `R`, `L`                  | Rocket radius + length (m)                         | geometry + inertia model        |
| `d_thruster`              | Lever arm for two‑thruster torque model (m)        | allocation + torque feasibility |
| `T_max`                   | Max thrust per engine (N)                          | constants                       |
| `u1_max`                  | Total max thrust (N), typically `2*T_max`          | constants                       |
| `USE_THRUSTER_ALLOCATION` | Whether to allocate $(u_1,\tau)$ into $(T_L,T_R)$   | RHS allocation step             |
| `tau_max_independent`     | Torque cap used only if allocation is disabled     | constants                       |

#### Initial conditions / attitude

| Parameter                    | Meaning                                      | Where it appears                          |
| ---------------------------- | -------------------------------------------- | ----------------------------------------- |
| `x0`, `vx0`                  | Initial horizontal position/velocity         | constants                                 |
| `vz0`                        | Initial vertical velocity                    | constants                                 |
| `theta0`, `omega0`           | Initial pitch angle and pitch rate           | constants                                 |
| `theta_tol_deg`, `omega_tol` | Attitude “settled” tolerances for ending LQR | constants + `event_attitude_settled(...)` |

#### Controllers / tuning

| Parameter                          | Meaning                                                  | Where it appears                                              |
| ---------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| `q_theta`, `q_omega`, `r_tau`      | LQR weights for attitude settle phase                    | `lqr_gain_attitude(...)`                                      |
| `a_des`, `kd_v`                    | Profiled descent parameters (vertical velocity tracking) | `make_ctrl_profiled_descent(...)`                             |
| `kp_theta_burn`, `kd_theta_burn`   | Attitude PD gains during burn                            | `make_ctrl_profiled_descent(..., kp_theta=..., kd_theta=...)` |
| `kp_z`, `kd_z`                     | Hover vertical PD gains                                  | `make_ctrl_hover(..., kp_z=..., kd_z=...)`                    |
| `kp_theta_hover`, `kd_theta_hover` | Attitude PD gains during hover                           | `make_ctrl_hover(..., kp_theta=..., kd_theta=...)`            |
| `t_max_phase`                      | Max duration allowed per phase                           | `run_segment(..., duration=t_max_phase)`                      |
| `max_step`, `rtol`, `atol`         | Integrator resolution / accuracy                         | `solve_ivp(...)`                                              |

#### 2D $z_{\text{burn}}$ + safety margins

| Parameter                   | Meaning                                              | Where it appears                            |
| --------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| `v_target`                  | Target touchdown vertical speed $v_z$ (m/s)          | `find_zburn(v_target=...)`                  |
| `tol`, `n_scan`, `max_iter` | Search tolerances and scan/bisection controls        | `find_zburn(...)`                           |
| `USE_SAFETY_MARGINS`        | Toggle safety behavior                               | constants + `choose_zburn_with_safety(...)` |
| `z_safety`                  | “Start burn earlier” margin (m)                      | `zb_safe = zb_nom + z_safety`               |
| `v_safety`                  | Safety shift applied to target touchdown speed (m/s) | `find_zburn(v_target + v_safety, ...)`      |

---

## 5. Plotting Notes

Both scripts:

- mark phase boundaries with vertical dashed lines
- use **phase coloring**
- render axis labels with **LaTeX‑style mathtext** (no external LaTeX install required)

In the 2D state plot, the legend is intentionally placed **only on the top subplot** to avoid duplication.

---

## 6. Assumptions and Limitations

- This is a point-mass translation model where lift and 3D wind is neglected.
- $I_{sp}$ and $\dot{m}$ are constant.
- Gravity is constant where $g = 9.80655\,\text{m/s}^2$.
- Drag model is simplified to constant $C_d$.
- Rocket engine is modeled as an instantaneous thrust command with saturation.
- No engine spool dynamics and gimbal limits beyond the simple torque model in 2D.
- Torque is modeled with an idealized two-thruster lever-arm approximation.
- Scenario A assumes 1D motion in which there are no attitude dynamics and the rocket is perfectly vertical.
- No horizontal motion in Scenario A, so all thrust is effectively vertical.
- The two thrusters are modeled as one thrust $T(t)$ in Scenario A.
- Rocket motion is planar.
- Thrust is bounded: $0 \le T \le T_{\max}$, and thrust is set to zero if $m \le m_{\text{dry}}$.

---

## 7. Reference

```text
J. Davidov, “ME200 Final Project: suicide-burn descent simulation (1D + 2D) with event-driven phases and z_burn tuning,” Dec. 2025.
```

```
shahar603, “Telemetry-Data: A collection of telemetry captured from SpaceX Launch Webcasts,” GitHub repository, last updated Jan. 24, 2020. [Online]. Available: https://github.com/shahar603/Telemetry-Data
. Accessed: Dec. 18, 2025.
```