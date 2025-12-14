
# ME200-FinalProject

This repository contains a **1D vertical “suicide burn” descent simulator** (Falcon 9–ish, upright rocket, no rotation) built with:

- A **3-phase descent model**: coast → profiled burn → hover (PD).
- A simple **exponential atmosphere + quadratic drag** model.
- **Thrust saturation** (`0 ≤ T ≤ T_max`) and **propellant limits** (no thrust below `m_dry`).
- **Event-driven phase transitions** using `solve_ivp()` events.
- A **z_burn optimizer** that scans + brackets + bisects to hit a target touchdown speed.
- A plotting utility that visualizes **mass, altitude, and velocity** across phases.

The script integrates the coupled ODE system for state `y = [z, v, m]` and reports whether a touchdown event was detected, the touchdown state, and the final state.


## 0. Description of the Final

The “final” deliverable for this repository is a working simulation that:

1. Starts from an initial altitude `z0` with an initial downward velocity `v0 < 0`.
2. Coasts with **zero thrust** until reaching a chosen burn altitude `z_burn`.
3. Ignites and performs a **profiled descent** toward a hover-start altitude
   - Hover-start altitude: `z_hover_start = z_floor + z_error`
   - Velocity reference (for `z > z_target`): `v_ref(z) = -sqrt(2 * a_des * (z - z_target))`
     - Then clamp with: `v_ref = max(v_ref, -v_cap)`
   (capped by `v_cap`) plus a velocity feedback term.
4. Switches into a **hover controller (PD)** at `z_hover_start` for `t_hover` seconds (or until touchdown).
5. Detects touchdown when `z(t)` crosses `z_floor` while descending.
6. Automatically selects a burn start altitude `z_burn` using `find_zburn()` so that the simulated touchdown speed is near a desired target (default `v_target = -0.25 m/s`), then adds a safety margin `z_safety`.

### Model (as implemented)

**Sign convention**
- `+z` is up
- descending means `v < 0`

**Equations of motion**
- `dz/dt = v`
- `dv/dt = (T(t,z,v,m) + F_D(v,z) - m g)/m`
- `dm/dt = -T/(Isp*g)` when burning and propellant remains

**Drag**
- Atmosphere: `rho(z) = rho_0 * exp(-z/H)`
- Drag force: `F_D = -0.5 * rho(z) * Cd * A_ref * v * |v|`
  - This opposes velocity, so for `v<0` drag tends to be positive (upward).

**Thrust constraints**
- `T` is clamped: `T = clip(T_raw, 0, T_max)`
- If `m ≤ m_dry`, thrust is forced to `0`.


## 1. Project Setup

### 1.1. Requirements

- **Python** ≥ 3.9
- **NumPy**
- **SciPy**
- **Matplotlib**

### 1.2. Create and Activate a Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell / CMD
````

### 1.3. Install Python Dependencies

```bash
pip install numpy scipy matplotlib
```

## 2. Running the Code & Available Input Parameters

> **Note:** The current script is configured primarily through **constants and function arguments** (it does not yet expose an argparse CLI like `--z0 ...`).

### 2.1. Basic Usage

From the repository root, run the main Python script (whatever filename you saved it as):

```bash
python <your_script_name>.py
```

You should see terminal output similar to:

* Initial mass and thrust-to-weight at ignition
* The chosen `z_burn`
* Whether touchdown was detected, with `(t_td, z_td, v_td, m_td)`
* A Matplotlib window with 3 stacked plots: **mass**, **altitude**, **velocity**

### 2.2. What to Edit for Different Scenarios (common knobs)

Most “inputs” live in the constants section near the top of the script.

| Parameter                          | Meaning                                               | Where it appears                                   |
| ---------------------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| `z0`                               | Initial altitude (m)                                  | constants                                          |
| `v0`                               | Initial vertical velocity (m/s), negative for descent | constants                                          |
| `z_floor`                          | Ground “floor” altitude (m) used for touchdown event  | constants + `event_touchdown()`                    |
| `z_error`                          | Hover-start offset above the floor (m)                | constants → `z_hover_start`                        |
| `m_dry`                            | Dry mass (kg)                                         | constants + `make_rhs()`                           |
| `prop_total` / `landing_prop_frac` | Landing prop allocation (kg)                          | constants → `m0`                                   |
| `Isp`                              | Specific impulse (s)                                  | constants + `mdot_from_thrust()`                   |
| `T_max`                            | Maximum thrust (N)                                    | constants + `make_rhs()`                           |
| `diameter_m`, `Cd`, `rho_0`, `H`   | Drag model parameters                                 | constants + `drag_force()`                         |
| `t_hover`                          | Hover duration (s)                                    | `simulate_system(..., t_hover=...)`                |
| `a_des`, `kd_v`, `v_cap`           | Profiled descent controller tuning                    | `thrust_profiled_descent(...)`                     |
| `kp_hover`, `kd_hover`             | Hover PD gains                                        | `simulate_system(..., kp_hover=..., kd_hover=...)` |
| `max_step`, `rtol`, `atol`         | Integrator resolution/accuracy                        | `solve_ivp(...)` calls                             |
| `v_target`, `tol`, `n_scan`        | z_burn search target + tolerances                     | `find_zburn(...)`                                  |

### 2.3. Key Functions (what they do)

* `simulate_system(z_burn, ...)`
  Runs phases 1–3 using event-triggered transitions:

  1. **coast**: `T=0` until `z=z_burn` (or touchdown)
  2. **burn (profiled)**: velocity-profile descent to `z_hover_start` (or touchdown)
  3. **hover (PD)**: hold `z≈z_hover_start`, `v≈0` for `t_hover` (or touchdown)

* `find_zburn(v_target=-0.25, tol=0.05, n_scan=41, max_iter=60)`
  Finds an approximate `z_burn` that makes touchdown velocity near `v_target`:

  * coarse scan over `n_scan` candidate altitudes
  * tries to find a sign-change bracket in `v_td - v_target`
  * bisection refine if bracket exists
  * local refinement otherwise
  * final safety margin is added in `__main__` via `z_burn += z_safety`

* `plot_segments(segments)`
  Produces the mass/altitude/velocity plots and marks phase boundaries with dashed vertical lines.

### 2.4. Example: Change the Target Touchdown Speed

In `__main__`, edit:

```python
z_burn = find_zburn(v_target=-0.25, tol=0.05)
```

For a “softer” target touchdown (closer to 0 from below), try:

```python
z_burn = find_zburn(v_target=-0.10, tol=0.05)
```

### 2.5. Example: Adjust the Descent Aggressiveness

Inside `simulate_system`, the burn controller is constructed as:

```python
burn_law = thrust_profiled_descent(z_target=z_hover_start, a_des=a_des, kd_v=kd_v)
```

Increase `a_des` for more aggressive deceleration (at the cost of more thrust/prop usage), or adjust `kd_v` for tighter/looser velocity tracking.

## 3. References

```
J. Davidov, “Prompt to ChatGPT 5.2 Thinking requesting README.md generation,” ChatGPT 5.2 Thinking (large language model), OpenAI, prompt: “Write a README.md file similar in style to the following … in respect to the repository at ‘[https://github.com/mydkm/ME200-FinalProject’](https://github.com/mydkm/ME200-FinalProject’) … ” Dec. 14, 2025.
```
