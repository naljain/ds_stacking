# DS Stacking — Change Log

**Project:** MEAM 6230 Final Project — Franka dual-arm cube stacking with learned DS  
**Team:** Nalini Jain, Shivank Gupta, Thomas Stephen Felix

---

## `configs/default.yaml`

### Shared stacking goal
Replaced separate per-arm goals with a single shared goal at the table centre.
Both arms transport their blocks to the same XY position, building one unified stack.
```yaml
# Before
goals:
  left:  [-0.30, 0.65]
  right: [ 0.30, 0.65]

# After
shared_goal: [0.0, 0.65]
```

### Block layout — randomised positions
Removed fixed block `pos` fields. Blocks now spawn at random positions within
per-arm workspace bounds each demo, giving the DS varied starting configurations.
```yaml
block_workspace:
  left:  {x_min: -0.50, x_max: -0.15, y_min: 0.20, y_max: 0.60}
  right: {x_min:  0.15, x_max:  0.50, y_min: 0.20, y_max: 0.60}
  min_block_spacing: 0.08
```

### Arm default joint poses
Added explicit default joint poses (raw radians) used for arm reset and IK seeding.
Left arm uses the reference pose; right arm mirrors joints 0 and 1.
```yaml
arms:
  default_joints_left:  [ 1.93,  1.10, -1.8639, -2.6820,  1.0833,  1.9480, -0.1237]
  default_joints_right: [-1.93, -1.10, -1.8639, -2.6820,  1.0833,  1.9480, -0.1237]
```

### Coordination yield radius
Added `yield_radius` to prevent both arms descending onto the stack simultaneously.
```yaml
coordination:
  yield_radius: 0.12   # metres — arm waits if other EE is within this of stack goal
```

---

## `src/env.py`

### Randomised block positions (`reset_blocks`)
`reset_blocks()` now samples random `(x, y)` within per-arm workspace bounds with
minimum separation enforcement. Accepts an `rng` parameter for reproducibility.

### Default joint pose on spawn (`_apply_default_joints`)
After `world.reset()`, all arms are teleported to their configured default joint
pose and allowed to settle. Call `reset_arms()` between demos to restore this pose.

### Block grasp quaternion (`get_block_grasp_quat`)
New method reads the block's world orientation quaternion, extracts the yaw, snaps
it to the nearest 90° face, and returns a gripper-down EE quaternion aligned to
the block. This replaces random yaw sampling during collection.

### Shared goal for both arms
Both arms' goal markers point to `cfg["shared_goal"]` instead of separate per-arm
goal positions.

---

## `src/franka_ik.py`

### Elbow consistency enforcement
`solve()` now checks that the returned solution has the same elbow sign (joint 1)
as the seed. If the elbow flips, the seed is nudged into the desired half-space
and the solve is retried up to 5 times. This prevents Lula from returning
inconsistent elbow configurations across demos, which was the primary cause of
messy training data.

---

## `src/ik_controller.py`

**Complete rewrite** — replaced RMPflow with a straight-line Cartesian IK controller.

### Why
RMPflow is a reactive potential-field controller that doesn't follow a specific path.
It arcs, jitters, and resolves redundancy on its own schedule, producing inconsistent
demonstrations that are hard for the DS to learn from.

### What it does now
- Interpolates EE position linearly from current to target (straight-line Cartesian path)
- Uses a **trapezoidal velocity profile** (ramp up / cruise / ramp down) to avoid
  joint-velocity spikes at path endpoints
- Solves Lula IK at every waypoint using the previous joint solution as warm-start,
  keeping the arm in the same solution branch throughout the move
- Tracks gripper state internally and writes all 9 joints (7 arm + 2 fingers) in a
  **single `apply_action` call**, fixing the gripper command dropping bug

### Arm-aware rest pose
Constructor takes `arm="left"/"right"` and seeds IK with the configured default
joint pose, keeping all solves in the elbow-up homotopy class.

---

## `src/lpv_ds.py`

**New file** — LPV-DS (Linear Parameter-Varying Dynamical System) as an alternative
to the neural DS, ported from the EPFL LASA MATLAB library (Figueroa & Billard, CoRL 2018).

### Why LPV-DS
The transport primitive has a clean converging structure (many start positions → one
fixed goal) that LPV-DS was designed for. Stability is guaranteed by construction
via an SDP constraint (`A_k + A_k^T ≺ 0`), with no need for Lyapunov projection
at runtime.

### Architecture
```
x_dot_xy = Σ_k h_k(x_xy) * A_k * (x_xy - x_goal_xy)
```
- Operates in **2D XY only** — Z is constant at `lift_h` during transport and
  including it caused the DS to output mostly vertical velocity (Z std = 0.007m
  vs XY std = 0.12m; any normalisation distorts the velocity direction)
- GMM fitted via BIC model selection on normalised XY positions
- SDP solved with `cvxpy` (CLARABEL solver) to enforce `A_k + A_k^T ≺ -εI`
- Isotropic normalisation (single scalar `x_std`) preserves velocity directions

### Key bugs fixed during development
1. **Zero-velocity first steps** — FD velocity fallback set `xdot[0] = 0` for the
   first step of every block segment. Fixed by always using consecutive-step FD and
   skipping `i=0` entirely.
2. **Outlier velocity filtering** — sim reset artifacts caused steps with `>1 m/s`
   EE speed. These are dropped before training.
3. **Z dimension dominance** — per-dimension or isotropic normalisation both failed
   because Z is nearly constant. Fixed by operating in 2D XY only.

---

## `src/coordinator.py`

### Shared `goal_z` counter
Previously each arm had its own stack height counter. Now there is one shared
`goal_z` that increments whenever *either* arm completes a place, building a single
unified stack rather than two separate ones at the same XY.

### `can_place()` yield gate
New method that returns `False` if the other arm's EE is within `yield_radius` of
the shared goal XY and the other arm is in `transport` or `place`. This prevents
simultaneous descent onto the same point without disrupting the continuous DS loop.

---

## `scripts/collect_ik.py`

### Transport-only collection
**Only the transport segment is recorded.** Reach, grasp, lift, and place are
executed via IK for setup/teardown but not saved. The DS learns only the
block-to-stack-position motion.

### Fixed IK seed
All Lula IK solves use `default_joints` as the seed instead of the current joint
state. This keeps all solutions in the same elbow homotopy class across all demos.

### Shared goal
`goal_xy` reads from `cfg["shared_goal"]` so both arms target the same position.

### Block position known perfectly
No XY noise on the grasp target. The block is picked up at its exact centre.
Small variation in transport start positions comes naturally from block layout
randomisation in `env.reset_blocks()`.

### EE velocity recording
`record()` now captures `ee_vel` via finite difference of consecutive `ee_pos`
values. Used by `train_lpvds.py`; the neural DS trainer ignores it.

### EE orientation aligned to block face
Uses `env.get_block_grasp_quat()` to read the block's actual world yaw and snap
the gripper to the nearest 90° face, replacing random yaw sampling.

---

## `scripts/train_lpvds.py`

**New file** — trains the LPV-DS model outside Isaac Sim (pure numpy/cvxpy).

- Reads `x_goal` from the `"target"` field stored in demo data (no IK re-run needed)
- Calls `LPVDS.fit()` with BIC-selected GMM and SDP optimisation
- Saves as `.pkl` (not `.pt` — no PyTorch needed at train or deploy time)

Usage:
```bash
python scripts/train_lpvds.py --arm left --K_max 5
```

---

## `scripts/clean_demos.py`

**New file** — filters demonstration data before training.

Computes per-segment statistics and rejects outliers using **Median Absolute Deviation**
(robust to extreme outliers, unlike mean ± std):
- **Path length** `Σ||Δq||` — detects demos that took a detour
- **Smoothness** `Σ||Δq_dot||²` — detects jerk spikes from IK branch jumps

Default threshold: `--sigma 2.5` (2.5 MADs from median).
Use `--plot` to visualise the distributions before committing to a threshold.

---

## `scripts/deploy_single_arm.py`

### Transport-only deployment
Removed the 5-primitive DS loop. Now:
1. **Reach / grasp / lift / place** — IK straight-line via `IKController`
2. **Transport** — DS (neural or LPV-DS)

### `--model lpvds` flag
Loads `{arm}_transport_lpvds.pkl` and uses `LPVDS.predict()` for transport.
Convergence check uses EE distance to `x_goal` in metres.

### IK velocity for LPV-DS deployment
Instead of computing a Jacobian pseudoinverse (which requires finite-difference
perturbations that disturb the sim), the LPV-DS output is used as:
```python
ee_next   = ee_pos + x_dot_des * dt
q_next, _ = ik.solve(ee_next, ...)
q_dot     = (q_next - q) / dt
```
This is equivalent to `J^+ @ x_dot` but uses Lula IK, is singularity-robust,
and never perturbs the simulation state.

### Lazy Isaac Sim imports
All `isaacsim` / `omni.isaac` imports are inside `main()`, after `SimulationApp`
is instantiated. Top-level imports caused `ModuleNotFoundError` because Isaac Sim
extensions aren't registered until after `SimulationApp.__init__` completes.

### Single `apply_action` for all 9 joints
Gripper state is tracked via `ik_motion._finger_width` and written together with
arm joints in one `ArticulationAction`, fixing the split-command dropping bug.

### `weights_only=False` for PyTorch checkpoints
Neural DS checkpoints contain numpy arrays (normalisation stats) which PyTorch 2.6+
blocks under the new `weights_only=True` default. Added explicit `weights_only=False`.

---

## `scripts/deploy_dual_arm.py`

### Transport-only with inter-arm modulation
Same IK-for-all-other-primitives structure as `deploy_single_arm.py`.
DS (neural or LPV-DS) runs only during transport, with Huber modulation applied
to both arms simultaneously:
```python
q_dot_modulated = InterArmModulation.modulate(q_dot_nominal, ee_self, ee_other, J)
```

### Per-arm state machine
Each arm progresses through `REACH → GRASP → LIFT → TRANSPORT → PLACE → RETRACT`
independently. IK stages run synchronously; transport steps are interleaved so
both arms can be modulated against each other each tick.

### `can_place()` yield gate
Prevents both arms descending onto the shared stack simultaneously. The hovering
arm stays at its `transport` position (DS holds it there) until the other arm
clears `yield_radius`.

---

## Pipeline Summary

```
# Data collection
python scripts/collect_ik.py --arm left  --n_demos 50
python scripts/collect_ik.py --arm right --n_demos 50

# Clean data
python scripts/clean_demos.py --arm left  --plot
python scripts/clean_demos.py --arm right --plot

# Train neural DS (joint space)
python scripts/train_ds.py --primitive transport --arm left
python scripts/train_ds.py --primitive transport --arm right

# Train LPV-DS (Cartesian 2D, alternative)
python scripts/train_lpvds.py --arm left
python scripts/train_lpvds.py --arm right

# Deploy single arm (smoke test)
python scripts/deploy_single_arm.py --arm left                  # neural DS
python scripts/deploy_single_arm.py --arm left --model lpvds    # LPV-DS

# Deploy dual arm
python scripts/deploy_dual_arm.py --use_safe
```
