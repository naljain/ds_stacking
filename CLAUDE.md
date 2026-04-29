# Project Context for Claude Code

## Course

MEAM 6230 (Penn) — final project. Team: Nalini Jain, Shivank Gupta, Thomas Stephen Felix.

## One-line summary

Two Franka Panda arms in Isaac Sim, each running a learned **joint-space Neural Dynamical System** with **Huber 2019 modulation** for inter-arm collision avoidance, evaluated on cube stacking under perturbations.

## Why these specific design choices

These aren't defaults — earlier iterations of the project used a Cartesian DS + RMPflow + a discrete FSM coordinator. We deliberately moved to the current architecture so the project is genuinely a *dynamical systems* project rather than "robotics that uses a DS as one component." If you're tempted to suggest changes that walk these back, push back on the user first.

### Joint-space DS (not Cartesian)

The DS is `q̇ = f_θ(q, q*)` where `q ∈ R^7` is the Franka joint configuration and `q*` is the target joint config for the current primitive. At deployment, network output is sent to the robot verbatim — **no IK at runtime, no Jacobian inversion**. The closed-loop joint dynamics ARE the trained DS modulo low-level actuator dynamics, so Lyapunov stability claims hold in the actual control space. Cartesian DS + Jacobian-pseudoinverse would push singularity / null-space issues into the closed loop and break the stability story.

### Huber 2019 modulation (not FSM coordination)

Inter-arm collision avoidance is **smooth, state-dependent velocity shaping**, not a discrete "if EEs are close, hold one arm" finite-state machine. We use the modulation framework of Huber, Billard, Slotine 2019 (*"Avoidance of Convex and Concave Obstacles with Convergence Ensured Through Contraction"*, IEEE RA-L), which extends Khansari-Zadeh & Billard 2012 with: (a) reference-direction basis (eliminates antipodal saddle-points), (b) tail-effect gating (no damping on outward-pointing motion), (c) contraction-based convergence guarantee. The closed loop is therefore a coupled DS, not a hybrid system, so we don't need dwell-time analysis.

### Lyapunov stability

`V(q, q*) = ||g([q,q*]) - g([q*,q*])||² + ε||q - q*||²`, positive-definite around `q*` by construction. Training enforces `dV/dt + α·V ≤ 0` on the data distribution as a soft loss. The `safe_velocity()` method projects f(x) onto the half-space where this holds exactly — this is opt-in via `--use_safe` at deployment so we can ablate soft vs hard stability.

## Repo structure

```
src/
  env.py             Isaac Sim scene (two Frankas + table + blocks). Builds from configs/default.yaml.
  primitives.py      5 primitives: reach, grasp, lift, transport, place. Cartesian targets + completion checks.
  ik_controller.py   RMPflow wrapper. Used ONLY at data-collection time. Not at deployment.
  franka_ik.py       Lula IK wrapper for q* lookup. Auto-discovers config paths across Isaac Sim 4.x/5.x.
  neural_ds.py       Joint-space Neural DS + Lyapunov network. StableNeuralDS class.
  modulation.py      Huber 2019 modulation. HuberModulation + InterArmModulation classes.
  coordinator.py     Slim primitive sequencer. NO collision logic — that's modulation's job.
  perturbations.py   BlockDisplacement, EEDisturbance, ArmBlock for evaluation.

scripts/
  smoke_test.py            Verify scene loads
  collect_ik.py            Generate (q, q*, q̇) demos with RMPflow
  teleop.py                Manual demo collection
  train_ds.py              Train one primitive
  train_all.sh             Train all 5 primitives
  deploy_single_arm.py     Single arm DS deployment
  deploy_dual_arm.py       Dual arm + Huber modulation
  evaluate.py              Full eval suite, logs diagnostics for plotting
  plot_modulation.py       3 figures: field, gamma timeseries, radial dot

configs/default.yaml       All hyperparameters and constants
data/demonstrations/       Saved (q, q*, q̇) trajectory pickles
data/checkpoints/          Trained model .pt files (one per arm × primitive)
data/results/              Eval JSON + diagnostic pickles + figures
```

## Pipeline

```
1. python scripts/smoke_test.py
2. python scripts/collect_ik.py --arm left  --n_demos 50
   python scripts/collect_ik.py --arm right --n_demos 50
3. bash scripts/train_all.sh
4. python scripts/deploy_single_arm.py --arm left      # sanity check
5. python scripts/deploy_dual_arm.py                   # full system
6. python scripts/evaluate.py --n_trials 10
7. python scripts/plot_modulation.py all --diag data/results/diag_<ts>.pkl
```

## Environment

- Conda env: `franka_isaac` (Python 3.11)
- Isaac Sim 5.1 via pip: `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com`
- PyTorch (CUDA), numpy, pyyaml, matplotlib

Always activate the conda env before running anything: `conda activate franka_isaac`.

## Important conventions

- **Never put omni / isaacsim imports at the top of a script.** They must come AFTER `SimulationApp(...)` is instantiated, otherwise omniverse complains about modules loaded too early. The pattern is: argparse → SimulationApp → imports → main logic.
- **Joint conventions** — Franka has 9 joints in the articulation (7 arm + 2 fingers). All DS work is on `q[:7]`. Fingers are controlled separately via `franka.gripper.apply_action(...)`.
- **`render=not args.headless`** — every `world.step()` and `env.step()` call needs this so headless mode actually skips rendering.
- **Configs over magic numbers** — table dimensions, primitive heights, training hyperparams etc. all live in `configs/default.yaml`, not as constants in code. If something needs tuning, check the config first.
- **One IK call per primitive transition.** The DS handles smooth motion within a primitive; Lula IK is only called when the primitive switches and a new `q*` is needed. Don't put IK in the inner loop.

## Known footguns

1. **`extsDeprecated` Lula configs are broken in Isaac Sim 5.x.** The YAML files at `.../extsDeprecated/omni.isaac.motion_generation/.../franka/rmpflow/robot_descriptor.yaml` are stubs. The real configs live under `isaacsim.robot_motion.motion_generation`. `franka_ik.py` already auto-discovers and prefers non-deprecated paths — don't hardcode paths there.

2. **Quaternion order is (w, x, y, z)** in Isaac Sim. The `FACE_TABLE` quaternion `[0.7071, 0, 0, 0.7071]` is +90° around Z. Don't confuse with `(x, y, z, w)` ordering used elsewhere (e.g. ROS).

3. **Mimic joint warning is harmless.** `Joint 'panda_finger_joint2' is specified as a mimic joint...` shows up at startup; ignore it. Lula doesn't model mimic constraints, but we control fingers manually anyway.

4. **Recording in collect_ik.py uses a closure over `prev_q`.** When refactoring, preserve the `nonlocal prev_q` pattern — losing it silently breaks finite-difference velocities.

5. **Jacobian via finite differences is slow.** `jacobian_finite_difference()` does 7 set-and-restore operations per call. Fine for evaluation but if it becomes a bottleneck, swap to Isaac Sim's analytical Jacobian (`articulation.get_jacobians()` — exact field/indexing has shifted between versions, check what works on the install).

## Evaluation conditions

```
nominal              no perturbations (baseline)
block_displacement   teleport target block by random XY mid-task
ee_disturbance       force impulse on EE during transport
arm_block            freeze one arm for ~1s
combined             all of the above
```

Ablations available via flags: `--no_modulation` (FSM-free, naive parallel), `--use_safe` (Lyapunov projection on).

## Metrics (evaluate.py)

- `stack_completion_rate` — fraction of trials with all 6 blocks placed
- `blocks_placed_avg` — average per trial
- `avg_time_per_cube` — total simulated time / blocks placed
- `grasp_failure_rate` — fraction of grasps that didn't pick up
- `collisions_avg` — number of EE-proximity events per trial
- `recovery_success_rate` — fraction completing despite perturbation

## Working style notes for Claude Code

- Be honest about whether a proposed change actually fits the DS framing. If a fix walks back the joint-space DS or replaces modulation with a coordinator, flag it explicitly rather than just doing it.
- Prefer adding a new ablation flag over removing existing behaviour. The whole point of having both `--no_modulation` and `--use_safe` is that the writeup needs them.
- The user is a robotics student, not an Isaac Sim expert. When something fails because of an Isaac Sim version quirk, explain the quirk briefly, don't just patch silently.
- When debugging Isaac Sim runtime issues, the fastest signal is usually the warning *before* the error in the log — the actual exception is often a downstream symptom.
