# Project Context for Claude Code

## Course

MEAM 6230 (Penn) — final project. Team: Nalini Jain, Shivank Gupta, Thomas Stephen Felix.

## One-line summary

Two Franka Panda arms in Isaac Sim using a hybrid controller: learned **joint-space Neural DS** for `reach` and `transport`, Lula joint-space control for `grasp`, `lift`, and `place`, Huber 2019 end-effector modulation, sampled-link safety holds, dynamic stack clearance, and return-home parking.

## Why these specific design choices

These aren't defaults — earlier iterations of the project used a Cartesian DS plus a discrete FSM coordinator. We deliberately moved to the current architecture so the project is genuinely a *dynamical systems* project rather than "robotics that uses a DS as one component." If you're tempted to suggest changes that walk these back, push back on the user first.

### Joint-space DS (not Cartesian)

The DS is `q̇ = f_θ(e)` where `e = q - q_goal`, `q ∈ R^7` is the Franka joint configuration, and `q_goal` is the target joint config for the current primitive. The model is parameterized as `f(e_n) = residual_θ(e_n) - stable_skip_gain * e_n`, so the convergent linear prior is inside the learned DS architecture rather than added as a deployment controller. At deployment, Lula IK is called once at each primitive transition to compute `q_goal`. For `reach` and `transport`, the inner loop uses the learned joint-space velocity field, not Cartesian IK or Jacobian inversion. Cartesian DS + Jacobian-pseudoinverse would push singularity / null-space issues into the closed loop and break the stability story.

### Huber 2019 modulation plus link safety

End-effector collision avoidance is smooth, state-dependent velocity shaping. We use the modulation framework of Huber, Billard, Slotine 2019 (*"Avoidance of Convex and Concave Obstacles with Convergence Ensured Through Contraction"*, IEEE RA-L), which extends Khansari-Zadeh & Billard 2012 with: (a) reference-direction basis (eliminates antipodal saddle-points), (b) tail-effect gating (no damping on outward-pointing motion), (c) contraction-based convergence guarantee. EE-only modulation was not enough for this geometry, so dual-arm deployment also samples Franka link poses and uses a discrete hold/release guard when elbows or forearms get too close. Treat this sampled-link hold as a pragmatic deployment safety guard, not as part of the continuous modulation proof.

### Lyapunov stability

`V(e) = ||g(e) - g(0)||² + ε||e||²`, positive-definite around `e = 0` by construction. Training enforces `dV/dt + α·V ≤ 0` on the data distribution as a soft loss. The `safe_velocity()` method projects f(x) onto the half-space where this holds exactly — this is opt-in via `--use_safe` at deployment so we can ablate soft vs hard stability.

## Repo structure

```
src/
  env.py             Isaac Sim scene (two Frankas + table + blocks). Builds from configs/default.yaml.
  primitives.py      5 primitives: reach, grasp, lift, transport, place. Cartesian targets + completion checks.
  franka_ik.py       Lula IK wrapper for q_goal lookup. Auto-discovers config paths across Isaac Sim 4.x/5.x.
  neural_ds.py       Joint-space Neural DS + Lyapunov network. StableNeuralDS class.
  modulation.py      Huber 2019 modulation. HuberModulation + InterArmModulation classes.
  coordinator.py     Primitive sequencer with block order, stack-slot reservation, stack clearance, and return-home state.
  perturbations.py   BlockDisplacement, EEDisturbance, ArmBlock for evaluation.

scripts/
  smoke_test.py            Verify scene loads
  collect_ik.py            Generate (q, q_goal, q̇) joint-space demos
  audit_demo_labels.py     Audit primitive/q_goal label consistency
  teleop.py                Manual demo collection
  train_ds.py              Train one learned DS primitive
  train_all.sh             Train reach/transport DS primitives
  deploy_single_arm.py     Single arm DS + Lula scripted deployment
  deploy_dual_arm.py       Dual arm + Huber modulation + sampled-link safety
  evaluate.py              Full eval suite, logs diagnostics for plotting
  plot_modulation.py       3 figures: field, gamma timeseries, radial dot

configs/default.yaml       All hyperparameters and constants
data/demonstrations/       Saved (q, q_goal, q̇) trajectory pickles
data/checkpoints/          Trained model .pt files for reach/transport
data/results/              Eval JSON + diagnostic pickles + figures
```

## Pipeline

```
1. python scripts/smoke_test.py
2. python scripts/collect_ik.py --arm left  --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
   python scripts/collect_ik.py --arm right --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
3. python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
4. bash scripts/train_all.sh
5. python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --cart_done_tol 0.02 --place_cart_done_tol 0.01 --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv
6. python scripts/deploy_dual_arm.py --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --cart_done_tol 0.02 --place_cart_done_tol 0.01 --mod_safe_radius 0.25 --mod_reactivity 2.0 --link_safety_radius 0.20
7. python scripts/evaluate.py --n_trials 10 --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25
8. python scripts/plot_ds.py --all --ckpt_arm both --use_safe --joints 0 1 --out_dir data/results/ds_plots
9. python scripts/plot_modulation.py all --diag data/results/diag_<ts>.pkl
```

If `collect_ik.py` behaviour changes, rerun from step 2 onward. Do not retrain
from old demos that may contain missed grasps or target-noise artifacts.

## Environment

- Conda env: `franka_isaac` (Python 3.11)
- Isaac Sim 5.1 via pip: `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com`
- PyTorch (CUDA), numpy, pyyaml, matplotlib

Always activate the conda env before running anything: `conda activate franka_isaac`.

Isaac Sim's Franka helper needs the Isaac robot assets. If the default assets
root is unreachable, set `assets.franka_usd` in `configs/default.yaml` to a
local `FrankaPanda/franka.usd`. `env.py` uses a local cuboid ground plane to
avoid `add_default_ground_plane()` pulling from the remote asset root.

## Important conventions

- **Never put omni / isaacsim imports at the top of a script.** They must come AFTER `SimulationApp(...)` is instantiated, otherwise omniverse complains about modules loaded too early. The pattern is: argparse → SimulationApp → imports → main logic.
- **Joint conventions** — Franka has 9 joints in the articulation (7 arm + 2 fingers). All DS work is on `q[:7]`. Fingers are controlled separately via `franka.gripper.apply_action(...)`.
- **`render=not args.headless`** — every `world.step()` and `env.step()` call needs this so headless mode actually skips rendering.
- **Configs over magic numbers** — table dimensions, primitive heights, training hyperparams etc. all live in `configs/default.yaml`, not as constants in code. If something needs tuning, check the config first.
- **One IK call per primitive transition.** Lula IK is only called when the primitive switches and a new `q_goal` is needed. `reach` and `transport` use the DS inside the primitive; `grasp`, `lift`, and `place` use the scripted Lula joint-space controller.
- **Coordinator stays slim.** It owns block order, primitive order, stack-slot reservation, dynamic stack clearance, return-home state, and the initial arm phase offset. Continuous EE avoidance belongs to modulation; sampled-link hold is an explicit deployment safety guard.
- **Dual-arm start is intentionally staggered.** `coordination.start_stagger_steps` delays the right arm so both arms do not reach the shared stack at the same time. This is phase scheduling, not priority arbitration.

## Known footguns

1. **`extsDeprecated` Lula configs are broken in Isaac Sim 5.x.** The YAML files under deprecated motion-generation extension paths are stubs. The real configs live under `isaacsim.robot_motion.motion_generation`. `franka_ik.py` already auto-discovers and prefers non-deprecated paths — don't hardcode paths there.

2. **Quaternion order is (w, x, y, z)** in Isaac Sim. The `FACE_TABLE` quaternion `[0.7071, 0, 0, 0.7071]` is +90° around Z. Don't confuse with `(x, y, z, w)` ordering used elsewhere (e.g. ROS).

3. **Mimic joint warning is harmless.** `Joint 'panda_finger_joint2' is specified as a mimic joint...` shows up at startup; ignore it. Lula doesn't model mimic constraints, but we control fingers manually anyway.

4. **Recording in collect_ik.py uses a closure over `prev_q`.** When refactoring, preserve the `nonlocal prev_q` pattern — losing it silently breaks finite-difference velocities.

5. **Target noise in collection can cause missed grasps.** `collect_ik.py --noise` is legacy target noise and should normally stay at 0. Default collection is conservative: no target noise and no block jitter. Once the base grasp is reliable, use `--block_xy_jitter` to move the physical blocks and widen the data distribution without commanding the gripper beside the cube.

6. **Collection separates motion demos from contact grasp validation.** By default `collect_ik.py` kinematically carries the active block after `grasp` so joint-space demos are not discarded due Isaac contact flakiness. Pass `--physical_grasp` to require the gripper/contact setup to actually lift the cube; in that mode failed lifts are discarded.

7. **Default collection should match deployment q_goal.** Collection computes the same Lula target used at deployment, then records a joint-space expert moving toward it. Transport collection uses the same dynamic stack clearance as deployment.

8. **`q_goal` labels must match the demonstrated attractor.** The collector stores `q_goal_lula`, `q_goal_settled`, and `q_goal_lula_error` metadata. If `audit_demo_labels.py` shows large final `||q-q_goal||` or many negative `cos(q_dot, -error)` samples, retrain only after recollecting or relabeling; otherwise the learned flow can diverge even when primitive names are correct.

9. **Training data partition is primitive-label based.** Demonstration pickles contain full trajectories. `train_ds.py` filters by `step["primitive"]`; the current learned DS checkpoints are for `reach` and `transport`, while `grasp`, `lift`, and `place` run through Lula. The checkpoint stores `data_manifest` with demo files, samples by file, samples by arm, samples by block, samples by stack slot, and label-source counts. Check this manifest before blaming model behavior on architecture.

10. **Shared stack slots are reserved before transport/place.** The coordinator reserves stack heights when an arm asks for a transport/place target, not only after placement completes. Otherwise two arms can target the same stack layer.

11. **Kinematic-carry release must use the reserved stack slot.** In debug deployment with `--kinematic_carry`, the carried cube follows the EE during motion, then snaps to `TaskSequencer.stack_target_position(arm)` when `place` opens the gripper. Do not release at the raw EE-plus-offset pose, or every block can appear to land near the same table-height pose even though the coordinator reserved increasing stack heights.

12. **Single-arm deploy aborts on timeout by default.** This is intentional. Advancing after a failed `reach` closes the gripper from the wrong pose and hides the real failure. Use `--advance_on_timeout` only for phase-flow debugging.

13. **Collection speed is intentionally modest.** The current defaults are `--joint_goal_gain 2.0`, `--collection_max_joint_vel 1.2`, and `sim.inter_primitive_pause_steps: 120`. Faster data can look fine visually but tends to create sharper finite-difference velocities and messier DS fits.

14. **Place completion should be Cartesian-tight.** The block can appear to snap onto the stack with kinematic carry even if the gripper releases from too far away. Use `--place_cart_done_tol 0.01` when judging deployment.

15. **Completed arms should return home.** Dual-arm deployment parks an arm after its last block by default. Disabling that can leave the arm next to the stack and block the other arm's final placements.

16. **Jacobian via finite differences is slow.** `jacobian_finite_difference()` does 7 set-and-restore operations per call. Fine for evaluation but if it becomes a bottleneck, swap to Isaac Sim's analytical Jacobian (`articulation.get_jacobians()` — exact field/indexing has shifted between versions, check what works on the install).

## Evaluation conditions

```
nominal              no perturbations (baseline)
block_displacement   teleport target block by random XY mid-task
ee_disturbance       force impulse on EE during transport
arm_block            freeze one arm for ~1s
combined             all of the above
```

Ablations available via flags: `--no_modulation`, `--no_link_safety_hold`, and
`--use_safe` (Lyapunov projection on). The learned DS setting for `reach` and
`transport` is `--ds_scale 1.0 --goal_gain 0.0`, optionally with `--use_safe`
for hard Lyapunov projection. `--goal_gain > 0`, `--ds_scale < 1`, and
`--ds_scale 0` are diagnostics for isolating IK/actuation/data issues; do not
present them as the method or main baseline.

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


## The project proposal

Reactive Dual-Arm Cube Stacking in Isaac Gym with Learned DS Motion Primitives
Nalini Jain, Shivank Gupta, Thomas Stephen Felix
MEAM 6230 Final Project Proposal

Motivation:
Fast manipulation tasks like cube stacking demand more than nominal trajectory replay, they require online adaptation to correct pose error, timing mismatch, and inter-arm interference. This project explores whether DS-based motion primitives can make dual-arm manipulation both fast and robust under perturbations.

Goal:
Our goal is to build a dual-arm robotic system in simulation that can stack cubes quickly while remaining stable, reactive, and collision-aware. We specifically want to test learned DS-based pick-and-place primitives from teleoperated demonstrations and deploy them under a shared coordination layer that handles task sequencing, synchronization, and inter-arm collision avoidance.

Approach:
We will first create a lightweight teleoperation interface in Isaac Gym to collect demonstrations of single-arm pick-and-place motions such as reaching, grasping, lifting, transporting, and placing. From these demonstrations, we will fit DS-based motion primitives for each arm, then deploy them in a dual-arm setup where a shared task-level coordination layer handles synchronization, task sequencing, and inter-arm collision avoidance. This allows us to learn local manipulation skills from demonstration while preserving the reactive DS structure needed for online adaptation.

Evaluation Plan:
We will evaluate the method entirely in simulation in Isaac Gym. Metrics will include stack completion rate, average time per cube, grasp failure rate, collision count, and recovery success after perturbations such as object displacement, timing mismatch, or temporary blocking of one arm. 



Anticipated Challenges:
One challenge will be collecting demonstrations that are simple enough to learn from but still representative of successful pick-and-place behavior. Another difficulty will be coordinating two arms in a shared workspace without sacrificing speed or safety, especially during simultaneous motions near the stack. A final challenge will be ensuring that the learned primitives remain stable and precise enough for repeated stacking rather than just producing approximate reaching behavior.

Implementation Details: 
The project will be implemented in Python using Isaac Gym for simulation. DS fitting will be done by learning a neural parameterization of the motion vector field from demonstrations. The teleoperation interface, task sequencer, and coordination layer will be developed from scratch. Development will be divided into four stages: (1) teleoperated demonstration collection, (2) DS primitive fitting and single-arm, (3) dual-arm coordinated stacking, (4) perturbation recovery and collision modulation.


  python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_ds_pure.csv