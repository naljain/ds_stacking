# Reactive Dual-Arm Cube Stacking with Learned Dynamical Systems

## 1. Motivation

Fast dual-arm manipulation requires more than replaying precomputed
trajectories. In tasks such as cube stacking, each arm must adapt online to
small object pose errors, timing mismatch, imperfect grasps, and interference
from the other arm in the shared workspace. Classical scripted pick-and-place
pipelines can solve the nominal version of this task, but they are brittle when
the environment changes or when both arms attempt to use the stack region at
the same time.

This project investigates whether learned dynamical-system motion primitives
can provide a more reactive control layer for dual-arm cube stacking. The
motivation is to combine the structure and convergence properties of DS-based
controllers with the flexibility of learning from demonstration. The shared
stacking task is a useful testbed because it exposes both local manipulation
requirements and multi-arm coordination challenges.

## 2. Goal

The goal of the project is to build a simulated dual-arm Franka Panda system
that stacks cubes on a shared goal location using learned motion primitives.
Each arm executes a sequence of pick-and-place primitives:

```text
reach -> grasp -> lift -> transport -> place
```

The main technical objective is to learn stable joint-space Neural Dynamical
Systems for the longer free-space motions, `reach` and `transport`, while using
Lula inverse kinematics for the short constrained motions, `grasp`, `lift`, and
`place`. The system should support single-arm deployment, dual-arm deployment,
collision-aware motion near the shared workspace, and evaluation under nominal
and perturbed conditions.

## 3. Method / Approach

### System Architecture

The system is organized as a hybrid manipulation pipeline:

- Isaac Sim environment with two Franka Panda arms, a shared table, dynamic
  cube blocks, and stack goal markers.
- Task coordinator that assigns blocks, tracks primitive progress, reserves
  stack slots, and computes Cartesian targets.
- Lula IK module that converts Cartesian primitive targets into joint-space
  attractors `q_goal`.
- Learned Neural DS controllers for `reach` and `transport`.
- Scripted Lula joint-space controller for `grasp`, `lift`, and `place`.
- Dual-arm deployment layer with protected-point modulation, optional
  sampled-link safety, priority/yield behavior near the shared stack,
  kinematic-carry debugging, and return-home parking.

### Arm Setup

Two Franka Panda arms are mounted on opposite sides of the same table and face
the shared workspace. Each arm has its own source blocks but both arms stack at
the same central goal. This creates a deliberate shared-workspace conflict near
the stack, which is useful for evaluating inter-arm coordination and collision
avoidance.

The simulated scene includes:

- two Franka Panda robots,
- a shared table,
- colored dynamic cubes,
- visual stack goal markers,
- a table-side camera view,
- optional kinematic carry for isolating motion planning from contact grasping.

### IK Data Collection

Demonstrations are collected using Lula IK. At the start of each primitive, the
collector computes a Cartesian target and solves for a joint-space attractor
`q_goal`. The expert trajectory then moves toward this attractor with a
clamped joint-space controller. Each recorded sample stores:

- joint position `q`,
- joint velocity `q_dot`,
- primitive label,
- block label,
- end-effector position,
- Cartesian target,
- joint attractor `q_goal`,
- stack-slot metadata when applicable.

The collection pipeline was improved to make the demonstrations cleaner:

- Lula is used consistently for collection and deployment.
- The robot pauses between primitives without recording those pause states.
- Collection velocities are slower to reduce noisy finite-difference labels.
- Physical block jitter can be used to expand the data distribution.
- Transport targets use dynamic stack clearance so collection matches
  deployment as the stack grows.

### Dynamical-System Framing

The learned DS is formulated in joint-error coordinates:

```text
e = q - q_goal
q_dot = f_theta(e)
```

This makes the attractor explicit and keeps the learned vector field in the same
space as the robot commands. The DS does not learn direct Cartesian motion.
Instead, Lula provides a joint-space goal at primitive transitions, and the DS
learns how to move the current joint state toward that goal.

Only `reach` and `transport` are represented by learned DS models. The shorter
motions, `grasp`, `lift`, and `place`, are small constrained moves and were
more reliable as scripted Lula joint-space motions than as separate learned
vector fields.

### Neural DS Architecture

The Neural DS is a multilayer perceptron that maps normalized joint error to
normalized joint velocity. The model is parameterized as:

```text
f(e_n) = residual_theta(e_n) - stable_skip_gain * e_n
```

The residual network subtracts its value at the origin, so the equilibrium at
`e = 0` is guaranteed structurally:

```text
f(0) = 0
```

The stable skip term provides a linear convergent prior, while the residual
learns the demonstrated nonlinear correction.

### Lyapunov Function

The current Lyapunov candidate is a fixed quadratic:

```text
V(e_n) = ||e_n||^2
```

Training combines imitation loss with a stability hinge loss:

```text
L = ||f_theta(e) - q_dot_demo||^2
    + lambda_stab [dV/dt + alpha V]_+
```

At deployment, an optional safe projection can project the learned velocity
onto the half-space that satisfies Lyapunov decrease.

This replaced an earlier learned Lyapunov feature network. The current training
uses a uniform state normalization scale across joints so decreasing `V`
corresponds to decreasing unnormalized joint-space error.

### Modulation Design

Dual-arm collision avoidance uses Huber, Billard, and Slotine obstacle
modulation. The current implementation uses protected points rather than only
the end effector: distal arm frames and gripper proxy offsets define point
sets, the closest point pairs are modulated, and the Cartesian correction is
mapped back into joint space through a damped Jacobian pseudoinverse.

In practice, modulation was useful for high-level reactive avoidance, but it
was not sufficient to solve all dual-arm conflicts. End-effector-only
modulation did not account for wrists, gripper bodies, elbows, or forearms, so
the code now includes protected-point modulation, lateral-order modulation, and
optional sampled-link holds. Priority/yield weights and stack keepout options
are available for difficult shared-stack cases.

### Deployment Design

Deployment switches between primitive targets using the task coordinator.
For each primitive:

1. The coordinator computes a Cartesian target.
2. Lula IK computes a joint goal `q_goal`.
3. If the primitive is `reach` or `transport`, the Neural DS outputs joint
   velocity.
4. If the primitive is `grasp`, `lift`, or `place`, a scripted joint-space Lula
   controller moves toward `q_goal`.
5. The gripper closes after `grasp` and opens after `place`.
6. The stack slot is reserved before transport/place so both arms do not target
   the same height.

Dual-arm deployment starts both arms together by default. A start stagger is
still available as an ablation/debug flag, but the current default relies on
per-arm checkpoints, protected-point modulation, priority/yield behavior, and
return-home parking.

### Task Coordinator

The task coordinator is intentionally simple. It owns:

- block order,
- primitive order,
- current primitive state,
- stack-slot reservation,
- dynamic stack height,
- return-home behavior after an arm finishes.

It does not learn a policy. It provides the task-level structure within which
the learned DS and Lula controllers operate.

## 4. Evaluation

### Proposed Metrics

The system should be evaluated with both learning metrics and task-level
metrics:

- training loss for each Neural DS,
- imitation loss and Lyapunov stability violation rate,
- DS vector-field plots,
- Lyapunov landscape plots,
- rollout convergence plots,
- pick success rate,
- number of blocks successfully stacked,
- stack completion rate,
- average time per cube,
- minimum inter-arm distance,
- average minimum closest distance between arms,
- number of safety/modulation events,
- failure mode classification.

### Neural DS Results

The training losses for `reach` and `transport` are used to evaluate whether
the networks fit the demonstrated velocity fields. Vector-field plots and
rollouts are used to check whether the learned DS converges to the origin in
joint-error space.

The most useful qualitative diagnostics are:

- 2D slices of the learned vector field,
- rollout trajectories from perturbed initial errors,
- Lyapunov value contours,
- whether vectors point toward the attractor,
- whether spurious attractors appear.

### Neural DS vs. LPV-DS

The project should compare the Neural DS against an LPV-DS baseline. LPV-DS is
a more classical stable dynamical-system formulation and can provide a useful
contrast against the learned neural representation. Relevant comparison points
include:

- convergence behavior,
- ease of training,
- sensitivity to data quality,
- presence of spurious attractors,
- rollout stability,
- deployment success rate.

The expected lesson is that Neural DS models are flexible but harder to train
robustly, while LPV-DS can be more structured and easier to reason about.

### Task-Level Evaluation

Task-level evaluation should report:

- single-arm pick-and-stack success,
- dual-arm stack completion rate,
- per-block success or failure,
- grasp success rate,
- placement success rate,
- average minimum distance between arms,
- whether failures come from DS convergence, IK, grasping, collision avoidance,
  or coordinator timing.

The evaluation cases should include:

- nominal single-arm deployment,
- nominal dual-arm deployment,
- dual-arm deployment with modulation disabled,
- dual-arm deployment with protected-point modulation,
- dual-arm deployment with optional start stagger,
- block position jitter,
- perturbations to block positions,
- timing mismatch between arms.

These cases demonstrate both the advantages and limitations of the approach.

## 5. Implementation Details

The project is implemented in Python using Isaac Sim. The main code structure
is:

```text
src/
  env.py             Isaac Sim environment
  primitives.py      primitive definitions and targets
  franka_ik.py       Lula IK wrapper
  neural_ds.py       Neural DS and quadratic Lyapunov helper
  modulation.py      Huber-style protected-point DS modulation
  coordinator.py     task sequencing and stack-slot reservation
  perturbations.py   evaluation perturbations

scripts/
  collect_ik.py            data collection
  audit_demo_labels.py     label and q_goal audit
  train_ds.py              train one DS primitive
  train_all.sh             train reach and transport
  deploy_single_arm.py     single-arm deployment
  deploy_dual_arm.py       dual-arm deployment
  plot_ds.py               DS plots and rollouts
  evaluate.py              evaluation script
```

The pipeline is:

1. Build the Isaac Sim scene.
2. Collect Lula-labeled demonstrations.
3. Audit primitive and `q_goal` consistency.
4. Train Neural DS models for `reach` and `transport`.
5. Plot vector fields, Lyapunov landscapes, and rollouts.
6. Deploy a single arm.
7. Deploy two arms with protected-point modulation and return-home parking.
8. Evaluate task success and safety metrics.

The README documents the current commands, file layout, and training/deployment
workflow.

## 6. Conclusions and Lessons Learned

The main conclusion is that learning stable Neural DS controllers for real
robot manipulation is difficult. Even when the demonstrations look clean, small
inconsistencies in `q_goal`, velocity labels, timing, or primitive boundaries
can create poor vector fields or spurious attractors. Data collection quality is
therefore as important as network architecture.

Joint-space DS control was technically appealing because it avoids Cartesian
Jacobian inversion inside the learned controller and makes the stability story
more direct. However, in practice, joint-space learning was not obviously easier
or more reliable than Cartesian approaches. The joint-space representation is
sensitive to IK null-space choices and can be difficult to interpret when the
robot has redundant configurations.

Modulation was also more complicated than expected. End-effector modulation can
help avoid direct gripper conflicts, but it does not fully solve whole-arm
collision avoidance. Applying modulation symmetrically to both arms can create
stalemates near the stack. The current code therefore uses protected points,
lateral-order modulation, priority/yield weighting, and optional sampled-link
holds rather than relying only on the raw end-effector modulation.

What worked well:

- Lula-labeled data collection produced clean joint-space demonstrations.
- Pauses between primitives improved data quality.
- Dynamic stack-height targets fixed collection/deployment mismatch.
- Neural DS vector-field plots were useful for diagnosing convergence.
- Scripted `grasp`, `lift`, and `place` were more reliable than learning DS
  models for those small motions.

What did not work as well:

- Learning all five primitives as DS models was unnecessary and unstable.
- Joint-space DS training was sensitive to data and attractor labels.
- Symmetric dual-arm modulation caused conflicts near the shared stack.
- Whole-arm safety could not be handled by end-effector modulation alone.

Future work:

- Compare Neural DS more systematically against LPV-DS.
- Add a stronger whole-arm collision avoidance method.
- Learn or optimize a higher-level scheduler for dual-arm timing.
- Improve physical grasping so deployment does not rely on kinematic carry.
- Explore Cartesian or task-space DS formulations with better null-space
  handling.
- Add held-out evaluation sets and more systematic perturbation trials.
