# Reactive Dual-Arm Cube Stacking with DS Modulation

MEAM 6230 Final Project — Nalini Jain, Shivank Gupta, Thomas Stephen Felix

## Overview

This project trains Dynamical System (DS) transport policies from
demonstrations so two Franka Panda arms can stack cubes in a shared workspace.
Inter-arm collision avoidance is handled by **DS modulation** (Huber/Billard/
Slotine-style state-dependent velocity shaping) rather than a stop-and-go
coordinator.

Two transport models are supported:

  - **3D Cartesian LPVDS**: learns `x_dot = f(x, x_goal)` in `(x, y, z)`.
    This is the easiest path for testing shared workspace avoidance because
    modulation is applied directly to Cartesian EE velocity before IK.
  - **Joint-space Neural DS**: learns `q_dot = f(q, q*)` and applies
    modulation through the translational Jacobian.

## Why joint-space DS?

The DS is trained directly on `q̇ = f(q, q*)` where `q` is the 7-DOF Franka
joint configuration and `q*` is the target joint configuration for the
current primitive. At deployment the network's output is sent to the robot
verbatim — no IK, no integration-and-retarget — so the closed-loop joint
dynamics are exactly the trained DS modulo low-level actuator dynamics.

This means:
  - Lyapunov stability claims hold in the actual control space.
  - No singularity / null-space issues from a Cartesian DS + Jacobian inverse.
  - The learned vector field is what's running, not RMPflow.

## Why DS modulation instead of a coordinator FSM?

A discrete "if EEs are close, hold one arm" coordinator would make the system
hybrid — and stability of hybrid systems requires extra dwell-time analysis.
Instead, we apply a state-dependent modulation matrix `M(x_self, x_other)`
that smoothly deflects each arm's velocity around a safety sphere centred on
the other arm's EE. The closed loop is therefore a coupled DS rather than a
hybrid system.

We use the **Huber, Billard & Slotine (2019)** modulation framework
(*"Avoidance of Convex and Concave Obstacles with Convergence Ensured Through
Contraction"*, IEEE RA-L), which extends Khansari-Zadeh & Billard 2012 with:

  - **Reference-direction basis** — `M = E·D·E^{-1}` is built around the
    direction from a reference point inside the obstacle to the agent, rather
    than the surface normal. This eliminates the antipodal saddle-point that
    can trap agents on the back side of an obstacle.
  - **Tail-effect gating** — when the nominal velocity already points away
    from the obstacle, no radial damping is applied, so the agent can leave
    the contention zone naturally.
  - **Contraction-based convergence guarantee** — under the construction
    above, the modulated DS still converges to its attractor.

```
q̇_self = q̇_nom + J_trans^+ · ( M(ee_self, ee_other) - I ) · J_trans · q̇_nom
```

When far apart, `M ≈ I` and each arm tracks its nominal DS exactly.
When close and approaching, `M` damps the radial component and boosts the
tangential component, deflecting motion around the other arm's EE.
When close but already moving outward, `M` falls back to identity (tail
effect), letting the arm separate cleanly.

## Directory Layout

```
ds_stacking/
├── src/
│   ├── env.py               # Isaac Sim scene with two Frankas + table + blocks
│   ├── primitives.py        # Primitive Cartesian targets + completion checks
│   ├── ik_controller.py     # Straight-line Cartesian IK controller
│   ├── franka_ik.py         # Lula IK wrapper for q* lookup
│   ├── neural_ds.py         # Joint-space Neural DS + Lyapunov network
│   ├── lpv_ds.py            # 3D Cartesian LPVDS transport model
│   ├── modulation.py        # DS modulation matrices for collision avoidance
│   ├── coordinator.py       # Slim primitive sequencer (no FSM)
│   └── perturbations.py     # Perturbation injectors for evaluation
├── scripts/
│   ├── smoke_test.py        # Verify the scene loads
│   ├── collect_ik.py        # Generate transport demos with IK
│   ├── teleop.py            # Manual demo collection
│   ├── train_ds.py          # Train Neural DS for one primitive
│   ├── train_lpvds.py       # Train 3D Cartesian LPVDS transport
│   ├── train_all.sh         # Train all 5 primitives sequentially
│   ├── deploy_single_arm.py # Run learned DS on one arm
│   ├── deploy_dual_arm.py   # Both arms with modulation
│   └── evaluate.py          # Full evaluation suite + ablations
├── configs/
│   └── default.yaml         # All hyperparameters and constants
├── data/
│   ├── demonstrations/      # Saved trajectory data (.pkl)
│   ├── checkpoints/         # Trained model weights (.pt)
│   └── results/             # Evaluation JSON files
├── requirements.txt
└── README.md
```

## Pipeline

The default block spawn regions in `configs/default.yaml` keep blocks on each
arm's own side of the table. Both arms then transport into one shared stacking
goal near the table centre, which creates the coordinated shared-workspace
portion of the task.

### Recommended: Cartesian LPVDS Transport

```bash
# 1. Smoke test — confirm scene loads
python scripts/smoke_test.py

# 2. Collect transport demos for both arms
python scripts/collect_ik.py --arm left  --n_demos 50
python scripts/collect_ik.py --arm right --n_demos 50

# 3. Train one 3D Cartesian LPVDS transport model per arm
python scripts/train_lpvds.py --arm left
python scripts/train_lpvds.py --arm right

# 4. Validate one arm
python scripts/deploy_single_arm.py --arm left --model lpvds

# 5. Deploy both arms with Cartesian modulation
python scripts/deploy_dual_arm.py --model lpvds

# Faster, less stall-prone rollout
python scripts/deploy_dual_arm.py --model lpvds \
  --speedup 1.5 \
  --done_tol 0.07 \
  --max_transport 4000 \
  --mod_radius 0.25 \
  --yield_mod_weight 0.7

# More conservative modulation-only avoidance rollout
python scripts/deploy_dual_arm.py --model lpvds \
  --speedup 1.0 \
  --done_tol 0.07 \
  --max_transport 4000 \
  --mod_radius 0.45 \
  --mod_reactivity 1.0 \
  --priority_mod_weight 1.0 \
  --yield_mod_weight 1.0

# 6. Ablation: run the same dual-arm deployment without modulation
python scripts/deploy_dual_arm.py --model lpvds --no_modulation
```

Dual-arm modulation is weighted by default: both arms react to each other, but
the priority arm only receives a light modulation while the non-priority arm
receives the stronger avoidance correction. Priority starts with the left arm
and alternates after each placed block. A transporting arm treats the other
arm's EE as a moving obstacle even when the other arm is reaching, grasping,
placing, or retracting. Blocks are randomized at deploy startup, and an arm
that has placed all of its cubes returns to its configured nominal joint pose.

The faster rollout command above is a good first command when the arms seem to
stall near the shared stack. `--speedup` raises IK and transport speeds,
`--done_tol 0.07` accepts transport completion a little earlier, and the smaller
`--mod_radius` / `--yield_mod_weight` pair makes avoidance less conservative
while still keeping the non-priority arm responsive. If the arms still slow each
other too much, reduce `--mod_radius` or increase `--mod_reactivity`; if they
move too aggressively, reduce `--speedup` back toward `1.0`.

Use the conservative command if the EEs get too close. It avoids duplicate
flags, starts modulation earlier with lower `--mod_reactivity`, and modulates
both arms fully during transport and IK primitives.

To start with the right arm as priority:

```bash
python scripts/deploy_dual_arm.py --model lpvds --priority_arm right
```

To tune the balance:

```bash
python scripts/deploy_dual_arm.py --model lpvds \
  --mod_radius 0.35 \
  --mod_reactivity 2.0 \
  --priority_mod_weight 0.25 \
  --yield_mod_weight 1.0
```

`train_lpvds.py` defaults to `data/demonstrations/{arm}_demos.pkl`. If you run
`scripts/clean_demos.py`, train on that file explicitly:

```bash
python scripts/clean_demos.py --arm left
python scripts/train_lpvds.py --arm left --use_clean
```

### Joint-Space Neural DS Path

`collect_ik.py` currently records the transport primitive. If you use the
Neural DS path as-is, train and deploy the transport policy:

```bash
python scripts/train_ds.py --primitive transport --arm left
python scripts/train_ds.py --primitive transport --arm right

python scripts/deploy_single_arm.py --arm left --model neural
python scripts/deploy_dual_arm.py --model neural
```

If you collect full primitive trajectories and train the combined checkpoint
with `scripts/train_all.sh`, deploy it explicitly with:

```bash
bash scripts/train_all.sh
python scripts/deploy_dual_arm.py --model neural --ckpt_arm both
```

### Evaluation And Plots

```bash
python scripts/evaluate.py --n_trials 10

python scripts/evaluate.py --no_modulation                # collision-avoidance ablation
python scripts/evaluate.py --use_safe                     # Lyapunov projection
```

LPVDS-specific visualizations:

```bash
# Learned 3D Cartesian DS velocity field
python scripts/plot_lpvds_3d.py --arm left
python scripts/plot_lpvds_3d.py --arm right

# Same field with the other EE depicted as a spherical obstacle
python scripts/plot_lpvds_3d.py --arm left --modulated --other_ee 0.0 0.45 0.99 --mod_radius 0.35

# Record a real dual-arm interaction and plot Gamma, velocity, weights, and obstacle spheres
python scripts/deploy_dual_arm.py --model lpvds --diag_out data/results/lpvds_interaction.pkl
python scripts/plot_lpvds_interaction.py --diag data/results/lpvds_interaction.pkl

# Animate the same interaction in 3D over time
python scripts/animate_lpvds_interaction.py --diag data/results/lpvds_interaction.pkl

# If ffmpeg is unavailable, write a GIF instead
python scripts/animate_lpvds_interaction.py --diag data/results/lpvds_interaction.pkl --out data/results/lpvds_interaction.gif

# Report-style homework figures: demos, GMM regions, modulation slice, diagnostics
MPLCONFIGDIR=/tmp/mpl python scripts/plot_homework_figures.py
```

## Setup

```bash
conda create -n franka_isaac python=3.11 -y
conda activate franka_isaac
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch numpy pyyaml tqdm matplotlib
```

## Transport And Height

The LPVDS transport model learns a full 3D Cartesian EE velocity. Training uses
a guarded normalisation scale for `z` so near-constant transport height does not
let tiny vertical noise dominate GMM assignment. In dual-arm LPVDS deployment,
the resulting 3D Cartesian velocity is modulated around the other arm's EE
before IK computes the next arm command.

The demonstrations still determine what the nominal DS learns. If the
collected transport demos stay almost exactly at `lift_h`, the learned nominal
`z_dot` will be small near that manifold. Vertical avoidance during deployment
comes from the 3D modulation layer and from any vertical variation present in
the demos.

## Primitives

Each pick-and-stack sequence decomposes into 5 learned primitives:

| Primitive | Cartesian goal | Joint goal |
|---|---|---|
| `reach`     | hover above source block | IK of hover pose |
| `grasp`     | descend to block surface | IK of grasp pose |
| `lift`      | raise to transport altitude | IK of lift pose |
| `transport` | move above stacking goal | IK of transport pose |
| `place`     | descend onto stack | IK of place pose |

Each primitive has its own `f_theta` and `V_phi` networks. At a primitive
transition, Lula IK is queried once to compute the new `q*`; the DS then
drives `q -> q*` smoothly until the next transition.

## Stability

Per-primitive: `V(q, q*)` is positive-definite around `q*` by construction;
training enforces `dV/dt + α·V <= 0` on the data distribution; inference can
optionally project `f(x)` onto the half-space where this holds exactly
(`--use_safe`).

Across primitives: switched-system stability via dwell-time. We measure the
per-primitive convergence time empirically and report it.

Across arms: modulation preserves the stability of each arm's DS individually
because `M(x)` is diagonal in the basis aligned with the obstacle normal and
the radial scaling is non-negative — so `dV/dt` of each arm's individual
Lyapunov function remains non-positive after modulation.

```
python scripts/deploy_dual_arm.py --model lpvds \
  --speedup 1.0 \
  --done_tol 0.07 \
  --max_transport 4000 \
  --mod_radius 0.25 \
  --mod_reactivity 2.0 \
  --priority_mod_weight 0.25 \
  --yield_mod_weight 0.8 \
  --status_every 120
```
