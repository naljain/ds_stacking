# Reactive Dual-Arm Cube Stacking with Joint-Space Neural DS

MEAM 6230 Final Project — Nalini Jain, Shivank Gupta, Thomas Stephen Felix

## Overview

This project trains **joint-space Neural Dynamical System (DS)** motion
primitives from demonstrations to enable two Franka Panda arms to stack
cubes independently in a shared workspace. Inter-arm collision avoidance is
handled by **DS modulation** (Khansari-Zadeh & Billard 2012-style state-
dependent velocity shaping) rather than a discrete coordinator, keeping the
closed-loop a pure dynamical system.

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
│   ├── ik_controller.py     # RMPflow wrapper (used at data-collection only)
│   ├── franka_ik.py         # Lula IK wrapper for q* lookup
│   ├── neural_ds.py         # Joint-space Neural DS + Lyapunov network
│   ├── modulation.py        # DS modulation matrices for collision avoidance
│   ├── coordinator.py       # Slim primitive sequencer (no FSM)
│   └── perturbations.py     # Perturbation injectors for evaluation
├── scripts/
│   ├── smoke_test.py        # Verify the scene loads
│   ├── collect_ik.py        # Generate (q, q*, q̇) demos with RMPflow
│   ├── teleop.py            # Manual demo collection
│   ├── train_ds.py          # Train Neural DS for one primitive
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

```bash
# 1. Smoke test — confirm scene loads
python scripts/smoke_test.py

# 2. Collect demos (joint-space)
python scripts/collect_ik.py --arm left  --n_demos 50
python scripts/collect_ik.py --arm right --n_demos 50

# 3. Train all 5 primitives (joint-space DS + Lyapunov)
bash scripts/train_all.sh

# 4. Validate on one arm
python scripts/deploy_single_arm.py --arm left

# 5. Deploy dual-arm with modulation
python scripts/deploy_dual_arm.py

# 6. Run evaluation suite
python scripts/evaluate.py --n_trials 10

# 7. Ablations for the writeup
python scripts/evaluate.py --no_modulation                # collision-avoidance ablation
python scripts/evaluate.py --use_safe                     # Lyapunov projection
```

## Setup

```bash
conda create -n franka_isaac python=3.11 -y
conda activate franka_isaac
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch numpy pyyaml tqdm matplotlib
```

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
