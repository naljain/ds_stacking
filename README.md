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

If collection settings change, rerun from step 2 onward. The trained DS is only
as good as the saved demonstrations, so demos with missed grasps, target-noise
artifacts, or jerky primitive transitions should not be reused.

## Setup

```bash
conda create -n franka_isaac python=3.11 -y
conda activate franka_isaac
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch numpy pyyaml tqdm matplotlib
```

Isaac Sim's Franka helper resolves the robot USD through the configured Isaac
assets root. If your install cannot reach the default assets root, set a local
Franka USD in `configs/default.yaml`:

```yaml
assets:
  franka_usd: /path/to/Assets/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
```

The ground plane is built from a local cuboid, so it does not require the Isaac
asset server.

## Data Collection Notes

`collect_ik.py` uses RMPflow only for demonstration collection. The default
collection path is conservative: it targets the observed block pose directly,
uses no block jitter, and uses no target noise. Keep legacy target noise
`--noise` at `0` for reliable grasp demos; nonzero target noise commands the
gripper beside the observed block and can produce failed grasps.

After the base grasp is reliable, `--block_xy_jitter` can be used to move the
physical blocks and widen the data distribution without commanding grasps away
from the cube.

Collection adds extra RMPflow settling steps before switching primitives. By
default, after `grasp` it kinematically carries the active block with the EE so
clean joint-space demos are not discarded because Isaac contact grasping is
flaky. Use `--physical_grasp` when you explicitly want to test whether the
parallel gripper/contact setup can lift the cube; in that mode failed lifts are
discarded.

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

## Dual-Arm Coordination

The coordinator remains deliberately slim: it chooses block order, primitive
order, and stack slots. It does not perform close-range hold/release collision
logic. Inter-arm collision avoidance remains the responsibility of continuous
DS modulation.

Dual-arm deployment uses a small initial phase offset by default:
`coordination.start_stagger_steps: 30`, roughly 0.25 s at 120 Hz. The left arm
starts immediately and the right arm starts after the stagger. This avoids a
perfectly symmetric race into the shared stack while preserving the DS +
modulation framing. Override it with `--stagger_steps`.

Shared stack layers are reserved before `transport/place`, so both arms cannot
target the same stack height when they arrive at the goal area concurrently.
The reserved height is the desired block center height. During kinematic-carry
debug deployment, the carried block is snapped to that reserved stack slot when
`place` opens the gripper; otherwise the visual block pose can reflect the EE
carry offset instead of the coordinator's stack layer.

If a newly trained DS does not converge at deployment, inspect
`cos→goal` in `deploy_single_arm.py`. Negative values mean the learned vector
field is pointing away from `q_goal`. For debugging or a stabilizing ablation,
run with `--goal_gain 1.0`, which adds a linear attraction term toward `q_goal`
while still evaluating the learned DS velocity.

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
