# Neural Dynamical System

How the joint-space Neural DS in `src/neural_ds.py` is built, trained, and
deployed.

## State And Goal

The DS operates on 7-DOF Franka joint error:

```text
e = q - q_goal
```

`q_goal` is computed by Lula IK at primitive transitions. The learned DS is
used for the configured DS primitives, currently `reach` and `transport` by
default. Short constrained primitives, `grasp`, `lift`, and `place`, are
normally executed by the Lula joint-space controller. Some physical-deploy
presets also put `reach` in the IK set so pickup starts from a deterministic
pre-grasp hover while `transport` remains learned.

## Velocity Field

`NeuralDS` is a small MLP over normalized error `e_n`:

```text
e_n (7) -> Linear(7,128) -> tanh -> Linear(128,128) -> tanh -> Linear(128,7)
```

The deployed field is:

```text
f(e_n) = net(e_n) - net(0) - stable_skip_gain * e_n
```

`net(e_n) - net(0)` guarantees `f(0) = 0` exactly. The stable skip is a
convergent linear prior inside the learned DS architecture, not a deployment
controller. The external `DS_GOAL_GAIN` is currently `0.0` and should only be
used as a diagnostic fallback.

## Lyapunov Candidate

`LyapunovNet` is now parameterless:

```text
V(e_n) = ||e_n||^2
```

This replaced the older learned feature-map Lyapunov candidate. Training now
uses a uniform scalar `state_std` for all joints, so decreasing `||e_n||^2` is
equivalent to decreasing joint-space error up to a constant scale. That is the
property deployment actually needs.

The `lyapunov_hidden` config key is still accepted for checkpoint
compatibility, but it is ignored by the current `LyapunovNet`.

## Training

`scripts/train_ds.py` trains one checkpoint for one `(arm, primitive)` pair:

```bash
python scripts/train_ds.py --primitive reach --arm left
python scripts/train_ds.py --primitive transport --arm right
```

`scripts/train_all.sh` is the standard command. It trains four checkpoints:

```text
data/checkpoints/left_reach.pt
data/checkpoints/left_transport.pt
data/checkpoints/right_reach.pt
data/checkpoints/right_transport.pt
```

Per-arm training is intentional. Pooling left and right demonstrations can
average together mirrored joint-space flows at similar error states.

The training script:

- filters samples by `step["primitive"]`
- holds out the last about 10% of demos per pickle for validation when there
  are at least five demos
- clips velocity outliers to `training.max_joint_vel`
- uses zero state mean
- uses uniform scalar `state_std = max(max_j std, 0.5)`
- uses per-joint `vel_scale` with a floor of `0.05`
- saves the checkpoint with the best validation imitation MSE, or best training
  total loss if no validation split exists
- stores a `data_manifest` with source files, sample counts, arm/block/slot
  breakdowns, and label-source counts

The loss is:

```text
L = MSE(f(e_n), q_dot_demo_n)
    + lambda_stab * [dV/dt + alpha * V]_+
```

The stability term uses `scale_factor = vel_scale / state_std` so `dV/dt` is
computed for the real normalized-state time derivative, not a mismatched dot
product in network units.

## Safe Velocity

At deployment, `--use_safe` calls `StableNeuralDS.safe_velocity`. This projects
the nominal network velocity onto the half-space:

```text
dV/dt <= -alpha * V
```

If the nominal velocity already satisfies the constraint, it is unchanged. If
not, only the offending component along `grad V` is removed.

This gives two useful modes:

- `--use_safe` off: pure learned field plus the built-in stable skip.
- `--use_safe` on: hard Lyapunov projection at every DS step.

## Deployment Integration

The deploy scripts integrate DS velocity into a persistent command state:

```text
q_cmd_state <- q_cmd_state + q_dot * physics_dt
```

This is different from commanding `q_measured + q_dot * dt` every frame. The
persistent command trajectory avoids stalls caused by Isaac's articulation
controller lagging tiny one-step targets.

At every primitive transition, after gripper waits, and after scripted helper
motions, the command state is reset from the measured joint state.

## Guarantees

| Property | Current mechanism |
|---|---|
| `f(0) = 0` | `net(e) - net(0)` subtraction |
| `V(e_n) >= 0`, `V(0)=0` | quadratic `||e_n||^2` |
| `V` monotone with joint error | uniform scalar `state_std` |
| Lyapunov decrease | soft during training, hard with `--use_safe` |

The system is still hybrid across primitives: learned DS for the selected DS
primitives, Lula joint-space control for IK primitives, and discrete task
switching from the coordinator.
