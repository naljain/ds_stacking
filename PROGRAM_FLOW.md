# Program Flow: Data Collection to Deployment

This project trains and deploys joint-space Neural Dynamical Systems (DS) for
Franka block stacking in Isaac Sim. The full pipeline is:

1. Build an Isaac Sim scene with one or two Franka arms, a table, blocks, and
   stack goals.
2. Collect demonstrations with a joint-space Lula-goal expert.
3. Convert demonstrations into joint-error and joint-velocity training pairs.
4. Train one Neural DS per primitive.
5. Deploy the trained DS models by switching between primitive goals.
6. Optionally evaluate the full dual-arm system under perturbations.

The important idea is that IK is used to generate and label demonstrations, but
deployment is meant to be a learned joint-space DS. At deployment time, the
program computes a joint goal `q_goal` for the active primitive, then the learned
DS outputs `q_dot` commands that should drive the robot toward that goal.

## Main Files

- `configs/default.yaml` defines scene geometry, block positions, primitive
  timing, training hyperparameters, and output paths.
- `src/env.py` builds the Isaac Sim environment.
- `src/primitives.py` defines the primitive order and Cartesian targets.
- `src/ik_controller.py` wraps RMPflow for the legacy collection option.
- `src/franka_ik.py` wraps Lula IK for computing joint goals.
- `scripts/collect_ik.py` records demonstration trajectories.
- `scripts/audit_demo_labels.py` checks that primitive labels and `q_goal`
  attractor labels are consistent before training.
- `scripts/train_ds.py` trains a Neural DS for one primitive.
- `scripts/train_all.sh` trains all primitive models.
- `src/neural_ds.py` defines the DS and Lyapunov networks.
- `src/coordinator.py` manages primitive sequencing and stack slots.
- `scripts/deploy_single_arm.py` deploys one arm.
- `scripts/deploy_dual_arm.py` deploys both arms.
- `scripts/evaluate.py` runs perturbation experiments and metrics.

## 1. Configuration

The pipeline starts from `configs/default.yaml`.

The config defines:

- Table size, height, and position.
- Block size, mass, colors, and initial positions.
- Franka arm spacing and base orientation.
- Stack goal locations.
- Primitive heights:
  - `hover`: height for `reach`
  - `grasp`: height for descending to the block
  - `lift`: height for carrying blocks
- Simulation timestep:
  - `physics_dt`
  - `rendering_dt`
- Collection step budgets for each primitive:
  - `reach`
  - `grasp`
  - `lift`
  - `transport`
  - `place`
- Training hyperparameters:
  - hidden dimensions
  - Lyapunov loss weight
  - learning rate
  - batch size
  - epochs
  - max joint velocity
- Output directories:
  - demonstrations go to `data/demonstrations`
  - checkpoints go to `data/checkpoints`
  - evaluation results go to `data/results`

The scene and scripts all read this same config, so changing a height or block
layout changes both collection and deployment behavior.

## 2. Environment Construction

`src/env.py` defines `DualArmEnv`.

Each script creates an Isaac `SimulationApp` first, then constructs `DualArmEnv`.
The environment builds:

- a local ground cuboid
- a table
- one or two Franka arms
- a pedestal under each arm
- dynamic cube blocks
- visual goal markers
- lighting and a camera

The active arms are controlled by the caller:

```python
DualArmEnv(config_path=args.config, arms=("left",))
DualArmEnv(config_path=args.config, arms=("left", "right"))
```

That is why the same environment supports single-arm debugging and dual-arm
deployment.

The environment also exposes utility methods:

- `step(render=True)` advances physics.
- `reset_blocks()` returns blocks to their initial poses.
- `get_block_positions()` returns block center positions.
- `get_block_poses()` returns positions and orientations.
- `get_ee_pose(arm)` returns the end-effector pose.
- `get_block_obj(name)` returns an Isaac object for a block.

## 3. Motion Primitives

`src/primitives.py` defines the task as five primitives:

```python
["reach", "grasp", "lift", "transport", "place"]
```

Each primitive maps the current task state to a Cartesian target:

- `reach`: move above the source block at hover height.
- `grasp`: descend to the block at grasp height.
- `lift`: raise the block to lift height.
- `transport`: move above the stack goal at lift height.
- `place`: descend to the stack height.

The gripper only actuates at the end of two primitives:

- after `grasp`, close the gripper
- after `place`, open the gripper

For `reach` and `grasp`, the gripper orientation is aligned to the block yaw.
For later primitives, the gripper uses the default downward orientation.

## 4. Data Collection

Data collection is handled by `scripts/collect_ik.py`.

Typical commands:

```bash
python scripts/collect_ik.py --arm left --n_demos 50 --headless
python scripts/collect_ik.py --arm right --n_demos 50 --headless
```

The collector defaults to `--motion_source joint_lula`. For each primitive, it
computes the same Lula `q_goal` that deployment will use, then records a
joint-space expert trajectory moving toward that target. This keeps the
demonstrated velocity and the saved attractor label in the same joint-space DS.

The older Cartesian RMPflow collection path is still available with
`--motion_source rmpflow`, but it is diagnostic for the pure joint-space DS
pipeline because RMPflow can settle in a different redundant-arm null-space
configuration than a fresh Lula solve.

For each demo, the script:

1. Resets the blocks.
2. Resets the selected arm.
3. Optionally jitters the starting joint pose.
4. Optionally jitters physical block XY positions.
5. Moves through each block.
6. For each block, executes:
   - `reach`
   - `grasp`
   - `lift`
   - `transport`
   - `place`
7. Records joint positions and finite-difference joint velocities at every
   controller step.
8. Saves all successful demos to a pickle file.

The saved paths are:

```text
data/demonstrations/left_demos.pkl
data/demonstrations/right_demos.pkl
```

Each recorded step stores:

- `q`: 7 arm joint positions
- `q_dot`: 7 joint velocities
- `ee_pos`: end-effector position
- `primitive`: active primitive name
- `block`: active block name
- `arm`: left or right
- `target`: Cartesian primitive target
- `q_goal`: joint-space attractor for the primitive

### How `q_goal` Is Labeled

At the start of each primitive, the collector labels `q_goal`. By default,
`--motion_source joint_lula --q_goal_source lula` uses the same Lula IK target
that deployment uses for that primitive transition.

This is important. The DS is trained on:

```python
x = q - q_goal
q_dot = demonstrated joint velocity
```

So the model learns a velocity field in joint-error space, not directly in
Cartesian space.

The collector also computes a Lula IK solution for the same Cartesian target and
stores it as metadata:

- `q_goal_settled`
- `q_goal_lula`
- `q_goal_lula_ok`
- `q_goal_lula_error`
- `q_goal_source`

This matters because the learned DS is only valid relative to its attractor. If
training labels a trajectory with the wrong null-space solution, the data can
contain velocities that move away from the saved attractor, which produces
divergent learned flows.

Use the audit script before training:

```bash
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
```

For each primitive, check:

- final `||q - q_goal||` should be small
- `cos(q_dot, -error)` should usually be positive
- `fraction moving away` should be close to zero

### Kinematic Carry During Collection

By default, collection kinematically carries the active block after `grasp`.
That means the block is attached to the end-effector pose in code rather than
requiring Isaac contact physics to hold the grasp.

This is intentional. The goal of collection is to get clean motion
demonstrations. Contact grasping can be flaky in simulation, and failed contact
grasps would corrupt or discard otherwise useful joint-space motion data.

Use `--physical_grasp` only when you specifically want to test the gripper and
contact setup.

## 5. Training Data Format

`scripts/train_ds.py` loads one or both demonstration files and filters samples
by primitive.

For every recorded step in the selected primitive, it builds:

```python
state = q - q_goal
velocity = q_dot
```

The state is 7-dimensional because it is only the Franka arm joint error. Finger
joints are handled separately by gripper commands.

The script clips velocity outliers to `training.max_joint_vel`. This prevents a
bad finite-difference spike from dominating the velocity scale.

Then it normalizes:

- states by per-joint `state_std`
- velocities by per-joint `vel_scale`

The checkpoint stores these normalization values because deployment must apply
the same scaling before calling the network.

## 6. How Data Is Partitioned Across DS Primitives

The collected pickle files are not saved as five separate datasets. Each demo is
saved as one full pick-and-stack trajectory containing all primitives in order.
Every recorded timestep has a `primitive` field:

```python
step["primitive"] in ["reach", "grasp", "lift", "transport", "place"]
```

`scripts/train_ds.py` partitions the data by filtering on that field. The
function that does this is `load_trajectories(demo_paths, primitive)`.

Conceptually, for a requested primitive such as `reach`, training does:

```python
states = []
velocities = []

for demo_file in demo_paths:
    demos = load_pickle(demo_file)
    for demo in demos:
        for step in demo["trajectory"]:
            if step["primitive"] != "reach":
                continue
            states.append(step["q"] - step["q_goal"])
            velocities.append(step["q_dot"])
```

So `both_reach.pt` is trained only from timesteps labeled `reach`.
`both_grasp.pt` is trained only from timesteps labeled `grasp`, and so on.

The primitive split is therefore:

| Checkpoint | Training samples used |
|---|---|
| `both_reach.pt` | all `reach` timesteps |
| `both_grasp.pt` | all `grasp` timesteps |
| `both_lift.pt` | all `lift` timesteps |
| `both_transport.pt` | all `transport` timesteps |
| `both_place.pt` | all `place` timesteps |

This means each DS learns a different local vector field:

- `reach` learns how to move from the current arm pose to a hover pose above a
  source block.
- `grasp` learns how to descend from hover to grasp height.
- `lift` learns how to raise from grasp height to carry height.
- `transport` learns how to move from source-side lift pose to goal-side lift
  pose.
- `place` learns how to descend from carry height to stack placement height.

### Arm Partitioning

The `--arm` argument controls which demonstration files are loaded:

```bash
python scripts/train_ds.py --primitive reach --arm left
```

loads:

```text
data/demonstrations/left_demos.pkl
```

and saves:

```text
data/checkpoints/left_reach.pt
```

This command:

```bash
python scripts/train_ds.py --primitive reach --arm right
```

loads:

```text
data/demonstrations/right_demos.pkl
```

and saves:

```text
data/checkpoints/right_reach.pt
```

This command:

```bash
python scripts/train_ds.py --primitive reach --arm both
```

loads both:

```text
data/demonstrations/left_demos.pkl
data/demonstrations/right_demos.pkl
```

and saves:

```text
data/checkpoints/both_reach.pt
```

The current `scripts/train_all.sh` uses `--arm both`, so the standard pipeline
trains shared primitive models from the union of left-arm and right-arm
demonstrations.

### Checkpoint Data Manifest

`scripts/train_ds.py` now makes this split explicit when training. For every
checkpoint, it prints and saves a `data_manifest` containing:

- the primitive label used for filtering
- the demonstration files loaded
- total sample count
- samples by source file
- samples by arm
- samples by block

For example, when training:

```bash
python scripts/train_ds.py --primitive grasp --arm left
```

the checkpoint should explicitly say that `left_grasp.pt` was trained only from
`grasp` timesteps in `data/demonstrations/left_demos.pkl`.

When training:

```bash
python scripts/train_ds.py --primitive grasp --arm both
```

the checkpoint should explicitly say that `both_grasp.pt` was trained only from
`grasp` timesteps pooled from:

```text
data/demonstrations/left_demos.pkl
data/demonstrations/right_demos.pkl
```

This makes it possible to inspect a checkpoint later and verify exactly which
collection source produced that DS.

### What Is Not Partitioned

There is no train/validation split in the current training script. All samples
for the requested primitive and arm selection are used for training. The script
does print post-training diagnostics on the same training set, such as
imitation error and Lyapunov stability violation rate, but those are not held-out
validation metrics.

There is also no block-specific DS. For example, all `reach` samples for all
blocks are pooled together into the same `reach` dataset. The state is
`q - q_goal`, so the model is expected to generalize across blocks by seeing the
joint-space error to the current primitive goal rather than the block identity.

## 7. Neural DS Training

`src/neural_ds.py` defines the model.

There are two learned components:

- `NeuralDS`: maps normalized joint error to normalized joint velocity.
- `LyapunovNet`: produces a positive-definite Lyapunov value around the goal.

The trained policy is:

```python
q_dot = f_theta(q - q_goal)
```

The current DS architecture includes a stable error-space prior:

```text
f(e_n) = residual_theta(e_n) - stable_skip_gain * e_n
```

This is still a pure DS: the stabilizing term is inside the learned primitive
model and is used during training. It is not the same as deployment
`--goal_gain`, which adds an external controller after the network output and
is only for diagnostics.

One model is trained per primitive. The standard training command for all five
primitive models is:

```bash
bash scripts/train_all.sh
```

That runs:

```bash
python scripts/train_ds.py --primitive reach --arm both
python scripts/train_ds.py --primitive grasp --arm both
python scripts/train_ds.py --primitive lift --arm both
python scripts/train_ds.py --primitive transport --arm both
python scripts/train_ds.py --primitive place --arm both
```

The output checkpoints are:

```text
data/checkpoints/both_reach.pt
data/checkpoints/both_grasp.pt
data/checkpoints/both_lift.pt
data/checkpoints/both_transport.pt
data/checkpoints/both_place.pt
```

Each checkpoint contains:

- model weights
- state normalization
- velocity normalization
- primitive name
- training config
- loss history
- `data_manifest` showing the exact demo files, arms, blocks, and sample counts
  used to train that checkpoint

## 8. Task Sequencing

`src/coordinator.py` defines `TaskSequencer`.

The sequencer does not control continuous motion. Its job is discrete task
bookkeeping:

- which block each arm is working on
- which primitive is active
- which stack height is reserved
- when to move to the next primitive

For each arm, `ArmTaskState` tracks:

- block order
- current block index
- current primitive
- current `q_goal`
- reserved stack height

The sequence is:

```text
reach -> grasp -> lift -> transport -> place
```

After `place`, the arm advances to the next block and returns to `reach`.

For `transport` and `place`, the sequencer reserves a stack slot. This prevents
two arms from targeting the same stack layer during dual-arm operation.

## 9. Single-Arm Deployment

`scripts/deploy_single_arm.py` runs the learned DS on one arm.

Pure learned-DS debug command:

```bash
python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_pure_ds.csv
```

At startup, the script:

1. Creates the Isaac simulation.
2. Builds a single-arm environment.
3. Loads all five primitive checkpoints.
4. Creates a `TaskSequencer`.
5. Computes the first primitive's `q_goal`.
6. Opens the gripper.
7. Starts the control loop.

At every control step:

1. Read the current joint state `q`.
2. Compute the current error:

   ```python
   x = q - q_goal
   ```

3. Normalize the error using checkpoint statistics.
4. Call the current primitive's DS model.
5. Optionally apply Lyapunov safe projection with `--use_safe`.
6. Optionally add linear goal attraction:

   ```python
   q_dot = q_dot - goal_gain * (q - q_goal)
   ```

7. Clip joint velocity.
8. Integrate one timestep:

   ```python
   q_cmd = q + q_dot * physics_dt
   ```

9. Send `q_cmd` to Isaac's articulation controller.
10. Step the simulation.

The primitive completes only when:

```python
||q - q_goal|| < done_tol
```

If a timeout happens before convergence, single-arm deployment now aborts by
default. That is the right behavior for pickup tests because advancing from a
failed `reach` to `grasp` would close the gripper from the wrong pose.

Use `--advance_on_timeout` only for debugging phase flow.

### Important Deployment Flags

- `--kinematic_carry`: after `grasp`, attach the active block to the end
  effector until `place`. This isolates motion planning from contact physics.
- `--use_safe`: project the learned velocity so it satisfies the Lyapunov
  decrease condition.
- `--goal_gain`: add a direct linear attraction toward `q_goal`. Keep this at
  `0.0` for pure learned-DS runs.
- `--ds_scale`: scale the learned DS output. Keep this at `1.0` for pure
  learned-DS runs. Use `--ds_scale 0 --goal_gain 3.0` only as a joint-attractor
  sanity check when isolating IK or actuation.
- `--debug_ik`: print Cartesian targets, IK success, seeds, and `q_goal`.
- `--print_every`: print convergence diagnostics every N steps.

The key diagnostic is `cos->goal` in the terminal output. Positive values mean
the commanded velocity points toward the goal. Negative values mean the velocity
field is pushing away from the goal.

## 10. Dual-Arm Deployment

`scripts/deploy_dual_arm.py` follows the same DS logic as single-arm deployment,
but runs both arms in the same scene.

Each arm has its own:

- task state
- current primitive
- current `q_goal`
- DS velocity

The coordinator reserves stack heights so both arms do not place blocks on the
same layer. The right arm can also start after a small stagger from the config
so both arms do not move symmetrically into the shared stack at the same time.

Dual-arm deployment can use DS modulation from `src/modulation.py` to alter
joint velocities when end-effectors get close. The goal is to keep the system
continuous rather than adding explicit finite-state collision holds.

## 11. Evaluation

`scripts/evaluate.py` runs batches of trials and records metrics.

Conditions include:

- `nominal`: no perturbation
- `block_displacement`: shift a target block
- `ee_disturbance`: apply an end-effector disturbance
- `arm_block`: freeze one arm temporarily
- `combined`: combine perturbations

Metrics include:

- stack completion rate
- average blocks placed
- average time per cube
- grasp failure rate
- end-effector proximity events
- recovery success rate

Evaluation results are written to:

```text
data/results/eval_<timestamp>.json
```

## 12. Why Pickup Can Fail

The pickup sequence depends on `reach` and `grasp` converging in joint space.
If `reach` does not converge, then `grasp` starts from the wrong pose. Closing
the gripper after that cannot reliably pick up the block.

A timeout with a large joint error means the DS did not actually reach the
primitive attractor. It usually means one of these is true:

- the learned velocity points away from the goal for that state
- the state is outside the training distribution
- the DS output is too small or saturated
- `q_goal` differs from the joint goals seen during collection
- the safe projection or velocity scaling is changing the motion too much
- joint command tracking is not following the integrated commands

The recommended debugging order is:

1. Run the pure learned DS with `--debug_ik --print_every 25`.
2. Check whether `reach` converges before timeout.
3. Check `cos->goal`.
4. If it fails, run pure attractor mode only as a diagnostic:

   ```bash
   python scripts/deploy_single_arm.py --arm left --kinematic_carry --ds_scale 0 --goal_gain 3.0 --done_tol 0.25 --print_every 25 --debug_ik
   ```

5. If pure attractor works, IK and actuation are likely fine, and the learned DS
   needs tuning or retraining.
6. If pure attractor fails, investigate IK goals, asset setup, joint limits, or
   Isaac articulation control.

## 13. Typical End-to-End Commands

Collect demonstrations:

```bash
python scripts/collect_ik.py --arm left --n_demos 50 --headless
python scripts/collect_ik.py --arm right --n_demos 50 --headless
```

Audit labels:

```bash
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
```

Train all DS models:

```bash
bash scripts/train_all.sh
```

Run single-arm pure learned-DS deployment:

```bash
python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_pure_ds.csv
```

Run left-only pure learned-DS deployment after left-only retraining:

```bash
python scripts/deploy_single_arm.py --arm left --ckpt_arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_pure_ds.csv
```

Run pure dual-arm deployment:

```bash
python scripts/deploy_dual_arm.py --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25
```

Plot all learned primitive DS fields:

```bash
python scripts/plot_ds.py --all --ckpt_arm both --use_safe --joints 0 1 --out_dir data/results/ds_plots
```

For each checkpoint, this writes:

- `01_loss.png`
- `02_phase_portrait.png`
- `03_lyapunov.png`
- `04_rollouts.png`

Use `--ckpt_arm left` after left-only retraining. Use different `--joints a b`
pairs to inspect different 2D slices through the 7D error-space DS.

Diagnostic commands such as `--ds_scale 0`, `--ds_scale 0.2`, or
`--goal_gain > 0` are useful for isolating IK, actuation, or data issues. They
are not pure learned-DS runs.

## 14. Mental Model

The program is best understood as two layers:

The discrete layer chooses what the robot should do next:

```text
which block -> which primitive -> which Cartesian target -> which q_goal
```

The continuous layer decides how joints move:

```text
q - q_goal -> Neural DS -> q_dot -> q_cmd
```

The DS is successful only when the continuous layer reliably drives the current
joint state to the primitive's `q_goal`. The task should only switch primitives
after that convergence happens.
