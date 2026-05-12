# Program Flow: Data Collection to Deployment

This project trains and deploys a hybrid joint-space controller for Franka
block stacking in Isaac Sim. The long free-space primitives, `reach` and
`transport`, use learned Neural Dynamical Systems (DS). The short constrained
primitives, `grasp`, `lift`, and `place`, use a Lula joint-space controller.
The full pipeline is:

1. Build an Isaac Sim scene with one or two Franka arms, a table, blocks, and
   stack goals.
2. Collect demonstrations with a joint-space Lula-goal expert.
3. Convert demonstrations into joint-error and joint-velocity training pairs.
4. Train Neural DS models for `reach` and `transport`.
5. Deploy by switching between learned DS primitives and scripted Lula
   primitives.
6. Optionally evaluate the full dual-arm system under perturbations.

The important idea is that every primitive is labeled with the same Lula
`q_goal` convention used at deployment. At deployment time, the program computes
`q_goal` for the active primitive. `reach` and `transport` pass the joint error
through the learned DS; `grasp`, `lift`, and `place` follow the same target with
a clamped joint-space Lula controller.

## Main Files

- `configs/default.yaml` defines scene geometry, block positions, primitive
  timing, training hyperparameters, and output paths.
- `configs/deploy_single_neural_physical.yaml` and
  `configs/deploy_neural_physical.yaml` provide current single/dual deploy
  argument defaults.
- `src/env.py` builds the Isaac Sim environment.
- `src/primitives.py` defines the primitive order and Cartesian targets.
- `src/franka_ik.py` wraps Lula IK for computing joint goals.
- `scripts/collect_ik.py` records demonstration trajectories.
- `scripts/audit_demo_labels.py` checks that primitive labels and `q_goal`
  attractor labels are consistent before training.
- `scripts/train_ds.py` trains a Neural DS for one learned primitive.
- `scripts/train_all.sh` trains the `reach` and `transport` DS models.
- `src/neural_ds.py` defines the DS and quadratic Lyapunov helper.
- `src/coordinator.py` manages primitive sequencing and stack slots.
- `scripts/deploy_single_arm.py` deploys one arm.
- `scripts/deploy_dual_arm.py` deploys both arms with protected-point
  modulation and optional safety guards.
- `scripts/evaluate.py` runs perturbation experiments and metrics.

## 1. Configuration

The pipeline starts from `configs/default.yaml`.

The config defines:

- Table size, height, and position.
- Block size, mass, colors, and initial positions.
- Franka arm spacing and base orientation.
- Stack goal locations.
- Dynamic stack clearance:
  - `stack.clearance_above_top`
- Primitive heights:
  - `hover`: height for `reach`
  - `grasp`: height for descending to the block
  - `lift`: height for carrying blocks
- Inter-primitive settling:
  - `sim.inter_primitive_pause_steps`
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

The scene and scripts all read this same config, so changing a height, stack
clearance, collection speed, or block layout changes both collection and
deployment behavior.

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
- `transport`: move above the stack goal at a clearance height that rises with
  the current stack.
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
python scripts/collect_ik.py --arm left --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
python scripts/collect_ik.py --arm right --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
```

The collector computes the same Lula `q_goal` that deployment will use, then
records a joint-space expert trajectory moving toward that target. This keeps
the demonstrated velocity and the saved attractor label in the same joint-space
convention.

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
7. Pauses between primitives without recording, so the arm settles without
   teaching the DS to stop at non-goal states.
8. Records joint positions and finite-difference joint velocities during the
   primitive controller steps.
9. Saves all successful demos to a pickle file.

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
- `stack_slot`: reserved stack slot, when applicable
- `stack_goal_z`: desired block-center stack height, when applicable
- collection speed metadata, including joint goal gain and joint velocity cap

Collection defaults are deliberately slower and cleaner than early debugging
runs:

```text
--joint_goal_gain 2.0
--collection_max_joint_vel 1.2
sim.inter_primitive_pause_steps: 120
```

Transport collection uses the same stack-clearance rule as deployment:

```text
transport_z = max(lift_h, existing_stack_top + stack.clearance_above_top)
```

### How `q_goal` Is Labeled

At the start of each primitive, the collector labels `q_goal` with the same
Lula IK target that deployment uses for that primitive transition.

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

- states by a uniform scalar `state_std` broadcast to all joints
- velocities by per-joint `vel_scale`

The checkpoint stores these normalization values because deployment must apply
the same scaling before calling the network. The uniform `state_std` is
important because the current Lyapunov candidate is `V(e_n)=||e_n||^2`; with a
single scale factor, decreasing `V` matches decreasing joint-space error.

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

So `left_reach.pt` is trained only from left-arm timesteps labeled `reach`, and
`right_transport.pt` is trained only from right-arm timesteps labeled
`transport`.

The primitive split is therefore:

| Checkpoint | Training samples used |
|---|---|
| `left_reach.pt` | left-arm `reach` timesteps |
| `left_transport.pt` | left-arm `transport` timesteps |
| `right_reach.pt` | right-arm `reach` timesteps |
| `right_transport.pt` | right-arm `transport` timesteps |

This means each learned DS covers one of the longer free-space motions:

- `reach` learns how to move from the current arm pose to a hover pose above a
  source block.
- `transport` learns how to move from source-side lift pose to goal-side lift
  pose.

`grasp`, `lift`, and `place` are short constrained motions and are executed
with the Lula joint-space controller instead of learned DS checkpoints.

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

`--arm both` is still supported for ablations, but it is not the standard
pipeline. The current `scripts/train_all.sh` trains per-arm checkpoints because
pooled left/right data can average mirrored joint-space flows.

### Checkpoint Data Manifest

`scripts/train_ds.py` now makes this split explicit when training. For every
checkpoint, it prints and saves a `data_manifest` containing:

- the primitive label used for filtering
- the demonstration files loaded
- total sample count
- samples by source file
- samples by arm
- samples by block
- samples by stack slot
- label-source counts

For example, when training:

```bash
python scripts/train_ds.py --primitive reach --arm left
```

the checkpoint should explicitly say that `left_reach.pt` was trained only from
`reach` timesteps in `data/demonstrations/left_demos.pkl`.

For an ablation with pooled data:

```bash
python scripts/train_ds.py --primitive transport --arm both
```

the checkpoint should explicitly say that `both_transport.pt` was trained only
from `transport` timesteps pooled from:

```text
data/demonstrations/left_demos.pkl
data/demonstrations/right_demos.pkl
```

This makes it possible to inspect a checkpoint later and verify exactly which
collection source produced that DS.

### What Is Not Partitioned

The current training script holds out the last about 10% of demos in each
pickle as validation when a file has at least five demos. Checkpoints are saved
by best validation imitation MSE. If there are too few demos for a validation
split, the script falls back to best training total loss.

There is also no block-specific DS. For example, all `reach` samples for all
blocks are pooled together into the same `reach` dataset. The state is
`q - q_goal`, so the model is expected to generalize across blocks by seeing the
joint-space error to the current primitive goal rather than the block identity.

## 7. Neural DS Training

`src/neural_ds.py` defines the model.

There is one learned velocity component and one fixed Lyapunov component:

- `NeuralDS`: maps normalized joint error to normalized joint velocity.
- `LyapunovNet`: parameterless quadratic `V(e_n)=||e_n||^2`.

The trained policy is:

```python
q_dot = f_theta(q - q_goal)
```

The current DS architecture includes a stable error-space prior:

```text
f(e_n) = residual_theta(e_n) - stable_skip_gain * e_n
```

This is still part of the learned DS: the stabilizing term is inside the
learned primitive model and is used during training. The deploy scripts keep
the external `DS_GOAL_GAIN` constant at `0.0`; change it in code only as a
diagnostic fallback.

One model is trained for each learned DS primitive and arm. The standard training
command is:

```bash
bash scripts/train_all.sh
```

That runs:

```bash
python scripts/train_ds.py --primitive reach --arm left
python scripts/train_ds.py --primitive transport --arm left
python scripts/train_ds.py --primitive reach --arm right
python scripts/train_ds.py --primitive transport --arm right
```

The output checkpoints are:

```text
data/checkpoints/left_reach.pt
data/checkpoints/left_transport.pt
data/checkpoints/right_reach.pt
data/checkpoints/right_transport.pt
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
- which stack slot is reserved
- whether a finished arm should return home
- when to move to the next primitive

For each arm, `ArmTaskState` tracks:

- block order
- current block index
- current primitive
- current `q_goal`
- reserved stack slot

The sequence is:

```text
reach -> grasp -> lift -> transport -> place
```

After `place`, the arm advances to the next block and returns to `reach`.

For `transport` and `place`, the sequencer reserves a stack slot. This prevents
two arms from targeting the same stack layer during dual-arm operation.

After the final `place` for an arm, deployment sends that arm back to its
initial home pose by default. This keeps a completed arm from lingering beside
the stack and blocking the remaining arm.

## 9. Single-Arm Deployment

`scripts/deploy_single_arm.py` runs one arm with learned DS for `reach` and
`transport`, and scripted Lula control for `grasp`, `lift`, and `place`.

Single-arm debug command:

```bash
python scripts/deploy_single_arm.py --arm left --ckpt_arm left --deploy_config configs/deploy_single_neural_physical.yaml --kinematic_carry --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv
```

At startup, the script:

1. Creates the Isaac simulation.
2. Builds a single-arm environment.
3. Loads the learned `reach` and `transport` checkpoints.
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
4. For `reach` and `transport`, call the current primitive's DS model.
5. Optionally apply Lyapunov safe projection with `--use_safe`.
6. Clip joint velocity.
7. Integrate one timestep into the persistent command state:

   ```python
   q_cmd_state = q_cmd_state + q_dot * physics_dt
   ```

8. Send `q_cmd_state` to Isaac's articulation controller.
9. Step the simulation.

The persistent command state is reset from the measured joints at primitive
transitions, after gripper waits, and after scripted helper motions. This avoids
the stall caused by repeatedly commanding only `q_measured + q_dot * dt`.

For `grasp`, `lift`, and `place`, the script skips the network call and follows
the primitive `q_goal` with the clamped Lula joint-space controller.

Learned DS primitives complete when:

```python
||q - q_goal|| < done_tol
```

Scripted primitives also use Cartesian completion checks. `place` should use a
tighter Cartesian tolerance, for example `--place_cart_done_tol 0.01`, so the
block is released close to the desired stack pose.

If a timeout happens before convergence, single-arm deployment now aborts by
default. That is the right behavior for pickup tests because advancing from a
failed `reach` to `grasp` would close the gripper from the wrong pose.

Use `--advance_on_timeout` only for debugging phase flow.

### Important Deployment Flags

- `--kinematic_carry`: after `grasp`, attach the active block to the end
  effector until `place`. This isolates motion planning from contact physics.
- `--use_safe`: project the learned velocity so it satisfies the Lyapunov
  decrease condition.
- `--ds_scale`: scale the learned DS output. Keep this at `1.0` when
  evaluating the learned `reach`/`transport` DS.
- `--cart_done_tol`: Cartesian completion tolerance for scripted IK
  primitives.
- `--place_cart_done_tol`: tighter Cartesian completion tolerance before
  releasing a block at the stack.
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
- DS or scripted Lula velocity

The coordinator reserves stack heights so both arms do not place blocks on the
same layer. Both arms start together by default
(`coordination.start_stagger_steps: 0`); `--stagger_steps` is available only for
ablation/debugging.

Dual-arm deployment uses DS modulation from `src/modulation.py` to alter joint
velocities when protected points get close. Protected points are built from
distal Lula FK frames and gripper proxy offsets. The closest point pairs are
modulated with Huber-style obstacle modulation and mapped back through a damped
Jacobian pseudoinverse.

The dual-arm script also supports:

- priority/yield modulation weights
- optional closest-to-stack priority switching
- lateral-order modulation to keep left/right arms on their own side
- optional stack keepout for the non-priority arm
- optional sampled-link hold via `--link_safety_hold`

Important dual-arm flags:

- `--mod_safe_radius`: protected-point modulation radius.
- `--mod_reactivity`: protected-point modulation strength.
- `--priority_mod_weight`: modulation blend weight for the priority arm.
- `--yield_mod_weight`: modulation blend weight for the yielding arm.
- `--yield_mod_speed_scale`: extra speed scale for the yielding arm.
- `--no_lateral_order_modulation`: disables side-order preservation.
- `--link_safety_radius`: sampled-link hold threshold.
- `--link_safety_hysteresis`: release margin for sampled-link hold.
- `--link_safety_hold`: enables the sampled-link hold.
- `--no_link_safety_hold`: disables the sampled-link hold.
- `--no_return_home_after_done`: leaves completed arms where they finish.

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

1. Run the hybrid controller with `--debug_ik --print_every 25`.
2. Check whether `reach` converges before timeout.
3. Check `cos->goal`.
4. If the DS points toward the goal but stalls, inspect command tracking and the
   integrated command state.
5. If the IK primitives fail too, investigate IK goals, asset setup, joint
   limits, or Isaac articulation control.

## 13. Typical End-to-End Commands

Collect demonstrations:

```bash
python scripts/collect_ik.py --arm left --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
python scripts/collect_ik.py --arm right --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
```

Audit labels:

```bash
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
```

Train all DS models:

```bash
bash scripts/train_all.sh
```

Run single-arm DS + Lula deployment:

```bash
python scripts/deploy_single_arm.py --arm left --ckpt_arm left --deploy_config configs/deploy_single_neural_physical.yaml --kinematic_carry --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv
```

Run left-only deployment after left-only retraining:

```bash
python scripts/deploy_single_arm.py --arm left --ckpt_arm left --deploy_config configs/deploy_single_neural_physical.yaml --kinematic_carry --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv
```

Run dual-arm deployment:

```bash
python scripts/deploy_dual_arm.py --deploy_config configs/deploy_neural_physical.yaml
```

Plot learned DS fields:

```bash
python scripts/plot_ds.py --all --ckpt_arm left --use_safe --joints 0 1 --out_dir data/results/ds_plots/left
python scripts/plot_ds.py --all --ckpt_arm right --use_safe --joints 0 1 --out_dir data/results/ds_plots/right
```

For each checkpoint, this writes:

- `01_loss.png`
- `02_phase_portrait.png`
- `03_lyapunov.png`
- `04_rollouts.png`

Use different `--joints a b` pairs to inspect different 2D slices through the
7D error-space DS.

Diagnostic commands such as `--ds_scale 0` or `--ds_scale 0.2` are useful for
isolating DS contribution. They are not the learned `reach`/`transport` DS
setting.

## 14. Mental Model

The program is best understood as two layers:

The discrete layer chooses what the robot should do next:

```text
which block -> which primitive -> which Cartesian target -> which q_goal
```

The continuous layer decides how joints move:

```text
q - q_goal -> Neural DS or Lula joint controller -> q_dot -> q_cmd
```

The DS part is successful only when the continuous layer reliably drives the
current joint state to the primitive's `q_goal` for `reach` and `transport`.
The task should only switch primitives after the relevant joint or Cartesian
completion check passes.
