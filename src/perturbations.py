"""
Perturbation injection utilities for evaluating DS robustness.

Three perturbation types:
  1. block_displacement — teleport a block by a small offset mid-task
  2. ee_disturbance     — apply a force impulse on the end-effector
  3. arm_block          — freeze one arm's controller for a duration
"""

import numpy as np


class BlockDisplacement:
    """Teleport a target block by a random XY offset, simulating a slip."""

    def __init__(self, max_offset=0.05):
        self.max_offset = max_offset

    def apply(self, env, block_name, rng=None):
        rng = rng or np.random.default_rng()
        offset = rng.uniform(-self.max_offset, self.max_offset, size=3)
        offset[2] = 0.0  # only XY
        block = env.get_block_obj(block_name)
        pos, rot = block.get_world_pose()
        block.set_world_pose(position=pos + offset, orientation=rot)
        block.set_linear_velocity(np.zeros(3))
        block.set_angular_velocity(np.zeros(3))
        return offset


class EEDisturbance:
    """Apply an impulse on the end-effector body, simulating a brief push."""

    def __init__(self, max_force=5.0, duration_s=0.2):
        self.max_force = max_force
        self.duration  = duration_s

    def apply(self, env, arm, rng=None, physics_dt=1/120, render=True):
        rng = rng or np.random.default_rng()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        magnitude = rng.uniform(0.5, 1.0) * self.max_force
        force = direction * magnitude

        franka = env.frankas[arm]
        steps  = int(self.duration / physics_dt)
        # Push by setting EE-body force; effect is approximate but visible
        for _ in range(steps):
            franka.end_effector.apply_force(force)
            env.step(render=render)
        return force


class ArmBlock:
    """Freeze an arm for a duration by locking its q_goal to the current joint
    configuration. The DS stays active but drives toward the current position,
    so the arm holds roughly in place. After the freeze, q_goal is restored via
    the caller's update_q_goal so the arm resumes its task."""

    def __init__(self, freeze_duration_s=1.0):
        self.duration = freeze_duration_s

    def apply(self, coordinator, arm, env, update_q_goal_fn,
              franka, physics_dt=1/120, render=True):
        """Hold the arm at its current configuration for freeze_duration_s.

        Args:
            coordinator      : TaskSequencer
            arm              : "left" or "right"
            env              : DualArmEnv
            update_q_goal_fn : callable(arm) that recomputes q_goal from IK
            franka           : the Franka articulation for this arm
            physics_dt       : simulation timestep
            render           : whether to render during freeze
        """
        task = coordinator.tasks[arm]
        task.q_goal = franka.get_joint_positions()[:7].copy()
        steps = int(self.duration / physics_dt)
        for _ in range(steps):
            env.step(render=render)
        update_q_goal_fn(arm)
