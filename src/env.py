"""
Dual-arm Franka Isaac Sim environment.

Block positions are randomised each demo within per-arm workspace bounds
defined in the config. Call reset_blocks(rng=...) at the start of each demo.
"""

import numpy as np
import yaml

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka


class DualArmEnv:

    def __init__(self, config_path="configs/default.yaml", arms=("left", "right")):
        self.cfg = self._load_config(config_path)
        self.arms_active = arms

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg["sim"]["physics_dt"],
            rendering_dt=self.cfg["sim"]["rendering_dt"],
        )
        self.world.scene.add_default_ground_plane()

        self.frankas         = {}
        self.blocks          = {}
        self.goals           = {}
        self._default_joints = {}

        self._build_table()
        self._build_arms()
        self._build_blocks()
        self._build_goal_markers()

        self.world.reset()
        self._apply_default_joints(settle_steps=30)
        self._set_viewport_camera()

    # ── Config ────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ── Default joint helpers ─────────────────────────────────────────────────
    def _apply_default_joints(self, settle_steps=30):
        for name, franka in self.frankas.items():
            q = self._default_joints[name]
            full = franka.get_joint_positions().copy()
            full[:7] = q
            franka.set_joint_positions(full)
            franka.set_joint_velocities(np.zeros_like(full))
        for _ in range(settle_steps):
            self.world.step(render=False)

    def reset_arms(self, settle_steps=60, render=False):
        """Reset all arms to their default pose. Call at the start of each demo."""
        self._apply_default_joints(settle_steps=settle_steps)

    # ── Scene construction ────────────────────────────────────────────────────
    def _build_table(self):
        t = self.cfg["table"]
        centre = np.array(t["centre"])

        self.world.scene.add(FixedCuboid(
            prim_path="/World/Table/top",
            name="table_top",
            position=centre + np.array([0.0, 0.0, t["height"] - t["thick"] / 2]),
            scale=np.array([t["width"], t["depth"], t["thick"]]),
            color=np.array([0.45, 0.30, 0.15]),
        ))

        leg_h = t["height"] - t["thick"]
        leg_w = 0.05
        for i, (dx, dy) in enumerate([
            ( t["width"] / 2 - 0.06,  t["depth"] / 2 - 0.06),
            ( t["width"] / 2 - 0.06, -t["depth"] / 2 + 0.06),
            (-t["width"] / 2 + 0.06,  t["depth"] / 2 - 0.06),
            (-t["width"] / 2 + 0.06, -t["depth"] / 2 + 0.06),
        ]):
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Table/leg_{i}",
                name=f"table_leg_{i}",
                position=centre + np.array([dx, dy, leg_h / 2]),
                scale=np.array([leg_w, leg_w, leg_h]),
                color=np.array([0.35, 0.22, 0.10]),
            ))

    def _build_arms(self):
        a = self.cfg["arms"]
        t = self.cfg["table"]
        face_quat = np.array(a["face_table_quat"])
        arm_x = {"left": -a["spacing"] / 2, "right": a["spacing"] / 2}

        for name in self.arms_active:
            x = arm_x[name]

            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Pedestal_{name}",
                name=f"pedestal_{name}",
                position=np.array([x, a["y"], t["height"] / 2]),
                scale=np.array([0.15, 0.15, t["height"]]),
                color=np.array([0.4, 0.4, 0.4]),
            ))

            franka = self.world.scene.add(Franka(
                prim_path=f"/World/Franka_{name}",
                name=f"franka_{name}",
                position=np.array([x, a["y"], t["height"]]),
                orientation=face_quat,
            ))
            self.frankas[name] = franka
            self._default_joints[name] = np.array(a[f"default_joints_{name}"])
            self.goals[name] = tuple(self.cfg["shared_goal"])

    def _build_blocks(self):
        """Spawn blocks at workspace centre — randomised before each demo."""
        block_z = self.cfg["table"]["height"] + self.cfg["block"]["size"] / 2
        front_y_min = self.cfg["block_workspace"].get(
            "front_y_min", self.cfg["arms"]["y"] + 0.30
        )
        for arm in self.arms_active:
            ws = self.cfg["block_workspace"][arm]
            cx = (ws["x_min"] + ws["x_max"]) / 2
            cy = (max(ws["y_min"], front_y_min) + ws["y_max"]) / 2
            for i, b in enumerate(self.cfg[f"{arm}_blocks"]):
                spawn_pos = np.array([cx + i * 0.10, cy, block_z])
                obj = self.world.scene.add(DynamicCuboid(
                    prim_path=f"/World/Blocks/{b['name']}",
                    name=b["name"],
                    position=spawn_pos,
                    scale=np.array([self.cfg["block"]["size"]] * 3),
                    color=np.array(b["color"]),
                    mass=self.cfg["block"]["mass"],
                ))
                self.blocks[b["name"]] = obj

    def _build_goal_markers(self):
        t = self.cfg["table"]
        marker_z = t["height"] - t["thick"] / 2 + 0.001
        for arm in self.arms_active:
            gx, gy = self.goals[arm]
            color = np.array([1.0, 0.0, 0.0]) if arm == "left" else np.array([0.0, 0.0, 1.0])
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Goals/goal_{arm}",
                name=f"goal_{arm}",
                position=np.array([gx, gy, marker_z]),
                scale=np.array([0.08, 0.08, 0.002]),
                color=color,
            ))

    def _set_viewport_camera(self):
        camera_cfg = self.cfg.get("sim", {}).get("camera")
        if not camera_cfg:
            return

        eye = np.array(camera_cfg.get("eye", [0.0, 1.85, 1.18]), dtype=float)
        target = np.array(camera_cfg.get("target", [0.0, 0.22, 0.83]), dtype=float)

        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            try:
                from omni.isaac.core.utils.viewports import set_camera_view
            except ImportError:
                return

        try:
            set_camera_view(eye=eye, target=target)
        except TypeError:
            set_camera_view(eye=eye.tolist(), target=target.tolist())

    # ── Public utilities ──────────────────────────────────────────────────────
    def step(self, render=True):
        self.world.step(render=render)

    def reset_blocks(self, settle_steps=30, render=False, rng=None):
        """Randomise block positions within per-arm workspace bounds.

        Blocks are placed at random (x, y) within their arm's workspace with
        a minimum separation enforced between all blocks. The Y lower bound is
        also clamped so cubes spawn far enough in front of the robot bases.
        """
        if rng is None:
            rng = np.random.default_rng()

        block_z  = self.cfg["table"]["height"] + self.cfg["block"]["size"] / 2
        min_sep  = self.cfg["block_workspace"]["min_block_spacing"]
        front_y_min = self.cfg["block_workspace"].get(
            "front_y_min", self.cfg["arms"]["y"] + 0.30
        )
        placed   = []

        for arm in self.arms_active:
            ws = self.cfg["block_workspace"][arm]
            y_min = max(ws["y_min"], front_y_min)
            y_max = ws["y_max"]
            if y_min >= y_max:
                raise ValueError(
                    f"{arm} block workspace has no valid forward spawn band: "
                    f"y_min={y_min:.3f}, y_max={y_max:.3f}"
                )
            for b in self.cfg[f"{arm}_blocks"]:
                for _ in range(200):
                    x = rng.uniform(ws["x_min"], ws["x_max"])
                    y = rng.uniform(y_min, y_max)
                    if all(np.linalg.norm([x - px, y - py]) >= min_sep
                           for px, py in placed):
                        break
                placed.append((x, y))
                obj = self.blocks[b["name"]]
                obj.set_world_pose(position=np.array([x, y, block_z]))
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))

        for _ in range(settle_steps):
            self.world.step(render=render)

    def get_block_positions(self):
        return {name: obj.get_world_pose()[0] for name, obj in self.blocks.items()}

    def get_block_poses(self):
        """Return {name: (position, orientation_wxyz)} for all blocks."""
        return {name: obj.get_world_pose() for name, obj in self.blocks.items()}

    def get_block_grasp_quat(self, block_name):
        """Return a gripper-down quaternion (w,x,y,z) with yaw snapped to the
        nearest 90° face of the block."""
        from scipy.spatial.transform import Rotation

        _, quat_wxyz = self.blocks[block_name].get_world_pose()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2],
                               quat_wxyz[3], quat_wxyz[0]])
        yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz")[2]
        snapped_yaw = round(yaw / (np.pi / 2)) * (np.pi / 2)
        q_xyzw = Rotation.from_euler("xz", [np.pi, snapped_yaw]).as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    def get_ee_pose(self, arm):
        return self.frankas[arm].end_effector.get_world_pose()

    def get_block_obj(self, name):
        return self.blocks[name]

    def is_running(self):
        return True

    def close(self):
        pass
