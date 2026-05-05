"""
Dual-arm Franka Isaac Sim environment.

Provides DualArmEnv class that scripts can import and use without re-creating
the scene every time. The scene contains:
  - Two Franka Panda arms mounted side-by-side on pedestals
  - A long shared table with both arms' workspaces accessible
  - Three blocks per arm, colour-coded by side
  - Goal markers on the far side of the table for stacking

Note: SimulationApp must be initialised in the calling script before importing
this module — the imports below depend on it being live.
"""

import numpy as np
import yaml
from pathlib import Path
from pxr import Gf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka


class DualArmEnv:
    """Dual-arm Franka environment with table and stackable blocks."""

    def __init__(self, config_path="configs/default.yaml", arms=("left", "right")):
        self.cfg = self._load_config(config_path)
        self.arms_active = arms  # which arms to spawn — useful for single-arm tests

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg["sim"]["physics_dt"],
            rendering_dt=self.cfg["sim"]["rendering_dt"],
        )

        self.frankas    = {}  # arm_name -> Franka object
        self.blocks     = {}  # block_name -> DynamicCuboid object
        self.block_init = {}  # block_name -> initial position (np.array)
        self.goals      = {}  # arm_name -> (x, y) goal location

        self._build_lighting()
        self._build_camera()
        self._build_ground()
        self._build_table()
        self._build_arms()
        self._build_blocks()
        self._build_goal_markers()

        self.world.reset()

    # ── Loading ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ── Scene construction ────────────────────────────────────────────────────
    def _build_lighting(self):
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(650.0)

        sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        sun.CreateIntensityAttr(2200.0)
        sun.CreateAngleAttr(0.35)
        sun_xform = UsdGeom.Xformable(sun.GetPrim())
        sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))

    def _build_camera(self):
        stage = self.world.stage
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        xform = UsdGeom.Xformable(camera.GetPrim())
        # Front-of-table view: camera is on the far side of the table looking
        # back toward the robots, with enough height to see the stack.
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.65, 1.45))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 180.0))
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    def _build_ground(self):
        """Local ground collider.

        Isaac's add_default_ground_plane() resolves a material/asset through
        the Isaac assets root. On offline installs that can fail before the
        scene is even built, so keep the ground entirely local.
        """
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.45, -0.015]),
            scale=np.array([3.0, 3.0, 0.02]),
            color=np.array([0.28, 0.28, 0.28]),
        ))

    def _build_table(self):
        t = self.cfg["table"]
        centre = np.array(t["centre"])

        # Table top
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Table/top",
            name="table_top",
            position=centre + np.array([0.0, 0.0, t["height"] - t["thick"] / 2]),
            scale=np.array([t["width"], t["depth"], t["thick"]]),
            color=np.array([0.45, 0.30, 0.15]),
        ))

        # Legs
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
        franka_usd = self.cfg.get("assets", {}).get("franka_usd")
        if franka_usd:
            franka_usd = str(Path(franka_usd).expanduser().resolve())

        arm_x = {
            "left":  -a["spacing"] / 2,
            "right":  a["spacing"] / 2,
        }

        for name in self.arms_active:
            x = arm_x[name]

            # Pedestal
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Pedestal_{name}",
                name=f"pedestal_{name}",
                position=np.array([x, a["y"], t["height"] / 2]),
                scale=np.array([0.15, 0.15, t["height"]]),
                color=np.array([0.4, 0.4, 0.4]),
            ))

            # Robot
            try:
                franka = self.world.scene.add(Franka(
                    prim_path=f"/World/Franka_{name}",
                    name=f"franka_{name}",
                    usd_path=franka_usd,
                    position=np.array([x, a["y"], t["height"]]),
                    orientation=face_quat,
                ))
            except RuntimeError as exc:
                if "assets root" in str(exc):
                    raise RuntimeError(
                        "Could not locate Isaac Sim robot assets for Franka. "
                        "Make the Isaac assets root reachable, or set "
                        "`assets.franka_usd` in configs/default.yaml to a "
                        "local FrankaPanda/franka.usd file."
                    ) from exc
                raise
            self.frankas[name] = franka

            # Goal location for this arm
            self.goals[name] = tuple(self.cfg["goals"][name])

    def _build_blocks(self):
        for arm in self.arms_active:
            block_list = self.cfg[f"{arm}_blocks"]
            for b in block_list:
                pos = np.array(b["pos"])
                obj = self.world.scene.add(DynamicCuboid(
                    prim_path=f"/World/Blocks/{b['name']}",
                    name=b["name"],
                    position=pos,
                    scale=np.array([self.cfg["block"]["size"]] * 3),
                    color=np.array(b["color"]),
                    mass=self.cfg["block"]["mass"],
                ))
                self.blocks[b["name"]]     = obj
                self.block_init[b["name"]] = pos.copy()

    def _build_goal_markers(self):
        t = self.cfg["table"]
        marker_z = t["height"] - t["thick"] / 2 + 0.001
        for arm in self.arms_active:
            gx, gy = self.goals[arm]
            color  = np.array([1.0, 0.0, 0.0]) if arm == "left" else np.array([0.0, 0.0, 1.0])
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Goals/goal_{arm}",
                name=f"goal_{arm}",
                position=np.array([gx, gy, marker_z]),
                scale=np.array([0.08, 0.08, 0.002]),
                color=color,
            ))

    # ── Public utilities ──────────────────────────────────────────────────────
    def step(self, render=True):
        self.world.step(render=render)

    def reset_blocks(self, settle_steps=30, render=False):
        """Teleport all blocks back to initial positions AND identity
        orientation, zero velocity. Without resetting orientation, blocks
        accumulate rotation across demos which silently changes
        grasp_quat_from_block(...) and corrupts training labels."""
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        for name, obj in self.blocks.items():
            obj.set_world_pose(position=self.block_init[name],
                               orientation=identity_quat)
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        for _ in range(settle_steps):
            self.world.step(render=render)

    def get_block_positions(self):
        return {name: obj.get_world_pose()[0] for name, obj in self.blocks.items()}

    def get_block_poses(self):
        """Returns {name: (position, orientation_wxyz)} for all blocks."""
        return {name: obj.get_world_pose() for name, obj in self.blocks.items()}

    def get_ee_pose(self, arm):
        return self.frankas[arm].end_effector.get_world_pose()

    def get_block_obj(self, name):
        return self.blocks[name]

    def is_running(self):
        return True  # caller checks simulation_app.is_running() externally

    def close(self):
        # Nothing to clean here — SimulationApp owns lifecycle
        pass
