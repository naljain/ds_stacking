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
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
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
        self._block_physics_material = None
        self._gripper_physics_material = None

        self._build_lighting()
        self._build_camera()
        self._build_ground()
        self._build_table()
        self._build_arms()
        self._build_blocks()
        self._build_goal_markers()
        self._apply_gripper_physics_material()

        self.world.reset()
        self._configure_viewport_camera()

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
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    def _configure_viewport_camera(self):
        """Point the interactive viewport at the same head-on view.

        The USD camera above is useful for render products, but Isaac's active
        perspective viewport can keep its previous camera unless we explicitly
        set it.
        """
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport

            camera_cfg = self.cfg.get("sim", {}).get("camera", {})
            eye = np.array(camera_cfg.get("eye", [0.0, 2.25, 1.45]), dtype=float)
            target = np.array(camera_cfg.get("target", [0.0, 0.25, 0.80]), dtype=float)

            try:
                set_camera_view(
                    eye=eye,
                    target=target,
                    camera_prim_path="/World/Camera",
                )
            except TypeError:
                set_camera_view(
                    eye=eye.tolist(),
                    target=target.tolist(),
                    camera_prim_path="/World/Camera",
                )

            viewport = get_active_viewport()
            if viewport is None:
                return
            set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path="/OmniverseKit_Persp",
                viewport_api=viewport,
            )
        except Exception as exc:
            print(f"[WARN] Could not configure viewport camera: {exc}")

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
        block_cfg = self.cfg["block"]
        self._block_physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/high_friction_blocks",
            name="high_friction_blocks",
            static_friction=block_cfg.get("static_friction", 2.0),
            dynamic_friction=block_cfg.get("dynamic_friction", 1.5),
            restitution=block_cfg.get("restitution", 0.0),
        )
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
                    physics_material=self._block_physics_material,
                ))
                self.blocks[b["name"]]     = obj
                self.block_init[b["name"]] = pos.copy()

    def _apply_gripper_physics_material(self):
        """Bind high-friction physics material to Franka finger collision prims.

        Pinch lifting depends on friction at both sides of the contact. Setting
        the cube material alone is not enough if the referenced Franka USD keeps
        default low-friction finger collision meshes.
        """
        grip_cfg = self.cfg.get("gripper", {})
        if not grip_cfg.get("high_friction_fingers", True):
            return

        self._gripper_physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/high_friction_gripper",
            name="high_friction_gripper",
            static_friction=grip_cfg.get("static_friction", 2.5),
            dynamic_friction=grip_cfg.get("dynamic_friction", 2.0),
            restitution=grip_cfg.get("restitution", 0.0),
        )

        stage = self.world.stage
        bound = 0
        candidates = []
        for arm in self.arms_active:
            root = f"/World/Franka_{arm}"
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                lower = path.lower()
                if not path.startswith(root):
                    continue
                if "finger" not in lower:
                    continue
                candidates.append((path, prim.GetTypeName()))
                is_finger_body = (
                    "panda_leftfinger" in lower
                    or "panda_rightfinger" in lower
                )
                is_finger_joint = prim.IsA(UsdPhysics.Joint)
                if is_finger_joint:
                    continue
                if not (
                    prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.IsA(UsdGeom.Boundable)
                    or is_finger_body
                ):
                    continue
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(
                    self._gripper_physics_material.material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound += 1
        if bound == 0:
            print("[WARN] No Franka finger collision prims found for high-friction material")
            if candidates:
                preview = ", ".join(
                    f"{path}<{type_name or 'typeless'}>"
                    for path, type_name in candidates[:12]
                )
                print(f"[WARN] Finger-like prim candidates: {preview}")
        else:
            print(f"[ENV] Bound high-friction gripper material to {bound} finger prims")

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

    def get_arm_link_positions(self, arm):
        """Return representative Franka link positions for coarse safety checks.

        Isaac's high-level Franka wrapper exposes the end effector directly but
        not a stable link-distance API across versions. The USD link prims are
        stable enough for a conservative sampled-link clearance check.
        """
        stage = self.world.stage
        positions = []
        for i in range(8):
            prim = stage.GetPrimAtPath(f"/World/Franka_{arm}/panda_link{i}")
            if not prim or not prim.IsValid():
                continue
            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            p = mat.ExtractTranslation()
            positions.append(np.array([p[0], p[1], p[2]], dtype=float))
        ee_pos = self.get_ee_pose(arm)[0]
        positions.append(ee_pos.copy())
        return positions

    def get_block_obj(self, name):
        return self.blocks[name]

    def is_running(self):
        return True  # caller checks simulation_app.is_running() externally

    def close(self):
        # Nothing to clean here — SimulationApp owns lifecycle
        pass
