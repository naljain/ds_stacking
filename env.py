
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka

world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/60)
world.scene.add_default_ground_plane()

#  Table dimensions 
# Long side runs along X axis, arms sit along negative Y edge
TABLE_WIDTH  = 1.20   # long side — along X
TABLE_DEPTH  = 0.70   # short side — along Y
TABLE_HEIGHT = 0.74
TABLE_THICK  = 0.05
LEG_H        = TABLE_HEIGHT - TABLE_THICK
LEG_W        = 0.05

# Table centre 
TABLE_CENTRE = np.array([0.0, 0.45, 0.0])

#  Table top 
world.scene.add(
    FixedCuboid(
        prim_path="/World/Table/top",
        name="table_top",
        position=TABLE_CENTRE + np.array([0.0, 0.0, TABLE_HEIGHT - TABLE_THICK / 2]),
        scale=np.array([TABLE_WIDTH, TABLE_DEPTH, TABLE_THICK]),
        color=np.array([0.45, 0.3, 0.15]),
    )
)

#  Table legs 
for i, (dx, dy) in enumerate([
    ( TABLE_WIDTH/2 - 0.06,  TABLE_DEPTH/2 - 0.06),
    ( TABLE_WIDTH/2 - 0.06, -TABLE_DEPTH/2 + 0.06),
    (-TABLE_WIDTH/2 + 0.06,  TABLE_DEPTH/2 - 0.06),
    (-TABLE_WIDTH/2 + 0.06, -TABLE_DEPTH/2 + 0.06),
]):
    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Table/leg_{i}",
            name=f"table_leg_{i}",
            position=TABLE_CENTRE + np.array([dx, dy, LEG_H / 2 - TABLE_HEIGHT / 2 + LEG_H / 2]),
            scale=np.array([LEG_W, LEG_W, LEG_H]),
            color=np.array([0.35, 0.22, 0.1]),
        )
    )

#  Both arms on the near side (Y=0), facing +Y toward the table 
# Default Franka faces +X 
# Quaternion (w, x, y, z) for +90° around Z:
FACE_TABLE = np.array([0.7071, 0.0, 0.0, 0.7071])

ARM_Y        = 0.0                  # arms sit at Y=0
ARM_SPACING  = 0.60                 # distance between the two arms along X
ARM_LEFT_X   = -ARM_SPACING / 2    # left arm at X=-0.30
ARM_RIGHT_X  =  ARM_SPACING / 2    # right arm at X=+0.30

#  Pedestals 
for name, x in [("pedestal_left", ARM_LEFT_X), ("pedestal_right", ARM_RIGHT_X)]:
    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array([x, ARM_Y, TABLE_HEIGHT / 2]),
            scale=np.array([0.15, 0.15, TABLE_HEIGHT]),
            color=np.array([0.4, 0.4, 0.4]),
        )
    )

#  Franka Left ─
franka_left = world.scene.add(
    Franka(
        prim_path="/World/Franka_L",
        name="franka_left",
        position=np.array([ARM_LEFT_X, ARM_Y, TABLE_HEIGHT]),
        orientation=FACE_TABLE,
    )
)

#  Franka Right 
franka_right = world.scene.add(
    Franka(
        prim_path="/World/Franka_R",
        name="franka_right",
        position=np.array([ARM_RIGHT_X, ARM_Y, TABLE_HEIGHT]),
        orientation=FACE_TABLE,
    )
)

#  Blocks ─
BLOCK_SIZE = 0.045
SURFACE_Z  = TABLE_HEIGHT + BLOCK_SIZE / 2 + 0.001

# Left arm's blocks — left half of the table
left_blocks = [
    ("block_red",    [-0.40,  0.30, SURFACE_Z], [1.0, 0.1, 0.1]),
    ("block_orange", [-0.40,  0.50, SURFACE_Z], [1.0, 0.5, 0.0]),
    ("block_pink",   [-0.25,  0.40, SURFACE_Z], [1.0, 0.4, 0.7]),
]

# Right arm's blocks — right half of the table
right_blocks = [
    ("block_blue",   [0.40,  0.30, SURFACE_Z], [0.1, 0.3, 1.0]),
    ("block_green",  [0.40,  0.50, SURFACE_Z], [0.1, 0.8, 0.1]),
    ("block_yellow", [0.25,  0.40, SURFACE_Z], [1.0, 0.85, 0.0]),
]

for name, pos, color in left_blocks + right_blocks:
    world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/Blocks/{name}",
            name=name,
            position=np.array(pos),
            scale=np.array([BLOCK_SIZE] * 3),
            color=np.array(color),
            mass=0.05,
        )
    )

#  Goal markers (where each arm stacks) 
MARKER_Z = TABLE_HEIGHT - TABLE_THICK / 2 + 0.001

world.scene.add(
    FixedCuboid(
        prim_path="/World/Goals/goal_left",
        name="goal_left",
        position=np.array([-0.30, 0.65, MARKER_Z]),
        scale=np.array([0.08, 0.08, 0.002]),
        color=np.array([1.0, 0.0, 0.0]),
    )
)

world.scene.add(
    FixedCuboid(
        prim_path="/World/Goals/goal_right",
        name="goal_right",
        position=np.array([0.30, 0.65, MARKER_Z]),
        scale=np.array([0.08, 0.08, 0.002]),
        color=np.array([0.0, 0.0, 1.0]),
    )
)

#  Helpers 
def get_block_positions():
    positions = {}
    for name, _, _ in left_blocks + right_blocks:
        obj = world.scene.get_object(name)
        if obj:
            positions[name] = obj.get_world_pose()[0]
    return positions

def get_arm_ee_poses():
    return {
        "left":  franka_left.end_effector.get_world_pose(),
        "right": franka_right.end_effector.get_world_pose(),
    }

#  Reset & run 
world.reset()

print("[INFO] Scene ready.")
print(f"[INFO] Left arm  @ X={ARM_LEFT_X:.2f}, blocks: {[b[0] for b in left_blocks]}")
print(f"[INFO] Right arm @ X={ARM_RIGHT_X:.2f}, blocks: {[b[0] for b in right_blocks]}")

while simulation_app.is_running():
    world.step(render=True)
    if world.is_stopped():
        break

simulation_app.close()