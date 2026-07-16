#!/usr/bin/env python3
"""
KR10 pick-and-place for Isaac Sim 5.1.0  (standalone Python, "magnet" gripper).

This version IMPORTS THE URDF ITSELF from Python -- you do NOT need the GUI
File->Import (which fails on this machine with "Accessed invalid null prim").

HOW TO RUN  (on the PC with Isaac Sim 5.1.0):
    cd <isaac-sim-install-folder>
    ./python.sh /home/nurtay/Desktop/kr10_arm_isaac/kr10_arm/kr10_pick_place.py

WHAT IT DOES
  1. Imports kr10_arm.urdf directly (no GUI).
  2. Puts a red cube on the ground as the object to move.
  3. Drives the 6 joints through waypoints.
  4. "Magnet" ON at the pick point (fixed joint between tip and cube),
     OFF at the place point.

FIRST RUN: set PRINT_JOINTS = True to discover joint angles for the waypoints
(jog the arm in the GUI, copy the printed numbers into the WAYPOINTS below),
then set PRINT_JOINTS = False and run again.
"""

# ===========================================================================
# CONFIG  --  EDIT THESE
# ===========================================================================
URDF_PATH = "/home/nurtay/Desktop/kr10_arm_isaac/kr10_arm/kr10_arm.urdf"  # <- path on the PC

END_EFFECTOR_LINK = "end_effector_v1_1"        # last link (the "magnet" tip)
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

PRINT_JOINTS = False   # True on the first run to discover joint angles

# Waypoints = 6 joint angles in RADIANS (order = JOINT_NAMES).
HOME        = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ABOVE_PICK  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # hover over the cube
AT_PICK     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # tip on the cube -> magnet ON
LIFT_PICK   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # lift up with the cube
ABOVE_PLACE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # hover over drop spot
AT_PLACE    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # lower to drop spot -> magnet OFF

CUBE_START = [0.6, 0.0, 0.05]     # where the cube starts (meters)
CUBE_SIZE = 0.05
STEPS_PER_MOVE = 120

# ===========================================================================
# Boot Isaac Sim
# ===========================================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.utils.extensions import enable_extension

# enable the URDF importer extension (the working core, not the flaky File->Import)
enable_extension("isaacsim.asset.importer.urdf")
simulation_app.update()

import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, Sdf


def _set(cfg, attr, val):
    """Set a config attribute if it exists (attribute names vary across versions)."""
    try:
        setattr(cfg, attr, val)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Import the URDF from Python
# ---------------------------------------------------------------------------
_, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
_set(import_config, "merge_fixed_joints", False)
_set(import_config, "fix_base", True)                 # base_link stays anchored
_set(import_config, "make_default_prim", True)
_set(import_config, "import_inertia_tensor", True)
_set(import_config, "distance_scale", 1.0)
_set(import_config, "create_physics_scene", True)

result, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
)
print(">>> URDF imported at prim path:", robot_prim_path)

# ---------------------------------------------------------------------------
# Build the scene
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

arm = SingleArticulation(prim_path=robot_prim_path, name="kr10_arm")
world.scene.add(arm)

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube",
        name="cube",
        position=np.array(CUBE_START),
        scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
        color=np.array([1.0, 0.2, 0.2]),
    )
)

world.reset()

stage = omni.usd.get_context().get_stage()
ee_prim_path = f"{robot_prim_path}/{END_EFFECTOR_LINK}"
dof_index = [arm.dof_names.index(n) for n in JOINT_NAMES]

MAGNET_JOINT_PATH = "/World/magnet_joint"


def magnet_on():
    if stage.GetPrimAtPath(MAGNET_JOINT_PATH):
        return
    j = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(MAGNET_JOINT_PATH))
    j.CreateBody0Rel().SetTargets([Sdf.Path(ee_prim_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/World/cube")])
    print(">>> MAGNET ON (cube attached)")


def magnet_off():
    if stage.GetPrimAtPath(MAGNET_JOINT_PATH):
        stage.RemovePrim(Sdf.Path(MAGNET_JOINT_PATH))
        print(">>> MAGNET OFF (cube released)")


def move_to(target_angles, settle=STEPS_PER_MOVE):
    full = np.zeros(arm.num_dof)
    for slot, ang in zip(dof_index, target_angles):
        full[slot] = ang
    arm.apply_action(ArticulationAction(joint_positions=full))
    for _ in range(settle):
        world.step(render=True)


# ---------------------------------------------------------------------------
# Discovery mode: print joint angles so you can build the waypoints
# ---------------------------------------------------------------------------
if PRINT_JOINTS:
    print("PRINT_JOINTS mode: jog the arm in the GUI, copy the printed numbers.")
    t = 0
    while simulation_app.is_running():
        world.step(render=True)
        t += 1
        if t % 60 == 0:
            q = arm.get_joint_positions()
            print("joint angles (rad):", [round(float(q[i]), 4) for i in dof_index])
    simulation_app.close()
    raise SystemExit

# ---------------------------------------------------------------------------
# Pick-and-place sequence
# ---------------------------------------------------------------------------
print("Starting pick-and-place...")
move_to(HOME)
move_to(ABOVE_PICK)
move_to(AT_PICK)
magnet_on()
move_to(LIFT_PICK)
move_to(ABOVE_PLACE)
move_to(AT_PLACE)
magnet_off()
move_to(ABOVE_PLACE)
move_to(HOME)
print("Done. Close the window to exit.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
