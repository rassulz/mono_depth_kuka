#!/usr/bin/env python3
"""Assemble the Anima pick-and-place cell into one USD, headless, Isaac Sim 5.1.

Composes:
  * ground plane + physics + lighting
  * the KR10 arm (referenced from the imported USD)          -> articulated
  * the stand/table (referenced from the .usdz)              -> static collider
  * the toy car (referenced from the .usdz)                  -> DYNAMIC rigid body
  * a RealSense D455 camera prim above the table             -> sensor pose

Run with the venv Isaac python (NOT system python3):

    cd ~/Anima_project
    ~/isaac/venv/bin/python scripts/build_scene.py
    #  or:  ~/isaac/IsaacLab/isaaclab.sh -p scripts/build_scene.py

Output: isaac_scene_anima.usd  (open it in the Isaac Sim GUI).

NOTE ON PLACEMENT: the transforms below are PLACEHOLDERS. This is a digital
twin -- for the camera->robot pick-and-place to work, these must match your
real cell. The script PRINTS the bounding box of each asset at load time so
you can read off the real table height etc. and fill in the constants marked
"MEASURE THIS".
"""

import os

# ============================ PLACEMENT CONSTANTS ==========================
# All in meters, world frame is Z-up, origin at the ground plane.
# >>> MEASURE THIS against your real cell / read the printed bounding boxes. <<<
TABLE_TOP_Z = 0.90                     # height of the table surface above floor
ROBOT_BASE_POS = (0.0, 0.0, TABLE_TOP_Z)   # where the arm is bolted on the table
ROBOT_BASE_RPY = (0.0, 0.0, 0.0)           # base orientation, degrees
STAND_POS = (0.0, 0.0, 0.0)                # stand sits on the floor
STAND_RPY = (0.0, 0.0, 0.0)
CAR_POS = (0.45, 0.20, TABLE_TOP_Z + 0.03)  # toy car on the table, near the arm
CAR_MASS_KG = 0.20                          # ~200 g toy car
CAM_POS = (0.30, 0.00, TABLE_TOP_Z + 1.20)  # RealSense on the stick, above table
CAM_RPY = (0.0, 0.0, 0.0)                   # (0,0,0) = looking straight DOWN (-Z)
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ROBOT_USD = os.path.join(PROJECT, "3d_model", "kr10_official_imported.usd")
STAND_USDZ = os.path.join(PROJECT, "3d_model", "arlan_project_srtand_usd_updated.usdz")
CAR_USDZ = os.path.join(PROJECT, "3d_model", "car_object_arlan_usd.usdz")
OUT_USD = os.path.join(PROJECT, "isaac_scene_anima.usd")

# Fall back to the USD that shipped inside the extracted zip if we haven't
# re-imported yet.
if not os.path.isfile(ROBOT_USD):
    _alt = os.path.join(PROJECT, "3d_model", "kr10_arm_isaac(2)", "kr10_arm", "kr10_arm", "kr10_arm.usd")
    if os.path.isfile(_alt):
        ROBOT_USD = _alt

# --- launch Kit BEFORE importing any omni/isaacsim/pxr module --------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysicsSchemaTools,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
)

# --- fresh stage: meters, Z-up, /World as default prim ---------------------
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())


def set_pose(prim, pos, rpy_deg=(0, 0, 0), scale=None):
    """Author translate/rotate/scale on any prim, regardless of its existing
    xformOp stack. XformCommonAPI silently no-ops on prims whose ops it does
    not recognize (e.g. the quaternion `orient` op the URDF importer authors
    on the robot root) -- the Kit TransformPrimSRT command handles all cases."""
    import omni.kit.commands

    if scale is None:
        # Preserve the prim's existing scale -- add_reference_to_stage authors
        # a unit-correction scale (e.g. 0.001 for mm assets) that must survive.
        m = UsdGeom.Xformable(prim).GetLocalTransformation()
        scale = (m.GetRow3(0).GetLength(), m.GetRow3(1).GetLength(), m.GetRow3(2).GetLength())
    omni.kit.commands.execute(
        "TransformPrimSRT",
        path=Sdf.Path(prim.GetPath()),
        new_translation=Gf.Vec3d(*pos),
        new_rotation_euler=Gf.Vec3d(*rpy_deg),
        new_rotation_order=Gf.Vec3i(0, 1, 2),
        new_scale=Gf.Vec3d(*scale),
    )


def snap_onto_surface(prim, center_xy, surface_top_z, gap=0.005):
    """Translate an outer wrapper prim so its world bbox is centered at
    center_xy and its bottom rests at surface_top_z + gap. Robust to the
    asset's internal pivot/vertex offset (that's what made the car float).
    The prim must be a wrapper with NO prior transform of its own."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()
    tx = center_xy[0] - (mn[0] + mx[0]) / 2.0
    ty = center_xy[1] - (mn[1] + mx[1]) / 2.0
    tz = (surface_top_z + gap) - mn[2]
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(tx, ty, tz))


def print_bbox(prim, label):
    """Print an asset's world-space bounding box so placement can be tuned."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()
    print(f"[bbox] {label:12s} min=({mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f})  "
          f"max=({mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f})  "
          f"size=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f})")


def add_colliders(root_prim, approximation):
    """Apply collision to every mesh under a referenced prim.
    approximation: 'convexHull' for dynamic, 'none' (triangle mesh) for static."""
    n = 0
    for p in Usd.PrimRange(root_prim):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(approximation)
            n += 1
    return n


# --- physics scene + gravity ----------------------------------------------
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# --- ground plane ----------------------------------------------------------
PhysicsSchemaTools.addGroundPlane(
    stage, "/World/groundPlane", "Z", 50.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
)

# --- lighting --------------------------------------------------------------
UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(1000)
UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(300)

# --- KR10 arm (articulated) ------------------------------------------------
if os.path.isfile(ROBOT_USD):
    robot = add_reference_to_stage(ROBOT_USD, "/World/kr10")
    set_pose(robot, ROBOT_BASE_POS, ROBOT_BASE_RPY)
    print_bbox(robot, "kr10")
else:
    print(f"[warn] robot USD not found ({ROBOT_USD}); run import_kr10.py first.")

# --- stand / table (static collider) --------------------------------------
stand = add_reference_to_stage(STAND_USDZ, "/World/stand")
set_pose(stand, STAND_POS, STAND_RPY)
n_stand = add_colliders(stand, "none")   # static -> full triangle mesh is fine
print_bbox(stand, "stand")
print(f"[phys] stand static colliders: {n_stand} meshes")

# --- toy car (DYNAMIC rigid body) -----------------------------------------
# Reference the car UNDER an outer wrapper we fully control, then snap it onto
# the table by bounding box -- this fixes the "car floating in the sky" issue
# caused by the asset's geometry being offset from its prim origin.
car = UsdGeom.Xform.Define(stage, "/World/car").GetPrim()
add_reference_to_stage(CAR_USDZ, "/World/car/model")
snap_onto_surface(car, (CAR_POS[0], CAR_POS[1]), TABLE_TOP_Z)
UsdPhysics.RigidBodyAPI.Apply(car)
UsdPhysics.MassAPI.Apply(car).CreateMassAttr(CAR_MASS_KG)
n_car = add_colliders(car, "convexHull")  # dynamic -> convex hull required
print_bbox(car, "car")
print(f"[phys] car dynamic colliders: {n_car} meshes, mass={CAR_MASS_KG} kg")

# --- RealSense D455 camera prim -------------------------------------------
# Intrinsics approximate the D455 RGB stream (HFOV ~90 deg, 16:9). The actual
# capture RESOLUTION is set at runtime when you attach a render product /
# isaacsim.sensors.camera.Camera to this prim (e.g. 1280x720). See notes below.
cam = UsdGeom.Camera.Define(stage, "/World/realsense_d455")
cam.CreateFocalLengthAttr(10.0)
cam.CreateHorizontalApertureAttr(20.0)     # HFOV = 2*atan(20/(2*10)) = 90 deg
cam.CreateVerticalApertureAttr(11.25)      # 16:9 -> VFOV ~ 58.7 deg
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
set_pose(cam.GetPrim(), CAM_POS, CAM_RPY)  # RPY (0,0,0) => looks straight down -Z
print(f"[cam ] realsense_d455 at {CAM_POS}, RPY {CAM_RPY} (0,0,0 = top-down)")

# --- save ------------------------------------------------------------------
omni.usd.get_context().save_as_stage(OUT_USD)
print(f"[save] {OUT_USD}")
print("Open it in the GUI:  ~/isaac/venv/bin/isaacsim  (File > Open)")

# --- Blackwell (RTX 5080) teardown workaround ------------------------------
import sys  # noqa: E402

sys.stdout.flush()
os._exit(0)
