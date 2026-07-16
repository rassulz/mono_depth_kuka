#!/usr/bin/env python3
"""Import the KR10 URDF into a clean USD, headless, on Isaac Sim 5.1.

Run with your venv's Isaac Sim python (NOT system python3):

    cd ~/Anima_project
    ~/isaac/venv/bin/python scripts/import_kr10.py
    # or:  ~/isaac/IsaacLab/isaaclab.sh -p scripts/import_kr10.py

Output: 3d_model/kr10_arm/kr10_arm_imported.usd

IMPORTANT: this will produce an INVISIBLE robot (frames only) until the mesh
files referenced by the URDF exist on disk. Run scripts/check_meshes.py first.
"""

import os

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
# Default: the official ros-industrial arm + custom magnet end-effector.
# Override with URDF_PATH / OUT_USD env vars to import something else.
URDF_PATH = os.environ.get(
    "URDF_PATH", os.path.join(PROJECT, "3d_model", "kr10_official", "kr10_with_ee.urdf"))
OUT_USD = os.environ.get(
    "OUT_USD", os.path.join(PROJECT, "3d_model", "kr10_official_imported.usd"))

# --- preflight: refuse to import if meshes are missing ---------------------
# (Importing anyway just reproduces the "only XYZ frames" problem.)
import xml.etree.ElementTree as ET

_urdf_dir = os.path.dirname(URDF_PATH)
_missing = []
for _m in ET.parse(URDF_PATH).getroot().findall(".//geometry/mesh"):
    _fn = _m.get("filename", "")
    if not _fn.startswith("package://"):
        _p = os.path.normpath(os.path.join(_urdf_dir, _fn))
        if not os.path.isfile(_p):
            _missing.append(_p)
if _missing:
    print("REFUSING TO IMPORT: the URDF references mesh files that do not exist.")
    print("You would get an invisible robot (coordinate frames only). Missing e.g.:")
    for _p in sorted(set(_missing))[:5]:
        print("   ", _p)
    print("Fix: put the .stl meshes there (see scripts/check_meshes.py), then re-run.")
    raise SystemExit(1)

# --- launch Kit BEFORE importing any omni/isaacsim/pxr module --------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdLux, UsdPhysics  # noqa: E402

# Fresh stage, meters + Z-up (URDF convention).
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# --- build an import config ------------------------------------------------
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False      # keep the kinematic tree intact
import_config.fix_base = True                 # arm is bolted down -> fixed base
import_config.make_default_prim = True
import_config.create_physics_scene = False    # the scene builder adds physics
import_config.import_inertia_tensor = True    # URDF has real inertias -> use them
import_config.distance_scale = 1.0            # URDF is already in meters
import_config.self_collision = False          # adjacent links overlap; avoid jitter
import_config.default_drive_strength = 1e7    # position-drive stiffness
import_config.default_position_drive_damping = 1e5

# --- run the import into the current stage ---------------------------------
result = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
)
# result is (status, prim_path) on 5.1; fall back to the stage default prim.
robot_prim_path = None
if isinstance(result, (list, tuple)) and len(result) >= 2 and isinstance(result[1], str):
    robot_prim_path = result[1]
if not robot_prim_path:
    dp = stage.GetDefaultPrim()
    robot_prim_path = dp.GetPath().pathString if dp else "/kr10_arm"
print(f"[import] robot imported at: {robot_prim_path}")

# A light so a subsequent GUI open isn't pitch black.
UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(500)

# --- save to a reusable USD ------------------------------------------------
os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)
omni.usd.get_context().save_as_stage(OUT_USD)
print(f"[import] saved: {OUT_USD}")

# --- Blackwell (RTX 5080) teardown workaround ------------------------------
# simulation_app.close() hangs on Kit shutdown on this GPU; the save above is
# already flushed, so hard-exit. (Same pattern as verify_isaac.py.)
import sys  # noqa: E402

sys.stdout.flush()
os._exit(0)
