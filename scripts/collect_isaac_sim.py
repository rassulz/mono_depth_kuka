"""
Isaac Sim Synthetic Dataset Collection for RealDepth — Diverse Edition

Loads multiple USD scenes, spawns random objects (primitives + Nucleus mesh assets),
randomizes lighting and materials, and moves a camera on random trajectories
to collect RGB + depth frame pairs with maximum domain diversity.

Usage (run from Isaac Sim's python):
    /home/nurtay/isaacsim/python.sh scripts/collect_isaac_sim.py \
        --usd_paths warehouse.usd digtwin.usd hospital.usd \
        --num_frames 5000 \
        --output_dir collected_dataset/isaac_diverse

    # Single scene (backward-compatible):
    /home/nurtay/isaacsim/python.sh scripts/collect_isaac_sim.py \
        --usd_path warehouse.usd --num_frames 3000

    # No scene (procedural fallback):
    /home/nurtay/isaacsim/python.sh scripts/collect_isaac_sim.py --headless --num_frames 3000
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- Parse args BEFORE SimulationApp (it consumes some args) ----
parser = argparse.ArgumentParser(description="Collect diverse RGB-D dataset in Isaac Sim")
parser.add_argument("--usd_path", type=str, default="",
                    help="Path to a single USD scene file (backward compat).")
parser.add_argument("--usd_paths", type=str, nargs="*", default=[],
                    help="Paths to multiple USD scene files. The collector "
                         "cycles through them for scene-level diversity.")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory (default: collected_dataset/<timestamp>)")
parser.add_argument("--num_frames", type=int, default=3000,
                    help="Total number of frames to collect")
parser.add_argument("--width", type=int, default=640,
                    help="Base image width (used as resolution budget when "
                         "randomizing aspect ratio)")
parser.add_argument("--height", type=int, default=480,
                    help="Fallback image height (used when aspect randomization is off)")
parser.add_argument("--fov_range", type=float, nargs=2, default=[50.0, 100.0],
                    help="Min / max horizontal FOV in degrees, sampled per scene")
parser.add_argument("--aspect_ratios", type=float, nargs="+",
                    default=[1.3333, 1.7778, 1.0, 0.75],
                    help="Set of width/height aspect ratios sampled per scene")
parser.add_argument("--scene_gap", type=int, default=50,
                    help="Frame-number gap inserted between scenes so "
                         "SequenceDataset/split_dataset treat each scene as a "
                         "separate sequence (one camera per sequence)")
parser.add_argument("--num_objects", type=int, default=40,
                    help="Number of random objects to spawn per scene")
parser.add_argument("--fps", type=int, default=30,
                    help="Simulation step rate")
parser.add_argument("--headless", action="store_true",
                    help="Run in headless mode (no GUI)")
parser.add_argument("--max_depth_mm", type=int, default=10000,
                    help="Max depth clamp in mm (default 10000 = 10 m)")
parser.add_argument("--camera_height_range", type=float, nargs=2,
                    default=[0.5, 2.5],
                    help="Min / max camera height in metres")
parser.add_argument("--arena_size", type=float, default=15.0,
                    help="Half-size of the arena the camera can roam (metres)")
parser.add_argument("--frames_per_scene", type=int, default=0,
                    help="Frames to collect per scene before switching. "
                         "0 = evenly split num_frames across scenes.")
parser.add_argument("--light_randomize_interval", type=int, default=10,
                    help="Re-randomize lighting every N frames")
parser.add_argument("--object_randomize_interval", type=int, default=50,
                    help="Re-randomize object poses every N frames")
parser.add_argument("--use_nucleus_assets", action="store_true",
                    help="Try to load mesh assets from Isaac Sim Nucleus")
args, unknown = parser.parse_known_args()

# Merge --usd_path into --usd_paths for unified handling
if args.usd_path and args.usd_path not in args.usd_paths:
    args.usd_paths.insert(0, args.usd_path)

# ---- Launch Isaac Sim ----
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now safe to import Omniverse / Isaac Sim modules
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

import cv2


def log(msg):
    sys.stderr.write(f"[RealDepth] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Nucleus mesh asset helpers
# ---------------------------------------------------------------------------

# Common Isaac Sim Nucleus asset paths (available in most installations)
NUCLEUS_ASSET_PATHS = [
    "/Isaac/Props/Blocks/basic_block.usd",
    "/Isaac/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
    "/Isaac/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
    "/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
    "/Isaac/Props/YCB/Axis_Aligned/024_bowl.usd",
    "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
    "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
    "/Isaac/Props/YCB/Axis_Aligned/037_scissors.usd",
]

# Additional Nucleus paths to search for available assets
NUCLEUS_SEARCH_DIRS = [
    "/Isaac/Props/",
    "/Isaac/Environments/Simple_Warehouse/Props/",
    "/Isaac/Props/YCB/Axis_Aligned/",
]


def _try_get_nucleus_server():
    """Try to find the default Nucleus server URL."""
    try:
        import omni.client
        # Try common Nucleus server URLs
        for server in ["omniverse://localhost", "omniverse://nucleus"]:
            result, _ = omni.client.stat(server)
            if result == omni.client.Result.OK:
                return server
    except Exception:
        pass
    return None


def _discover_nucleus_assets():
    """Discover available assets from Nucleus server."""
    available = []
    try:
        import omni.client
        nucleus_server = _try_get_nucleus_server()
        if nucleus_server is None:
            log("No Nucleus server found — skipping Nucleus asset discovery")
            return available

        for search_dir in NUCLEUS_SEARCH_DIRS:
            full_path = nucleus_server + search_dir
            result, entries = omni.client.list(full_path)
            if result == omni.client.Result.OK:
                for entry in entries:
                    if entry.relative_path.endswith(".usd") or entry.relative_path.endswith(".usda"):
                        available.append(search_dir + entry.relative_path)
        log(f"Discovered {len(available)} Nucleus assets")
    except Exception as e:
        log(f"Nucleus asset discovery failed: {e}")
    return available


def _add_nucleus_reference(stage, prim_path, asset_path, nucleus_server=None):
    """Add a USD reference from Nucleus to the stage."""
    try:
        prim = stage.DefinePrim(prim_path)
        if nucleus_server:
            prim.GetReferences().AddReference(nucleus_server + asset_path)
        else:
            prim.GetReferences().AddReference(asset_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Procedural warehouse (fallback when no USD is provided)
# ---------------------------------------------------------------------------
def _build_procedural_warehouse(stage, arena_size):
    def _add_box(path, center, size, color=(0.5, 0.5, 0.5)):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.AddTranslateOp().Set(Gf.Vec3f(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*(s / 2.0 for s in size)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))

    S = arena_size
    wall_h, wall_t = 4.0, 0.3
    _add_box("/World/Env/WallN", (0, S, wall_h / 2), (2 * S, wall_t, wall_h), (0.6, 0.6, 0.55))
    _add_box("/World/Env/WallS", (0, -S, wall_h / 2), (2 * S, wall_t, wall_h), (0.6, 0.6, 0.55))
    _add_box("/World/Env/WallE", (S, 0, wall_h / 2), (wall_t, 2 * S, wall_h), (0.6, 0.6, 0.55))
    _add_box("/World/Env/WallW", (-S, 0, wall_h / 2), (wall_t, 2 * S, wall_h), (0.6, 0.6, 0.55))

    rng = np.random.default_rng(42)
    for i in range(6):
        x = rng.uniform(-S * 0.7, S * 0.7)
        y = rng.uniform(-S * 0.7, S * 0.7)
        h = rng.uniform(1.5, 3.5)
        w = rng.uniform(2.0, 5.0)
        d = rng.uniform(0.5, 1.0)
        color = tuple(rng.uniform(0.3, 0.8, size=3))
        _add_box(f"/World/Env/Shelf_{i}", (x, y, h / 2), (w, d, h), color)


# ---------------------------------------------------------------------------
# Lighting randomization
# ---------------------------------------------------------------------------
_light_prims = []


def _setup_lights(stage):
    """Create a set of lights that we'll randomize."""
    global _light_prims
    _light_prims = []

    # Dome light for ambient / HDR environment lighting
    dome_path = "/World/Lights/DomeLight"
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    _light_prims.append(("dome", dome_path))

    # Distant light (sun-like)
    dist_path = "/World/Lights/DistantLight"
    dist = UsdLux.DistantLight.Define(stage, dist_path)
    dist.CreateIntensityAttr(3000.0)
    dist.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(dist_path))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
    _light_prims.append(("distant", dist_path))

    # Several point/spot lights at random positions
    for i in range(4):
        ppath = f"/World/Lights/PointLight_{i}"
        pl = UsdLux.SphereLight.Define(stage, ppath)
        pl.CreateIntensityAttr(5000.0)
        pl.CreateRadiusAttr(0.1)
        pl.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ppath))
        xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 3.0))
        _light_prims.append(("point", ppath))

    log(f"Created {len(_light_prims)} lights for randomization")


def _randomize_lighting(stage, rng):
    """Randomize all light properties."""
    for ltype, lpath in _light_prims:
        prim = stage.GetPrimAtPath(lpath)
        if not prim.IsValid():
            continue

        if ltype == "dome":
            light = UsdLux.DomeLight(prim)
            # Randomize intensity (wide range for different ambient levels)
            light.GetIntensityAttr().Set(rng.uniform(200, 5000))
            # Randomize color temperature (warm to cool)
            r = rng.uniform(0.7, 1.0)
            g = rng.uniform(0.7, 1.0)
            b = rng.uniform(0.7, 1.0)
            light.GetColorAttr().Set(Gf.Vec3f(float(r), float(g), float(b)))
            # Randomize dome rotation for different ambient direction
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddRotateYOp().Set(float(rng.uniform(0, 360)))

        elif ltype == "distant":
            light = UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(rng.uniform(500, 8000))
            r = rng.uniform(0.8, 1.0)
            g = rng.uniform(0.75, 1.0)
            b = rng.uniform(0.7, 1.0)
            light.GetColorAttr().Set(Gf.Vec3f(float(r), float(g), float(b)))
            # Randomize sun angle
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(
                float(rng.uniform(-80, -10)),
                float(rng.uniform(-180, 180)),
                0.0,
            ))

        elif ltype == "point":
            light = UsdLux.SphereLight(prim)
            light.GetIntensityAttr().Set(rng.uniform(500, 15000))
            light.GetRadiusAttr().Set(float(rng.uniform(0.05, 0.5)))
            r = rng.uniform(0.6, 1.0)
            g = rng.uniform(0.6, 1.0)
            b = rng.uniform(0.6, 1.0)
            light.GetColorAttr().Set(Gf.Vec3f(float(r), float(g), float(b)))
            # Randomize position
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3f(
                float(rng.uniform(-args.arena_size * 0.8, args.arena_size * 0.8)),
                float(rng.uniform(-args.arena_size * 0.8, args.arena_size * 0.8)),
                float(rng.uniform(1.5, 5.0)),
            ))

            # Randomly enable/disable some lights for variety
            if rng.random() < 0.3:
                light.GetIntensityAttr().Set(0.0)


# ---------------------------------------------------------------------------
# Material randomization
# ---------------------------------------------------------------------------
_material_cache = {}


def _create_random_material(stage, mat_path, rng):
    """Create a UsdPreviewSurface material with random properties."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    # Random diffuse color
    r, g, b = rng.uniform(0.05, 0.95, size=3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(r), float(g), float(b))
    )

    # Random roughness
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(rng.uniform(0.1, 1.0))
    )

    # Random metallic
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        float(rng.choice([0.0, 0.0, 0.0, 0.5, 0.9]))  # mostly non-metallic
    )

    # Connect shader to material
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    return mat


def _apply_random_material(stage, prim_path, rng, mat_index):
    """Apply a random material to a prim."""
    mat_path = f"/World/Materials/RandomMat_{mat_index:04d}"
    if mat_path not in _material_cache:
        _material_cache[mat_path] = _create_random_material(stage, mat_path, rng)
    mat = _material_cache[mat_path]
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(mat)


# ---------------------------------------------------------------------------
# Random object spawning (primitives + Nucleus meshes)
# ---------------------------------------------------------------------------
def _spawn_random_objects(stage, num_objects, arena_size, rng, nucleus_assets=None):
    """Spawn diverse objects at random positions with random materials."""
    shape_types = ["Cube", "Cylinder", "Sphere", "Cone", "Capsule"]
    object_paths = []

    # Decide how many will be Nucleus mesh assets vs primitives
    num_nucleus = 0
    if nucleus_assets:
        num_nucleus = min(num_objects // 3, len(nucleus_assets))

    nucleus_server = None
    if num_nucleus > 0:
        nucleus_server = _try_get_nucleus_server()

    for i in range(num_objects):
        path = f"/World/Objects/Obj_{i:03d}"

        x = float(rng.uniform(-arena_size * 0.8, arena_size * 0.8))
        y = float(rng.uniform(-arena_size * 0.8, arena_size * 0.8))
        scale = float(rng.uniform(0.1, 0.8))
        z = scale

        use_nucleus = (i < num_nucleus) and nucleus_assets
        if use_nucleus:
            asset_path = rng.choice(nucleus_assets)
            success = _add_nucleus_reference(stage, path, asset_path, nucleus_server)
            if not success:
                use_nucleus = False

        if not use_nucleus:
            shape = rng.choice(shape_types)
            if shape == "Cube":
                prim = UsdGeom.Cube.Define(stage, path)
            elif shape == "Cylinder":
                prim = UsdGeom.Cylinder.Define(stage, path)
            elif shape == "Sphere":
                prim = UsdGeom.Sphere.Define(stage, path)
            elif shape == "Cone":
                prim = UsdGeom.Cone.Define(stage, path)
            else:  # Capsule
                prim = UsdGeom.Capsule.Define(stage, path)

        # Transform
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))

        # Anisotropic random scale
        sx = scale * float(rng.uniform(0.5, 2.0))
        sy = scale * float(rng.uniform(0.5, 2.0))
        sz = scale * float(rng.uniform(0.5, 2.0))
        xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

        # Random rotation
        xform.AddRotateXYZOp().Set(Gf.Vec3f(
            float(rng.uniform(0, 360)),
            float(rng.uniform(0, 360)),
            float(rng.uniform(0, 360)),
        ))

        # Apply random material
        _apply_random_material(stage, path, rng, i)

        # Physics
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))

        object_paths.append(path)

    return object_paths


def _rerandomize_objects(stage, object_paths, arena_size, rng):
    """Re-randomize positions, scales, rotations, and materials of existing objects."""
    for idx, path in enumerate(object_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        x = float(rng.uniform(-arena_size * 0.8, arena_size * 0.8))
        y = float(rng.uniform(-arena_size * 0.8, arena_size * 0.8))
        scale = float(rng.uniform(0.1, 0.8))
        z = scale

        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        sx = scale * float(rng.uniform(0.5, 2.0))
        sy = scale * float(rng.uniform(0.5, 2.0))
        sz = scale * float(rng.uniform(0.5, 2.0))
        xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(
            float(rng.uniform(0, 360)),
            float(rng.uniform(0, 360)),
            float(rng.uniform(0, 360)),
        ))

        # Re-randomize material with some probability
        if rng.random() < 0.5:
            new_mat_idx = int(rng.integers(0, 200))
            _apply_random_material(stage, path, rng, new_mat_idx)


# ---------------------------------------------------------------------------
# Camera random walk
# ---------------------------------------------------------------------------
class _CameraRandomWalk:
    def __init__(self, arena_size=15.0, height_range=(0.5, 2.5), dt=1 / 30,
                 max_speed=1.5, direction_change_rate=0.3):
        self.arena = arena_size * 0.75
        self.h_min, self.h_max = height_range
        self.dt = dt
        self.max_speed = max_speed
        self.dir_rate = direction_change_rate
        self.rng = np.random.default_rng()

        self.pos = np.array([0.0, 0.0, np.mean(height_range)])
        self.vel = self.rng.uniform(-0.5, 0.5, size=3)
        self.yaw = 0.0
        self.pitch = -0.1
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0

    def reset(self):
        """Reset to a new random starting position and orientation."""
        self.pos = np.array([
            self.rng.uniform(-self.arena * 0.5, self.arena * 0.5),
            self.rng.uniform(-self.arena * 0.5, self.arena * 0.5),
            self.rng.uniform(self.h_min, self.h_max),
        ])
        self.vel = self.rng.uniform(-0.5, 0.5, size=3)
        self.yaw = self.rng.uniform(0, 2 * math.pi)
        self.pitch = self.rng.uniform(-0.4, 0.1)
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0

    def step(self):
        rng = self.rng
        dt = self.dt

        accel = rng.normal(0, 1.5, size=3)
        self.vel += accel * dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed

        self.pos += self.vel * dt
        self.pos[0] = np.clip(self.pos[0], -self.arena, self.arena)
        self.pos[1] = np.clip(self.pos[1], -self.arena, self.arena)
        self.pos[2] = np.clip(self.pos[2], self.h_min, self.h_max)

        for i in range(2):
            if abs(self.pos[i]) >= self.arena * 0.95:
                self.vel[i] *= -0.5
        if self.pos[2] <= self.h_min + 0.1 or self.pos[2] >= self.h_max - 0.1:
            self.vel[2] *= -0.5

        self.yaw_vel += rng.normal(0, self.dir_rate) * dt
        self.yaw_vel *= 0.95
        self.yaw += self.yaw_vel

        self.pitch_vel += rng.normal(0, self.dir_rate * 0.5) * dt
        self.pitch_vel *= 0.95
        self.pitch = np.clip(self.pitch + self.pitch_vel, -0.6, 0.3)

        euler_deg = np.array([0.0, math.degrees(self.pitch), math.degrees(self.yaw)])
        quat = rot_utils.euler_angles_to_quats(euler_deg, degrees=True)

        return self.pos.copy(), quat


# ---------------------------------------------------------------------------
# Save intrinsics
# ---------------------------------------------------------------------------
def _save_intrinsics(output_dir, width, height, camera):
    try:
        intrinsic_matrix = camera.get_intrinsics_matrix()
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
    except Exception:
        fov_h = 69.4
        fx = fy = width / (2.0 * math.tan(math.radians(fov_h / 2)))
        cx = width / 2.0
        cy = height / 2.0

    path = output_dir / "intrinsics.txt"
    with open(path, "w") as f:
        f.write("Color Camera Intrinsics:\n")
        f.write(f"  Width: {width}\n")
        f.write(f"  Height: {height}\n")
        f.write(f"  fx: {fx:.4f}\n")
        f.write(f"  fy: {fy:.4f}\n")
        f.write(f"  cx: {cx:.4f}\n")
        f.write(f"  cy: {cy:.4f}\n")
        f.write(f"\nDepth Scale: 0.001\n")
        f.write(f"\nSource: Isaac Sim (synthetic, diverse domain randomization)\n")
    log(f"Saved intrinsics to {path}")


def _resolution_for_aspect(aspect, base_width):
    """Pick an (even) (width, height) for a target width/height aspect ratio,
    using base_width as the resolution budget."""
    width = int(round(base_width / 2.0)) * 2
    height = int(round((base_width / aspect) / 2.0)) * 2
    return max(width, 2), max(height, 2)


def _set_camera_fov(stage, prim_path, fov_h_deg, width, height):
    """Set the USD camera's aperture/focal so its horizontal FOV matches
    fov_h_deg, with square pixels for the given resolution.

    FOV depends only on aperture/focal ratio, so absolute units are irrelevant.
    """
    cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    focal = 24.0
    h_aperture = 2.0 * focal * math.tan(math.radians(fov_h_deg) / 2.0)
    v_aperture = h_aperture * (height / float(width))  # square pixels
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(h_aperture)
    cam.GetVerticalApertureAttr().Set(v_aperture)


def _scene_intrinsics(camera, width, height, fov_h_deg):
    """Return {fx, fy, cx, cy, width, height} for the current camera, reading
    back the actual intrinsics with a FOV-based fallback."""
    try:
        m = camera.get_intrinsics_matrix()
        fx, fy = float(m[0, 0]), float(m[1, 1])
        cx, cy = float(m[0, 2]), float(m[1, 2])
    except Exception:
        fx = fy = width / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
        cx, cy = width / 2.0, height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": int(width), "height": int(height)}


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------
def _load_scene(usd_path, arena_size, rng, num_objects, nucleus_assets):
    """Load a USD scene (or build procedural one), spawn objects and lights."""
    import omni.usd as omni_usd
    import omni.kit.app as omni_app

    stage = omni_usd.get_context().get_stage()

    if usd_path:
        resolved = usd_path
        if not resolved.startswith(("http://", "https://", "omniverse://", "/")):
            resolved = str(Path(resolved).resolve())
        log(f"Loading USD scene: {resolved}")
        omni_usd.get_context().open_stage(resolved)
        omni_app.get_app().update()
        omni_app.get_app().update()
        stage = omni_usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
    else:
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        _build_procedural_warehouse(stage, arena_size)

    # Setup lights
    _setup_lights(stage)

    # Spawn objects
    object_paths = _spawn_random_objects(stage, num_objects, arena_size, rng, nucleus_assets)

    return world, stage, object_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    project_root = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng()

    # ---- Output dirs ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "collected_dataset" / f"isaac_{timestamp}"

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {output_dir}")

    # ---- Discover Nucleus assets ----
    nucleus_assets = None
    if args.use_nucleus_assets:
        nucleus_assets = _discover_nucleus_assets()
        if not nucleus_assets:
            # Fall back to the known asset paths list
            nucleus_assets = NUCLEUS_ASSET_PATHS
            log(f"Using {len(nucleus_assets)} predefined Nucleus asset paths")

    # ---- Scene list ----
    scenes = args.usd_paths if args.usd_paths else [""]
    num_scenes = len(scenes)

    if args.frames_per_scene > 0:
        frames_per_scene = args.frames_per_scene
    else:
        frames_per_scene = max(1, args.num_frames // num_scenes)

    log(f"Scenes: {scenes if scenes[0] else ['<procedural>']}")
    log(f"Frames per scene: {frames_per_scene}")

    # ---- Camera walk state ----
    cam_state = _CameraRandomWalk(
        arena_size=args.arena_size,
        height_range=args.camera_height_range,
        dt=1.0 / args.fps,
    )

    # ---- Collection ----
    frame_count = 0      # actual frames saved (drives the num_frames budget)
    name_index = 0       # number embedded in the filename (gets a gap per scene)
    scene_idx = 0
    intrinsics_meta = {}  # filename stem -> per-scene intrinsics dict

    while frame_count < args.num_frames and simulation_app.is_running():
        usd_path = scenes[scene_idx % num_scenes]
        scene_label = usd_path if usd_path else "<procedural>"
        log(f"\n=== Scene {scene_idx + 1}: {scene_label} ===")

        # Load scene
        world, stage, object_paths = _load_scene(
            usd_path, args.arena_size, rng, args.num_objects, nucleus_assets
        )

        # --- Per-scene camera randomization: FOV + aspect ratio ---
        scene_fov = float(rng.uniform(args.fov_range[0], args.fov_range[1]))
        scene_aspect = float(rng.choice(args.aspect_ratios))
        scene_w, scene_h = _resolution_for_aspect(scene_aspect, args.width)
        log(f"Camera: FOV={scene_fov:.1f} deg, aspect={scene_aspect:.3f}, "
            f"resolution={scene_w}x{scene_h}")

        # Create camera at the sampled resolution
        camera = Camera(
            prim_path="/World/DepthCamera",
            position=np.array([0.0, 0.0, 1.5]),
            frequency=args.fps,
            resolution=(scene_w, scene_h),
        )
        world.reset()
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()

        # Apply the sampled FOV to the camera prim
        _set_camera_fov(stage, "/World/DepthCamera", scene_fov, scene_w, scene_h)

        # Let physics settle
        log("Letting physics settle...")
        for _ in range(60):
            world.step(render=True)

        # Read back the actual intrinsics for this scene's camera
        scene_intr = _scene_intrinsics(camera, scene_w, scene_h, scene_fov)

        # Reset camera walk for new scene
        cam_state.reset()

        # Randomize lighting initially
        _randomize_lighting(stage, rng)

        # Collect frames for this scene
        scene_frames = 0
        remaining = args.num_frames - frame_count

        target_frames = min(frames_per_scene, remaining)

        while scene_frames < target_frames and simulation_app.is_running():
            # Periodic randomizations
            if scene_frames > 0 and scene_frames % args.light_randomize_interval == 0:
                _randomize_lighting(stage, rng)

            if scene_frames > 0 and scene_frames % args.object_randomize_interval == 0:
                _rerandomize_objects(stage, object_paths, args.arena_size, rng)
                # Let physics settle briefly after rerandomization
                for _ in range(10):
                    world.step(render=True)

            # Update camera pose
            position, orientation = cam_state.step()
            camera.set_world_pose(position=position, orientation=orientation)

            world.step(render=True)

            # Get sensor data
            rgba = camera.get_rgba()
            depth = camera.get_depth()

            if rgba is None or depth is None:
                continue

            # RGB
            if rgba.dtype == np.uint8:
                rgb_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                rgb_bgr = cv2.cvtColor(
                    (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )

            # Depth → uint16 mm
            depth_clean = np.where(np.isfinite(depth), depth, 0.0)
            depth_mm = np.clip(depth_clean * 1000.0, 0, args.max_depth_mm).astype(np.uint16)

            stem = f"{name_index:06d}"
            fname = f"{stem}.png"
            cv2.imwrite(str(rgb_dir / fname), rgb_bgr)
            cv2.imwrite(str(depth_dir / fname), depth_mm)
            intrinsics_meta[stem] = scene_intr

            frame_count += 1
            scene_frames += 1
            name_index += 1

            if frame_count % 100 == 0:
                log(f"  {frame_count}/{args.num_frames} frames collected "
                    f"(scene: {scene_label}, scene_frame: {scene_frames}/{target_frames})")

        # Numbering gap between scenes so each scene is a separate sequence.
        name_index += args.scene_gap
        scene_idx += 1

    # ---- Save intrinsics ----
    # Per-frame intrinsics (machine-readable; used by split_dataset/SequenceDataset)
    with open(output_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_meta, f)
    log(f"Saved per-frame intrinsics for {len(intrinsics_meta)} frames to "
        f"{output_dir / 'intrinsics.json'}")
    # Human-readable summary for the last scene's camera
    _save_intrinsics(output_dir, scene_w, scene_h, camera)

    log(f"\nDone! {frame_count} frames saved to {output_dir}")
    log(f"Scenes used: {num_scenes}")
    log("Run split_dataset.py next to create train/val/test splits.")

    simulation_app.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n\nFATAL ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        simulation_app.close()