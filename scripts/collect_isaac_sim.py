"""Isaac Sim RGB-D dataset collection for the KUKA pick-and-place project.

Adapted from RealDepth's scripts/collect_isaac_sim.py (the diverse-domain
collector) for this project's fixed deployment rig:
  - one fixed scene: isaac_version_project_anima.usd (table, KR10, tray, car)
  - the camera is NOT on a random walk — it sits at the authored RealSense
    D455 mount pose above the table (45 deg yaw, looking straight down),
    with D455 RGB geometry (90 x 65 deg FOV, 16:10)
  - instead of spawning random objects, the toy car teleports to a different
    spot on the table for every frame (uniform over the camera-visible,
    reachable tabletop, keeping the whole car in frame); its orientation stays
    fixed at the authored, tray-fitting yaw (the vision model only outputs
    position) unless --random_yaw is passed
  - mild indoor lighting randomization (can be disabled), no material swaps
  - physics is never stepped (render-only), so the arm stays parked upright
  - NEW: labels.json with ground-truth car pose per frame (world xyz + yaw +
    pixel coordinates + depth), for training the car-coordinate model

Output layout (unchanged, consumed by RealDepth's split_dataset.py):
  <out>/rgb/NNNNNN.png          8-bit BGR
  <out>/depth/NNNNNN.png        uint16, millimetres, 0 = invalid
  <out>/intrinsics.json         per-frame {stem: {fx fy cx cy width height}}
  <out>/intrinsics.txt          human-readable summary
  <out>/labels.json             per-frame ground-truth car pose (extra file)

Usage:
    ~/isaac/venv/bin/python scripts/collect_isaac_sim.py --headless \
        --num_frames 1000 --seed 0
    # quick self-check (captures a few frames, verifies GT projection):
    ~/isaac/venv/bin/python scripts/collect_isaac_sim.py --headless --test
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- Parse args BEFORE SimulationApp (it consumes some args) ----
parser = argparse.ArgumentParser(description="Collect KUKA-scene RGB-D dataset")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory (default: collected_dataset/kuka_<timestamp>)")
parser.add_argument("--num_frames", type=int, default=1000)
parser.add_argument("--width", type=int, default=640,
                    help="Image width; height follows the D455 16:10 aspect")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max_depth_mm", type=int, default=10000)
parser.add_argument("--light_randomize_interval", type=int, default=25,
                    help="Re-randomize lighting every N frames (0 = never)")
parser.add_argument("--settle_renders", type=int, default=6,
                    help="Render-only updates after each car move before capture")
parser.add_argument("--random_yaw", action="store_true",
                    help="Randomize the car's yaw per frame. Default is OFF: the "
                         "vision model only outputs position, so the car always "
                         "keeps its authored (tray-fitting) orientation")
parser.add_argument("--test", action="store_true",
                    help="Capture 8 frames and verify GT labels against the "
                         "rendered depth instead of collecting the full set")
args, unknown = parser.parse_known_args()

# ---- Launch Isaac Sim ----
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# Render with PATH TRACING, not the real-time (RTX) mode: the real-time
# denoiser keeps per-pixel radiance history that leaves semi-transparent
# "ghost" copies of the car at its previous positions for dozens of frames
# after a teleport (survives AA/DLSS being disabled). Path tracing
# accumulates samples only within a frame — no cross-frame history, no
# ghosts — and gives cleaner lighting for the dataset.
import carb.settings

_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 256)
_s.set("/rtx/pathtracing/totalSpp", 256)
_s.set("/rtx/pathtracing/clampSpp", 256)
# The PT denoiser is TEMPORAL (previous output bleeds into new frames);
# disabled, 256 spp keeps noise low.
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
# MOTION BLUR is the root cause of the "transparent extra cars": a teleport
# is a huge instantaneous velocity, so the renderer smears the car across the
# shutter — semi-transparent copies at old positions, in RGB only (depth AOVs
# are not motion blurred, which is why depth was always clean).
_s.set("/rtx/post/motionblur/enabled", False)
_s.set("/rtx/pathtracing/mbEnabled", False)
_s.set("/omni/replicator/captureMotionBlur", False)

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

import cv2

SCENE = "/home/rassul_pc/mono_depth_kuka/isaac_version_project_anima.usd"
CAR = "/World/car_object_arlan_usd"
REALSENSE = "/World/Realsense"
CAM_PATH = "/World/DatasetCamera"

# Tabletop and rig geometry (grounded by scripts/inspect_scene.py, see CLAUDE.md)
TABLE_X = (-0.427, 0.780)
TABLE_Y = (-0.387, 0.767)
TABLE_MARGIN = 0.12          # keep the whole car on the table
TRAY_X = (-0.34, -0.11)      # tray fixture bbox + margin: don't drop the car onto it
TRAY_Y = (0.50, 0.73)
ROBOT_X = (0.40, 0.90)       # robot column footprint + margin
ROBOT_Y = (0.07, 0.49)
CAR_HALF_DIAG_MARGIN = 0.16  # keep the whole car inside the camera view
                             # (sized for tilted views too, where the
                             # metre-to-pixel scale varies across the frame)


def log(msg):
    sys.stderr.write(f"[RealDepth] {msg}\n")
    sys.stderr.flush()


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Scene setup (render-only: physics is never stepped, the arm stays parked)
# ---------------------------------------------------------------------------
open_stage(SCENE)
stage = omni.usd.get_context().get_stage()

bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
car_prim = stage.GetPrimAtPath(CAR)
car_box = bbc.ComputeWorldBound(car_prim).ComputeAlignedRange()
car_min, car_max = np.array(car_box.GetMin()), np.array(car_box.GetMax())
CAR_CENTER0 = (car_min + car_max) / 2.0
CAR_HALF_HEIGHT = (car_max[2] - car_min[2]) / 2.0
TABLE_TOP_Z = car_min[2]

# Camera pose: replicate the RealSense asset's Camera_Pseudo_Depth prim (the
# user aligned it to the intended view). It lives inside the cloud-referenced
# RSD455 asset, so it only exists on the composed stage at runtime.
pseudo_prim = None
for p in stage.Traverse():
    if p.GetTypeName() == "Camera" and "pseudo" in p.GetName().lower():
        pseudo_prim = p
        break
if pseudo_prim is None:
    log("FATAL: no *pseudo* Camera prim found under the RealSense asset — "
        "did the cloud asset load (needs internet)?")
    simulation_app.close()
    import os
    os._exit(3)

ps_cache = UsdGeom.XformCache()
ps_m = ps_cache.GetLocalToWorldTransform(pseudo_prim)
CAM_POS = np.array(ps_m.ExtractTranslation())
R_USD_TARGET = np.array([[ps_m[r][c] for c in range(3)] for r in range(3)]).T
log(f"replicating {pseudo_prim.GetPath()}: pos {CAM_POS.round(4)}, "
    f"view dir {(-R_USD_TARGET[:, 2]).round(3)}")

# Copy its FOV too (D455 depth geometry), keeping square pixels.
ps_cam = UsdGeom.Camera(pseudo_prim)
PS_FOCAL = float(ps_cam.GetFocalLengthAttr().Get() or 24.0)
PS_HAP = float(ps_cam.GetHorizontalApertureAttr().Get() or 20.955)
PS_VAP = float(ps_cam.GetVerticalApertureAttr().Get() or (PS_HAP * 10 / 16))
log(f"pseudo-depth camera optics: focal {PS_FOCAL}, aperture {PS_HAP} x {PS_VAP}")

# Resolution follows the pseudo-depth camera's aspect ratio (square pixels).
WIDTH = int(round(args.width / 2.0)) * 2
HEIGHT = int(round((WIDTH * PS_VAP / PS_HAP) / 2.0)) * 2

world = World(stage_units_in_meters=1.0)
car = SingleXFormPrim(prim_path=CAR, name="car")


def rot_to_quat(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    # trace near -1 (180 deg rotation): use the largest-diagonal branch
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


# The camera pose must exist BEFORE world.reset() so the renderer picks it up
# (late set_world_pose writes were not reflected in renders). The constructor
# routes orientation through set_world_pose with camera_axes="world"
# (+X forward, +Z image-up), so convert the target USD-frame rotation:
# R_in = R_usd_target @ inv(W_U_TRANSFORM).
W_U_INV = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
camera = Camera(
    prim_path=CAM_PATH,
    position=CAM_POS,
    orientation=rot_to_quat(R_USD_TARGET @ W_U_INV),
    resolution=(WIDTH, HEIGHT),
)
world.reset()
car.initialize(world.physics_sim_view)
camera.initialize()
camera.add_distance_to_image_plane_to_frame()

# Copy the pseudo-depth camera's optics (FOV) onto the render camera.
cam_usd = UsdGeom.Camera(stage.GetPrimAtPath(CAM_PATH))
cam_usd.GetFocalLengthAttr().Set(PS_FOCAL)
cam_usd.GetHorizontalApertureAttr().Set(PS_HAP)
cam_usd.GetVerticalApertureAttr().Set(PS_HAP * HEIGHT / float(WIDTH))
# CRITICAL: a fresh USD Camera prim falls back to clippingRange (1, 1e6) —
# 1 METER near plane in this stage, which clips away the whole table scene
# (the camera is only ~0.5 m above it). D455 min range is ~0.4 m, but keep
# the near plane tiny so nothing in the rig ever clips.
cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

for _ in range(10):  # let the renderer warm up
    world.render()

# Intrinsics (read back, with FOV fallback like the original collector)
try:
    m = camera.get_intrinsics_matrix()
    FX, FY = float(m[0, 0]), float(m[1, 1])
    CX, CY = float(m[0, 2]), float(m[1, 2])
except Exception:
    FX = FY = WIDTH * PS_FOCAL / PS_HAP
    CX, CY = WIDTH / 2.0, HEIGHT / 2.0
INTR = {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "width": WIDTH, "height": HEIGHT}
log(f"intrinsics: fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f} {WIDTH}x{HEIGHT}")

# Camera extrinsics in the ROS optical convention (+Z forward, +X right,
# +Y down), derived from the SAME USD transform the renderer uses.
# Gf matrices are row-vector convention: row i = image of basis vector i,
# so the USD-frame axis columns are the transposed rotation block.
cam_m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(CAM_PATH))
R_usd = np.array([[cam_m[r][c] for c in range(3)] for r in range(3)]).T
CAM_T = np.array(cam_m.ExtractTranslation())
# ROS optical axes from USD camera axes: x=+X, y=-Y (down), z=-Z (forward)
CAM_R = R_usd @ np.diag([1.0, -1.0, -1.0])  # optical -> world


def project_to_pixel(p_world):
    pc = CAM_R.T @ (np.asarray(p_world, dtype=float) - CAM_T)
    u = FX * pc[0] / pc[2] + CX
    v = FY * pc[1] / pc[2] + CY
    return float(u), float(v), float(pc[2])


# ---------------------------------------------------------------------------
# Car placement (same bookkeeping as pick_place.py)
# ---------------------------------------------------------------------------
# Move the car by writing its USD xform attributes DIRECTLY: the Isaac
# set_world_pose path writes through Fabric, which the render pipeline's
# change-detection can miss — the accumulation then never resets and keeps a
# semi-transparent copy of the car at its old position. Plain USD writes fire
# notices that Hydra reliably picks up. (Re-fetch the stage: the pre-World
# handle can go stale, see CLAUDE.md.)
stage_live = omni.usd.get_context().get_stage()
_car_xf_ops = {op.GetOpName(): op
               for op in UsdGeom.Xformable(stage_live.GetPrimAtPath(CAR)).GetOrderedXformOps()}
_op_t = _car_xf_ops["xformOp:translate"]
_op_o = _car_xf_ops["xformOp:orient"]
_T_DOUBLE = _op_t.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
_O_DOUBLE = _op_o.GetPrecision() == UsdGeom.XformOp.PrecisionDouble


def _write_car_pose_usd(pos, quat_wxyz):
    w, x, y, z = (float(a) for a in quat_wxyz)
    _op_t.Set(Gf.Vec3d(*pos) if _T_DOUBLE else Gf.Vec3f(*[float(a) for a in pos]))
    _op_o.Set(Gf.Quatd(w, x, y, z) if _O_DOUBLE else Gf.Quatf(w, x, y, z))


car_pos0, car_quat0 = car.get_world_pose()
car_pos0 = np.asarray(car_pos0, dtype=float)
CAR_QUAT0 = np.asarray(car_quat0, dtype=float)
CAR_OFFSET0 = CAR_CENTER0 - car_pos0

rng = np.random.default_rng(args.seed)


def set_car_center(center_xy, yaw, z_bottom=TABLE_TOP_Z):
    center = np.array([center_xy[0], center_xy[1], z_bottom + CAR_HALF_HEIGHT])
    pos = center - rot_z(yaw) @ CAR_OFFSET0
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = CAR_QUAT0
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    _write_car_pose_usd(pos, q)
    return center


def sample_car_xy():
    """Uniform over the tabletop, whole car inside the camera view, away from
    the tray fixture and the robot column. Rejection sampling."""
    pad_u = CAR_HALF_DIAG_MARGIN * FX / (CAM_T[2] - TABLE_TOP_Z)
    pad_v = CAR_HALF_DIAG_MARGIN * FY / (CAM_T[2] - TABLE_TOP_Z)
    for _ in range(1000):
        x = rng.uniform(TABLE_X[0] + TABLE_MARGIN, TABLE_X[1] - TABLE_MARGIN)
        y = rng.uniform(TABLE_Y[0] + TABLE_MARGIN, TABLE_Y[1] - TABLE_MARGIN)
        # inside the camera view at table height, margin so the whole car fits
        u, v, _ = project_to_pixel([x, y, TABLE_TOP_Z])
        if not (pad_u <= u < WIDTH - pad_u and pad_v <= v < HEIGHT - pad_v):
            continue
        if TRAY_X[0] <= x <= TRAY_X[1] and TRAY_Y[0] <= y <= TRAY_Y[1]:
            continue
        if ROBOT_X[0] <= x <= ROBOT_X[1] and ROBOT_Y[0] <= y <= ROBOT_Y[1]:
            continue
        return np.array([x, y])
    raise RuntimeError("sample_car_xy: rejection sampling failed — check zones")


# ---------------------------------------------------------------------------
# Mild indoor lighting randomization (no material swaps, no object spawns)
# ---------------------------------------------------------------------------
def setup_lights():
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(500.0)
    dist = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    dist.CreateIntensityAttr(1000.0)
    UsdGeom.Xformable(dist.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))


def randomize_lighting():
    # ranges tuned low: these stack on the scene's own default light, and the
    # mostly-white table overexposes easily
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/DomeLight"))
    dome.GetIntensityAttr().Set(float(rng.uniform(100, 1200)))
    dome.GetColorAttr().Set(Gf.Vec3f(
        1.0, float(rng.uniform(0.85, 1.0)), float(rng.uniform(0.75, 1.0))))
    dist_prim = stage.GetPrimAtPath("/World/Lights/DistantLight")
    dist = UsdLux.DistantLight(dist_prim)
    dist.GetIntensityAttr().Set(float(rng.uniform(200, 2500)))
    xf = UsdGeom.Xformable(dist_prim)
    xf.ClearXformOpOrder()
    # keep the sun steep so the car's shadow stays compact (the background-
    # difference gate excludes a limited zone around the car)
    xf.AddRotateXYZOp().Set(Gf.Vec3f(
        float(rng.uniform(-75, -45)), float(rng.uniform(-180, 180)), 0.0))


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
def capture_frame():
    rgba = camera.get_rgba()
    depth = camera.get_depth()
    if rgba is None or depth is None or np.asarray(rgba).size == 0:
        return None, None
    if rgba.dtype == np.uint8:
        rgb_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        rgb_bgr = cv2.cvtColor(
            (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    depth_clean = np.where(np.isfinite(depth), depth, 0.0)
    depth_mm = np.clip(depth_clean * 1000.0, 0, args.max_depth_mm).astype(np.uint16)
    return rgb_bgr, depth_mm


# Hiding spot for the background capture: BELOW the ground plane, where the
# car is optically sealed off (under the tabletop it stays visible between
# the table legs and shades the floor).
HIDE_XY = (0.60, -0.30)
HIDE_Z_BOTTOM = -0.6

# ---------------------------------------------------------------------------
# Freshness beacon: the RGB annotator lags scene changes by an arbitrary,
# sometimes very large number of renders (depth is prompt) — the root cause
# of every phantom-car artifact. A small cube pinned in the image corner
# changes color each capture phase; an RGB readback is provably fresh exactly
# when the beacon shows the phase's color. The final saved capture is taken
# with the beacon HIDDEN and verified gone.
# ---------------------------------------------------------------------------
BEACON_PATH = "/World/FreshnessBeacon"
BEACON_PX = (24, 24)
_bdir_cam = np.array([(BEACON_PX[0] - CX) / FX, (BEACON_PX[1] - CY) / FY, 1.0])
_bdir_w = CAM_R @ _bdir_cam
BEACON_POS = CAM_T + _bdir_w / np.linalg.norm(_bdir_w) * 0.35
_beacon_cube = UsdGeom.Cube.Define(stage_live, BEACON_PATH)
_beacon_cube.CreateSizeAttr(0.02)
UsdGeom.Xformable(_beacon_cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*BEACON_POS))
_beacon_color_attr = _beacon_cube.CreateDisplayColorAttr()
_beacon_img = UsdGeom.Imageable(_beacon_cube.GetPrim())

BEACON_COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0)}


def beacon_show(name):
    _beacon_color_attr.Set([Gf.Vec3f(*BEACON_COLORS[name])])
    _beacon_img.MakeVisible()


def beacon_hide():
    _beacon_img.MakeInvisible()


def beacon_reads(img_bgr, name):
    """Does the beacon patch show the given color (dominant channel test)?"""
    u, v = BEACON_PX
    patch = img_bgr[max(0, v - 8):v + 8, max(0, u - 8):u + 8].astype(float)
    b, g, r = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
    if name == "red":
        return r > g + 25 and r > b + 25
    return g > r + 25 and g > b + 25


def capture_validated(center_xy, yaw):
    """Ghost-proof capture: for each frame, first hide the car UNDER the
    table and capture a car-free background, then place the car and capture
    again. Accept only when:
      - depth: the window at the labeled pixel contains a surface at or
        closer than the car's expected optical-axis depth (tilt-aware), and
      - rgb: pixels that differ from the background are confined to the car's
        neighbourhood — any ghost / stale copy / frozen accumulation artifact
        ANYWHERE else in the image is a difference outside that region and
        forces a retry with more renders.
    This is renderer-agnostic: it detects bad frames by construction instead
    of trusting any particular accumulation setting."""
    car_img = UsdGeom.Imageable(stage_live.GetPrimAtPath(CAR))

    def step_render(subframes):
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0,
                              pause_timeline=True)

    def capture_when(pred, subframes, tries=8):
        """Step + capture until pred(rgb) says the readback is fresh."""
        for _ in range(tries):
            step_render(subframes)
            rgb, depth = capture_frame()
            if rgb is not None and pred(rgb):
                return rgb, depth
        return None, None

    for attempt in range(4):
        sub = 8 + attempt * 8
        # phase 1: car hidden + beacon RED -> a capture showing RED is a
        # provably fresh car-free background
        car_img.MakeInvisible()
        set_car_center(HIDE_XY, 0.0, z_bottom=HIDE_Z_BOTTOM)
        beacon_show("red")
        bg_bgr, _ = capture_when(lambda im: beacon_reads(im, "red"), sub)
        if bg_bgr is None:
            log(f"  gate: no fresh RED background (attempt {attempt + 1})")
            continue

        # phase 2: car revealed at target + beacon GREEN -> fresh reveal
        center = set_car_center(center_xy, yaw)
        u, v, cam_depth = project_to_pixel(center)
        iu, iv = int(round(u)), int(round(v))
        car_img.MakeVisible()
        beacon_show("green")
        probe, _ = capture_when(lambda im: beacon_reads(im, "green"), sub)
        if probe is None:
            log(f"  gate: no fresh GREEN reveal (attempt {attempt + 1})")
            continue

        # phase 3: beacon hidden -> a capture showing NEITHER color is the
        # fresh, beacon-free image that gets saved
        beacon_hide()
        rgb_bgr, depth_mm = capture_when(
            lambda im: not beacon_reads(im, "green") and not beacon_reads(im, "red"),
            sub)
        if rgb_bgr is None:
            log(f"  gate: beacon did not disappear (attempt {attempt + 1})")
            continue

        win = depth_mm[max(0, iv - 30):iv + 31,
                       max(0, iu - 30):iu + 31].astype(float)
        valid = win[win > 0] / 1000.0
        depth_ok = (valid.size and 0.1 < float(valid.min()) < cam_depth + 0.02)

        g_now = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        g_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        changed = np.abs(g_now - g_bg) > 10
        ys, xs = np.mgrid[0:changed.shape[0], 0:changed.shape[1]]
        car_zone = (xs - u) ** 2 + (ys - v) ** 2 < 130 ** 2  # car + shadow + reflection
        beacon_zone = ((xs - BEACON_PX[0]) ** 2 + (ys - BEACON_PX[1]) ** 2 < 45 ** 2)
        outside = float(changed[~car_zone & ~beacon_zone].mean())
        core = changed[max(0, iv - 35):iv + 36, max(0, iu - 35):iu + 36]
        car_visible = float(core.mean()) > 0.06
        if depth_ok and car_visible and outside < 0.01 and float(rgb_bgr.mean()) >= 5.0:
            return rgb_bgr, depth_mm, center, u, v, cam_depth, float(valid.min())
        log(f"  gate: depth_ok={depth_ok} car_visible={car_visible} "
            f"outside_frac={outside:.4f} (attempt {attempt + 1})")
    return None


def main():
    project_root = Path(__file__).resolve().parent.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "collected_dataset" / f"kuka_{timestamp}"
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {output_dir}")

    setup_lights()
    if args.light_randomize_interval:
        randomize_lighting()

    num_frames = 8 if args.test else args.num_frames
    intrinsics_meta = {}
    labels = {}


    for i in range(num_frames):
        if not simulation_app.is_running():
            break
        if (args.light_randomize_interval and i > 0
                and i % args.light_randomize_interval == 0):
            randomize_lighting()

        result = None
        for pose_attempt in range(3):
            xy = sample_car_xy()
            yaw = float(rng.uniform(-np.pi, np.pi)) if args.random_yaw else 0.0
            result = capture_validated(xy, yaw)
            if result is not None:
                break
            log(f"frame {i}: capture gate failed, resampling pose "
                f"(attempt {pose_attempt + 1})")
        if result is None:
            raise RuntimeError(
                f"frame {i}: could not produce a clean validated frame — "
                "renderer/scene issue, aborting instead of saving bad labels")
        rgb_bgr, depth_mm, center, u, v, cam_depth, win_min = result

        stem = f"{i:06d}"
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb_bgr)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
        intrinsics_meta[stem] = INTR

        if i < 6:
            # human-viewable previews: RGB | colorized depth side by side
            # (the raw depth PNGs are uint16 millimetres and look black in
            # normal image viewers)
            preview_dir = output_dir / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            dv = depth_mm.astype(float)
            valid = dv > 0
            norm = np.zeros(dv.shape, dtype=np.uint8)
            if valid.any() and dv[valid].max() > dv[valid].min():
                lo, hi = dv[valid].min(), dv[valid].max()
                norm[valid] = (255 * (dv[valid] - lo) / (hi - lo)).astype(np.uint8)
            dcol = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(preview_dir / f"{stem}.png"),
                        np.concatenate([rgb_bgr, dcol], axis=1))
        labels[stem] = {
            "car_center_world": [float(center[0]), float(center[1]), float(center[2])],
            "car_yaw_world": yaw,
            "car_center_px": [u, v],
            "car_center_cam_depth_m": cam_depth,
            "table_top_z": float(TABLE_TOP_Z),
        }

        if args.test:
            log(f"  test frame {stem}: car world ({center[0]:+.3f},{center[1]:+.3f}) "
                f"px ({int(round(u))},{int(round(v))}) car min depth {win_min:.3f} "
                f"(validated)")
        elif (i + 1) % 100 == 0:
            log(f"  {i + 1}/{num_frames} frames collected")

    # ---- Metadata (same files/format as the original collector) ----
    with open(output_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_meta, f)
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=1)
    with open(output_dir / "intrinsics.txt", "w") as f:
        f.write("Color Camera Intrinsics:\n")
        f.write(f"  Width: {WIDTH}\n  Height: {HEIGHT}\n")
        f.write(f"  fx: {FX:.4f}\n  fy: {FY:.4f}\n  cx: {CX:.4f}\n  cy: {CY:.4f}\n")
        f.write("\nDepth Scale: 0.001\n")
        f.write("\nCamera extrinsics (ROS optical frame, world coords):\n")
        f.write(f"  position: {CAM_T.tolist()}\n")
        f.write(f"  rotation matrix (optical->world):\n{CAM_R}\n")
        f.write("\nSource: Isaac Sim (KUKA table scene, fixed D455 overhead camera)\n")

    log(f"\nDone! {len(intrinsics_meta)} frames saved to {output_dir}")
    if args.test:
        # every saved frame passed capture_validated (car rendered at its
        # labeled pixel); a systemic failure would have aborted with an error
        log("TEST RESULT: PASS (all frames validated: car rendered at its GT pixel)")
    else:
        log("Run RealDepth's split_dataset.py next to create train/val/test splits.")

    # simulation_app.close() can hang/terminate mid-teardown (see CLAUDE.md);
    # everything is written, so flush and hard-exit.
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n\nFATAL ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        import os
        os._exit(2)
