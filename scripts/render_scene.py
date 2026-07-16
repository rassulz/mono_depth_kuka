#!/usr/bin/env python3
"""Render the assembled Anima scene to PNG images, headless, Isaac Sim 5.1.

Opens isaac_scene_anima.usd and captures two views with the RTX renderer:
  * renders/overview/rgb_0000.png  -- 3/4 perspective of the whole cell
  * renders/topdown/rgb_0000.png   -- straight-down view from the D455 position

Run:  ~/isaac/venv/bin/python scripts/render_scene.py

NOTE: the KUKA arm will NOT appear -- its URDF meshes are missing (see
check_meshes.py). You will see the stand/table, the toy car, and the ground.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
USD = os.path.join(PROJECT, "isaac_scene_anima.usd")
OUT_DIR = os.path.join(PROJECT, "renders")

# --- launch Kit (rendering enabled) BEFORE any omni/isaacsim/pxr import ----
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdGeom  # noqa: E402

# --- open the assembled scene ---------------------------------------------
omni.usd.get_context().open_stage(USD)
stage = omni.usd.get_context().get_stage()

# --- define dedicated cameras and aim them --------------------------------
# Scene content: table top ~0.9 m, robot on top (reaches ~1.5), arm span ~1.3
UsdGeom.Camera.Define(stage, "/World/cam_overview")
UsdGeom.Camera.Define(stage, "/World/cam_topdown")
set_camera_view([3.4, -3.0, 2.9], [0.20, 0.30, 1.00], "/World/cam_overview")
set_camera_view([0.30, 0.0, 3.10], [0.30, 0.001, 0.90], "/World/cam_topdown")

# Let the stage, materials and textures stream in before rendering.
for _ in range(60):
    simulation_app.update()

VIEWS = [("overview", "/World/cam_overview"), ("topdown", "/World/cam_topdown")]
for name, cam_path in VIEWS:
    out = os.path.join(OUT_DIR, name)
    rp = rep.create.render_product(cam_path, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=out, rgb=True)
    writer.attach([rp])
    # rt_subframes lets the RTX real-time denoiser accumulate -> clean still.
    rep.orchestrator.step(rt_subframes=64)
    for _ in range(3):          # flush the file write
        simulation_app.update()
    writer.detach()
    rp.destroy()
    pngs = [f for f in os.listdir(out)] if os.path.isdir(out) else []
    print(f"[render] {name}: wrote {pngs} in {out}")

# --- Blackwell (RTX 5080) teardown workaround ------------------------------
import sys  # noqa: E402

sys.stdout.flush()
os._exit(0)
