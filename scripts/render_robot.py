#!/usr/bin/env python3
"""Render JUST the imported KR10 arm, auto-framed, so we can judge its geometry
and home pose in isolation (no table occluding it). Isaac Sim 5.1, headless.

Run:  ~/isaac/venv/bin/python scripts/render_robot.py
Out:  renders/robot/rgb_0000.png
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ROBOT_USD = os.environ.get(
    "ROBOT_USD", os.path.join(PROJECT, "3d_model", "kr10_official_imported.usd"))
OUT = os.path.join(PROJECT, "renders", "robot")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, Sdf  # noqa: E402

omni.usd.get_context().open_stage(ROBOT_USD)
stage = omni.usd.get_context().get_stage()

# add lights (the bare import may have none)
UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(1500)
UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(400)

for _ in range(40):
    simulation_app.update()

# frame the default prim (the whole arm)
prim = stage.GetDefaultPrim()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
mn, mx = r.GetMin(), r.GetMax()
cx, cy, cz = (mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2
d = max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 0.5)
print(f"[robot] bbox min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
      f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) size~{d:.3f}")

set_camera_view([cx + 1.6*d, cy - 1.6*d, cz + 1.1*d], [cx, cy, cz], "/OmniverseKit_Persp")

# Synchronous capture: drive the render with orchestrator.step (this is what
# populates the annotator), then read the pixels back and save with PIL.
rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])
rep.orchestrator.step(rt_subframes=64)   # converge the RTX denoiser
data = annot.get_data()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

arr = np.asarray(data)
os.makedirs(OUT, exist_ok=True)
out_png = os.path.join(OUT, "rgb_0000.png")
if arr.ndim == 3 and arr.shape[2] >= 3 and arr.size > 0:
    Image.fromarray(arr[:, :, :3].astype("uint8")).save(out_png)
    print(f"[robot] saved {out_png} shape={arr.shape}")
else:
    print(f"[robot] NO DATA (shape={getattr(arr,'shape',None)})")

import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
