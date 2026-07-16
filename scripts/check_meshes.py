#!/usr/bin/env python3
"""Preflight check for a URDF before importing into Isaac Sim.

This is PLAIN Python -- it does NOT launch Isaac Sim. Run it with the system
python to instantly find out WHY your robot imports as bare coordinate frames
with no visible mesh:

    python3 scripts/check_meshes.py

It reports, for every <mesh filename="..."> in the URDF:
  * whether the referenced mesh file actually exists on disk, and
  * whether each joint axis looks like a clean canonical axis or a tilted
    CAD-export axis (a red flag for fusion2urdf exports).

If meshes are MISSING, that is the reason you only see XYZ frames in Isaac Sim.
No import script can fix missing mesh data -- you must supply the mesh files.
"""

import os
import sys
import xml.etree.ElementTree as ET

URDF = os.environ.get(
    "URDF_PATH",
    os.path.join(os.path.dirname(__file__), "..", "3d_model", "kr10_arm.urdf"),
)


def main() -> int:
    urdf_path = os.path.abspath(URDF)
    if not os.path.isfile(urdf_path):
        print(f"URDF not found: {urdf_path}")
        return 2

    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    print(f"URDF:      {urdf_path}")
    print(f"URDF dir:  {urdf_dir}")
    print("=" * 70)

    # ---- 1. Mesh existence ------------------------------------------------
    meshes = root.findall(".//geometry/mesh")
    missing, found = [], []
    for m in meshes:
        fn = m.get("filename", "")
        # URDF mesh paths are relative to the URDF file (unless package:// URI)
        if fn.startswith("package://"):
            resolved = fn  # cannot resolve without a ROS workspace
            exists = False
        else:
            resolved = os.path.normpath(os.path.join(urdf_dir, fn))
            exists = os.path.isfile(resolved)
        (found if exists else missing).append((fn, resolved))

    print(f"MESH REFERENCES: {len(meshes)} total, "
          f"{len(found)} found, {len(missing)} MISSING")
    for fn, resolved in missing:
        print(f"  [MISSING] {fn}")
        print(f"            expected at: {resolved}")
    for fn, resolved in found:
        size = os.path.getsize(resolved)
        print(f"  [ok]      {fn}  ({size/1024:.0f} KB)")

    # ---- 2. Joint axis sanity --------------------------------------------
    print("=" * 70)
    print("JOINT AXES (canonical = aligned to X/Y/Z; tilted = CAD-frame export):")
    tilted = 0
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")
        axis_el = j.find("axis")
        if axis_el is None:
            print(f"  {name} ({jtype}): no <axis> (fixed?)")
            continue
        axis = [float(x) for x in axis_el.get("xyz", "0 0 0").split()]
        # canonical means exactly one component is ~1 and the rest ~0
        near = [abs(abs(a) - 1.0) < 1e-3 for a in axis]
        zero = [abs(a) < 1e-3 for a in axis]
        canonical = sum(near) == 1 and sum(zero) == 2
        flag = "canonical" if canonical else ">>> TILTED (non-canonical)"
        if not canonical:
            tilted += 1
        print(f"  {name} ({jtype}): [{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}]  {flag}")

    # ---- Verdict ----------------------------------------------------------
    print("=" * 70)
    if missing:
        print("VERDICT: Meshes are MISSING. This is why Isaac Sim shows only the")
        print("         coordinate frames and no robot body. Provide the mesh files")
        print("         (put them where the URDF expects them, above) and re-import.")
    else:
        print("VERDICT: All meshes present. If the body is still invisible, the")
        print("         problem is scale/units or camera framing, not missing meshes.")
    if tilted:
        print(f"WARNING: {tilted} joint axes are non-canonical (tilted). This is a")
        print("         fusion2urdf export whose CAD was not aligned to world axes.")
        print("         The arm will be very hard to control and will NOT match the")
        print("         real KUKA joint conventions. Strongly consider the official")
        print("         ros-industrial 'kuka_kr10_support' URDF instead.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
