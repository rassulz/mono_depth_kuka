# mono_depth_kuka

Isaac Sim project: an Intel RealSense D455 mounted above a table detects a toy car's position; a KUKA KR10 R1100-2 with a **magnetic end effector** (points straight down, magnets the car on contact) picks it and places it at a predefined spot on the table. Eventually coordinates flow camera → ROS 2 → manipulator; motion is kinematics-based (IK, not RL).

Project stages:
1. Pick-and-place of the car in Isaac Sim, car respawns at a random table location each cycle.
2. Generate a 1000-image RGB + depth dataset (car at a different location per image) for training a custom vision model.

## Environment

- **Isaac Sim 5.1.0** (pip install) — venv at `~/isaac/venv`, Python 3.11.
- **Isaac Lab 2.3.2** at `~/isaac/IsaacLab` (installed editable into the same venv). Not needed for stages 1–2; scripts are Isaac Sim *standalone* (`from isaacsim import SimulationApp` first, all other omni/isaacsim/pxr imports strictly after).
- Run scripts with: `~/isaac/venv/bin/python scripts/<script>.py` (add `--headless` for no GUI). Never system python.
- First boot after cache clear is slow (extension loading + shader warmup); allow several minutes.
- To inspect USD without booting Kit, `pxr` is importable from the bundled libs:
  `PYTHONPATH=<venv>/site-packages/isaacsim/extscache/omni.usd.libs-*/ LD_LIBRARY_PATH=<that>/bin:~/.local/share/uv/python/cpython-3.11*/lib`

## Main scene: `isaac_version_project_anima.usd`

Z-up, meters. defaultPrim `/World`. Everything is already placed (do not move robot/table/camera; only the car gets moved).

| Prim | What | Key facts (world frame) |
|---|---|---|
| `/World/KR10_R1100_2_updated` | KUKA KR10 R1100-2 articulation | base_link origin (0.093, 0.465, 1.102); articulation root API on `root_joint`; 7 revolute joints `Revolute_1..7` — 1–6 are the arm, **7 is the magnetic gripper mount** (keep at 0) |
| `/World/car_object_arlan_usd` | toy car (static mesh, **no physics APIs**) | bbox ~0.20×0.09×0.08 m; sits on table top at z ≈ 0.836; mesh has internal offset — move it via translate delta, not absolute bbox target |
| `/World/arlan_project_srtand_usd_updated` | table + camera stand | table top surface z ≈ 0.835; tabletop spans roughly x ∈ (−0.43, 0.78), y ∈ (−0.39, 0.77) |
| `/World/Realsense` | RealSense D455 (references Nucleus cloud asset over https — needs internet) | mounted at (−0.314, −0.180, 1.374), ~0.54 m above table, looking down |
| `/World/kr10_arm` | **stale/broken** — references nonexistent `/home/rassul_pc/Anima_project/...` path | empty Xform, harmless warning at load; ignore |
| `/World/GroundPlane`, `/physicsScene`, `/Environment/defaultLight` | standard | |

## Robot description (URDF source)

`3d_model/KR10_R1100_2_updated_description/KR10_R1100-2_updated_description/urdf/KR10_R1100-2_updated.urdf`
- CAD (Onshape-style) export: URDF joint names have spaces ("Revolute 1"); USD import renamed them `Revolute_1`.
- Joint axes carry CAD noise (joint 5 axis tilted ~1°, joints 6/7 ~2° off Z) → **no analytic IK; use numeric/Jacobian IK via the physics engine**.
- KR10 R1100-2 reach ≈ 1.1 m; axis-1 stands at world (−0.03, 0.28). Keep targets well inside reach.
- **Traps found the hard way** (all handled inside `pick_place.py`, re-apply if writing new control scripts):
  - `effort="100"` in the URDF became a 100 N·m drive force limit on import — far too weak for the CAD-inflated masses (~180 kg arm; real KR10 is ~55 kg). Without rewriting drive `maxForce` in the stage, the arm collapses onto its J2 limit under gravity.
  - Imported drive gains are stiff but almost undamped (ζ≈0.07) → raise kd ~10×, and command an integrated reference trajectory with an anti-windup governor, never feed measured q back.
  - Home posture (all zeros) is straight up = elbow singularity, and the tool axis is antipodal to "down".
  - A 3-row cross-product orientation task implicitly forbids yaw about the tool axis; with J6 camped on its −π limit this deadlocks alignment at ~10° tilt. Use a 5-row task (position + 2 tilt rows, yaw free) + nullspace wrist centering.
  - `SimulationApp` default `fast_shutdown` kills the process inside `close()` — code after it never runs (exit codes lost). Use `{"fast_shutdown": False}` + `os._exit(code)` (plain `sys.exit` hangs at teardown).
  - Python stdout is block-buffered when piped; prints must use `flush=True` (or run `python -u`) or they're lost on shutdown.

## Asset path notes

Source assets live in this repo under `3d_model/` (`car_object_arlan_usd.usdz`, `arlan_project_srtand_usd_updated.usdz`, KR10 description). The main scene USD already embeds/references what it needs, except the RealSense cloud asset (and the broken `/World/kr10_arm` prim, which still points at an old `/home/rassul_pc/Anima_project/...` path — harmless).

## Scripts

- `scripts/pick_place.py` — stage 1: IK pick-and-place loop, car randomized on the table each cycle, magnetic grip modeled as kinematic attach (car has no rigid body). `--headless --cycles N --seed S`; prints per-cycle PASS/FAIL and exits nonzero on failure.

## Conventions

- Follow the `isaac` skill (Isaac Sim 5.1 / Isaac Lab 2.3 API names — `isaacsim.*`, not `omni.isaac.*`).
- Verify changes by actually running headless with the built-in success checks, not by reading code.
- Prim paths always start with `/`.
