"""Headless scene inspection for isaac_version_project_anima.usd.

Grounds the constants pick_place.py needs: DOF names/order, joint limits,
default positions, gripper body pose & local bbox (magnet face offset),
jacobian shape/indexing, and a resolved-rate IK servo smoke test.

Run:
    ~/isaac/venv/bin/python scripts/inspect_scene.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=240, help="servo steps for the IK smoke test")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import open_stage
import omni.usd
from pxr import Usd, UsdGeom

SCENE = "/home/rassul_pc/mono_depth_kuka/isaac_version_project_anima.usd"
ROBOT = "/World/KR10_R1100_2_updated"
GRIPPER = ROBOT + "/magnetic_gripper_1"
CAR = "/World/car_object_arlan_usd"

open_stage(SCENE)
stage = omni.usd.get_context().get_stage()

# --- pre-sim USD facts (valid before Fabric takes over) ---
print("=" * 70)
print("PRE-SIM USD FACTS")
cache = UsdGeom.XformCache()
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

car_prim = stage.GetPrimAtPath(CAR)
car_xf = cache.GetLocalToWorldTransform(car_prim)
car_box = bbc.ComputeWorldBound(car_prim).ComputeAlignedRange()
print(f"car translate (local op): {car_xf.ExtractTranslation()}")
print(f"car world bbox min {car_box.GetMin()} max {car_box.GetMax()}")
car_center = (np.array(car_box.GetMin()) + np.array(car_box.GetMax())) / 2
print(f"car bbox center: {car_center}, bottom z: {car_box.GetMin()[2]:.4f}")

grip_prim = stage.GetPrimAtPath(GRIPPER)
grip_local_box = bbc.ComputeLocalBound(grip_prim).ComputeAlignedRange()
print(f"gripper LOCAL bbox min {grip_local_box.GetMin()} max {grip_local_box.GetMax()}")

world = World(stage_units_in_meters=1.0)
robot = world.scene.add(SingleArticulation(prim_path=ROBOT, name="kr10"))
gripper_body = SingleRigidPrim(prim_path=GRIPPER, name="gripper_body")
world.reset()
gripper_body.initialize(world.physics_sim_view)

print("=" * 70)
print("ARTICULATION")
print(f"num_dof: {robot.num_dof}")
print(f"dof_names: {robot.dof_names}")
view = robot._articulation_view
try:
    print(f"body_names: {view.body_names}")
except Exception as e:
    print(f"body_names failed: {e}")
print(f"default joint positions: {robot.get_joint_positions()}")
try:
    limits = view.get_dof_limits()
    print(f"dof limits:\n{np.asarray(limits)}")
except Exception as e:
    print(f"get_dof_limits failed: {e}")
try:
    gains = view.get_gains()
    print(f"gains (kp, kd): {np.asarray(gains[0])}, {np.asarray(gains[1])}")
except Exception as e:
    print(f"get_gains failed: {e}")

pos, quat = gripper_body.get_world_pose()
print(f"gripper world pos at home: {pos}")
print(f"gripper world quat (wxyz) at home: {quat}")


def quat_to_rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


R = quat_to_rot(np.asarray(quat, dtype=float))
print(f"gripper rotation matrix at home:\n{R}")
print(f"gripper local axes in world: X={R[:, 0]}, Y={R[:, 1]}, Z={R[:, 2]}")

print("=" * 70)
print("JACOBIAN")
try:
    jshape = view.get_jacobian_shape()
    print(f"get_jacobian_shape(): {jshape}")
except Exception as e:
    print(f"get_jacobian_shape failed: {e}")
jac = view.get_jacobians()
jac = np.asarray(jac)
print(f"get_jacobians() ndarray shape: {jac.shape}")
body_names = list(view.body_names)
ee_body_idx = body_names.index("magnetic_gripper_1")
print(f"magnetic_gripper_1 body index in body_names: {ee_body_idx}")

# Fixed-base articulations exclude the root body from the jacobian rows.
n_jac_bodies = jac.shape[1]
jac_idx = ee_body_idx - (len(body_names) - n_jac_bodies)
print(f"jacobian body rows: {n_jac_bodies} (of {len(body_names)} bodies) -> ee jac index {jac_idx}")

print("=" * 70)
print("IK SERVO SMOKE TEST (resolved-rate, position only)")
# Servo the gripper toward a hover point 15 cm above the car's current center.
target = np.array([car_center[0], car_center[1], car_box.GetMax()[2] + 0.15])
print(f"target: {target}")

controller = robot.get_articulation_controller()
q = np.asarray(robot.get_joint_positions(), dtype=float)
arm_dofs = list(range(min(6, robot.num_dof)))  # joints 1..6; keep gripper joint (7) fixed
dt = 1.0 / 60.0

from isaacsim.core.utils.types import ArticulationAction

err_log = []
for i in range(args.steps):
    pos, quat = gripper_body.get_world_pose()
    pos = np.asarray(pos, dtype=float)
    err = target - pos
    err_log.append(np.linalg.norm(err))
    jac = np.asarray(view.get_jacobians())[0, jac_idx, :3, :]  # linear rows, world frame
    J = jac[:, arm_dofs]
    # damped least squares
    lam = 0.05
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), np.clip(err, -0.10, 0.10) * 4.0)
    q_now = np.asarray(robot.get_joint_positions(), dtype=float)
    q_cmd = q_now.copy()
    for k, d in enumerate(arm_dofs):
        q_cmd[d] = q_now[d] + dq[k] * dt
    controller.apply_action(ArticulationAction(joint_positions=q_cmd))
    world.step(render=False)
    if i % 40 == 0:
        print(f"  step {i:4d}  |err| = {err_log[-1]:.4f} m  ee={pos}")

pos, quat = gripper_body.get_world_pose()
final_err = np.linalg.norm(target - np.asarray(pos, dtype=float))
R = quat_to_rot(np.asarray(quat, dtype=float))
print(f"final |err| = {final_err:.4f} m after {args.steps} steps")
print(f"final gripper axes in world: X={R[:, 0]}, Y={R[:, 1]}, Z={R[:, 2]}")
print("SMOKE:", "PASS" if final_err < 0.02 else "FAIL")

simulation_app.close()
