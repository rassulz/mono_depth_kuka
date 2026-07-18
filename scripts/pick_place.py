"""KUKA KR10 R1100-2 pick-and-place of the toy car in Isaac Sim 5.1.

Stage 1 of the project (see readme.md / CLAUDE.md):
  - The toy car appears at a random location on the table each cycle, always
    in its authored (tray-fitting) orientation — the vision model only
    outputs position, so orientation must stay known.
  - The arm moves by kinematics (resolved-rate Jacobian IK on position drives).
  - The magnetic end effector points straight down, "magnets" the car on
    contact (modeled as a kinematic attach — the car mesh has no rigid body),
    lifts it, and sets it down fitted into the car-shaped tray fixture
    (MeshInstance_11), always in the same orientation — the place pose is read
    from the car's authored (in-tray) pose in the scene at startup.

Run (GUI):
    ~/isaac/venv/bin/python scripts/pick_place.py
Run (verification):
    ~/isaac/venv/bin/python -u scripts/pick_place.py --headless --cycles 6 --corners --seed 0

Exits nonzero if any cycle fails its placement check.

Constants below were grounded with scripts/inspect_scene.py:
  - dof names Revolute_1..7 (7 = the gripper's rotary actuator, used for the
    place-yaw task), home q = zeros
  - jacobian shape (1, 7, 6, 7), fixed base excluded -> gripper row 6,
    rows = [linear; angular], world frame
  - the working surface is the long rectangular magnet plane (0.25 x 0.054 m)
    perpendicular to the joint-7 rotary axis; tool axis = that axis (~local
    +Z with a real 2 deg CAD tilt), TCP at the plane center — see
    TOOL_AXIS_LOCAL / MAGNET_TCP_LOCAL
  - the URDF's effort="100" became a 100 Nm drive force limit on import —
    far too weak for the CAD-inflated link masses, the arm falls onto its J2
    limit -> maxForce is rewritten in the stage before physics loads
  - drive gains from the URDF import are stiff but almost undamped
    (zeta ~ 0.07) -> kd is raised 10x at startup, commands go through an
    integrated reference trajectory (anti-windup governor, never measured
    joint feedback)
  - axis-1 of the arm stands at roughly (-0.03, 0.28) in world XY; targets
    0.4-0.6 m away are comfortably inside the 1.1 m reach envelope
"""

import argparse
import os

parser = argparse.ArgumentParser(description="KR10 magnetic pick-and-place")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--cycles", type=int, default=5, help="number of pick-place cycles")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--corners", action="store_true",
                    help="pick-zone corners for the first 4 cycles (worst-case reach test)")
args = parser.parse_args()

from isaacsim import SimulationApp

# fast_shutdown would terminate the process inside close(), losing our exit code
simulation_app = SimulationApp({"headless": args.headless, "fast_shutdown": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# ----------------------------------------------------------------------------
# Scene constants
# ----------------------------------------------------------------------------
SCENE = "/home/rassul_pc/mono_depth_kuka/isaac_version_project_anima.usd"
ROBOT = "/World/KR10_R1100_2_updated"
GRIPPER = ROBOT + "/magnetic_gripper_1"
CAR = "/World/car_object_arlan_usd"

# Region on the table where the car may appear (bbox-center coordinates, world).
# Inside the RealSense D455 view cone (camera nadir ~(-0.31, -0.18)) and within
# comfortable arm reach of axis 1 at (-0.03, 0.28).
PICK_ZONE_X = (-0.30, 0.05)
PICK_ZONE_Y = (-0.25, 0.08)

# The place target is the car-shaped tray fixture (MeshInstance_11 on the
# stand). The car is authored in the scene already fitted into that tray, so
# its authored pose IS the place pose: PLACE_XY is read from the stage at
# startup, and PLACE_YAW = 0 means "the authored orientation".
PLACE_XY = None  # set after the scene is measured
PLACE_YAW = 0.0
YAW_TOL = 0.03  # rad, ~1.7 deg

# The magnet's working surface is the long rectangular plane of the bar
# (0.25 x 0.054 m), perpendicular to the rotary-actuator (joint 7) axis and on
# the far side of it from the wrist. Its outward normal IS the joint-7 axis
# direction (the ~2 deg tilt off local +Z is real, from the CAD export), and
# its center sits at local (0.000, -0.002, 0.0695). Both measured directly
# from magnetic_gripper_1.stl in the body frame.
TOOL_AXIS_LOCAL = np.array([0.000597, -0.034639, 0.9994])
TOOL_AXIS_LOCAL /= np.linalg.norm(TOOL_AXIS_LOCAL)
MAGNET_TCP_LOCAL = np.array([0.0, -0.0024, 0.0695])

HOVER = 0.15          # hover height above car top / place point [m]
CONTACT_GAP = 0.004   # magnet face standoff at "contact" [m]
POS_TOL_HOVER = 0.010
POS_TOL_CONTACT = 0.005
AXIS_TOL = 0.999      # min dot(tool_axis, down) to accept a waypoint
PLACE_TOL = 0.020     # cycle success: car bbox center within this of PLACE_XY
STEP_BUDGET = 1200    # max sim steps per waypoint before declaring failure

DT = 1.0 / 60.0
MAX_LIN_VEL = 0.25    # EE speed cap [m/s]
MAX_ANG_VEL = 1.2     # tool-axis correction cap [rad/s]
MAX_JOINT_VEL = 0.9   # per-joint reference speed cap [rad/s]
LIN_GAIN = 2.5
ROT_GAIN = 3.0
DLS_LAMBDA = 0.05
GOVERNOR = 0.2        # max lead of the reference over measured joints [rad]
NULLSPACE_W = np.array([0.0, 0.0, 0.0, 0.15, 0.15, 0.5, 0.15])  # wrist-centering weights
DOWN = np.array([0.0, 0.0, -1.0])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Scene setup
# ----------------------------------------------------------------------------
open_stage(SCENE)
stage = omni.usd.get_context().get_stage()

# The URDF declares effort="100" on every joint, which the importer turned into
# a 100 Nm drive force limit — nowhere near enough to hold the CAD-inflated
# link masses (~180 kg arm) against gravity: the arm collapses onto its J2
# limit. Give the drives real authority before physics parses the stage.
DRIVE_MAX_FORCE = [5e4, 5e4, 5e4, 5e3, 5e3, 5e3, 2e3]
for i, f in enumerate(DRIVE_MAX_FORCE, start=1):
    jp = stage.GetPrimAtPath(f"{ROBOT}/joints/Revolute_{i}")
    drive = UsdPhysics.DriveAPI.Get(jp, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
    drive.CreateMaxForceAttr().Set(float(f))

# Pre-sim USD measurements (authored pose: car resting on the table).
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
car_prim = stage.GetPrimAtPath(CAR)
car_box = bbc.ComputeWorldBound(car_prim).ComputeAlignedRange()
car_min, car_max = np.array(car_box.GetMin()), np.array(car_box.GetMax())
CAR_CENTER0 = (car_min + car_max) / 2.0
CAR_HALF_HEIGHT = (car_max[2] - car_min[2]) / 2.0
TABLE_TOP_Z = car_min[2]
PLACE_XY = CAR_CENTER0[:2].copy()  # authored pose = fitted into the tray fixture
print(f"[scene] car bbox center {CAR_CENTER0}, half-height {CAR_HALF_HEIGHT:.4f}, "
      f"table top z {TABLE_TOP_Z:.4f}", flush=True)
print(f"[scene] place target (tray fixture): ({PLACE_XY[0]:+.4f}, {PLACE_XY[1]:+.4f})",
      flush=True)

world = World(stage_units_in_meters=1.0)
robot = world.scene.add(SingleArticulation(prim_path=ROBOT, name="kr10"))
gripper_body = SingleRigidPrim(prim_path=GRIPPER, name="gripper_body")
car = SingleXFormPrim(prim_path=CAR, name="car")
world.reset()
gripper_body.initialize(world.physics_sim_view)
car.initialize(world.physics_sim_view)

view = robot._articulation_view
body_names = list(view.body_names)
ee_body_idx = body_names.index("magnetic_gripper_1")
JAC_EE_ROW = ee_body_idx - (len(body_names) - np.asarray(view.get_jacobians()).shape[1])
NUM_DOF = robot.num_dof
# Revolute_1..6 are the arm; Revolute_7 is the gripper's own rotary actuator —
# controlled too, because J6 alone (range 6.1 rad < 2*pi) has a dead band
# where neither rotation direction can reach the place yaw.
ARM_DOFS = list(range(7))
Q_HOME = np.asarray(robot.get_joint_positions(), dtype=float).copy()

lim = np.asarray(view.get_dof_limits()).reshape(-1, 2)
DOF_LOWER, DOF_UPPER = lim[:, 0].copy(), lim[:, 1].copy()

# The URDF-import drives are extremely stiff (kp up to 1.2e6) and nearly
# undamped (zeta ~ 0.07) -> visible end-effector vibration. Soften kp to 10%
# and raise kd 10x: the damping ratio goes up ~30x combined, and the outer
# servo loop absorbs the small gravity sag softer drives allow.
kps, kds = view.get_gains()
kps = np.asarray(kps, dtype=float).reshape(1, -1) * 0.1
kds = np.asarray(kds, dtype=float).reshape(1, -1) * 10.0
view.set_gains(kps=kps, kds=kds)
print(f"[setup] smoothed drives: kp = {kps.ravel()}, kd = {kds.ravel()}", flush=True)

controller = robot.get_articulation_controller()

# Car pose bookkeeping: bbox center = xform_pos + R_extra @ offset0 where
# R_extra is the rotation applied on top of the authored orientation.
car_pos0, car_quat0 = car.get_world_pose()
car_pos0 = np.asarray(car_pos0, dtype=float)
CAR_QUAT0 = np.asarray(car_quat0, dtype=float)
CAR_ROT0 = quat_to_rot(CAR_QUAT0)
CAR_OFFSET0 = CAR_CENTER0 - car_pos0

rng = np.random.default_rng(args.seed)


def set_car_center(center_xy, yaw, z_bottom=TABLE_TOP_Z):
    """Place the car so its bbox center lands at (x, y) with its wheels on the table."""
    center = np.array([center_xy[0], center_xy[1], z_bottom + CAR_HALF_HEIGHT])
    pos = center - rot_z(yaw) @ CAR_OFFSET0
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    car.set_world_pose(position=pos, orientation=quat_mul(q_yaw, CAR_QUAT0))
    return center


def car_pose_now():
    """Current bbox center and the extra yaw applied on top of the authored pose."""
    pos, quat = car.get_world_pose()
    pos = np.asarray(pos, dtype=float)
    R_extra = quat_to_rot(np.asarray(quat, dtype=float)) @ CAR_ROT0.T
    center = pos + R_extra @ CAR_OFFSET0
    yaw = float(np.arctan2(R_extra[1, 0], R_extra[0, 0]))
    return center, yaw


# ----------------------------------------------------------------------------
# Magnetic gripper: kinematic attach (car follows the magnet face rigidly)
# ----------------------------------------------------------------------------
class MagneticGripper:
    def __init__(self):
        self.attached = False
        self._rel_pos = None
        self._rel_quat = None

    def attach(self):
        gp, gq = gripper_body.get_world_pose()
        cp, cq = car.get_world_pose()
        gq = np.asarray(gq, dtype=float)
        Rg = quat_to_rot(gq)
        self._rel_pos = Rg.T @ (np.asarray(cp, dtype=float) - np.asarray(gp, dtype=float))
        self._rel_quat = quat_mul(quat_conj(gq), np.asarray(cq, dtype=float))
        self.attached = True

    def release(self):
        self.attached = False

    def follow(self):
        if not self.attached:
            return
        gp, gq = gripper_body.get_world_pose()
        gp, gq = np.asarray(gp, dtype=float), np.asarray(gq, dtype=float)
        pos = gp + quat_to_rot(gq) @ self._rel_pos
        car.set_world_pose(position=pos, orientation=quat_mul(gq, self._rel_quat))


magnet = MagneticGripper()

# ----------------------------------------------------------------------------
# Resolved-rate IK servo (position + tool-axis-down task)
# ----------------------------------------------------------------------------
q_ref = Q_HOME.copy()  # integrated reference trajectory, the only thing we command
JITTER_BUF = []        # per-step magnet-face positions, for the smoothness metric

# slew-rate limits on the commanded twist: velocity ramps instead of stepping
# at waypoint switches, which removes the largest acceleration spikes
LIN_ACC = 0.6   # m/s^2
ANG_ACC = 4.0   # rad/s^2
v_cmd_prev = np.zeros(3)
w_cmd_prev = np.zeros(3)


def tool_axis_world():
    _, gq = gripper_body.get_world_pose()
    return quat_to_rot(np.asarray(gq, dtype=float)) @ TOOL_AXIS_LOCAL


def magnet_face_pos():
    gp, gq = gripper_body.get_world_pose()
    R = quat_to_rot(np.asarray(gq, dtype=float))
    return np.asarray(gp, dtype=float) + R @ MAGNET_TCP_LOCAL


def step_sim():
    world.step(render=not args.headless)
    magnet.follow()


def servo_to(target_face_pos, pos_tol, label="", yaw_target=None):
    """Drive the magnet face to target_face_pos with the tool axis pointing down.

    With yaw_target set (and the car attached), additionally rotate about the
    tool axis until the carried car's yaw reaches it.
    """
    global q_ref
    target_face_pos = np.asarray(target_face_pos, dtype=float)
    for i in range(STEP_BUDGET):
        face = magnet_face_pos()
        JITTER_BUF.append(face)
        e_pos = target_face_pos - face
        a = tool_axis_world()
        dot = float(a @ DOWN)
        yaw_err = 0.0
        if yaw_target is not None:
            _, car_yaw = car_pose_now()
            yaw_err = wrap_angle(yaw_target - car_yaw)
        if (np.linalg.norm(e_pos) < pos_tol and dot > AXIS_TOL
                and abs(yaw_err) < YAW_TOL):
            return True
        if dot < -0.5:
            # Antipodal escape: tool points up (zero-gradient point of the
            # cross-product task) -> nudge about a fixed horizontal axis.
            e_rot = np.array([1.0, 0.0, 0.0])
        else:
            e_rot = np.cross(a, DOWN)
        global v_cmd_prev, w_cmd_prev
        v = np.clip(e_pos * LIN_GAIN, -MAX_LIN_VEL, MAX_LIN_VEL)
        w = np.clip(e_rot * ROT_GAIN, -MAX_ANG_VEL, MAX_ANG_VEL)
        v = v_cmd_prev + np.clip(v - v_cmd_prev, -LIN_ACC * DT, LIN_ACC * DT)
        w = w_cmd_prev + np.clip(w - w_cmd_prev, -ANG_ACC * DT, ANG_ACC * DT)
        v_cmd_prev, w_cmd_prev = v, w
        jac = np.asarray(view.get_jacobians())[0, JAC_EE_ROW]  # 6 x NUM_DOF, world frame
        J6 = jac[:, ARM_DOFS]
        # Constrain only the two tilt components of the angular task: yaw about
        # the tool axis stays free, otherwise a wrist joint pinned at a limit
        # (J6 at -pi) deadlocks the alignment (solver forbids parasitic yaw).
        t1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(a, t1)
        rows = [J6[:3], t1 @ J6[3:], t2 @ J6[3:]]
        task = list(v) + [t1 @ w, t2 @ w]
        if yaw_target is not None:
            # 6th row: yaw rate about world +Z, driving the carried car toward
            # the fixed place orientation. NOT about the tool axis — that
            # points down (~ -Z), which would flip the sign and settle the
            # yaw 180 deg off.
            rows.append(J6[5])
            task.append(np.clip(yaw_err * ROT_GAIN, -MAX_ANG_VEL, MAX_ANG_VEL))
        J = np.vstack(rows)
        task = np.asarray(task)
        JJt_inv = np.linalg.inv(J @ J.T + DLS_LAMBDA**2 * np.eye(len(task)))
        dq = J.T @ JJt_inv @ task
        # nullspace bias: pull the wrist joints toward mid-range so they do
        # not camp on their limits (uses the free yaw direction)
        q_arm = np.asarray(robot.get_joint_positions(), dtype=float)[:len(ARM_DOFS)]
        N = np.eye(len(ARM_DOFS)) - J.T @ JJt_inv @ J
        dq += N @ (NULLSPACE_W * (0.0 - q_arm))
        dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL)
        for k, d in enumerate(ARM_DOFS):
            q_ref[d] += dq[k] * DT
        # anti-windup governor: the reference may lead the real joints by at
        # most GOVERNOR rad, so it can never run away when the arm is blocked
        qm = np.asarray(robot.get_joint_positions(), dtype=float)
        q_ref = np.clip(q_ref, qm - GOVERNOR, qm + GOVERNOR)
        q_ref = np.clip(q_ref, DOF_LOWER, DOF_UPPER)
        controller.apply_action(ArticulationAction(joint_positions=q_ref))
        step_sim()
    print(f"  [fail] servo '{label}' not converged: |err|="
          f"{np.linalg.norm(target_face_pos - magnet_face_pos()):.4f} m, "
          f"axis dot={float(tool_axis_world() @ DOWN):.4f}", flush=True)
    return False


def park(q_target, steps=240):
    """Joint-space interpolation to a rest posture."""
    global q_ref
    q_start = q_ref.copy()
    for i in range(steps):
        s = min(1.0, (i + 1) / (steps * 0.85))
        s = s * s * (3 - 2 * s)  # smoothstep
        q_ref = q_start + (q_target - q_start) * s
        controller.apply_action(ArticulationAction(joint_positions=q_ref))
        step_sim()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def random_car_xy(cycle):
    if args.corners and cycle < 4:
        corners = [(PICK_ZONE_X[0], PICK_ZONE_Y[0]), (PICK_ZONE_X[0], PICK_ZONE_Y[1]),
                   (PICK_ZONE_X[1], PICK_ZONE_Y[0]), (PICK_ZONE_X[1], PICK_ZONE_Y[1])]
        return np.array(corners[cycle])
    return np.array([rng.uniform(*PICK_ZONE_X), rng.uniform(*PICK_ZONE_Y)])


for _ in range(30):  # settle
    step_sim()

# One-time staging move: servo from the singular straight-up home to a hover
# above the middle of the pick zone, then remember that well-conditioned
# posture as the between-cycles rest pose.
STAGING = np.array([(PICK_ZONE_X[0] + PICK_ZONE_X[1]) / 2,
                    (PICK_ZONE_Y[0] + PICK_ZONE_Y[1]) / 2,
                    TABLE_TOP_Z + 0.40])
if not servo_to(STAGING, 0.02, "initial-staging"):
    print("FATAL: could not reach the staging pose above the table", flush=True)
    simulation_app.close()
    os._exit(2)
Q_READY = np.asarray(robot.get_joint_positions(), dtype=float).copy()
Q_READY[6] = Q_HOME[6]
print(f"[setup] ready posture q = {np.round(Q_READY, 3)}", flush=True)

results = []
for cycle in range(args.cycles):
    xy = random_car_xy(cycle)
    # The vision model only outputs position, so in the real system the car's
    # orientation must be known a priori: it always spawns in the authored
    # (tray-fitting) orientation. The yaw task during the carry still holds it
    # there against any drift.
    yaw = 0.0
    center = set_car_center(xy, yaw)
    print(f"[cycle {cycle}] car at ({center[0]:+.3f}, {center[1]:+.3f}), "
          f"yaw {np.degrees(yaw):+.0f} deg", flush=True)
    for _ in range(10):
        step_sim()

    car_top = center[2] + CAR_HALF_HEIGHT
    ok = servo_to([center[0], center[1], car_top + HOVER], POS_TOL_HOVER, "hover-pick")
    ok = ok and servo_to([center[0], center[1], car_top + CONTACT_GAP], POS_TOL_CONTACT,
                         "descend-pick")
    if ok:
        magnet.attach()
        print("  magnet ON", flush=True)
        place_top = TABLE_TOP_Z + 2 * CAR_HALF_HEIGHT
        ok = (servo_to([center[0], center[1], car_top + HOVER], POS_TOL_HOVER, "lift",
                       yaw_target=PLACE_YAW)
              and servo_to([PLACE_XY[0], PLACE_XY[1], place_top + HOVER], POS_TOL_HOVER,
                           "hover-place", yaw_target=PLACE_YAW)
              and servo_to([PLACE_XY[0], PLACE_XY[1], place_top + CONTACT_GAP],
                           POS_TOL_CONTACT, "descend-place", yaw_target=PLACE_YAW))
        magnet.release()
        print("  magnet OFF", flush=True)
        # settle the car exactly onto the table in the predefined orientation
        # (yaw was already servoed there; on failure keep whatever it has)
        c, yaw_now = car_pose_now()
        set_car_center(c[:2], PLACE_YAW if ok else yaw_now)
        servo_to([PLACE_XY[0], PLACE_XY[1], place_top + HOVER], POS_TOL_HOVER, "retreat")
    else:
        magnet.release()
    park(Q_READY)

    if len(JITTER_BUF) > 2:
        p = np.asarray(JITTER_BUF)
        acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) / DT**2
        print(f"  EE smoothness: acc rms {acc.std() + acc.mean():.2f} m/s^2, "
              f"p95 {np.percentile(acc, 95):.2f}", flush=True)
    JITTER_BUF.clear()
    final, final_yaw = car_pose_now()
    err = np.linalg.norm(final[:2] - PLACE_XY)
    yaw_err = abs(wrap_angle(final_yaw - PLACE_YAW))
    passed = bool(ok and err < PLACE_TOL and yaw_err < YAW_TOL)
    results.append(passed)
    print(f"[cycle {cycle}] {'PASS' if passed else 'FAIL'}  "
          f"car xy err {err * 1000:.1f} mm, yaw err {np.degrees(yaw_err):.1f} deg",
          flush=True)

n_pass = sum(results)
print(f"\n{'=' * 50}\nRESULT: {n_pass}/{len(results)} cycles passed", flush=True)
simulation_app.close()
# plain sys.exit hangs the interpreter at teardown after Kit shutdown; os._exit
# is the verified-working way to report failures to the shell (stdout is already
# flushed — every print above uses flush=True)
os._exit(0 if n_pass == len(results) else 1)
