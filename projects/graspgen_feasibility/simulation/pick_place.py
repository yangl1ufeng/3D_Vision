"""Contact-driven pick/place with prescribed Cartesian waypoints, SI units.

Select the Panda bootstrap or the UR5/community 2F-140 adapter. No object
attachment, teleportation after initialization, vision, or obstacle planner is used.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data

PROJECT = Path(__file__).resolve().parents[1]
DT = 1 / 240


class Demo:
    def __init__(self, output, gui=False, no_grasp=False, robot_model="panda", cube_size=0.05):
        self.output, self.gui, self.no_grasp = output, gui, no_grasp
        self.robot_model, self.cube_size = robot_model, cube_size
        self.adapter = None
        self.max_mimic_error = 0.0
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.rows, self.stages = [], []
        self.step = 0
        self.stage = "setup"
        self.width = 0.04
        self.robot_table_collision = False
        self.hold_min_height = float("inf")
        self.hold_contact_steps = 0
        self.grasp_verified = False
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        # GUI dragging and keyboard shortcuts must not alter the physics run.
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        p.resetDebugVisualizerCamera(1.55, 45, -28, [0.35, 0.05, 0.3])
        self.box([0, 0, -0.04], [0.14, 0.16, 0.04], 0, [0.3, 0.35, 0.4, 1])
        self.table = self.box([0.55, 0, -0.04], [0.35, 0.45, 0.04], 0, [0.65, 0.7, 0.75, 1])
        # The colored drop-zone has no collision geometry.
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.065, 0.065, 0.0003],
                                     rgbaColor=[0.2, 0.7, 0.35, 1])
        p.createMultiBody(0, -1, visual, [0.5, 0.22, 0.0003])
        # Relative loading also works on Windows installations with Unicode paths.
        if robot_model == "ur5_robotiq140":
            from .ur5_robotiq import UR5Robotiq140
            self.adapter = UR5Robotiq140()
            self.robot, self.arm = self.adapter.body, self.adapter.arm
            self.fingers, self.tcp = self.adapter.pads, self.adapter.tcp
            home = self.adapter.home
            self.open_width = self.width = self.adapter.max_width
            self.grasp_z = cube_size / 2 + 0.02
            self.adapter.write_manifest(output)
        else:
            original_cwd = Path.cwd()
            try:
                os.chdir(pybullet_data.getDataPath())
                self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
            finally:
                os.chdir(original_cwd)
            self.arm = list(range(7))
            self.fingers = [9, 10]
            self.tcp = 11
            home = [0, -0.4, 0, -2.2, 0, 1.8, math.pi / 4]
            self.open_width = 0.04
            self.grasp_z = cube_size / 2 + 0.003
            for joint, angle in zip(self.arm, home):
                p.resetJointState(self.robot, joint, angle)
            for joint in self.fingers:
                p.resetJointState(self.robot, joint, self.width)
                p.changeDynamics(self.robot, joint, lateralFriction=1.0)
        self.cube = self.box([0.5, 0, cube_size / 2], [cube_size / 2] * 3, 0.1, [0.9, 0.35, 0.12, 1])
        p.changeDynamics(self.cube, -1, lateralFriction=0.8, restitution=0)
        # Symmetric top-down grasp with a yaw that stays away from UR5 wrist limits.
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, math.pi if self.adapter else 0])
        self.command(home)
        self.wall_start = time.perf_counter()
        self.tick(240)
        self.initial_z = p.getBasePositionAndOrientation(self.cube)[0][2]

    @staticmethod
    def box(position, half, mass, color):
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return p.createMultiBody(mass, shape, visual, position)

    def command(self, joints):
        if self.adapter:
            self.adapter.command(joints, self.width)
            return
        p.setJointMotorControlArray(self.robot, self.arm, p.POSITION_CONTROL,
                                    targetPositions=list(joints[:7]), forces=[87] * 7,
                                    positionGains=[0.15] * 7)
        p.setJointMotorControlArray(self.robot, self.fingers, p.POSITION_CONTROL,
                                    targetPositions=[self.width] * 2, forces=[20] * 2)

    def tick(self, count):
        for _ in range(count):
            if not p.isConnected():
                raise RuntimeError("Simulation window disconnected before task verification completed")
            p.stepSimulation()
            if self.adapter:
                self.max_mimic_error = max(self.max_mimic_error, self.adapter.mimic_error())
            self.step += 1
            xyz = p.getBasePositionAndOrientation(self.cube)[0]
            if self.stage == "transfer" and xyz[2] < self.initial_z + 0.08:
                raise RuntimeError("Object dropped during transfer")
            contacts = p.getContactPoints(self.robot, self.cube)
            fingers_touching = {c[3] for c in contacts if c[9] > 0.01 and c[3] in self.fingers}
            table_contacts = p.getContactPoints(self.robot, self.table)
            if any(c[3] >= 0 and c[8] < -0.001 for c in table_contacts):
                self.robot_table_collision = True
            if self.stage == "hold":
                self.hold_min_height = min(self.hold_min_height, xyz[2] - self.initial_z)
                self.hold_contact_steps += int(len(fingers_touching) == 2)
            if self.step % 12 == 0:
                self.rows.append({"time_s": self.step * DT, "stage": self.stage,
                                  "cube_x": xyz[0], "cube_y": xyz[1], "cube_z": xyz[2],
                                  "finger_contacts": len(fingers_touching),
                                  **{f"joint_{j}": p.getJointState(self.robot, j)[0] for j in self.arm}})
            if self.gui:
                # Windows may round each 4 ms sleep upward. Pace against an
                # absolute deadline so this does not accumulate across 5k steps.
                remaining = self.wall_start + self.step * DT - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    def move(self, stage, target, duration=2):
        self.stage = stage
        start = np.array(p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4])
        for index in range(1, int(duration / DT / 4) + 1):
            fraction = index / int(duration / DT / 4)
            fraction = fraction * fraction * (3 - 2 * fraction)
            xyz = start + fraction * (np.array(target) - start)
            joints = p.calculateInverseKinematics(self.robot, self.tcp, xyz.tolist(),
                                                  self.orientation, maxNumIterations=100,
                                                  residualThreshold=1e-5)
            self.command(joints)
            self.tick(4)
        self.tick(120)
        actual = np.array(p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[4])
        error = float(np.linalg.norm(actual - target))
        quat = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)[5]
        orientation_error = 2 * math.acos(float(np.clip(abs(np.dot(quat, self.orientation)), 0, 1)))
        self.stages.append({"stage": stage, "tcp_error_m": error, "orientation_error_rad": orientation_error})
        print(f"{stage}: TCP error {error:.4f} m", flush=True)
        if error > 0.015:
            raise RuntimeError(f"TCP failed to reach {stage}: {error:.4f} m")
        if orientation_error > math.radians(3):
            raise RuntimeError(f"TCP orientation failed at {stage}: {orientation_error:.4f} rad")

    def snapshot(self, name):
        view = p.computeViewMatrixFromYawPitchRoll([0.35, 0.05, 0.3], 1.55, 45, -28, 0, 2)
        projection = p.computeProjectionMatrixFOV(50, 960 / 720, 0.01, 5)
        camera = p.getCameraImage(960, 720, view, projection, renderer=p.ER_TINY_RENDERER)
        pixels = np.asarray(camera[2], dtype=np.uint8).reshape(720, 960, 4)
        Image.fromarray(pixels[:, :, :3]).save(self.output / f"{name}.png")

    def run(self):
        self.snapshot("01_initial")
        self.move("pregrasp", [0.5, 0, 0.22])
        self.move("approach", [0.5, 0, self.grasp_z])
        self.stage = "close"
        self.width = self.open_width if self.no_grasp else 0
        self.command([p.getJointState(self.robot, j)[0] for j in self.arm])
        self.tick(240)
        self.move("lift", [0.5, 0, 0.2])
        self.stage = "hold"
        self.tick(720)
        self.snapshot("02_hold")
        grasp_ok = self.hold_min_height >= 0.1 and self.hold_contact_steps >= 684
        self.grasp_verified = grasp_ok
        if not grasp_ok:
            return {"grasp_success": False, "task_success": False, "failure_stage": "hold"}
        self.move("transfer", [0.5, 0.22, 0.2])
        self.move("lower", [0.5, 0.22, self.grasp_z])
        self.stage = "release"
        self.width = self.open_width
        self.command([p.getJointState(self.robot, j)[0] for j in self.arm])
        self.tick(240)
        self.move("retreat", [0.5, 0.22, 0.22])
        self.stage = "settle"
        self.tick(480)
        self.snapshot("03_placed")
        xyz = p.getBasePositionAndOrientation(self.cube)[0]
        linear, angular = p.getBaseVelocity(self.cube)
        placed = (abs(xyz[0] - 0.5) < 0.04 and abs(xyz[1] - 0.22) < 0.04
                  and abs(xyz[2] - self.cube_size / 2) < 0.005
                  and np.linalg.norm(linear) < 0.01 and np.linalg.norm(angular) < 0.1)
        success = bool(placed and not self.robot_table_collision)
        return {"grasp_success": True, "task_success": success,
                "failure_stage": None if success else "placement_or_table_collision",
                "final_cube_position_m": list(xyz)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show real-time interactive simulation")
    parser.add_argument("--keep-open", action="store_true", help="Keep GUI open after completion")
    parser.add_argument("--no-grasp", action="store_true", help="Negative control: keep fingers open")
    parser.add_argument("--robot", choices=["panda", "ur5_robotiq140"], default="panda")
    parser.add_argument("--cube-size", type=float, default=0.05, help="Cube edge in meters")
    parser.add_argument("--planning", action="store_true", help="UR5 full-arm collision checking and RRT-Connect")
    parser.add_argument("--scenario", choices=['free', 'barrier', 'blocked_goal', 'unreachable'], default='barrier')
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    if args.keep_open and not args.gui:
        parser.error("--keep-open requires --gui")
    if args.planning and args.robot != 'ur5_robotiq140':
        parser.error('--planning currently requires --robot ur5_robotiq140')
    maximum = 0.07 if args.robot == "panda" else 0.11
    if not math.isfinite(args.cube_size) or not 0.03 <= args.cube_size <= maximum:
        parser.error(f"--cube-size must be in [0.03, {maximum}] m for this baseline")
    output = (args.output or PROJECT / "runs" / "simulation" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    demo = None
    result = {"task_success": False, "grasp_success": False}
    try:
        if args.planning:
            from .planned_demo import PlannedDemo
            demo = PlannedDemo(output, args.gui, args.no_grasp, args.robot, args.cube_size, args.scenario)
        else:
            demo = Demo(output, args.gui, args.no_grasp, args.robot, args.cube_size)
        result.update(demo.run())
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", failure_stage=demo.stage if demo else "setup",
                      grasp_success=demo.grasp_verified if demo else False)
    finally:
        result.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      robot=args.robot,
                      input="prescribed poses; no vision", planner="RRT-Connect + checked Cartesian segments" if args.planning else "Cartesian waypoint IK; no obstacle planner",
                      object_attachment=False, no_grasp_control=args.no_grasp, gui=args.gui,
                      python=platform.python_version(), pybullet=version("pybullet"),
                      timestep_s=DT, solver_iterations=150, cube_mass_kg=0.1,
                      cube_size_m=args.cube_size, cube_friction=0.8, finger_friction=1.0,
                      finger_force_limit_n=20 if args.robot == "panda" else None,
                      required_lift_m=0.1, required_hold_s=3,
                      self_collision_checked=args.planning)
        if demo:
            if args.planning:
                planning = demo.save_planning()
                result.update(scenario=args.scenario, execution_collision=planning['execution_collision'],
                              replanned_segments=sum(r.get('method') == 'rrt_connect' for r in planning['plans']))
                demo.scene.close()
            if demo.adapter:
                result.update(gripper_motor_torque_limit_nm=demo.adapter.motor_torque,
                              max_mimic_error_rad=demo.max_mimic_error,
                              measured_model_opening_m=demo.adapter.max_width,
                              graspgen_frame_mapping_verified=False)
            result.update(stages=demo.stages, robot_table_collision=demo.robot_table_collision,
                          hold_min_lift_m=demo.hold_min_height if math.isfinite(demo.hold_min_height) else None,
                          hold_both_fingers_contact_fraction=demo.hold_contact_steps / 720)
            if demo.rows:
                with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(demo.rows[0]))
                    writer.writeheader()
                    writer.writerows(demo.rows)
        (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=True))
        print(f"Output: {output}")
        if args.gui and args.keep_open and p.isConnected():
            print("Close the simulation window or press Ctrl+C to exit.")
            try:
                while p.isConnected():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        if p.isConnected():
            p.disconnect()
    return 0 if result["task_success"] else 1
