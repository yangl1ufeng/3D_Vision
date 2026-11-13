"""UR5 + community 2F-140 model adapter. No upstream controller is executed.

The downloaded model includes added pads: its measured opening is ~128.5 mm,
not the nominal hardware 140 mm. Dynamics corrections are simulation assumptions.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

PROJECT = Path(__file__).resolve().parents[1]
COMMIT = "37f7d66d3196490618bf89ffe2b49dcb7e07aec3"
REPOSITORY = "https://github.com/ElectronicElephant/pybullet_ur5_robotiq"
SOURCE = PROJECT / "third_party" / f"pybullet_ur5_robotiq-{COMMIT}"
ARM_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
PAD_NAMES = ["left_inner_finger_pad", "right_inner_finger_pad"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inertial(link, mass, diagonal):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    node = ET.SubElement(link, "inertial")
    ET.SubElement(node, "mass", value=str(mass))
    ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(node, "inertia", ixx=str(diagonal[0]), iyy=str(diagonal[1]),
                  izz=str(diagonal[2]), ixy="0", ixz="0", iyz="0")


def prepare_model():
    source = SOURCE / "urdf/ur5_robotiq_140.urdf"
    if not source.is_file():
        raise FileNotFoundError("Robot assets missing. Run download_simulation_models.ps1 first.")
    tree = ET.parse(source)
    root = tree.getroot()
    # Upstream pads have mass=1 kg and singular inertia tensors. Give each added
    # pad an explicit 10 g uniform-box inertia, retaining its collision geometry.
    for name in PAD_NAMES:
        link = root.find(f"link[@name='{name}']")
        x, y, z = map(float, link.find("collision/geometry/box").get("size").split())
        mass = 0.01
        inertial(link, mass, [mass * (y*y+z*z)/12, mass * (x*x+z*z)/12, mass * (x*x+y*y)/12])
    # ee_link is a coordinate marker, not the default 1 kg body assigned by Bullet.
    inertial(root.find("link[@name='ee_link']"), 0, [0, 0, 0])
    # Preserve the flange geometry and upstream mounting rotation. Add a named,
    # fixed task TCP in gripper-base axes; +z points from palm toward fingertips.
    tcp = ET.SubElement(root, "link", name="project_grasp_tcp")
    inertial(tcp, 0, [0, 0, 0])
    joint = ET.SubElement(root, "joint", name="project_grasp_tcp_joint", type="fixed")
    ET.SubElement(joint, "parent", link="robotiq_arg2f_base_link")
    ET.SubElement(joint, "child", link="project_grasp_tcp")
    ET.SubElement(joint, "origin", xyz="0 0 0.2", rpy="0 0 0")
    destination = source.with_name("ur5_robotiq_140_project.urdf")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    paths = {source, SOURCE / "LICENSE", SOURCE / "README.md"}
    for mesh in root.findall(".//mesh"):
        paths.add((source.parent / mesh.get("filename")).resolve())
    manifest = {
        "repository": REPOSITORY, "commit": COMMIT,
        "files_sha256": {str(f.relative_to(SOURCE)).replace('\\', '/'): sha256(f) for f in sorted(paths)},
        "derived_urdf_sha256": sha256(destination),
        "changes": ["added pads: 0.01 kg each with positive diagonal box inertia",
                    "ee_link coordinate marker: zero mass instead of Bullet default 1 kg",
                    "fixed project_grasp_tcp at gripper-base (0, 0, 0.2) m"],
        "T_gripper_base_tcp": [[1,0,0,0], [0,1,0,0], [0,0,1,0.2], [0,0,0,1]],
        "graspgen_frame_mapping_verified": False,
    }
    return destination, root, manifest


class UR5Robotiq140:
    def __init__(self):
        model, xml, self.manifest = prepare_model()
        previous = Path.cwd()
        try:
            os.chdir(SOURCE)
            self.body = p.loadURDF(str(model.relative_to(SOURCE)), useFixedBase=True)
        finally:
            os.chdir(previous)
        infos = [p.getJointInfo(self.body, i) for i in range(p.getNumJoints(self.body))]
        self.joints = {i[1].decode(): i[0] for i in infos}
        self.links = {i[12].decode(): i[0] for i in infos}
        self.arm = [self.joints[n] for n in ARM_NAMES]
        self.pads = [self.links[n] for n in PAD_NAMES]
        self.tcp = self.links["project_grasp_tcp"]
        self.parent = self.joints["finger_joint"]
        self.mimic = {self.joints[j.get("name")]: float(j.find("mimic").get("multiplier"))
                      for j in xml.findall("joint") if j.find("mimic") is not None}
        self.home = [0, -1.54, 1.34, -1.37, -math.pi / 2, 0]
        self.forces = [infos[j][10] for j in self.arm]
        self.limits = np.array([[infos[j][8], infos[j][9]] for j in self.arm])
        self.motor_torque = 2.0
        self.q_grid = np.linspace(infos[self.parent][8], infos[self.parent][9], 141)
        self.gaps = []
        for q in self.q_grid:
            self.reset_fingers(q)
            self.gaps.append(self.pad_gap())
        self.gaps = np.array(self.gaps)
        if not np.all(np.diff(self.gaps) < 0):
            raise RuntimeError("Gripper geometry is not monotonically closing")
        self.max_width = float(self.gaps[0])
        self.manifest.update(measured_opening_m=self.max_width, closed_gap_m=float(self.gaps[-1]),
                             motor_torque_limit_nm=self.motor_torque,
                             mimic_joint_names={infos[j][1].decode(): m for j, m in self.mimic.items()})
        self.reset_fingers(0)
        for j, angle in zip(self.arm, self.home):
            p.resetJointState(self.body, j, angle)
        for info in infos:
            if info[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(self.body, info[0], p.VELOCITY_CONTROL, force=0)
        for child, multiplier in self.mimic.items():
            constraint = p.createConstraint(self.body, self.parent, self.body, child,
                                             p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
            p.changeConstraint(constraint, gearRatio=-multiplier, maxForce=50, erp=0.8)
        for pad in self.pads:
            p.changeDynamics(self.body, pad, lateralFriction=1.0, restitution=0)

    def reset_fingers(self, q):
        """Initialization and isolated geometric calibration only; never during a task."""
        p.resetJointState(self.body, self.parent, q)
        for joint, multiplier in self.mimic.items():
            p.resetJointState(self.body, joint, multiplier * q)

    def pad_gap(self):
        base = p.getLinkState(self.body, self.links["robotiq_arg2f_base_link"], computeForwardKinematics=True)
        inverse = p.invertTransform(base[4], base[5])
        bounds = []
        for pad in self.pads:
            state = p.getLinkState(self.body, pad, computeForwardKinematics=True)
            pos, quat = p.multiplyTransforms(*inverse, state[4], state[5])
            rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            radius = float(np.abs(rotation[1]) @ np.array([0.015, 0.035, 0.00375]))
            bounds.append((pos[1] - radius, pos[1] + radius))
        bounds.sort()
        return float(bounds[1][0] - bounds[0][1])

    def angle_for_width(self, width):
        if not math.isfinite(width) or width < 0 or width > self.max_width + 1e-6:
            raise ValueError(f"Width must be in [0, {self.max_width:.6f}] m for this model")
        return float(np.interp(width, self.gaps[::-1], self.q_grid[::-1]))

    def command(self, joints, width):
        q = np.array(joints[:6], dtype=float, copy=True)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite UR5 joint angles")
        # Select a legal equivalent angle close to the present configuration.
        # Crossing +/-pi by blindly wrapping would command a full wrist turn.
        for index, joint in enumerate(self.arm):
            current = p.getJointState(self.body, joint)[0]
            candidates = [q[index] + k * 2 * math.pi for k in range(-2, 3)]
            candidates = [v for v in candidates if self.limits[index, 0] <= v <= self.limits[index, 1]]
            if not candidates:
                raise ValueError("UR5 joint limit violation")
            q[index] = min(candidates, key=lambda v: abs(v - current))
            if abs(q[index] - current) > 0.25:
                raise ValueError(f"Discontinuous IK at {ARM_NAMES[index]}; replan before execution")
        p.setJointMotorControlArray(self.body, self.arm, p.POSITION_CONTROL,
                                    targetPositions=q.tolist(), forces=self.forces, positionGains=[0.15]*6)
        p.setJointMotorControl2(self.body, self.parent, p.POSITION_CONTROL,
                                targetPosition=self.angle_for_width(width), force=self.motor_torque,
                                maxVelocity=1.0)

    def mimic_error(self):
        parent = p.getJointState(self.body, self.parent)[0]
        return max(abs(p.getJointState(self.body, j)[0] - multiplier * parent)
                   for j, multiplier in self.mimic.items())

    def write_manifest(self, output):
        (output / "robot_model.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
