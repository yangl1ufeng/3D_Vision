"""Plan before moving; keep contact-driven execution separate from query geometry."""
import json
import math
import time

import numpy as np
import pybullet as p

from .pick_place import Demo
from .planning import CollisionScene, PlanningError, plan_rrt, validate_path


class PlannedDemo(Demo):
    def __init__(self, output, gui=False, no_grasp=False, robot_model='ur5_robotiq140', cube_size=.05, scenario='barrier'):
        self.scene = None
        self.plan_records = []
        self.scenario = scenario
        self.execution_checks = 0
        self.execution_collision = None
        super().__init__(output, gui, no_grasp, robot_model, cube_size)
        # The synthetic target width is known here. Leave 10 mm on either side
        # instead of unnecessarily approaching with the fully extended linkage.
        self.open_width = self.width = min(self.adapter.max_width, cube_size + .02)
        self.command([p.getJointState(self.robot, j)[0] for j in self.arm])
        self.tick(240)
        obstacles = []
        if scenario == 'barrier':
            obstacles = [('barrier', [.5, .11, .12], [.1, .008, .12])]
        elif scenario == 'blocked_goal':
            obstacles = [('goal_block', [.5, 0, .22], [.05, .05, .035])]
        for _, position, half in obstacles:
            self.box(position, half, 0, [.65, .15, .2, 1])
        self.scene = CollisionScene(self, obstacles)
        self.obstacles = obstacles

    def tick(self, count):
        if self.scene is None:
            return super().tick(count)
        for index in range(0, count, 4):
            super().tick(min(4, count-index))
            self.scene.configure(self.stage)
            q = [p.getJointState(self.robot, j)[0] for j in self.arm]
            reason = self.scene.collision(q, margin=-.001)
            self.execution_checks += 1
            if reason:
                self.execution_collision = reason
                raise PlanningError(f'execution_collision: {reason}')

    def move(self, stage, target, duration=2):
        self.stage = stage
        self.scene.configure(stage)
        if stage == 'pregrasp' and self.scenario == 'unreachable':
            target = [2, 0, .22]
        start = np.array([p.getJointState(self.robot, j)[0] for j in self.arm])
        began = time.perf_counter()
        record = {'stage': stage, 'target_m': target, 'status': 'planning', 'execution_started': False}
        record.update(start_joint_positions=[p.getJointState(self.robot, i)[0] for i in self.scene.infos],
                      target_pose_at_plan_start=p.getBasePositionAndOrientation(self.cube),
                      attachment_tcp=self.scene.attachment if self.scene.attached else None,
                      allow_pad_contact=self.scene.allow_pads, allow_table_support=self.scene.allow_support)
        self.plan_records.append(record)
        try:
            goal = self.scene.ik(target, self.orientation, start)
            if stage == 'lower':
                # Account for the measured grasp offset/slip and cube orientation.
                # Stop just above support, then release; do not plan a penetrating
                # payload pose merely because the original grasp TCP was lower.
                bottom = self.scene.b.getAABB(self.scene.target)[0][2]
                record['requested_target_m'] = list(target)
                target = list(target)
                target[2] += getattr(self, 'table_top', 0.) + .002 - bottom
                record['target_m'] = target
                goal = self.scene.ik(target, self.orientation, start)
            if stage in {'approach', 'lift', 'lower', 'retreat'}:
                self.scene.set_q(start)
                origin = np.array(self.scene.b.getLinkState(self.scene.robot, self.tcp, computeForwardKinematics=True)[4])
                count = max(1, math.ceil(float(np.linalg.norm(np.array(target)-origin))/.005))
                path = [start]
                for t in np.linspace(0, 1, count+1)[1:]:
                    q = self.scene.ik((origin*(1-t)+np.array(target)*t).tolist(), self.orientation, path[-1])
                    if np.max(np.abs(q-path[-1])) > .25:
                        raise PlanningError('cartesian_ik_discontinuity')
                    path.append(q)
                info = {'method': 'cartesian', 'direct_blocked': None}
            else:
                path, info = plan_rrt(self.scene, start, goal)
            validate_path(self.scene, path)
            record.update(info, status='validated', joint_path=[q.tolist() for q in path],
                          planning_seconds=time.perf_counter()-began,
                          payload_included=self.scene.attached, fine_resolution_rad=.005)
        except Exception as exc:
            record.update(status='rejected', reason=str(exc), collision=self.scene.last_reason,
                          planning_seconds=time.perf_counter()-began)
            raise
        # Physics stayed still during planning; exclude that time from GUI pacing.
        self.wall_start += time.perf_counter()-began
        record['execution_started'] = True
        try:
            for a, b in zip(path, path[1:]):
                count = max(1, math.ceil(float(np.max(np.abs(b-a)))/.006))
                for t in np.linspace(0, 1, count+1)[1:]:
                    self.command(a*(1-t)+b*t)
                    self.tick(4)
            self.tick(120)
            state = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)
            error = float(np.linalg.norm(np.array(target)-state[4]))
            quat = np.array(state[5]); quat /= np.linalg.norm(quat)
            angle = 2*math.acos(float(np.clip(abs(np.dot(quat, self.orientation)), 0, 1)))
            self.stages.append({'stage': stage, 'tcp_error_m': error, 'orientation_error_rad': angle})
            if error > .015 or angle > math.radians(3):
                raise PlanningError('execution_tracking_error')
            record['status'] = 'executed'
            print(f'{stage}: {info["method"]}, {len(path)} waypoints, error={error:.5f} m', flush=True)
        except Exception as exc:
            record.update(status='execution_failed', reason=str(exc))
            raise

    def save_planning(self):
        data = {'scenario': self.scenario, 'obstacles': self.obstacles, 'plans': self.plan_records,
                'collision_calls': self.scene.calls, 'execution_checks': self.execution_checks,
                'execution_collision': self.execution_collision,
                'planning_clearance_m': .004, 'execution_penetration_limit_m': .001,
                'search_resolution_rad': .04, 'validation_resolution_rad': .005,
                'execution_check_period_steps': 4, 'seed': 7,
                'pregrasp_width_m': self.open_width,
                'self_collision_exclusions': 'same rigid group, adjacent groups, same-side four-bar closure pairs',
                'payload': 'fixed TCP-relative collision proxy only in planner; no physical object constraint',
                'limitation': 'discrete collision sampling; not a continuous collision guarantee'}
        (self.output/'planning.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        return data
