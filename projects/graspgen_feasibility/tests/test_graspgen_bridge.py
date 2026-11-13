import unittest
import numpy as np
import pybullet as p
from simulation.ur5_robotiq import UR5Robotiq140
from simulation.graspgen_bridge import verify_mapping, quaternion, T_G_TCP


class FrameBridgeTests(unittest.TestCase):
    def test_official_pad_mesh_matches_simulation_fk(self):
        client = p.connect(p.DIRECT)
        try:
            adapter = UR5Robotiq140()
            report = verify_mapping(adapter)
            self.assertTrue(report['passed'])
            self.assertLess(max(report['pad_center_errors_m']), .0002)
            # Omitting the 90-degree correction must fail the same landmarks.
            corrected = np.asarray(report['transformed_simulation_open_pad_centers_m'])
            raw = corrected @ T_G_TCP[:3, :3]
            official = np.asarray(report['official_mesh_open_pad_centers_m'])
            self.assertGreater(np.linalg.norm(raw - official, axis=1).min(), .05)
        finally:
            p.disconnect(client)

    def test_matrix_to_quaternion_preserves_candidate_orientation(self):
        for euler in ([2.9, .7, -2.3], [0, 1.57079632679, .8], [3.14, 0, 3.14]):
            expected = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(euler))).reshape(3, 3)
            actual = np.array(p.getMatrixFromQuaternion(quaternion(expected))).reshape(3, 3)
            np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
