"""Small GPU check against analytically known transformed surface positions."""
import numpy as np
from offline_core.config import load_config
from offline_core.runtime import initialize
from offline_core.collision import CollisionChecker

initialize(load_config())
checker = CollisionChecker.__new__(CollisionChecker)
checker.samples_local_m = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
checker.cfg = {'threshold_m': .02, 'batch_size': 1}
poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
poses[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
poses[0, :3, 3] = [2, 3, 0]  # First surface point is exactly (2,4,0).
poses[1, :3, 3] = [10, 10, 0]  # Far away.
poses[2, :3, 3] = [1.01, 4, 0]  # First surface point 1 cm away from obstacle.
actual = checker.check(np.array([[2, 4, 0]], dtype=np.float32), poses)
np.testing.assert_array_equal(actual, [False, True, False])
print('GPU collision geometry: rotated/translated collision, far clear, 1 cm collision passed.')
