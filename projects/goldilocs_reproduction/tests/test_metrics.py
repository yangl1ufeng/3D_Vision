import unittest
from goldilocs_local.metrics import detection_ap


class MetricsTests(unittest.TestCase):
    def test_missed_image_remains_in_denominator(self):
        g = {'a': [[0, 0, 10, 10]], 'b': [[0, 0, 10, 10]]}
        p = [{'image_id': 'a', 'bbox': [0, 0, 10, 10], 'score': 1.}]
        self.assertEqual(detection_ap(g, p)['ap'], .5)

    def test_duplicate_is_false_positive(self):
        g = {'a': [[0, 0, 10, 10]], 'b': [[0, 0, 10, 10]]}
        p = [{'image_id': i, 'bbox': [0, 0, 10, 10], 'score': s}
             for i, s in [('a', 1.), ('a', .9), ('b', .8)]]
        r = detection_ap(g, p)
        self.assertAlmostEqual(r['ap'], 5/6)
        self.assertEqual(r['false_positives'], 1)

    def test_empty_predictions_and_no_ground_truth(self):
        self.assertEqual(detection_ap({'a': [[0, 0, 1, 1]]}, [])['ap'], 0.)
        self.assertIsNone(detection_ap({'a': []}, [])['ap'])


if __name__ == '__main__':
    unittest.main()
