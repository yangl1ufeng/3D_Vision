import json
from pathlib import Path
import tempfile
import unittest
from run_rc3d_revision import completed_sample_matches


class ResumeTests(unittest.TestCase):
    def fixture(self, folder):
        (folder/'run.json').write_text(json.dumps({'status':'complete',
            'sam_checkpoint_sha256':'weights-a', 'input_sha256':['image-a','image-b']}))
        for view in ('view_0','view_1'):
            (folder/view).mkdir()
            for name in ('prediction.json','labels.png','object_masks.npz'):
                (folder/view/name).touch()

    def test_fresh_and_completed_samples_distinguished(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.assertFalse(completed_sample_matches(folder,'weights-a',['image-a','image-b']))
            self.fixture(folder)
            self.assertTrue(completed_sample_matches(folder,'weights-a',['image-a','image-b']))

    def test_changed_checkpoint_or_inputs_cannot_mix_into_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            for weights, inputs in [('weights-b',['image-a','image-b']),
                                    ('weights-a',['image-a','changed-image'])]:
                with self.subTest(weights=weights, inputs=inputs), self.assertRaises(ValueError):
                    completed_sample_matches(folder,weights,inputs)

    def test_completion_marker_cannot_hide_missing_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture(folder)
            (folder/'view_0/labels.png').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete completed sample'):
                completed_sample_matches(folder,'weights-a',['image-a','image-b'])
