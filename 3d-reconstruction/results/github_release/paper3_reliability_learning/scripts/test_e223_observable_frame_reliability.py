import importlib.util
import pathlib
import unittest

import numpy as np


SCRIPT = pathlib.Path(__file__).with_name("e223_observable_frame_reliability.py")
SPEC = importlib.util.spec_from_file_location("e223", SCRIPT)
e223 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e223)


class OracleVisibilityDiagnosticTest(unittest.TestCase):
    def test_predictor_features_are_invariant_to_gt_and_teacher_arrays_only(self):
        rng = np.random.default_rng(7)
        shape = (3, 3, 12, 12)
        baseline = rng.random(shape, dtype=np.float32)
        f3d = rng.random(shape, dtype=np.float32)
        visibility = rng.random((3, 6, 6)) > 0.35
        cameras = np.repeat(np.eye(4)[None], 3, axis=0)
        cameras[1, 0, 3] = 0.1
        cameras[2, 1, 3] = 0.2
        gt = rng.random(shape, dtype=np.float32)
        teacher = rng.random(shape, dtype=np.float32)

        rows_a = e223.build_frame_rows(
            "scene-a",
            baseline,
            f3d,
            visibility,
            cameras,
            gt,
            teacher,
            min_mask_pixels=1,
        )
        rows_b = e223.build_frame_rows(
            "scene-a",
            baseline,
            f3d,
            visibility,
            cameras,
            np.zeros_like(gt),
            np.ones_like(teacher),
            min_mask_pixels=1,
        )

        self.assertEqual(len(rows_a), len(rows_b))
        features_a = np.asarray(
            [[row[name] for name in e223.FEATURE_NAMES] for row in rows_a]
        )
        features_b = np.asarray(
            [[row[name] for name in e223.FEATURE_NAMES] for row in rows_b]
        )
        np.testing.assert_array_equal(features_a, features_b)

    def test_visibility_changes_predictor_features_and_crosses_boundary(self):
        rng = np.random.default_rng(11)
        baseline = rng.random((3, 16, 16), dtype=np.float32)
        f3d = rng.random((3, 16, 16), dtype=np.float32)
        cameras = np.repeat(np.eye(4)[None], 2, axis=0)
        visibility_a = np.zeros((16, 16), dtype=bool)
        visibility_a[:, :8] = True
        visibility_b = ~visibility_a

        features_a = e223.diagnostic_features(
            baseline, f3d, visibility_a, cameras[1], cameras[0]
        )
        features_b = e223.diagnostic_features(
            baseline, f3d, visibility_b, cameras[1], cameras[0]
        )

        self.assertNotEqual(features_a, features_b)
        self.assertEqual(
            e223.EVIDENCE_TIER, "RGB disagreement + oracle-visibility diagnostic"
        )
        self.assertIn("complete target clip", e223.VISIBILITY_PROVENANCE)
        self.assertNotIn("observable", e223.EVIDENCE_TIER.lower())

    def test_grouped_folds_never_overlap_scenes(self):
        scene_ids = np.asarray(["a", "a", "b", "b", "c"])
        folds = list(e223.leave_one_scene_out(scene_ids))

        self.assertEqual(len(folds), 3)
        for held_out, train_idx, test_idx in folds:
            train_scenes = set(scene_ids[train_idx])
            test_scenes = set(scene_ids[test_idx])
            self.assertEqual(test_scenes, {held_out})
            self.assertTrue(train_scenes.isdisjoint(test_scenes))
            self.assertEqual(len(train_idx) + len(test_idx), len(scene_ids))


if __name__ == "__main__":
    unittest.main()
