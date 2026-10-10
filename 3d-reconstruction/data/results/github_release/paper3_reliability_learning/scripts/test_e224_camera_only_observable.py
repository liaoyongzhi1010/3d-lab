import importlib.util
import json
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).with_name("e224_camera_only_observable.py")
SPEC = importlib.util.spec_from_file_location("e224", SCRIPT)
e224 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e224)


class CameraOnlyObservableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "results/E-149_frames_b5678_aug.json").open() as handle:
            cls.rows = json.load(handle)

    def test_feature_whitelist_contains_only_camera_fields(self):
        self.assertEqual(e224.FEATURE_NAMES, ("cam_trans", "cam_rot_deg"))
        forbidden = {
            "base_inv",
            "teacher_inv",
            "delta",
            "label",
            "f3d_vis",
            "f3d_inv",
            "vis_gap",
            "vis_frac",
            "disocc_frac",
            "f3d_vi_ratio",
            "f3d_vi_prod",
        }
        self.assertTrue(forbidden.isdisjoint(e224.FEATURE_NAMES))
        features = e224.feature_matrix(self.rows[:3])
        np.testing.assert_array_equal(
            features,
            [[row["cam_trans"], row["cam_rot_deg"]] for row in self.rows[:3]],
        )

    def test_grouped_folds_hold_out_each_scene_exactly_once(self):
        groups = np.asarray([row["sid"] for row in self.rows])
        folds = list(e224.leave_one_scene_out(groups))
        self.assertEqual(len(folds), 74)
        covered = []
        for held_out, train_idx, test_idx in folds:
            self.assertEqual(set(groups[test_idx]), {held_out})
            self.assertTrue(set(groups[train_idx]).isdisjoint(groups[test_idx]))
            covered.extend(test_idx.tolist())
        self.assertEqual(sorted(covered), list(range(2898)))

    def test_reproduces_released_camera_only_metrics(self):
        result = e224.run(self.rows)
        metrics = result["metrics"]
        policy = result["policy_at_0.5"]
        self.assertEqual(result["frame_count"], 2898)
        self.assertEqual(result["scene_count"], 74)
        self.assertAlmostEqual(metrics["accuracy"], 0.6118012422360248)
        self.assertAlmostEqual(metrics["roc_auc"], 0.6495999059112509)
        self.assertAlmostEqual(metrics["pr_auc_trapezoidal"], 0.6631235562612292)
        self.assertAlmostEqual(policy["mean_gain_db"], 0.71524745871212)
        self.assertAlmostEqual(policy["worst_gain_db"], -23.094440460205078)
        self.assertEqual(policy["teacher_calls"], 2028)
        self.assertEqual(policy["calls_saved"], 870)
        self.assertEqual(len(result["threshold_frontier"]), 31)


if __name__ == "__main__":
    unittest.main()
