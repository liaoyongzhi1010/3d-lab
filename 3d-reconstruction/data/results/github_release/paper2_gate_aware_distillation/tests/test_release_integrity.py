import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudentDataTests(unittest.TestCase):
    def test_missing_baseline_fails_with_expected_path(self):
        student = load_script("e031b_student_fullnpy.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir = root / "teacher"
            f3d_dir = root / "f3d"
            data_dir = root / "data"
            sid = "test_scene"
            (data_dir / sid).mkdir(parents=True)
            teacher_dir.mkdir()
            f3d_dir.mkdir()
            rgb = np.zeros((2, 3, 8, 8), dtype=np.float32)
            vis = np.zeros((2, 8, 8), dtype=np.float32)
            np.save(teacher_dir / f"adaptive2_{sid}.npy", rgb)
            np.save(teacher_dir / f"gt_{sid}.npy", rgb)
            np.save(f3d_dir / f"f3d_{sid}.npy", rgb)
            np.save(data_dir / sid / "visibility.npy", vis)
            expected = teacher_dir / f"baseline_{sid}.npy"

            with self.assertRaisesRegex(FileNotFoundError, str(expected)):
                student.load_scene(
                    sid, str(teacher_dir), str(f3d_dir), str(data_dir), 8
                )


class TrainingProvenanceTests(unittest.TestCase):
    def test_training_manifest_discloses_reproduction_boundary(self):
        manifest = (ROOT / "TRAINING_MANIFEST.md").read_text()
        for required in (
            "--base_mode f3d",
            "--gate_aware",
            "--frame_dataset <FRAME_LABEL_JSON>",
            "--holdout <FOUR_COMMA_SEPARATED_SCENE_IDS>",
            "4 scenes and 161 frames",
            "external teacher arrays",
            "original holdout manifest",
            "not included",
            "template, not a self-contained exact reproduction command",
        ):
            self.assertIn(required, manifest)

        for path in (
            ROOT / "README.md",
            ROOT / "MODEL_CARD.md",
            ROOT / "paper/README.md",
        ):
            text = path.read_text()
            self.assertIn("checkpoint evaluation is reproducible", text.lower())
            self.assertIn("exact retraining", text.lower())
            self.assertIn("TRAINING_MANIFEST.md", text)

        checker = load_script("check_claim_integrity.py")
        self.assertEqual(checker.check_training_manifest(manifest), [])


class ClaimIntegrityTests(unittest.TestCase):
    def test_forbidden_claims_are_detected_across_line_breaks(self):
        checker = load_script("check_claim_integrity.py")
        failures = checker.scan_forbidden_claims(
            "A learned\nrouter chooses the student at test time.", "synthetic.md"
        )
        self.assertTrue(any("router claim" in failure for failure in failures))

    def test_stale_e211_value_is_rejected(self):
        checker = load_script("check_claim_integrity.py")
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "n_frames": 759,
                        "scenes": 21,
                        "lpips_backbone": "vgg",
                        "size": 256,
                        "student": {
                            "lpips": 0.283921458,
                            "fid": 33.61412,
                            "time_s": 0.087,
                        },
                    }
                )
            )
            text = "Student LPIPS is 0.999 and FID is 33.6 at 0.087 s on 759 frames."
            failures = checker.check_e211_headlines(text, metrics_path, "synthetic.md")
        self.assertTrue(any("student LPIPS" in failure for failure in failures))

    def test_checkpoint_headlines_are_derived_from_checkpoint(self):
        checker = load_script("check_claim_integrity.py")
        facts = checker.checkpoint_facts(ROOT / "checkpoints" / "student.pt")
        self.assertEqual(facts["parameters"], 46_371)
        self.assertEqual(
            facts["sha256"],
            "1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2",
        )


if __name__ == "__main__":
    unittest.main()
