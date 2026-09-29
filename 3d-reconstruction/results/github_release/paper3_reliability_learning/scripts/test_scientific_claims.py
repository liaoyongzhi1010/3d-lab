import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScientificClaimIntegrityTest(unittest.TestCase):
    def test_camera_only_predictor_has_strict_observable_whitelist(self):
        module = load_script("e224_camera_only_observable.py")
        self.assertEqual(module.FEATURE_NAMES, ("cam_trans", "cam_rot_deg"))
        self.assertTrue(
            module.FORBIDDEN_PREDICTOR_FIELDS.isdisjoint(module.FEATURE_NAMES)
        )

    def test_quality_probe_scripts_do_not_claim_observability_or_deployment(self):
        for name in (
            "e040_reliability_net.py",
            "e142_build_combined.py",
            "e144_build_frame_dataset.py",
            "e145b_frame_net_full.py",
            "e149_augment_features.py",
            "e208_pixel_reliability.py",
            "e221_granularity_figure.py",
            "e222_routing_strip.py",
        ):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8").lower()
            self.assertIn("quality probe", text, name)
            self.assertNotIn("features (all test-time observable)", text, name)
            self.assertNotIn("deployable", text, name)

    def test_scene_and_frame_vis_gap_semantics_are_distinguished(self):
        scene_text = (ROOT / "scripts/e142_build_combined.py").read_text(
            encoding="utf-8"
        )
        frame_text = (ROOT / "scripts/e144_build_frame_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("scene `vis_gap = f3d_vis - base_vis`", scene_text)
        self.assertIn("frame `vis_gap = f3d_vis - base_inv`", scene_text)
        self.assertIn("f3d_vis - base_inv", frame_text)

    def test_population_caveats_cover_full_and_selected_populations(self):
        paths = (
            ROOT / "README.md",
            ROOT / "PROTOCOL.md",
            ROOT / "paper/main.tex",
            ROOT / "paper/paper3_full_writeup.md",
            ROOT / "scripts/e208_pixel_reliability.py",
            ROOT / "scripts/e221_granularity_figure.py",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("74 scenes", text, path)
            self.assertIn("21 selected high-gain scenes", text, path)

    def test_public_materials_define_four_evidence_tiers(self):
        paths = (
            ROOT / "README.md",
            ROOT / "PROTOCOL.md",
            ROOT / "paper/main.tex",
            ROOT / "paper/paper3_full_writeup.md",
            ROOT / "paper/README.md",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("e-224", text, path)
            self.assertIn("oracle true gain", text, path)
            self.assertIn("gt-quality-probe diagnostic", text, path)
            self.assertIn("camera-only observable", text, path)
            self.assertIn("rgb disagreement + oracle-visibility diagnostic", text, path)

    def test_e223_is_never_described_as_observable(self):
        paths = (
            tuple(ROOT.rglob("*.md"))
            + tuple(ROOT.rglob("*.tex"))
            + tuple(
                path for path in ROOT.rglob("*.py") if not path.name.startswith("test_")
            )
        )
        forbidden = ("observable rgb", "rgb observable", "observable model at 0.5")
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase!r} in {path}")
        result = json.loads(
            (ROOT / "results/E-223_observable_frame_reliability.json").read_text()
        )
        self.assertEqual(
            result["evidence_tier"],
            "RGB disagreement + oracle-visibility diagnostic",
        )
        self.assertIn("complete target clip", result["visibility_provenance"])

    def test_e223_manifest_matches_released_scene_counts(self):
        result = json.loads(
            (ROOT / "results/E-223_observable_frame_reliability.json").read_text()
        )
        manifest = json.loads(
            (ROOT / "results/E-223_selected_scene_manifest.json").read_text()
        )
        self.assertEqual(manifest["selection"], "selected, not population-aligned")
        self.assertIn("teacher_npy", manifest["source_directory_identifier"])
        self.assertEqual(manifest["scene_frame_counts"], result["scene_frame_counts"])
        self.assertEqual(len(manifest["scene_ids"]), 21)
        self.assertEqual(manifest["scene_ids"], sorted(result["scene_frame_counts"]))
        self.assertEqual(manifest["frame_count"], 910)
        self.assertRegex(manifest["scene_counts_sha256"], r"^[0-9a-f]{64}$")

    def test_metric_names_distinguish_ap_from_trapezoidal_pr_auc(self):
        e223 = json.loads(
            (ROOT / "results/E-223_observable_frame_reliability.json").read_text()
        )
        e224 = json.loads(
            (ROOT / "results/E-224_camera_only_observable.json").read_text()
        )
        self.assertIn("average_precision", e223["metrics"])
        self.assertNotIn("pr_auc", e223["metrics"])
        self.assertIn("pr_auc_trapezoidal", e224["metrics"])
        for path in (ROOT / "README.md", ROOT / "PROTOCOL.md", ROOT / "paper/main.tex"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("average precision", text, path)
            self.assertIn("trapezoidal pr-auc", text, path)

    def test_reproduction_commands_write_generated_outputs_to_tmp(self):
        for path in (ROOT / "README.md", ROOT / "PROTOCOL.md"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("python3 scripts/e22"):
                    self.assertIn("/tmp/", line, f"tracked output in {path}: {line}")

    def test_scripts_use_current_numpy_and_pillow_apis(self):
        for path in (ROOT / "scripts").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("np." + "trapz", text, path)
            self.assertNotIn("Image." + "NEAREST", text, path)

    def test_public_narrative_forbids_scheduler_and_matched_budget_claims(self):
        paths = (
            ROOT / "README.md",
            ROOT / "PROTOCOL.md",
            ROOT / "paper/main.tex",
            ROOT / "paper/paper3_full_writeup.md",
            ROOT / "paper/README.md",
        )
        forbidden = (
            "deployable scheduler",
            "matched-budget",
            "matched budget",
            "framenet (ours)",
            "frame is the sweet spot",
            "frames the learnable sweet spot",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase!r} in {path}")
            self.assertIn("diagnostic", text, path)
            self.assertIn("observable", text, path)

    def test_released_vis_gap_definition_is_documented_exactly(self):
        expected = "f3d_vis - base_inv"
        for path in (
            ROOT / "README.md",
            ROOT / "PROTOCOL.md",
            ROOT / "paper/paper3_full_writeup.md",
        ):
            self.assertIn(expected, path.read_text(encoding="utf-8"), path)


if __name__ == "__main__":
    unittest.main()
