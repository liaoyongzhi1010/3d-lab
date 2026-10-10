from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if parent.name == "3d-reconstruction"
)
EXPECTED_TOP_LEVEL = {"README.md", "data", "code", "onepage", "papers"}
REQUIRED_EXPERIMENTS = {
    "selective-injection.md",
    "full-image-quality.md",
    "acid-boundary.md",
    "tiny-student.md",
    "reliability-audit.md",
    "cyclefusion-pending.md",
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.paths: list[str] = []
        self.images_without_alt: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"] or "")
        if tag == "a" and attributes.get("href"):
            self.paths.append(attributes["href"] or "")
        if tag == "img":
            src = attributes.get("src") or ""
            if src:
                self.paths.append(src)
            if not (attributes.get("alt") or "").strip():
                self.images_without_alt.append(src or "<missing src>")


def markdown_targets(path: Path) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8"))


class WorkspaceLayoutTests(unittest.TestCase):
    def test_project_has_only_required_top_level_entries(self) -> None:
        self.assertEqual(
            {path.name for path in PROJECT_ROOT.iterdir()},
            EXPECTED_TOP_LEVEL,
        )

    def test_release_evidence_remains_intact(self) -> None:
        release = PROJECT_ROOT / "data" / "results" / "github_release"
        with (
            release
            / "paper1_selective_generation"
            / "results"
            / "E-203_bootstrap_N166.json"
        ).open(encoding="utf-8") as handle:
            paper1 = json.load(handle)
        with (
            release
            / "paper3_reliability_learning"
            / "results"
            / "E-224_camera_only_observable.json"
        ).open(encoding="utf-8") as handle:
            paper3 = json.load(handle)
        self.assertEqual(sum(bucket["n"] for bucket in paper1["buckets"].values()), 166)
        self.assertEqual(paper3["frame_count"], 2898)
        self.assertTrue(
            (
                release
                / "paper2_gate_aware_distillation"
                / "checkpoints"
                / "student.pt"
            ).is_file()
        )

    def test_onepage_experiment_records_are_complete(self) -> None:
        experiments = PROJECT_ROOT / "onepage" / "experiments"
        records = {path.name: path for path in experiments.glob("*.md")}
        self.assertEqual(set(records), REQUIRED_EXPERIMENTS)
        required_sections = {
            "## 目的",
            "## 数据与代码版本",
            "## 配置",
            "## 运行方式",
            "## 指标",
            "## 效果图",
            "## 观察",
            "## 失败点",
            "## 结论",
        }
        for name, path in records.items():
            text = path.read_text(encoding="utf-8")
            self.assertTrue(required_sections.issubset(set(text.splitlines())), name)
            self.assertIn("../", text, name)
        cycle = records["cyclefusion-pending.md"].read_text(encoding="utf-8")
        self.assertIn("证据状态：原始产物待恢复", cycle)

    def test_onepage_sections_and_local_links(self) -> None:
        index = PROJECT_ROOT / "onepage" / "index.html"
        parser = LinkParser()
        parser.feed(index.read_text(encoding="utf-8"))
        self.assertEqual(parser.images_without_alt, [])
        self.assertTrue(
            {
                "overview",
                "data",
                "method",
                "tests",
                "results",
                "failures",
                "next",
            }.issubset(set(parser.ids))
        )
        missing: list[str] = []
        for target in parser.paths:
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            if not (index.parent / unquote(parsed.path)).resolve().exists():
                missing.append(target)
        self.assertEqual(missing, [])

    def test_onepage_headline_metrics_match_json(self) -> None:
        page = (PROJECT_ROOT / "onepage" / "index.html").read_text(encoding="utf-8")
        release = PROJECT_ROOT / "data" / "results" / "github_release"
        e203 = json.loads(
            (
                release
                / "paper1_selective_generation/results/E-203_bootstrap_N166.json"
            ).read_text()
        )
        e210 = json.loads(
            (
                release
                / "paper1_selective_generation/results/E-210_standard_metrics.json"
            ).read_text()
        )
        e215 = json.loads(
            (
                release / "paper1_selective_generation/results/E-215_acid_metrics.json"
            ).read_text()
        )
        e211 = json.loads(
            (
                release
                / "paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json"
            ).read_text()
        )
        e224 = json.loads(
            (
                release
                / "paper3_reliability_learning/results/E-224_camera_only_observable.json"
            ).read_text()
        )
        expected = [
            f"{e203['buckets']['hard(<12)']['mean']:.3f}",
            f"{e203['buckets']['easy(>=20)']['mean']:.3f}",
            f"{e210['ours']['lpips']:.3f}",
            f"{e215['mechanism']['mean_gain']:.3f}",
            f"{e211['student']['time_s']:.3f}",
            f"{e224['metrics']['roc_auc']:.3f}",
            f"{e224['policy_at_0.5']['worst_gain_db']:.3f}",
        ]
        for value in expected:
            self.assertIn(value, page)

    def test_project_markdown_links_resolve(self) -> None:
        missing: list[str] = []
        files = [PROJECT_ROOT / "README.md"]
        files.extend((PROJECT_ROOT / "onepage" / "experiments").glob("*.md"))
        files.extend(
            [
                PROJECT_ROOT / "code" / "README.md",
                PROJECT_ROOT / "data" / "README.md",
                PROJECT_ROOT / "papers" / "README.md",
            ]
        )
        for path in files:
            for target in markdown_targets(path):
                parsed = urlparse(target)
                if parsed.scheme or parsed.netloc or target.startswith("#"):
                    continue
                if not (path.parent / unquote(parsed.path)).resolve().exists():
                    missing.append(f"{path.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_eval_suite_has_no_broken_orchestrator(self) -> None:
        suite = PROJECT_ROOT / "code" / "eval-suite"
        self.assertFalse((suite / "run_all.sh").exists())
        for path in [suite / "README.md", suite / "docs" / "USAGE.md"]:
            self.assertNotIn("bash run_all.sh", path.read_text(encoding="utf-8"))

    def test_eval_patches_parse_on_python_39(self) -> None:
        import ast

        patches = PROJECT_ROOT / "code" / "eval-suite" / "patches"
        for path in [patches / "catsplat_evaluate.py", patches / "flash3d_evaluate.py"]:
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))

    def test_obsolete_duplicate_papers_are_removed(self) -> None:
        papers = PROJECT_ROOT / "papers"
        for name in ["paper1_latex", "paper2_latex", "paper3_latex"]:
            self.assertFalse((papers / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
