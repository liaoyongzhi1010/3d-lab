from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TOP_LEVEL = {"README.md", "data", "code", "onepage", "papers"}
REQUIRED_EXPERIMENTS = {
    "dpc-main-n100.md",
    "epsilon-lambda-ablation.md",
    "depth-transfer-n40.md",
    "defense-n20.md",
    "region-analysis-n80.md",
    "vggt-transfer-n30.md",
}
REQUIRED_QUALITATIVE_VARIANTS = {
    "perceptual/scene7_perceptual.png",
    "perceptual/scene12_perceptual.png",
    "v2/scene0_dpc_v2.png",
    "v2/scene3_dpc_v2.png",
    "v2/scene7_dpc_v2.png",
    "v2/scene12_dpc_v2.png",
    "v2/scene20_dpc_v2.png",
    "v3/scene7_v3.png",
    "v3/scene12_v3.png",
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


class RepositoryLayoutTests(unittest.TestCase):
    def test_project_has_only_reviewable_top_level_entries(self) -> None:
        entries = {path.name for path in PROJECT_ROOT.iterdir()}
        self.assertEqual(entries, EXPECTED_TOP_LEVEL)

    def test_core_evidence_and_experiment_records_are_preserved(self) -> None:
        results = PROJECT_ROOT / "data" / "results"
        with (results / "main_dpc_n100.json").open(encoding="utf-8") as handle:
            main = json.load(handle)
        self.assertEqual(main["n_scenes"], 100)
        self.assertEqual(main["dpc_cfg"]["method"], "dpc")
        self.assertFalse((results / "BB-002-vggt-transfer.json").exists())
        self.assertTrue((results / "invalid" / "BB-002-vggt-transfer.json").exists())

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
            self.assertNotIn("../../../data/results", text, name)

    def test_qualitative_variants_are_archived(self) -> None:
        root = PROJECT_ROOT / "data" / "qualitative" / "archive"
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*.png")}
        self.assertEqual(actual, REQUIRED_QUALITATIVE_VARIANTS)

    def test_readme_serves_project_root_for_onepage_links(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--directory 3d-reconstruction-security", readme)
        self.assertIn("http://127.0.0.1:8765/onepage/", readme)

    def test_gpu_experiment_commands_use_explicit_split(self) -> None:
        experiments = PROJECT_ROOT / "onepage" / "experiments"
        for name in [
            "dpc-main-n100.md",
            "epsilon-lambda-ablation.md",
            "depth-transfer-n40.md",
            "defense-n20.md",
            "region-analysis-n80.md",
        ]:
            text = (experiments / name).read_text(encoding="utf-8")
            self.assertIn(
                "--split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt",
                text,
                name,
            )

    def test_summary_reads_relocated_result_data(self) -> None:
        completed = subprocess.run(
            [sys.executable, "attacks/summarize_results.py"],
            cwd=PROJECT_ROOT / "code" / "flash3d-attack",
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("MAIN: DPC eps=8/255 (n=100)", completed.stdout)
        self.assertIn(
            "Depth-transfer via shared UniDepth prior (n=40)", completed.stdout
        )
        self.assertIn("Defense detector (n=20)", completed.stdout)

    def test_paper_abstract_matches_released_main_result(self) -> None:
        paper = (PROJECT_ROOT / "papers" / "paper_dpc.md").read_text(encoding="utf-8")
        abstract = paper.split("## 1. Introduction", maxsplit=1)[0]
        self.assertRegex(abstract, re.compile(r"4\.8.+5\.2.+5\.0 dB", re.DOTALL))
        self.assertIn("2.6 dB", abstract)

    def test_invalid_vggt_result_is_not_presented_as_evidence(self) -> None:
        record = (
            PROJECT_ROOT / "onepage" / "experiments" / "vggt-transfer-n30.md"
        ).read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "onepage" / "index.html").read_text(encoding="utf-8")
        paper = (PROJECT_ROOT / "papers" / "paper_dpc.md").read_text(encoding="utf-8")
        self.assertIn("审计状态：无效", record)
        self.assertIn("VGGT 迁移（审计未通过）", page)
        self.assertNotIn("transfers across feed-forward 3DGS architectures", paper)
        self.assertNotIn("stealthy against training-free defenses", paper)
        self.assertNotIn("full-3204 MINE table", paper)
        for overclaim in [
            "keeps the source-view reconstruction nearly intact",
            "imperceptible budget",
            "the geometry-corruption signature",
            "geometry-level corruption fingerprint",
        ]:
            self.assertNotIn(overclaim, paper)
        method = (PROJECT_ROOT / "code" / "flash3d-attack" / "METHOD_DPC.md").read_text(
            encoding="utf-8"
        )
        for overclaim in [
            "nearly invisible in the source reconstruction",
            "source view looks fine to a human/defender",
            "direct evidence it corrupts parallax-dependent geometry",
        ]:
            self.assertNotIn(overclaim, method)

    def test_vggt_rerun_uses_current_scene_and_records_seed(self) -> None:
        source = (
            PROJECT_ROOT / "code" / "flash3d-attack" / "eval_vggt_transfer.py"
        ).read_text(encoding="utf-8")
        loop = source.split("for inputs in loader:", maxsplit=1)[1]
        self.assertIn("local = dict(inputs)", loop)
        self.assertNotIn("local = dict(adapter.example_inputs)", loop)
        self.assertIn('parser.add_argument("--seed", type=int, default=0)', source)
        self.assertIn("torch.manual_seed(args.seed)", source)
        self.assertIn("np.random.seed(args.seed)", source)
        self.assertIn("seed=args.seed", source)
        self.assertIn('"seed": args.seed', source)
        record = (
            PROJECT_ROOT / "onepage" / "experiments" / "vggt-transfer-n30.md"
        ).read_text(encoding="utf-8")
        self.assertIn("cd 3d-reconstruction-security/code/flash3d-attack", record)
        self.assertNotIn("--out ../../data/results/invalid/", record)
        self.assertIn("--out /tmp/BB-002-vggt-transfer-rerun.json", record)
        runner = (
            PROJECT_ROOT / "code" / "flash3d-attack" / "run_bb_vggt_transfer.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--seed 0", runner)
        self.assertIn("/tmp/BB-002-vggt-transfer-rerun.json", runner)

    def test_onepage_local_links_and_images_resolve(self) -> None:
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
            resolved = (index.parent / unquote(parsed.path)).resolve()
            if not resolved.exists():
                missing.append(target)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
