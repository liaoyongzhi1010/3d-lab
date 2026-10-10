#!/usr/bin/env python3
"""Validate the self-contained public release without external datasets."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
import yaml
from generate_manifests import generated_manifest_text
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2"

RELEASED_JSONS = [
    "paper1_selective_generation/results/E-142_combined_N169.json",
    "paper1_selective_generation/results/E-203_bootstrap_N166.json",
    "paper1_selective_generation/results/E-207_baseline_ablation_N166.json",
    "paper1_selective_generation/results/E-209_fullimage_psnr.json",
    "paper1_selective_generation/results/E-210_standard_metrics.json",
    "paper1_selective_generation/results/E-216_gate_sensitivity.json",
    "paper1_selective_generation/results/E-215_acid_metrics.json",
    "paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json",
    "paper3_reliability_learning/results/E-040_reliability_N166.json",
    "paper3_reliability_learning/results/E-145b_N2898_ext.json",
    "paper3_reliability_learning/results/E-149_frames_b5678_aug.json",
    "paper3_reliability_learning/results/E-223_observable_frame_reliability.json",
    "paper3_reliability_learning/results/E-224_camera_only_observable.json",
]


class TinyStudent(nn.Module):
    def __init__(self, in_ch=8, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def load_json(relative_path):
    with (ROOT / relative_path).open() as handle:
        return json.load(handle)


def check_readme_links():
    missing = []
    for readme in ROOT.rglob("README.md"):
        markdown = readme.read_text(encoding="utf-8")
        links = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", markdown)
        for raw_target in links:
            target = unquote(raw_target.split("#", 1)[0].strip())
            parsed = urlparse(target)
            if not target or parsed.scheme or parsed.netloc:
                continue
            if not (readme.parent / target).exists():
                source = readme.relative_to(ROOT)
                missing.append(f"{source}: {target}")
    if missing:
        raise AssertionError(f"missing README-linked files: {', '.join(missing)}")


def check_citation_metadata():
    with (ROOT / "CITATION.cff").open(encoding="utf-8") as handle:
        citation = yaml.safe_load(handle)
    assert citation["cff-version"] == "1.2.0"
    assert citation["title"]
    assert citation["message"]
    assert citation["authors"]
    assert citation.get("type", "software") in {"software", "dataset"}
    assert "version" not in citation
    assert "date-released" not in citation


def check_manifests():
    generated = generated_manifest_text(ROOT)
    for relative_path, expected_text in generated.items():
        actual_text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert actual_text == expected_text, f"stale manifest: {relative_path}"

    frame_manifest = json.loads(generated["manifests/frame_reliability_manifest.json"])
    split_scene_ids = frame_manifest["scene_ids_by_source_split"]
    frame_counts = frame_manifest["frame_count_by_scene_id"]
    assert set(frame_counts) == {
        scene_id for scene_ids in split_scene_ids.values() for scene_id in scene_ids
    }
    assert sum(frame_counts.values()) == frame_manifest["frame_count"]


def assert_close(actual, expected, tolerance=1e-9):
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def main():
    released = {path: load_json(path) for path in RELEASED_JSONS}

    scenes = released[RELEASED_JSONS[0]]
    frames = released["paper3_reliability_learning/results/E-149_frames_b5678_aug.json"]
    assert len(scenes) == 166
    assert len({row["sid"] for row in scenes}) == 166
    assert len(frames) == 2898
    assert (
        released["paper3_reliability_learning/results/E-145b_N2898_ext.json"]["n"]
        == 2898
    )

    check_manifests()

    checkpoint = ROOT / "paper2_gate_aware_distillation/checkpoints/student.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model = TinyStudent().eval()
    model.load_state_dict(state_dict)
    with torch.inference_mode():
        output = model(torch.zeros(1, 8, 64, 64))
    assert tuple(output.shape) == (1, 3, 64, 64)

    check_readme_links()
    check_citation_metadata()

    bootstrap = released[
        "paper1_selective_generation/results/E-203_bootstrap_N166.json"
    ]
    ablation = released[
        "paper1_selective_generation/results/E-207_baseline_ablation_N166.json"
    ]
    fullimage = released[
        "paper1_selective_generation/results/E-209_fullimage_psnr.json"
    ]
    metrics = released[
        "paper1_selective_generation/results/E-210_standard_metrics.json"
    ]
    gate_sweep = released[
        "paper1_selective_generation/results/E-216_gate_sensitivity.json"
    ]
    student = released[
        "paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json"
    ]
    routing = released["paper3_reliability_learning/results/E-145b_N2898_ext.json"]
    observable_rgb = released[
        "paper3_reliability_learning/results/E-223_observable_frame_reliability.json"
    ]
    observable_camera = released[
        "paper3_reliability_learning/results/E-224_camera_only_observable.json"
    ]
    assert_close(bootstrap["mechanism_r"]["r"], 0.9817429139463754)
    assert ablation["N"] == 166
    assert_close(
        ablation["deltas_vs_gen3r"]["Quality-gap oracle (gap>5; selects Ours)"]["mean"],
        1.6584672099192257,
    )
    assert fullimage["N"] == 166
    assert_close(fullimage["Ours (always inject)"]["mean"], 14.345163167413856)
    assert metrics["n_frames"] == 759
    assert metrics["lpips_backbone"] == "vgg"
    assert_close(metrics["ours"]["lpips"], 0.34142624096986646)
    tau5 = next(row for row in gate_sweep["sweep"] if row["tau"] == 5.0)
    assert tau5["inject"] == 87
    assert_close(tau5["mean"], 1.6584672099192257)
    assert student["n_frames"] == 759
    assert student["scenes"] == 21
    assert_close(student["student"]["lpips"], 0.2839214580528664)
    assert routing["n"] == 2898
    assert_close(routing["roc_auc"], 0.946553541655721)
    assert observable_rgb["scene_count"] == 21
    assert observable_rgb["frame_count"] == 910
    assert_close(observable_rgb["metrics"]["roc_auc"], 0.4255410022779043)
    assert observable_rgb["policies"]["diagnostic_model_at_0.5"]["teacher_calls"] == 910
    assert (
        observable_camera["evidence_tier"]
        == "genuinely observable camera-only baseline"
    )
    assert observable_camera["scene_count"] == 74
    assert observable_camera["frame_count"] == 2898
    assert_close(observable_camera["metrics"]["roc_auc"], 0.6495999059112509)
    assert observable_camera["policy_at_0.5"]["teacher_calls"] == 2028
    assert_close(
        observable_camera["policy_at_0.5"]["worst_gain_db"], -23.094440460205078
    )

    print(f"Paper1 scenes: {len(scenes)}")
    print(f"Paper3 frames: {len(frames)}")
    print(f"TinyStudent output: {tuple(output.shape)}")
    print(f"TinyStudent SHA256: {digest}")
    print(f"Paper1 mechanism r: {bootstrap['mechanism_r']['r']:.3f}")
    print(
        f"Paper1 diagnostic LPIPS/FID: {metrics['ours']['lpips']:.3f} / {metrics['ours']['fid']:.1f}"
    )
    print(
        f"Paper3 quality-probe ROC-AUC/PR-AUC: "
        f"{routing['roc_auc']:.3f} / {routing['pr_auc']:.3f}"
    )
    print(
        f"Paper3 RGB/oracle-visibility ROC-AUC/calls: "
        f"{observable_rgb['metrics']['roc_auc']:.3f} / "
        f"{observable_rgb['policies']['diagnostic_model_at_0.5']['teacher_calls']}"
    )
    print(
        f"Paper3 camera-only ROC-AUC/calls/worst gain: "
        f"{observable_camera['metrics']['roc_auc']:.3f} / "
        f"{observable_camera['policy_at_0.5']['teacher_calls']} / "
        f"{observable_camera['policy_at_0.5']['worst_gain_db']:.3f} dB"
    )
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
