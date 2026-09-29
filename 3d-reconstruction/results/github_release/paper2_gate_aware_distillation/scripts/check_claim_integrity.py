#!/usr/bin/env python3
"""Validate publication claims against released E-211 data and checkpoint."""

import hashlib
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_FILES = (
    ROOT / "README.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "paper/README.md",
    ROOT / "paper/main.tex",
    ROOT / "paper/paper2_full_writeup.md",
    ROOT / "paper/figs/fig_pipeline.tex",
)
METRICS_PATH = ROOT / "results/E-211_student_standard_metrics.json"
CHECKPOINT_PATH = ROOT / "checkpoints/student.pt"
TRAINING_MANIFEST_PATH = ROOT / "TRAINING_MANIFEST.md"
FORBIDDEN = {
    "gate-aware framing": re.compile(r"gate\s*[- ]\s*aware", re.IGNORECASE),
    "inference gating": re.compile(
        r"gat(?:e|ed|ing)[\s\S]{0,40}(?:inference|test\s*time)|(?:inference|test\s*time)[\s\S]{0,40}gat(?:e|ed|ing)",
        re.IGNORECASE,
    ),
    "router claim": re.compile(
        r"(?:learned\s+|logistic\s+|reliability\s+)?router", re.IGNORECASE
    ),
    "reliability gate": re.compile(r"reliability\s+gate", re.IGNORECASE),
    "accept/fallback claim": re.compile(
        r"(?:gate|router)[\s\S]{0,40}(?:accept|fall(?:s)?\s+back|fallback)|(?:accept|fall(?:s)?\s+back|fallback)[\s\S]{0,40}(?:gate|router)",
        re.IGNORECASE,
    ),
    "deployment claim": re.compile(r"deployment\s+(?:claim|principle)", re.IGNORECASE),
    "companion validation": re.compile(
        r"companion\s+reliability|learned\s+variant[\s\S]{0,40}validat", re.IGNORECASE
    ),
}


def location(text, start):
    return text.count("\n", 0, start) + 1


def scan_forbidden_claims(text, display_path):
    prose = re.sub(r"```[\s\S]*?```", "", text)
    prose = re.sub(r"`[^`]*`", "", prose)
    failures = []
    for label, pattern in FORBIDDEN.items():
        for match in pattern.finditer(prose):
            excerpt = " ".join(match.group(0).split())[:120]
            failures.append(
                f"{display_path}:{location(prose, match.start())}: {label}: {excerpt}"
            )
    return failures


def checkpoint_facts(path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Expected a non-empty state_dict mapping in {path}")
    parameters = sum(value.numel() for value in state_dict.values())
    return {
        "parameters": parameters,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def require_value(text, label, value, pattern, display_path):
    matches = [
        float(match) for match in re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    ]
    if not matches:
        return [f"{display_path}: missing duplicated headline value for {label}"]
    if any(abs(match - value) > 5e-4 for match in matches):
        return [f"{display_path}: stale {label}: found {matches}, expected {value}"]
    return []


def check_e211_headlines(text, metrics_path=METRICS_PATH, display_path="publication"):
    if not metrics_path.is_file():
        raise FileNotFoundError(f"E-211 metrics JSON not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    student = metrics["student"]
    failures = []
    failures += require_value(
        text,
        "student LPIPS",
        round(student["lpips"], 3),
        r"student.{0,180}?LPIPS.{0,30}?(0\.\d+)",
        display_path,
    )
    failures += require_value(
        text,
        "student FID",
        round(student["fid"], 1),
        r"student.{0,180}?FID.{0,30}?([0-9]+\.[0-9]+)",
        display_path,
    )
    failures += require_value(
        text,
        "student runtime",
        student["time_s"],
        r"student.{0,180}?(0\.\d+)\s*(?:s|\\,s|seconds?)",
        display_path,
    )
    e211_sections = re.findall(
        r"(?:E-211|fixed\s+custom)[\s\S]{0,300}", text, re.IGNORECASE
    )
    e211_text = "\n".join(e211_sections)
    normalized = re.sub(r"[${}*]", "", e211_text)
    for label, expected, pattern in (
        (
            "E-211 frame count",
            metrics["n_frames"],
            r"(\d+)(?:-frame|\s+disocclusion-heavy\s+frames)",
        ),
        ("E-211 scene count", metrics["scenes"], r"(\d+)(?:-scene|\s+scenes)"),
        ("E-211 image size", metrics["size"], r"(\d+)\s*(?:x|\\times)\s*\d+"),
    ):
        values = [
            int(value) for value in re.findall(pattern, normalized, re.IGNORECASE)
        ]
        if not values:
            failures.append(
                f"{display_path}: missing duplicated headline value for {label}"
            )
        elif any(value != expected for value in values):
            failures.append(
                f"{display_path}: stale {label}: found {values}, expected {expected}"
            )
    return failures


def check_training_manifest(text, display_path="TRAINING_MANIFEST.md"):
    required = (
        "--base_mode f3d",
        "--gate_aware",
        "--frame_dataset <FRAME_LABEL_JSON>",
        "--holdout <FOUR_COMMA_SEPARATED_SCENE_IDS>",
        "4 scenes and 161 frames",
        "external teacher arrays",
        "original holdout manifest",
        "not included",
        "template, not a self-contained exact reproduction command",
    )
    return [
        f"{display_path}: missing training provenance disclosure: {value}"
        for value in required
        if value not in text
    ]


def check_checkpoint_headlines(text, facts, display_path):
    failures = []
    counts = [
        int(value.replace(",", ""))
        for value in re.findall(r"([0-9][0-9,]*)[- ]parameter", text)
    ]
    if counts and any(value != facts["parameters"] for value in counts):
        failures.append(
            f"{display_path}: stale checkpoint parameter count: found {counts}, expected {facts['parameters']}"
        )
    hashes = re.findall(r"\b[0-9a-f]{64}\b", text, re.IGNORECASE)
    if hashes and any(value.lower() != facts["sha256"] for value in hashes):
        failures.append(
            f"{display_path}: stale checkpoint SHA256: expected {facts['sha256']}"
        )
    return failures


def main():
    missing = [
        path
        for path in (*PUBLICATION_FILES, TRAINING_MANIFEST_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Publication files not found:\n" + "\n".join(str(path) for path in missing)
        )
    facts = checkpoint_facts(CHECKPOINT_PATH)
    failures = check_training_manifest(TRAINING_MANIFEST_PATH.read_text())
    for path in PUBLICATION_FILES:
        text = path.read_text()
        display_path = str(path.relative_to(ROOT))
        failures += scan_forbidden_claims(text, display_path)
        if path in {
            ROOT / "README.md",
            ROOT / "paper/main.tex",
            ROOT / "paper/paper2_full_writeup.md",
        }:
            failures += check_e211_headlines(text, METRICS_PATH, display_path)
        failures += check_checkpoint_headlines(text, facts, display_path)
    if failures:
        raise SystemExit("Claim integrity failures:\n" + "\n".join(failures))
    print("CLAIM INTEGRITY PASS")


if __name__ == "__main__":
    main()
