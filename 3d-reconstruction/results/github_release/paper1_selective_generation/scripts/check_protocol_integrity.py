"""Check Paper 1 sources for scientific-validity labeling regressions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROSE_SOURCES = [
    ROOT / "README.md",
    ROOT / "PROTOCOL.md",
    ROOT / "paper" / "README.md",
    ROOT / "paper" / "paper1_full_writeup.md",
    ROOT / "paper" / "main.tex",
    ROOT / "paper" / "figs" / "fig_pipeline.tex",
]
SCRIPT_SOURCES = sorted(
    path for path in (ROOT / "scripts").glob("*.py") if path.name != Path(__file__).name
)
TEXT_SOURCES = PROSE_SOURCES + SCRIPT_SOURCES

FORBIDDEN = {
    "exact visible no-harm": "the mask is exact only on the latent path, not decoded RGB",
    "provably unchanged": "the decoder can mix spatially, so final RGB invariance is not proven",
    "visible psnr remains 14.48 db exactly": "the large-scale visible value was assigned, not independently measured",
    "visible-region psnr | 14.48 to 14.48 exactly": "the large-scale visible value was assigned, not independently measured",
    "optional three-layer": "the learned injection-weight source and results are not released",
    "injectionweightnet": "the learned injection-weight source and results are not released",
    "observable gate": "GT-derived PSNR gates are not observable",
    "observable relative": "visible PSNR is a target-GT quality probe",
    "default reliability gate": "the released gap>5 policy is an oracle analysis",
    "deployed gate": "no deployed routing policy is released",
    "gate-rejected": "the qualitative failure has no released routing decision",
    "gate rejects": "the qualitative failure has no released routing decision",
    "selected output: gen3r fallback": "the qualitative failure has no released routing decision",
    "+3.41": "ACID bucket value is absent from released E-215 output",
    "-1.38": "ACID bucket value is absent from released E-215 output",
    "-1.90": "ACID bucket value is absent from released E-215 output",
    "+0.85": "ACID gated value is absent from released E-215 output",
    "worst case 0.00": "ACID gated value is absent from released E-215 output",
}

REQUIRED_E215 = ["8 scenes", "-0.443", "3/8"]
CONTRIBUTION_CLAIMS = [
    "Concept: oracle-visibility selective injection as a mechanism diagnostic, not deployable single-view inference.",
    "Mechanism: an exact visible latent bypass, with decoded RGB behavior measured rather than guaranteed.",
    "Evidence: a scale-stable quality-gap mechanism and perceptual component gains on a fixed long-sequence diagnostic, bounded by failed zero-shot ACID transfer.",
]
E210_DIRECT_LABELS = ["always injected candidate", "injected output"]
POSITIONING_SOURCES = [
    "arXiv:2406.04343v2 Table 2",
    "camera-ready/retrained README",
    "arXiv:2403.14627v2 Table 1; checkpoint N/A",
    "arXiv:2410.13862v3 Table 6; checkpoint N/A",
]

AMBIGUOUS_SCRIPT_LABELS = {
    "ours (selective, gap>5)": "GT-quality-gap selection must be labeled oracle",
    "ours (selective gap>5)": "GT-quality-gap selection must be labeled oracle",
    "ours(gap5)": "GT-quality-gap selection must be labeled oracle",
    "ours(gate)": "GT-quality-gap selection must be labeled oracle",
    "selective gate (gap>5)": "GT-quality-gap selection must be labeled oracle",
}
GT_GAP_PATTERN = re.compile(
    r"f3d_inv(?:['\"]\]|\b).*?-.*?base_inv(?:['\"]\]|\b)", re.DOTALL
)


def main() -> int:
    errors: list[str] = []
    texts = {path: path.read_text() for path in TEXT_SOURCES}
    for path, text in texts.items():
        for phrase, reason in FORBIDDEN.items():
            if phrase.lower() in text.lower():
                errors.append(f"{path.relative_to(ROOT)}: {reason}: {phrase!r}")

    for path in SCRIPT_SOURCES:
        text = texts[path]
        lowered = text.lower()
        for phrase, reason in AMBIGUOUS_SCRIPT_LABELS.items():
            if phrase in lowered:
                errors.append(f"{path.relative_to(ROOT)}: {reason}: {phrase!r}")
        if GT_GAP_PATTERN.search(text) and not (
            "quality-gap oracle" in lowered or "oracle diagnostic" in lowered
        ):
            errors.append(
                f"{path.relative_to(ROOT)}: GT-quality-gap use must be labeled "
                "'quality-gap oracle' or 'oracle diagnostic'"
            )

    main_tex = texts[ROOT / "paper" / "main.tex"]
    writeup = texts[ROOT / "paper" / "paper1_full_writeup.md"]
    if "PSNR & SSIM & LPIPS & FID & KID & Runtime" not in main_tex:
        errors.append("paper/main.tex: component table must include KID and runtime")
    if (
        "| Resolution | Metric type | PSNR | SSIM | LPIPS-VGG | FID | KID | Runtime | Status |"
        not in writeup
    ):
        errors.append(
            "paper/paper1_full_writeup.md: component table must match LaTeX columns"
        )
    for label, text in (
        ("paper/main.tex", main_tex),
        ("paper/paper1_full_writeup.md", writeup),
    ):
        lowered = " ".join(text.lower().split())
        if "quality-gap oracle" not in lowered:
            errors.append(f"{label}: missing 'quality-gap oracle' labeling")
        if "oracle selectivity" not in lowered:
            errors.append(f"{label}: missing 'oracle selectivity' labeling")
        if (
            "failed zero-shot transfer" not in lowered
            or "dataset-dependent" not in lowered
        ):
            errors.append(
                f"{label}: ACID must be framed as failed, dataset-dependent transfer"
            )
        for value in REQUIRED_E215:
            if value not in text:
                errors.append(f"{label}: missing released E-215 value {value!r}")
    if "visible-quality proxy" not in main_tex.lower():
        errors.append("paper/main.tex: missing 'visible-quality proxy' labeling")

    boundary_phrases = [
        "complete target clip",
        "oracle visibility",
        "crosses the single-image inference boundary",
        "source-only visibility estimator",
        "+0.016",
        "construction assumption",
    ]
    for label, text in (
        ("README.md", texts[ROOT / "README.md"]),
        ("PROTOCOL.md", texts[ROOT / "PROTOCOL.md"]),
        ("paper/main.tex", main_tex),
        ("paper/paper1_full_writeup.md", writeup),
    ):
        normalized = " ".join(text.lower().split())
        for phrase in boundary_phrases:
            if phrase.lower() not in normalized:
                errors.append(
                    f"{label}: missing inference-boundary disclosure {phrase!r}"
                )

    e012 = texts[ROOT / "scripts" / "e012_vesg.py"]
    if "learned arm is unavailable in this release" not in e012.lower():
        errors.append(
            "scripts/e012_vesg.py: learned arm must fail with an actionable release error"
        )
    if "from e025_inject_net import" in e012:
        errors.append(
            "scripts/e012_vesg.py: broken unreleased learned-arm import remains reachable"
        )

    manifest = ROOT / "results" / "E-218_selected_qual_manifest.json"
    if not manifest.exists():
        errors.append(
            "results/E-218_selected_qual_manifest.json: missing selected-example mapping"
        )
    else:
        manifest_text = manifest.read_text()
        for asset in [
            "01_window_recovery.png",
            "02_doorway_recovery.png",
            "03_sofa_window_recovery.png",
        ]:
            if asset not in manifest_text:
                errors.append(f"selected-example manifest missing {asset!r}")

    readme = texts[ROOT / "README.md"]
    for claim in CONTRIBUTION_CLAIMS:
        for label, text in (
            ("README.md", readme),
            ("paper/main.tex", main_tex),
            ("paper/paper1_full_writeup.md", writeup),
        ):
            normalized = " ".join(
                text.replace("\\textbf{", "").replace("}", "").split()
            )
            if claim not in normalized:
                errors.append(
                    f"{label}: contribution claim is not synchronized: {claim!r}"
                )

    for label in E210_DIRECT_LABELS:
        if label not in main_tex.lower() or label not in writeup.lower():
            errors.append(
                f"paper sources: E-210 must use direct-output label {label!r}"
            )
    component_block = main_tex[
        main_tex.find("\\label{tab:component}") - 1800 : main_tex.find(
            "\\label{tab:component}"
        )
    ]
    if "oracle-selective" in component_block.lower():
        errors.append(
            "paper/main.tex: E-210 component table must not be oracle-selective"
        )
    for label, text in (
        ("paper/main.tex", main_tex),
        ("paper/paper1_full_writeup.md", writeup),
        ("PROTOCOL.md", texts[ROOT / "PROTOCOL.md"]),
    ):
        normalized = " ".join(text.lower().split())
        if re.search(r"759.{0,400}oracle-select", normalized):
            errors.append(
                f"{label}: 759-frame E-210 result must not be oracle-selective"
            )

    release_sentence = "teaser is constrained to released gt/gen3r/injected assets"
    for label, text in (
        ("paper/main.tex", main_tex),
        ("PROTOCOL.md", texts[ROOT / "PROTOCOL.md"]),
    ):
        if release_sentence not in text.lower():
            errors.append(f"{label}: missing explicit teaser release-asset exception")

    for source in POSITIONING_SOURCES:
        if source not in main_tex or source not in texts[ROOT / "PROTOCOL.md"]:
            errors.append(
                f"positioning metadata missing from paper/protocol: {source!r}"
            )

    if "fig_failures_limitations.pdf" not in main_tex:
        errors.append("paper/main.tex: missing failure/limitations composite")
    for value in [
        ".0251",
        ".0267",
        "9/21",
        "8 scenes",
        "-0.443",
        "3/8",
        "no raw acid images",
    ]:
        if value not in main_tex.lower():
            errors.append(
                f"paper/main.tex: limitations figure/caption missing {value!r}"
            )

    if "& 1 & +10 & 25.94 & .857 & .133 \\" not in main_tex:
        errors.append("paper/main.tex: missing official Flash3D +10 quoted bucket")
    pipeline = texts[ROOT / "paper" / "figs" / "fig_pipeline.tex"].lower()
    if "oracle/proposed" not in pipeline or "target gt" not in pipeline:
        errors.append(
            "paper/figs/fig_pipeline.tex: fallback must be oracle/proposed and target-GT labeled"
        )

    if errors:
        print("Protocol integrity check failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Protocol integrity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
