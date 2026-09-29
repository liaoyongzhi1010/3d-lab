"""E-142: build combined scene-level router/reliability dataset across all batches.

Merges test16 (E-018b + E-019 probe) with train batches 1-4 (direct + probe pairs).
Each row is one scene with component measurements and a teacher-improves label.
The f3d/base PSNR fields use target GT; downstream quality-gap selection is an
oracle diagnostic, not a test-time-observable or deployable gate. Field schema
matches E-140_combined_router_dataset.json.
"""

import json
import os
import argparse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
EXP = os.path.join(RESULTS, "expanded")

# (split, vesg_per_scene_json, probe_json)
SOURCES = [
    (
        "test",
        os.path.join(RESULTS, "E-018b_e012_vesg.json"),
        os.path.join(RESULTS, "E-019_probe.json"),
    ),
    (
        "train1",
        os.path.join(EXP, "E-131_train20_direct.json"),
        os.path.join(EXP, "E-130_train20_probe.json"),
    ),
    (
        "train2",
        os.path.join(EXP, "E-133_train20_b02_direct.json"),
        os.path.join(EXP, "E-132_train20_b02_probe.json"),
    ),
    (
        "train3",
        os.path.join(EXP, "E-135_train20_b03_direct.json"),
        os.path.join(EXP, "E-134_train20_b03_probe.json"),
    ),
    (
        "train4",
        os.path.join(EXP, "E-137_train20_b04_direct.json"),
        os.path.join(EXP, "E-136_train20_b04_probe.json"),
    ),
    (
        "train5",
        os.path.join(EXP, "E-139_train20_b05_direct.json"),
        os.path.join(EXP, "E-138_train20_b05_probe.json"),
    ),
    (
        "train6",
        os.path.join(EXP, "E-147_train20_b06_direct.json"),
        os.path.join(EXP, "E-146_train20_b06_probe.json"),
    ),
    (
        "train7",
        os.path.join(EXP, "E-171_train20_b07_direct.json"),
        os.path.join(EXP, "E-170_train20_b07_probe.json"),
    ),
    (
        "train8",
        os.path.join(EXP, "E-173_train20_b08_direct.json"),
        os.path.join(EXP, "E-172_train20_b08_probe.json"),
    ),
]


def load_probe(path):
    rows = json.load(open(path))
    return {r["scene"]: r for r in rows}


def build_split(split, vesg_path, probe_path):
    e = json.load(open(vesg_path))["per_scene"]
    probe = load_probe(probe_path)
    rows = []
    for scene, b in e["baseline"].items():
        sid = scene.replace("test_", "")
        a = e["adaptive2"].get(scene, {})
        p = probe.get(sid, {}) or probe.get(scene, {})
        if b.get("invis_psnr") is None or a.get("invis_psnr") is None:
            continue
        if p.get("f3d_vis_psnr") is None:
            continue
        delta = a["invis_psnr"] - b["invis_psnr"]
        rows.append(
            {
                "sid": sid,
                "split": split,
                "base_inv": b["invis_psnr"],
                "teacher_inv": a["invis_psnr"],
                "delta": delta,
                "label": int(delta > 0.1),
                "f3d_vis": p["f3d_vis_psnr"],
                "base_vis": b["vis_psnr"],
                "vis_gap": p["f3d_vis_psnr"] - b["vis_psnr"],
                "vis_frac": p.get("mean_vis_frac", None),
                "f3d_inv": p.get("f3d_inv_psnr", None),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.join(EXP, "E-142_combined_router_dataset.json")
    )
    args = ap.parse_args()
    all_rows = []
    seen = set()
    for split, vesg, probe in SOURCES:
        if not (os.path.exists(vesg) and os.path.exists(probe)):
            print(f"{split:8s} SKIP (missing files)")
            continue
        rows = build_split(split, vesg, probe)
        added = 0
        for r in rows:
            if r["sid"] in seen:
                continue
            seen.add(r["sid"])
            all_rows.append(r)
            added += 1
        print(f"{split:8s} added={added:2d} (from {len(rows)})")
    json.dump(all_rows, open(args.out, "w"), indent=2)
    pos = sum(r["label"] for r in all_rows)
    print(f"\nsaved {args.out} rows={len(all_rows)} positive={pos}")


if __name__ == "__main__":
    main()
