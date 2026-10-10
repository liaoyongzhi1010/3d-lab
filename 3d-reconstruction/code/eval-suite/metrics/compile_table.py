"""E-052 evaluation table compiler.

Reads per-method JSONs produced by:
  - _e052_eval_external.py  (official Flash3D evaluator: PSNR/SSIM/LPIPS-VGG, 5% crop)
  - _e052_eval_genquality.py (cleanfid FID/KID + DISTS + LPIPS, full-image)
and emits a comparison table (markdown + merged json) per gap bucket.

All numbers trace back to the source JSONs; this script only formats.

Usage:
  python _e052_compile_table.py \
    --recon flash3d=/home/data/E-052_external/flash3d_wide700.json \
    --recon catsplat=/home/data/E-052_external/catsplat_wide700.json \
    --genq  flash3d=/home/data/E-052_external/flash3d_genq.json \
    --genq  catsplat=/home/data/E-052_external/catsplat_genq.json \
    --out   /home/data/E-052_external/COMPARISON.md
"""

import argparse
import json
from pathlib import Path


def load(p):
    with open(p) as f:
        return json.load(f)


def parse_kv(items):
    d = {}
    for it in items or []:
        k, v = it.split("=", 1)
        d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recon", action="append", help="method=path.json (external/official metrics)"
    )
    ap.add_argument(
        "--genq", action="append", help="method=path.json (gen-quality metrics)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--title",
        default="Cross-Method Evaluation (unified official metrics)",
        help="table title (e.g. 'official MINE present split')",
    )
    args = ap.parse_args()

    recon = {k: load(v) for k, v in parse_kv(args.recon).items()}
    genq = {k: load(v) for k, v in parse_kv(args.genq).items()}
    methods = list(dict.fromkeys(list(recon.keys()) + list(genq.keys())))

    lines = []
    lines.append(f"# {args.title}\n")
    lines.append("All methods scored by the SAME official metric code:")
    lines.append(
        "- Recon: Flash3D/CATSplat official `evaluation/evaluator.py` (PSNR/SSIM/LPIPS-VGG, 5% border crop)"
    )
    lines.append(
        "- GenQuality: cleanfid FID/KID + DISTS_pytorch (latentSplat) + LPIPS-VGG, full-image\n"
    )

    buckets = ["tgt5", "tgt10", "tgt_rand"]

    # --- Recon table (PSNR/SSIM/LPIPS per bucket) ---
    lines.append("## Reconstruction accuracy (official evaluator, 5% crop)\n")
    header = (
        "| Method | "
        + " | ".join([f"{b} PSNR / SSIM / LPIPS" for b in buckets])
        + " | novel PSNR |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (len(buckets) + 2))
    for m in methods:
        if m not in recon:
            continue
        r = recon[m]["summary"] if "summary" in recon[m] else recon[m]
        bk = r["buckets"]
        cells = []
        for b in buckets:
            cells.append(
                f"{bk[b]['psnr']:.2f} / {bk[b]['ssim']:.3f} / {bk[b]['lpips']:.3f}"
            )
        nov = r.get("novel_mean", {}).get("psnr", float("nan"))
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {nov:.2f} |")
    lines.append("")

    # --- GenQuality table (FID/KID/DISTS per bucket) ---
    if genq:
        lines.append("## Generation quality (full-image; lower is better)\n")
        header = (
            "| Method | "
            + " | ".join([f"{b} FID / KID / DISTS" for b in buckets])
            + " | novel FID |"
        )
        lines.append(header)
        lines.append("|" + "---|" * (len(buckets) + 2))
        for m in methods:
            if m not in genq:
                continue
            g = genq[m]
            bk = g["buckets"]
            cells = []
            for b in buckets:
                cells.append(
                    f"{bk[b]['fid']:.1f} / {bk[b]['kid']:.4f} / {bk[b]['dists']:.3f}"
                )
            nov = g.get("novel_overall", {}).get("fid", float("nan"))
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {nov:.1f} |")
        lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    # also dump merged json
    out.with_suffix(".json").write_text(
        json.dumps({"recon": recon, "genq": genq}, indent=2)
    )
    print("\n".join(lines))
    print(f"\n-> {out}\n-> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
