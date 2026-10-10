"""
Generate a WIDE-baseline eval split from RE10K test scenes for the Representation Oracle
(and later Paper A wide-baseline eval). Unlike MINE (+5/+10/rand), this samples targets at large
frame gaps to create substantial occluded/out-of-frustum regions.

Output format matches MINE loader: "key src tgt_a tgt_b tgt_c" (5 cols). We reuse the loader's
_load_split_indices which expects exactly src + 3 targets.

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  python /root/sv3d-lab/paper_a_explicit3d/oracle/make_widebaseline_split.py \
      --n_scenes 200 --out /root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide.txt
"""

import sys
import gzip
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/root/projects/flash3d/data/RealEstate10K")
    ap.add_argument(
        "--present_split",
        default="/root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt",
    )
    ap.add_argument("--n_scenes", type=int, default=200)
    ap.add_argument(
        "--gaps",
        default="20,40,60",
        help="frame gaps for the 3 targets (wide baseline)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gaps = [int(g) for g in args.gaps.split(",")]

    # scenes present on disk (from the present split — first column)
    present_scenes = []
    seen = set()
    with open(args.present_split) as f:
        for line in f:
            key = line.split()[0]
            if key not in seen:
                seen.add(key)
                present_scenes.append(key)

    # load test metadata to get sequence lengths
    data_path = Path(args.data_path)
    with gzip.open(data_path / "test.pickle.gz", "rb") as f:
        seq_data = pickle.load(f)

    lines = []
    for key in present_scenes:
        if key not in seq_data:
            continue
        timestamps = seq_data[key]["timestamps"]
        seq_len = len(timestamps)
        max_gap = max(gaps)
        if seq_len <= max_gap + 5:
            continue  # too short for wide baseline
        # place source so that src + max_gap fits
        src = (seq_len - max_gap) // 2
        tgts = [min(src + g, seq_len - 1) for g in gaps]
        # verify all chosen frames exist on disk
        frame_dir = data_path / "test" / key
        idxs = [src] + tgts
        if not all((frame_dir / f"{timestamps[i]}.jpg").exists() for i in idxs):
            continue
        lines.append(f"{key} {src} {tgts[0]} {tgts[1]} {tgts[2]}")
        if len(lines) >= args.n_scenes:
            break

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} wide-baseline scenes (gaps={gaps}) to {args.out}")
    if lines:
        print("Example:", lines[0])


if __name__ == "__main__":
    main()
