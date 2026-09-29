"""E-148: concatenate per-batch frame-level datasets (E-144 outputs) into one.

De-dupes by (sid, frame_idx). Used to grow the Paper-3 frame-level training set
across batches.
"""

import json
import os
import argparse
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="frame dataset jsons")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seen = set()
    rows = []
    for pat in args.inputs:
        for path in sorted(glob.glob(pat)):
            if not os.path.exists(path):
                continue
            data = json.load(open(path))
            added = 0
            for r in data:
                key = (r["sid"], r["frame_idx"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
                added += 1
            print(f"{os.path.basename(path):48s} +{added} rows")
    json.dump(rows, open(args.out, "w"), indent=2)
    pos = sum(r["label"] for r in rows)
    scenes = len(set(r["sid"] for r in rows))
    print(f"\nsaved {args.out} frames={len(rows)} scenes={scenes} positive={pos}")


if __name__ == "__main__":
    main()
