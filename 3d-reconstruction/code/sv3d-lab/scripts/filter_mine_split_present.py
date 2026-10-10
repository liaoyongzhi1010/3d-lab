"""Filter the MINE RE10K test split to scenes whose frame folders are present on
this server's archive. Writes a filtered split next to the original and prints how
many samples/scenes were dropped. Fair-repro rationale: our method will be
evaluated on the SAME filtered split, so baseline-vs-ours stays comparable; the
dropped scenes are simply absent from the local data archive (not cherry-picking).
"""

import os
import sys

FLASH3D = "/root/projects/flash3d"
SRC = f"{FLASH3D}/splits/re10k_mine_filtered/test_files.txt"
BASE = "/home/data/RealEstate10K/test"
OUT = f"{FLASH3D}/splits/re10k_mine_filtered/test_files_present.txt"


def main():
    lines = [l.rstrip("\n") for l in open(SRC) if l.strip()]
    kept, dropped = [], []
    dropped_scenes = set()
    for l in lines:
        scene = l.split()[0]
        if os.path.isdir(os.path.join(BASE, scene)):
            kept.append(l)
        else:
            dropped.append(l)
            dropped_scenes.add(scene)
    with open(OUT, "w") as f:
        f.write("\n".join(kept) + "\n")
    all_scenes = set(l.split()[0] for l in lines)
    print(f"total samples: {len(lines)}  kept: {len(kept)}  dropped: {len(dropped)}")
    print(f"total scenes: {len(all_scenes)}  dropped scenes: {len(dropped_scenes)}")
    print(f"dropped scene ids: {sorted(dropped_scenes)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
