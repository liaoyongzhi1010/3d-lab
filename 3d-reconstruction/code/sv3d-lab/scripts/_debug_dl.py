import sys, json, os

sys.path.insert(0, "scripts")
from download_re10k_ytdlp import process_video, extract_frames, download_video, OUT_DIR
import tempfile

idx = json.load(open("/home/data/RealEstate10K_full/video_index.json"))
url = "https://www.youtube.com/watch?v=2lEzOcj8oQo"
scenes = idx[url]
print(f"video has {len(scenes)} scenes in index")
# check how many already have frames
for sid, ts in scenes[:5]:
    d = OUT_DIR / sid
    n = len(list(d.glob("*.jpg"))) if d.exists() else 0
    print(f"  {sid}: {n}/{len(ts)} existing")

print("--- running process_video ---")
result = process_video(url, scenes, "/home/data/RealEstate10K_full/dl_tmp")
print(f"result: {result}")
for sid, ts in scenes[:5]:
    d = OUT_DIR / sid
    n = len(list(d.glob("*.jpg"))) if d.exists() else 0
    print(f"  after {sid}: {n}/{len(ts)} jpg")
