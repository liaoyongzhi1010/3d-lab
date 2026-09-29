"""Download RE10K training frames from YouTube, deduplicated by video.

Key efficiency: 65,835 missing scenes come from only ~6,510 unique YouTube
videos (avg ~10 scenes/video). We download each video ONCE, then extract
frames for ALL scenes that reference it, at the exact microsecond timestamps
in each scene's meta file.

Uses yt-dlp (robust, maintained) + OpenCV frame extraction.

Usage:
    python download_re10k_ytdlp.py --workers 6 --max_videos 0
    # --max_videos N to cap for testing; 0 = all
    # resumable: skips scenes whose frames already exist (>=90%)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

META_DIR = Path("/home/data/RealEstate10K/RealEstate10K/train")
OUT_DIR = Path("/home/data/RealEstate10K/train")


def parse_meta(meta_file):
    with open(meta_file) as f:
        lines = f.readlines()
    url = lines[0].strip()
    timestamps = []
    for line in lines[1:]:
        parts = line.strip().split()
        if parts:
            timestamps.append(int(parts[0]))
    return url, timestamps


def scene_done(sid, timestamps):
    d = OUT_DIR / sid
    if d.exists():
        existing = set(int(f.stem) for f in d.glob("*.jpg"))
        if len(existing) >= max(1, int(len(timestamps) * 0.9)):
            return True
    return False


def build_video_index():
    """Map url -> list of (sid, timestamps) for scenes still missing frames."""
    url2scenes = defaultdict(list)
    metas = sorted(META_DIR.glob("*.txt"))
    for mf in metas:
        sid = mf.stem
        url, ts = parse_meta(mf)
        if not ts:
            continue
        if scene_done(sid, ts):
            continue
        url2scenes[url].append((sid, ts))
    return url2scenes


def download_video(url, tmp_dir):
    """Download one YouTube video via yt-dlp. Returns local path or None.

    Rate-limit hardened: random sleep between requests, retries with backoff,
    and rotating player clients to reduce 429 / bot-check failures.
    """
    out_tpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--sleep-requests",
        "1",
        "--min-sleep-interval",
        "1",
        "--max-sleep-interval",
        "4",
        "--retries",
        "3",
        "--retry-sleep",
        "exp=2:30",
        "--extractor-args",
        "youtube:player_client=tv,web_safari,android_vr",
        "--force-ipv4",
        "-o",
        out_tpl,
        url,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=360,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    # find the downloaded file
    files = [f for f in os.listdir(tmp_dir) if f.endswith((".mp4", ".mkv", ".webm"))]
    if not files:
        return None
    return os.path.join(tmp_dir, files[0])


def extract_frames(video_path, scenes):
    """Extract frames for all scenes referencing this video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        return 0, 0
    n_ok_scenes = 0
    n_frames = 0
    for sid, timestamps in scenes:
        out_scene = OUT_DIR / sid
        out_scene.mkdir(parents=True, exist_ok=True)
        existing = set(int(f.stem) for f in out_scene.glob("*.jpg"))
        got = 0
        for ts in timestamps:
            if ts in existing:
                got += 1
                continue
            frame_num = int(round(ts / 1_000_000.0 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(
                    str(out_scene / f"{ts}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                got += 1
                n_frames += 1
        if got >= max(1, int(len(timestamps) * 0.9)):
            n_ok_scenes += 1
    cap.release()
    return n_ok_scenes, n_frames


def process_video(url, scenes, tmp_root):
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        vpath = download_video(url, tmp_dir)
        if vpath is None:
            return ("dl_fail", 0, 0)
        try:
            n_scenes, n_frames = extract_frames(vpath, scenes)
            return ("ok", n_scenes, n_frames)
        except Exception as e:
            return (f"extract_err:{str(e)[:60]}", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all")
    ap.add_argument("--start_from", type=int, default=0)
    ap.add_argument(
        "--index_cache",
        type=str,
        default="/home/data/RealEstate10K_full/video_index.json",
    )
    ap.add_argument("--rebuild_index", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_root = "/home/data/RealEstate10K_full/dl_tmp"
    os.makedirs(tmp_root, exist_ok=True)

    if args.rebuild_index or not os.path.exists(args.index_cache):
        print("Building video index (scanning metas + existing frames)...", flush=True)
        url2scenes = build_video_index()
        json.dump({k: v for k, v in url2scenes.items()}, open(args.index_cache, "w"))
        print(f"Cached index -> {args.index_cache}", flush=True)
    else:
        url2scenes = json.load(open(args.index_cache))
        print(f"Loaded cached index: {args.index_cache}", flush=True)

    videos = list(url2scenes.items())
    videos = videos[args.start_from :]
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    total_scenes = sum(len(v) for _, v in videos)
    print(
        f"Videos to process: {len(videos)} (covering {total_scenes} scenes)", flush=True
    )

    stats = {"ok_videos": 0, "dl_fail": 0, "ok_scenes": 0, "frames": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for url, scenes in videos:
            f = pool.submit(process_video, url, scenes, tmp_root)
            futures[f] = url

        for i, f in enumerate(as_completed(futures)):
            status, n_scenes, n_frames = f.result()
            if status == "ok":
                stats["ok_videos"] += 1
                stats["ok_scenes"] += n_scenes
                stats["frames"] += n_frames
            elif status == "dl_fail":
                stats["dl_fail"] += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(videos):
                el = time.time() - t0
                rate = (i + 1) / el * 3600
                print(
                    f"  [{i + 1}/{len(videos)}] ok_vid={stats['ok_videos']} "
                    f"dl_fail={stats['dl_fail']} ok_scenes={stats['ok_scenes']} "
                    f"frames={stats['frames']} | {rate:.0f} vid/hr",
                    flush=True,
                )

    el = time.time() - t0
    print(f"\nDone in {el / 3600:.2f}h. {stats}", flush=True)


if __name__ == "__main__":
    main()
