"""RE10K data loader for Paper B real training / evaluation.

Wraps Flash3D's official dataset creation with MINE-present split support.
Returns batches compatible with SourceInput + TargetCameras + target_rgb supervision.
Guarantees scene-disjoint train/dev/test splits. Exposes preregistered qualitative IDs.

Uses the official MINE split for test (3204 samples, 641 scenes) and derives
scene-disjoint train/dev from the full training meta.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, "/root/projects/flash3d")

from .model.reconstruction_interface import SourceInput, TargetCameras

MINE_TEST_SPLIT = "/root/projects/flash3d/splits/re10k_mine_filtered/test_files.txt"
RE10K_META_DIR = "/home/data/RealEstate10K/RealEstate10K"
RE10K_FRAMES_DIR = "/home/data/RealEstate10K_full/frames/train"
FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"

FLASH3D_CONFIG_DIR = "/root/projects/flash3d/configs"
FLASH3D_EXPERIMENT = "layered_re10k"

STAGE_SPLIT_PATH = {
    "dev": "splits/re10k_mine_filtered/val_files_present.txt",
    "test": "splits/re10k_mine_filtered/test_files_present.txt",
}

STAGE_TO_SPLIT = {"train": "train", "dev": "test", "test": "test"}

TARGET_FRAME_IDS = (1, 2, 3)

QUALITATIVE_IDS = [
    "000c3ab189999a83_5_10_25",
    "000c3ab189999a83_10_15_20",
    "0a0b25e1e3cafa87_0_5_15",
    "0a0b25e1e3cafa87_5_10_20",
    "0a5d86b5c tried2d88_0_10_30",
    "0b17baf37f34e971_0_5_10",
    "0b23b45ba6f4a1a3_5_15_25",
    "0c7e5f4c6dcfa9c1_0_5_10",
]

RESOLUTION = (256, 384)


def _load_mine_test_scenes() -> set:
    """Parse MINE test split to extract scene IDs for disjoint guarantees."""
    scenes = set()
    p = Path(MINE_TEST_SPLIT)
    if p.exists():
        for line in p.read_text().strip().split("\n"):
            parts = line.strip().split()
            if parts:
                scenes.add(parts[0])
    return scenes


def _load_train_scene_list(meta_dir: str) -> List[str]:
    """Load all training scene IDs from the meta directory."""
    meta_path = Path(meta_dir) / "train"
    if not meta_path.exists():
        meta_path = Path(meta_dir)
    scenes = []
    for f in sorted(meta_path.glob("*.txt")):
        scenes.append(f.stem)
    return scenes


def get_scene_disjoint_splits(
    meta_dir: str = RE10K_META_DIR,
    dev_fraction: float = 0.05,
    seed: int = 42,
):
    """Return scene-disjoint train/dev/test scene ID lists.

    Test = MINE split scenes (641). Dev = 5% of remaining train scenes.
    Train = rest. Guarantees no scene overlap between any pair.
    """
    test_scenes = _load_mine_test_scenes()
    all_train = _load_train_scene_list(meta_dir)
    available = [s for s in all_train if s not in test_scenes]

    import random

    rng = random.Random(seed)
    rng.shuffle(available)
    n_dev = max(1, int(len(available) * dev_fraction))
    dev_scenes = available[:n_dev]
    train_scenes = available[n_dev:]
    return {
        "train": train_scenes,
        "dev": dev_scenes,
        "test": sorted(test_scenes),
    }


def build_flash3d_cfg(
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    stage: str = "train",
    extra_overrides: Optional[List[str]] = None,
):
    """Compose the official Flash3D layered_re10k config via hydra.

    Mirrors paper_a_explicit3d/eval_full.py (lines 33-47) and the attack's
    eval_dpc_crossmodel.py (lines 104-120): initialize_config_dir on the real
    Flash3D configs dir, +experiment=layered_re10k, and the same dataset /
    data_loader overrides. dev/test additionally point test_split_path at a
    present-filtered MINE split.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    overrides = [
        f"+experiment={FLASH3D_EXPERIMENT}",
        "+dataset.crop_border=true",
        "model.depth.version=v1",
        f"data_loader.batch_size={batch_size}",
        f"data_loader.num_workers={num_workers}",
        "++eval.save_vis=false",
    ]
    split_path = STAGE_SPLIT_PATH.get(stage)
    if split_path is not None:
        overrides.append(f"dataset.test_split_path={split_path}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=FLASH3D_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def build_re10k_dataloader(
    stage: str,
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    meta_dir: str = RE10K_META_DIR,
    frames_dir: str = RE10K_FRAMES_DIR,
    seed: int = 42,
    max_scenes: Optional[int] = None,
    flash3d_cfg=None,
):
    """Build the official Flash3D RE10K DataLoader for a given stage.

    Uses the proven hydra config + datasets.util.create_datasets pattern (exact
    match to paper_a_explicit3d/eval_full.py and the attack's
    eval_dpc_crossmodel.py) rather than hand-instantiating the dataset. Returns
    Flash3D-format batch dicts (keys like ("color_aug",0,0), ("color",fid,0),
    ("K_src",0), ("K_tgt",fid), ("cam_T_cam",0,fid)).

    stage: "train" | "dev" | "test". Maps to Flash3D's create_datasets split
    ("train" -> train loader; "dev"/"test" -> test loader over a present split).
    max_scenes: optionally cap the loader to the first N batches (smoke tests).
    """
    from datasets.util import create_datasets

    if stage not in STAGE_TO_SPLIT:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {list(STAGE_TO_SPLIT)}"
        )

    if flash3d_cfg is None:
        flash3d_cfg = build_flash3d_cfg(
            batch_size=batch_size, num_workers=num_workers, stage=stage
        )

    split = STAGE_TO_SPLIT[stage]
    _dataset, loader = create_datasets(flash3d_cfg, split=split)

    if max_scenes is not None:
        loader = _CappedLoader(loader, max_scenes)

    return loader


class _CappedLoader:
    """Wrap a DataLoader to yield at most ``max_batches`` batches (smoke tests)."""

    def __init__(self, loader, max_batches: int):
        self._loader = loader
        self._max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self._loader), self._max_batches)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def batch_to_supervision(inputs: dict, target_frame_ids=TARGET_FRAME_IDS):
    """Extract SourceInput, TargetCameras, and target_rgb from a Flash3D batch.

    inputs: Flash3D-format batch dict from create_datasets (keys like
            ("color_aug",0,0), ("color",fid,0), ("K_src",0), ("K_tgt",fid),
            ("cam_T_cam",0,fid)). Source frame id is 0; targets default to the
            layered_re10k novel frames (1, 2, 3).

    The frozen backbone produces Gaussians in the SOURCE-camera frame, so the
    source pose is the identity and each target camera pose is expressed relative
    to it via inv(cam_T_cam[0->fid]).

    Returns:
      source: SourceInput
      targets: TargetCameras
      target_rgb: (B, V, 3, H, W) ground truth
    """
    src_color = inputs["color", 0, 0]
    B = src_color.shape[0]
    device = src_color.device
    H, W = src_color.shape[2:]

    present = [
        fid
        for fid in target_frame_ids
        if ("color", fid, 0) in inputs and ("cam_T_cam", 0, fid) in inputs
    ]
    if not present:
        present = list(target_frame_ids)

    source = SourceInput(
        source_rgb=src_color,
        K_src=inputs[("K_src", 0)],
        T_c2w_src=torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1),
    )

    K_list = []
    T_list = []
    rgb_list = []
    for fid in present:
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        K_list.append(K_tgt)

        cam_T_cam = inputs.get(("cam_T_cam", 0, fid))
        if cam_T_cam is not None:
            T_c2w_tgt = torch.linalg.inv(cam_T_cam.float())
        else:
            T_c2w_tgt = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
        T_list.append(T_c2w_tgt)

        tgt_color = inputs.get(("color", fid, 0))
        if tgt_color is not None:
            rgb_list.append(tgt_color)
        else:
            rgb_list.append(torch.zeros(B, 3, H, W, device=device))

    targets = TargetCameras(
        K=torch.stack(K_list, dim=1),
        T_c2w=torch.stack(T_list, dim=1),
    )
    target_rgb = torch.stack(rgb_list, dim=1)

    return source, targets, target_rgb
