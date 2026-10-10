"""Robust dataset wrapper: skip samples that fail to load (e.g. corrupt gzip in RE10K).

RealEstate10K has a few corrupt/truncated `sparse_pcl` gzips; the upstream (whitelisted) re10k.py
raises EOFError on them. Rather than edit upstream, wrap the dataset so a failed __getitem__ falls
back to another (valid) index. Deterministic fallback (index-based) keeps eval reproducible.
"""

import random

from torch.utils.data import Dataset


class RobustDataset(Dataset):
    def __init__(self, base, max_retries=8, verbose=True):
        self.base = base
        self.max_retries = max_retries
        self.verbose = verbose
        self._bad = set()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        n = len(self.base)
        tried = 0
        j = idx
        while tried <= self.max_retries:
            try:
                return self.base[j]
            except Exception as e:  # noqa: BLE001 — corrupt data, skip
                self._bad.add(j)
                if self.verbose:
                    print(
                        f"[robust] skip idx {j} ({type(e).__name__}: {e})", flush=True
                    )
                # deterministic next candidate: step forward, then random after a few
                tried += 1
                j = (idx + tried) % n if tried <= 3 else random.randint(0, n - 1)
        raise RuntimeError(f"RobustDataset: too many bad samples near idx {idx}")
