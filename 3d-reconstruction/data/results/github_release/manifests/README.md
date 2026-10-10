# Released Manifests

These manifests are deterministic exports of released JSONs, not reconstructed
from private data. Regenerate them with `python3 scripts/generate_manifests.py`,
or verify exact checked-in content with `python3 scripts/generate_manifests.py --check`.

- [`re10k_scene_manifest.json`](re10k_scene_manifest.json) lists the 166 sorted
  scene identifiers and their source split from Paper 1's
  `E-142_combined_N169.json` (the historical filename says N169; the released
  valid row count is 166).
- [`frame_reliability_manifest.json`](frame_reliability_manifest.json) lists the
  74 sorted scene identifiers, per-scene frame counts, total 2,898 frames,
  released feature names, source splits, and grouped leave-one-scene-out rule
  from Paper 3's `E-149_frames_b5678_aug.json`.

The source JSONs do not contain random seeds or upstream checkpoint versions.
Those fields are recorded as unavailable rather than filled with assumptions.
Expected headline values are referenced from released result JSONs and checked
by `scripts/smoke_test.py`.
