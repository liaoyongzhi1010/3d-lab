# Model Zoo

| Component | Included | Source | SHA256 | Role |
|---|---|---|---|---|
| TinyStudent | Yes | [`student.pt`](paper2_gate_aware_distillation/checkpoints/student.pt) | `1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2` | Paper 2 feed-forward inference |
| Gen3R | No | Official Gen3R repository/checkpoint | External; not redistributed | Generative reconstruction backbone and teacher |
| Flash3D | No | Official Flash3D repository/checkpoint | External; not redistributed | Feed-forward geometry evidence |
| VGGT | No | Official VGGT repository/checkpoint | External; not redistributed | Visibility preprocessing |

## TinyStudent

- Architecture: four 3x3 convolutions with ReLU after the first three layers.
- Input: `[B, 8, H, W]`, concatenating Flash3D RGB (3), baseline RGB (3),
  visibility (1), and invisibility (1).
- Output: `[B, 3, H, W]` image-space residual/output tensor.
- Hidden width: 48.
- Parameter count: 46,371.
- Released checkpoint source: this repository's Paper 2 training run.
- Validation: `python3 scripts/smoke_test.py` verifies the checksum, loads the
  state dict, and runs a synthetic `[1, 8, 64, 64]` forward pass.

The checkpoint is specific to the released architecture. It does not bundle or
replace the Gen3R, Flash3D, VGGT, RealEstate10K, or ACID dependencies needed for
fresh end-to-end rendering. Upstream checkpoint hashes were not recorded in the
released result JSONs and are therefore not invented here.
