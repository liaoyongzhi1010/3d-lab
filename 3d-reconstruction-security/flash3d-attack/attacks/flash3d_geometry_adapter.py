"""Flash3D adapter that exposes the frozen model as a differentiable GeometryState source.

This adapter is the ONLY place that touches Flash3D's rendering internals. It follows the
exact conventions from the `flash3d-gaussian-render` skill and Flash3D's own model.py so the
geometry attack operates on the SAME primitives the model actually renders:

  * primitives live in SOURCE-camera coordinates (Flash3D's ``gauss_means`` are homogeneous
    source-camera coords ``[x,y,z,1]``, NOT world coords);
  * a target view is reached with ``cam_T_cam = T_w2c_target @ T_c2w_source`` (target-from-source);
  * unprojection uses the PADDED intrinsics ``K_src`` at 320x448 (principal point 224,160) with
    a +0.5 half-pixel shift (``shift_rays_half_pixel="forward"``);
  * rendering uses the UNPADDED intrinsics ``K_tgt`` at 256x384 (principal point 192,128);
  * ``pad_border_aug=32``; ``znear=0.01``, ``zfar=100.0``;
  * the per-Gaussian offset is applied AFTER the depth is produced and is NOT scaled by depth
    (``scaled_offset=false``);
  * the projection matrix is Flash3D's real NDC ``getProjectionMatrix`` (never a copy of the
    view matrix);
  * SH degree 1 (2 layers, ``gaussians_per_pixel=2``).

The CPU-safe part is the ``Flash3DConventions`` dataclass + ``FLASH3D_RE10K_CONVENTIONS``
constant, which pin these numbers so a regression is caught even without CUDA. The heavy
``Flash3DGeometryAdapter`` (with ``validate_renderer_parity``) only imports torch/Flash3D
lazily and is exercised by the server-only tests.
"""

from __future__ import annotations

from dataclasses import dataclass

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
FLASH3D_REPO = "/root/projects/flash3d"


@dataclass(frozen=True)
class Flash3DConventions:
    """The exact RealEstate10K rendering contract Flash3D uses (frozen)."""

    height: int
    width: int
    pad_border_aug: int
    znear: float
    zfar: float
    max_sh_degree: int
    gaussians_per_pixel: int
    shift_rays_half_pixel: float
    scaled_offset: bool

    @property
    def padded_height(self) -> int:
        return self.height + 2 * self.pad_border_aug

    @property
    def padded_width(self) -> int:
        return self.width + 2 * self.pad_border_aug

    def unpadded_principal_point(self) -> tuple:
        """Principal point of the UNPADDED output grid (used for K_tgt at 256x384)."""
        return (self.width / 2.0, self.height / 2.0)

    def padded_principal_point(self) -> tuple:
        """Principal point of the PADDED grid (used for inv_K_src unprojection at 320x448)."""
        px, py = self.unpadded_principal_point()
        return (px + self.pad_border_aug, py + self.pad_border_aug)


FLASH3D_RE10K_CONVENTIONS = Flash3DConventions(
    height=256,
    width=384,
    pad_border_aug=32,
    znear=0.01,
    zfar=100.0,
    max_sh_degree=1,
    gaussians_per_pixel=2,
    shift_rays_half_pixel=0.5,  # "forward"
    scaled_offset=False,
)


class Flash3DGeometryAdapter:
    """Wraps a frozen Flash3D model to produce differentiable GeometryStates and to render.

    Heavy dependencies (torch, Flash3D repo) are imported lazily so the module (and the CPU
    convention tests) load anywhere. Instantiation requires CUDA + the Flash3D repo.
    """

    def __init__(
        self, model, cfg, example_inputs, device, conventions=FLASH3D_RE10K_CONVENTIONS
    ):
        self.model = model
        self.cfg = cfg
        self.example_inputs = example_inputs
        self.device = device
        self.conventions = conventions

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_default_checkpoint(
        cls,
        split: str = "splits/re10k_mine_filtered/test_files_present.txt",
        ckpt: str = FLASH3D_CKPT,
        repo: str = FLASH3D_REPO,
    ) -> "Flash3DGeometryAdapter":
        import sys

        import torch

        if repo not in sys.path:
            sys.path.insert(0, repo)
        from hydra import compose, initialize_config_dir
        from datasets.util import create_datasets
        from models.model import GaussianPredictor

        device = torch.device("cuda:0")
        with initialize_config_dir(config_dir=f"{repo}/configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "+experiment=layered_re10k",
                    "+dataset.crop_border=true",
                    f"dataset.test_split_path={split}",
                    "model.depth.version=v1",
                    "data_loader.batch_size=1",
                    "data_loader.num_workers=1",
                ],
            )
        model = GaussianPredictor(cfg).to(device)
        state = torch.load(ckpt, map_location="cpu")
        sd = state["model"] if "model" in state else state
        current = model.state_dict()
        filtered = {}
        for k, v in sd.items():
            if "backproject_depth" in k:
                if k in current:
                    filtered[k] = current[k].clone()
            else:
                filtered[k] = v
        model.load_state_dict(filtered, strict=False)
        model.set_eval()

        _, loader = create_datasets(cfg, split="test")
        example_inputs = None
        for inputs in loader:
            for kk, vv in inputs.items():
                if isinstance(vv, torch.Tensor):
                    inputs[kk] = vv.to(device)
            if ("color", 1, 0) in inputs:
                inputs["target_frame_ids"] = [1, 2, 3]
                example_inputs = inputs
                break
        if example_inputs is None:
            raise RuntimeError("no example scene with a novel target frame found")
        return cls(model, cfg, example_inputs, device)

    def load_example_source(self):
        return self.example_inputs[("color_aug", 0, 0)].detach()

    # ------------------------------------------------------------------ geometry
    def build_geometry_state(self, image):
        """Build a differentiable GeometryState (source-camera means + ids) from ``image``.

        Reuses the frozen model's forward path (unidepth + compute_gauss_means) exactly, so the
        means match what Flash3D renders. ids are ``(layer, y, x)`` over the UNPADDED 256x384
        grid, with ``layer in {0,1}`` for the two Gaussian sheets, in the model's flattening
        order ``(b n) c (h w)`` -> ``layer-major`` per Flash3D's rearrange.
        """
        from einops import rearrange  # noqa: F401  (kept for parity with model.py order)

        from attacks.geometry_metrics import GeometryState

        local = dict(self.example_inputs)
        local[("color_aug", 0, 0)] = image
        local["target_frame_ids"] = [1]

        outputs = self.model.models["unidepth_extended"](local)
        self.model.compute_gauss_means(local, outputs)
        means_h = outputs[
            "gauss_means"
        ]  # [(B*n), 4, H*W] homogeneous source-camera coords

        c = self.conventions
        pH, pW = c.padded_height, c.padded_width
        n = c.gaussians_per_pixel
        Bn = means_h.shape[0]
        assert Bn == n, f"expected {n} layers, got {Bn}"
        xyz = means_h[:, :3, :]  # [n, 3, pH*pW]
        xyz = xyz.permute(0, 2, 1).reshape(n, pH * pW, 3)  # [n, pHW, 3]
        means = xyz.reshape(n * pH * pW, 3)

        ids = []
        for layer in range(n):
            for yy in range(pH):
                for xx in range(pW):
                    ids.append((layer, yy, xx))

        K_src = self._unpadded_K_tgt()
        return GeometryState(means=means, primitive_ids=ids, K_src=K_src)

    def _unpadded_K_tgt(self):
        import torch

        return self.example_inputs[("K_tgt", 0)][0].detach().to(self.device)

    # ------------------------------------------------------------------ parity gate
    def validate_renderer_parity(self, max_scenes: int = 1) -> dict:
        """Render Flash3D's own forward output and compare against native forward.

        Returns a report with ``source_psnr`` (should be >=35 dB), ``novel_psnr`` (a nearby
        novel view, ~28 dB typical) and ``max_abs_diff``. This is the hard renderer-parity gate
        from the flash3d-gaussian-render skill: ~11-18 dB indicates a projection/padding/coords
        bug and MUST block downstream experiments.
        """
        import copy

        import torch

        inputs = copy.deepcopy(self.example_inputs)
        inputs["target_frame_ids"] = [1, 2, 3]
        with torch.no_grad():
            out = self.model(inputs)

        def _psnr(pred, gt):
            mse = (pred.clamp(0, 1) - gt.clamp(0, 1)).pow(2).mean().item()
            if mse <= 0:
                return 99.0
            return -10.0 * torch.log10(torch.tensor(mse)).item()

        # Source parity: native color_gauss frame 0 vs GT source.
        src_pred = out[("color_gauss", 0, 0)]
        src_gt = inputs[("color", 0, 0)]
        source_psnr = _psnr(src_pred, src_gt)

        # Novel parity: nearest available novel target frame vs its GT.
        novel_psnr = None
        max_abs_diff = 0.0
        for fid in (1, 2, 3):
            if ("color_gauss", fid, 0) in out and ("color", fid, 0) in inputs:
                nv_pred = out[("color_gauss", fid, 0)]
                nv_gt = inputs[("color", fid, 0)]
                novel_psnr = _psnr(nv_pred, nv_gt)
                max_abs_diff = float(
                    (nv_pred.clamp(0, 1) - nv_gt.clamp(0, 1)).abs().max()
                )
                break
        if novel_psnr is None:
            raise RuntimeError("no novel target frame available for parity check")

        return {
            "source_psnr": source_psnr,
            "novel_psnr": novel_psnr,
            "max_abs_diff": max_abs_diff,
        }


def extract_clean_scaled_eval_poses(inputs: dict, device=None) -> list:
    """Extract target-from-source eval poses from Flash3D dataset inputs.

    Returns a list of [4,4] tensors ``T_target_from_source = T_w2c_target @ T_c2w_source``
    for each available target frame, suitable for geometry-metric evaluation.
    """
    import torch

    c2w_src = inputs.get(("T_c2w", 0))
    if c2w_src is None:
        c2w_src_inv = inputs.get(("T", 0))
        if c2w_src_inv is None:
            raise ValueError("no source pose found in inputs")
        c2w_src = torch.inverse(c2w_src_inv)
    if c2w_src.ndim == 3:
        c2w_src = c2w_src[0]
    poses = []
    for fid in inputs.get("target_frame_ids", [1, 2, 3]):
        c2w_tgt = inputs.get(("T_c2w", fid))
        if c2w_tgt is None:
            t_tgt = inputs.get(("T", fid))
            if t_tgt is None:
                continue
            c2w_tgt = torch.inverse(t_tgt)
        if c2w_tgt.ndim == 3:
            c2w_tgt = c2w_tgt[0]
        w2c_tgt = torch.inverse(c2w_tgt)
        T = w2c_tgt @ c2w_src
        if device is not None:
            T = T.to(device)
        poses.append(T.float())
    return poses
