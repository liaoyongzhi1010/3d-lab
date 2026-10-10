"""Smear-correcting feed-forward refinement head for Flash3D gaussians.

KEY INSIGHT (from E-127 diagnosis): Flash3D's novel-view failure is SMEAR — gaussians
that are too large / too anisotropic / mis-oriented, producing stretched high-alpha
streaks at novel views. Prior refine heads (E-114) only touched means/color/opacity and
got +0.11dB. This head ALSO corrects SCALES and ROTATIONS — the actual cause of smear —
and is trained with an anti-mean-collapse objective (LPIPS + optional adversarial) so it
produces SHARP corrections rather than blurred means.

The head is a per-gaussian MLP on Flash3D's source features + the gaussian's own params.
Residuals are zero-initialized so the model starts exactly at Flash3D (safe init, and
deletion of the head restores the backbone render = causally load-bearing).

One shared scene: build_scene runs once from the source; corrections are view-independent
(no target leakage). Rendering any target view uses the same corrected gaussians.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmearRefineHead(nn.Module):
    """Per-gaussian residual head correcting xyz, scale, rotation, opacity, color.

    Input per gaussian: [source_feature(C) ; own params encoded]. Output: bounded residuals.
    Zero-initialized final layer => starts at identity (= Flash3D).
    """

    def __init__(
        self,
        feat_dim=256,
        hidden=256,
        n_layers=4,
        max_dxyz=0.1,
        max_dlogscale=1.0,
        max_dcolor=0.3,
    ):
        super().__init__()
        self.max_dxyz = max_dxyz
        self.max_dlogscale = max_dlogscale
        self.max_dcolor = max_dcolor

        # encode own gaussian params: xyz(3)+logscale(3)+rot(4)+opacity(1)+color(3)=14
        self.param_enc = nn.Linear(14, hidden)
        self.feat_proj = nn.Linear(feat_dim, hidden)
        blocks = []
        for _ in range(n_layers):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        # heads: dxyz(3), dlogscale(3), drot(4), dopacity(1), dcolor(3)
        self.head = nn.Linear(hidden, 14)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Reliability gate: predicts per-gaussian UNreliability in [0,1]; corrections are
        # scaled by this gate so the head only edits gaussians it deems smeared/unreliable
        # and LEAVES already-good gaussians untouched (fixes the "hurts good regions" issue).
        self.gate_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 0.0)  # sigmoid(0)=0.5 neutral start
        # Free-color GENERATION branch (E-131): in disocclusion/smear regions the correct
        # color is NOT a small residual off the (wrong, brown) source color -- it must be
        # regenerated from scratch. This head outputs an absolute color; a per-gaussian
        # regen weight (gated) blends between (residual-corrected color) and (regen color).
        self.regen_color_head = nn.Linear(hidden, 3)
        self.regen_weight_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.regen_color_head.weight)
        nn.init.zeros_(self.regen_color_head.bias)
        nn.init.zeros_(self.regen_weight_head.weight)
        nn.init.constant_(
            self.regen_weight_head.bias, -4.0
        )  # sigmoid(-4)~0.018: start ~off

    def forward(self, gauss: dict, feat: torch.Tensor) -> dict:
        """gauss: dict of [N,*] tensors (xyz,scales,rotations,opacity,color_rgb).
        feat: [N, feat_dim] per-gaussian source features. Returns corrected gauss dict.
        """
        xyz = gauss["xyz"]
        scales = gauss["scales"]
        rot = gauss["rotations"]
        opacity = gauss["opacity"]
        color = gauss["color_rgb"]

        logscale = torch.log(scales.clamp(min=1e-6))
        params = torch.cat([xyz, logscale, rot, opacity, color], dim=-1)  # [N,14]
        h = self.param_enc(params) + self.feat_proj(feat)
        for blk in self.blocks:
            h = h + blk(h)
        d = self.head(h)  # [N,14]
        gate = torch.sigmoid(self.gate_head(h))  # [N,1] in [0,1]; 1 = unreliable/edit

        dxyz = self.max_dxyz * torch.tanh(d[:, 0:3]) * gate
        dlogscale = self.max_dlogscale * torch.tanh(d[:, 3:6]) * gate
        drot = 0.1 * torch.tanh(d[:, 6:10]) * gate
        dopacity_logit = d[:, 10:11] * gate
        dcolor = self.max_dcolor * torch.tanh(d[:, 11:14]) * gate

        new_xyz = xyz + dxyz
        # scale correction is MULTIPLICATIVE in log space -> can SHRINK smeared gaussians
        new_scales = torch.exp(logscale + dlogscale).clamp(max=0.5)
        new_rot = F.normalize(rot + drot, dim=-1)
        opacity_logit = torch.log(
            opacity.clamp(1e-4, 1 - 1e-4) / (1 - opacity.clamp(1e-4, 1 - 1e-4))
        )
        new_opacity = torch.sigmoid(opacity_logit + dopacity_logit)
        residual_color = (color + dcolor).clamp(0, 1)
        # Free regenerated color, blended in by a gated regen weight (strong only where
        # the head decides the source color is unusable, e.g. brown smear in disocclusions).
        regen_color = torch.sigmoid(self.regen_color_head(h))  # absolute [0,1]
        regen_w = torch.sigmoid(self.regen_weight_head(h)) * gate  # [N,1], gated
        new_color = ((1 - regen_w) * residual_color + regen_w * regen_color).clamp(0, 1)

        return {
            "xyz": new_xyz,
            "scales": new_scales,
            "rotations": new_rot,
            "opacity": new_opacity,
            "color_rgb": new_color,
            "gate": gate,
            "regen_w": regen_w,
            "_residual_norm": dxyz.norm(dim=-1).mean() + dlogscale.abs().mean(),
        }


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator on rendered novel views.

    Provides the anti-mean-collapse (sharpness) signal: pushes the rendered novel view
    from refined gaussians toward the distribution of real sharp images, breaking the
    blur that L1/LPIPS-only regression collapses to. 3D-gaussian-driven: gradient flows
    render -> gaussians.
    """

    def __init__(self, in_ch=3, base=32):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 4, stride=s, padding=1),
                nn.InstanceNorm2d(o) if i != in_ch else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            blk(in_ch, base, 2),
            blk(base, base * 2, 2),
            blk(base * 2, base * 4, 2),
            blk(base * 4, base * 4, 1),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)
