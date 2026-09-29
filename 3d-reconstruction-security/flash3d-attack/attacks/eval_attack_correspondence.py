"""Recompute correspondence-aware geometry metrics from saved clean/adv primitive dumps.

This is a thin, model-free verification tool: given two ``.pt`` files each holding a saved
GeometryState payload (``means``, ``primitive_ids``, ``K_src``) for the clean and adversarial
scenes plus a target-from-source transform, it recomputes the aligned point displacement,
reprojection displacement, mutual-visibility correspondence count, and the hard near/far order
reversal rate (with the frozen clean-eligible denominator).

Because it consumes saved tensors, it runs on CPU and lets an independent verifier recompute
the geometry construct WITHOUT trusting the attack code or the model. It is the correspondence
analogue of a raw-row recomputation.

Payload format (torch.save of a dict):
    {
      "clean_means": FloatTensor[N,3],   # source-camera coords
      "clean_ids":   list[(layer,y,x)],
      "adv_means":   FloatTensor[M,3],
      "adv_ids":     list[(layer,y,x)],
      "K_src":       FloatTensor[3,3],
      "T_target_from_source": FloatTensor[4,4],  # optional, defaults to identity
      "hw":          (H, W),                      # optional, for visibility filtering
    }

Usage:
    python attacks/eval_attack_correspondence.py --payload clean_adv.pt --out corr.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.geometry_metrics import (
    GeometryState,
    OrderReversalConfig,
    aligned_point_displacement,
    mutual_visibility_correspondences,
    order_reversal_rate,
    raster_depth_visibility,
    reprojection_displacement,
)


def compute_correspondence_report(
    payload: dict, order_cfg: OrderReversalConfig
) -> dict:
    """Recompute all correspondence-aware geometry metrics from a saved payload."""
    K_src = torch.as_tensor(payload["K_src"], dtype=torch.float32)
    clean = GeometryState(
        means=torch.as_tensor(payload["clean_means"], dtype=torch.float32),
        primitive_ids=[tuple(i) for i in payload["clean_ids"]],
        K_src=K_src,
    )
    adv = GeometryState(
        means=torch.as_tensor(payload["adv_means"], dtype=torch.float32),
        primitive_ids=[tuple(i) for i in payload["adv_ids"]],
        K_src=K_src,
    )
    T = torch.as_tensor(
        payload.get("T_target_from_source", torch.eye(4)), dtype=torch.float32
    )

    clean_t = clean.target_points(T)
    adv_t = adv.target_points(T)

    report = {
        "aligned_point_displacement": aligned_point_displacement(
            adv.means, clean.means, adv.primitive_ids, clean.primitive_ids
        ),
        "reprojection_displacement": reprojection_displacement(
            adv.means, clean.means, K_src, adv.primitive_ids, clean.primitive_ids
        ),
    }

    if "hw" in payload:
        hw = tuple(payload["hw"])
        clean_vis = raster_depth_visibility(clean_t, K_src, hw, clean.primitive_ids)
        adv_vis = raster_depth_visibility(adv_t, K_src, hw, adv.primitive_ids)
        corr = mutual_visibility_correspondences(clean_vis, adv_vis)
        report["clean_visible"] = len(clean_vis.visible_ids)
        report["adv_visible"] = len(adv_vis.visible_ids)
        report["mutual_visible"] = len(corr)

    order = order_reversal_rate(clean, adv, T, order_cfg)
    report["order_reversal_rate"] = order.rate
    report["order_eligible_pairs"] = order.n_eligible_pairs
    report["order_reversed"] = order.n_reversed
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--payload", required=True, help="torch.save dict of clean/adv primitives"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean_margin_rel", type=float, default=0.05)
    ap.add_argument("--adv_reverse_rel", type=float, default=0.02)
    args = ap.parse_args()

    payload = torch.load(args.payload, map_location="cpu")
    order_cfg = OrderReversalConfig(
        clean_margin_rel=args.clean_margin_rel,
        adv_reverse_rel=args.adv_reverse_rel,
    )
    report = compute_correspondence_report(payload, order_cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
