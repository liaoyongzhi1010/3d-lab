import json, numpy as np, sys

d = json.load(open(sys.argv[1]))
for k in sorted(d.keys()):
    if k in ("n_scenes", "cfg", "geometry_rows"):
        continue
    v = d[k]
    if isinstance(v, dict) and "psnr" in v:
        print(
            "%-20s psnr=%.2f ssim=%.3f lpips=%.3f"
            % (k, v["psnr"], v["ssim"], v["lpips"])
        )
print("n_scenes", d.get("n_scenes"))
gr = d.get("geometry_rows", [])
for key in [
    "aligned_point_displacement",
    "reprojection_displacement",
    "order_reversal_rate",
    "order_eligible_pairs",
    "order_reversed",
    "linf",
]:
    vals = [r[key] for r in gr if r.get(key) is not None]
    if vals:
        print(
            "geom %-28s mean=%.4f median=%.4f n=%d"
            % (key, float(np.mean(vals)), float(np.median(vals)), len(vals))
        )
