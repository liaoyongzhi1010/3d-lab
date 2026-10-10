# C0 Gen3R-Fuse (flagship)

Scene: bedroom 0a9f2831a3e73de8 (RE10K real trajectory).

## Idea
Gen3R (frozen) outputs per-pixel `pcds` (world points, unprojected from its VGGT depth head)
plus `point_masks` (reliability). Build an EXPLICIT 3DGS from the reliable points + their RGB,
render novel views. Compare to Flash3D's fuzzy feed-forward gaussians at large parallax.

## Pipeline
1. `c0_dump2.py` (env: /root/projects/Gen3R/.venv_gen3r, diffusers 0.33.1)
   - Runs Gen3RPipeline on frame-0 input along the RE10K real trajectory (49 frames).
   - Saves: pcds.npy [F,H,W,3], point_masks.npy [F,H,W], rgbs.npy [F,H,W,3],
     vggt_extrinsics.npy [F,3,4] (w2c OpenCV), vggt_intrinsics.npy [F,3,3].
   - CRITICAL FIX vs first attempt: must save the VGGT-ESTIMATED cameras (sample.cameras),
     NOT the input real-trajectory poses. pcds are unprojected in VGGT's own coordinate frame;
     rendering with input-traj poses gives total misalignment (that bug gave PSNR ~7).
2. `c0_render2.py` (env: dr3d, gsplat 1.5.3)
   - Gaussians = reliable points (mask>0.5) from seed frames [0,6,12,18,24,30].
   - Per-point isotropic scale = (distance_to_cam / focal) * 0.8  (projected pixel footprint).
     NOTE: knn-based scale FAILS on merged multi-frame clouds (inflated) — use depth/focal.
   - Render frames 36/48 with the matching VGGT extrinsic/intrinsic.

## Result
- frame36: Gen3R-Fuse PSNR 15.1 / LPIPS 0.303  vs Flash3D PSNR 20.7 / LPIPS 0.208
- frame48: Gen3R-Fuse PSNR 12.9 / LPIPS 0.367  vs Flash3D PSNR 19.1 / LPIPS 0.253

## Honest visual assessment
- Geometry is CORRECTLY pose-aligned (sanity_render_00.png reproduces the input view).
- In VALID (non-disoccluded) regions Gen3R-Fuse is clearly SHARPER and cleaner than Flash3D:
  headboard, wall plates, picture frame, blanket pattern are crisp; NO ghosting streaks.
- FAILURE MODE: black disocclusion holes (upper-left at f36/f48) where no reliable point
  projects to the novel view. Also mild speckle from sparse points at grazing angles.
- These holes tank global PSNR/LPIPS despite better perceptual sharpness in valid regions.
- => Motivates the trainable module: a small disocclusion inpainting / hole-filling net
  (or chaining C2's Difix refiner) on top of the explicit Gen3R geometry.
