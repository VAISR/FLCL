# Dataset layout

The evaluation launchers look for datasets under this directory by default
(relative to `src/`). Symlink or copy the benchmarks here:

```
src/data/
├── 7scenes/        # 7-Scenes (test split, RGB-D)                 --root_7scenes
├── neural_rgbd/    # NRGBD / Neural RGB-D surface reconstruction  --root_nrgbd
├── eth3d/          # ETH3D multi-view stereo (undistorted video)  --root_eth3d
├── Long3D/         # Long3D ultra-long sequences                  --root (launch_long3d.py)
└── eval/           # video-depth / pose benchmarks
    ├── bonn/       #   Bonn RGB-D dynamic
    ├── kitti/      #   KITTI depth selection
    └── tum/        #   TUM RGB-D
```

Preparation follows the evaluation guides of
[MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md),
[CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/eval.md) and
[StreamVGGT](https://github.com/wzzheng/StreamVGGT). The exact paths used by the
video-depth and pose launchers are listed in
`src/eval/video_depth/metadata.py` and `src/eval/pose_evaluation/metadata.py`.
