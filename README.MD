# Robot Gallery

This repository collects robot model assets for MuJoCo and URDF workflows.

For URDF visualization, the recommended option is the VS Code extension [URDF Visualizer](https://github.com/MorningFrog/urdf-visualizer), which can be installed directly from the VS Code marketplace.

## Scripts

Convex decompose an STL with CoACD:

```bash
conda activate robot_gallery
./scripts/convex_decompose_stl.py input.stl --resolution 2000 --max-convex-hull 1
```

Output: `input_convex/input_convex_000.stl`, etc.

## Preview

| Model | URDF Visual | URDF Collision | MJCF Visual | MJCF Collision |
| --- | --- | --- | --- | --- |
| GO2 | ![GO2 URDF Visual](doc/go2/urdf_vis.png) | ![GO2 URDF Collision](doc/go2/urdf_col.png) | ![GO2 MJCF Visual](doc/go2/mjcf_vis.png) | ![GO2 MJCF Collision](doc/go2/mjcf_col.png) |
| X5 | ![X5 URDF Visual](doc/x5/urdf_vis.png) | ![X5 URDF Collision](doc/x5/urdf_col.png) | ![X5 MJCF Visual](doc/x5/mjcf_vis.png) | ![X5 MJCF Collision](doc/x5/mjcf_col.png) |
| piper_l | ![piper_l URDF Visual](doc/piper_l/urdf_vis.png) | ![piper_l URDF Collision](doc/piper_l/urdf_col.png) |  |  |
