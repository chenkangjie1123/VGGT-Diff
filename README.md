<div align="center">

# VGGT-Diff

### Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis

[Kangjie Chen](https://github.com/chenkangjie1123)<sup>1*</sup>&nbsp;&nbsp; [Xiangyu Li](https://github.com/chenkangjie1123/VGGT-Diff)<sup>1*</sup>&nbsp;&nbsp; [Dongbin Zhang](https://scholar.google.com/citations?user=U1cdnYUAAAAJ&hl=zh-CN)<sup>1</sup>&nbsp;&nbsp; [Chaoda Zheng](https://scholar.google.com/citations?user=3YuWG1QAAAAJ&hl=en)<sup>1</sup>&nbsp;&nbsp; [Shijia Chen](https://github.com/chenkangjie1123/VGGT-Diff)<sup>1</sup><br>
[Jinhao Deng](https://scholar.google.com/citations?user=4lD_AkgAAAAJ&hl=en)<sup>1</sup>&nbsp;&nbsp; [Hongbin Lin](https://scholar.google.com/citations?user=LqX1k5QAAAAJ&hl=en)<sup>2</sup>&nbsp;&nbsp; [Choo Sin Wai](https://scholar.google.com/citations?hl=zh-CN&user=XM2n3scAAAAJ)<sup>3</sup>&nbsp;&nbsp; [Minqi Wang](https://github.com/chenkangjie1123/VGGT-Diff)<sup>2</sup>&nbsp;&nbsp; [Minghao Yang](https://github.com/chenkangjie1123/VGGT-Diff)<sup>3</sup><br>
[Dake Zhong](https://github.com/chenkangjie1123/VGGT-Diff)<sup>3</sup>&nbsp;&nbsp; [Guorui Song](https://scholar.google.com/citations?user=qOOnZAoAAAAJ&hl=en)<sup>3</sup>&nbsp;&nbsp; [Yu Zhang](https://github.com/chenkangjie1123/VGGT-Diff)<sup>1</sup>&nbsp;&nbsp; [Xianming Liu](https://scholar.google.com/citations?user=697UEEIAAAAJ&hl=en)<sup>1</sup>&nbsp;&nbsp; [Boyang Wang](https://github.com/chenkangjie1123/VGGT-Diff)<sup>1&dagger;</sup>

<sup>1</sup> XPeng Motors&nbsp;&nbsp;&nbsp; <sup>2</sup> The Chinese University of Hong Kong&nbsp;&nbsp;&nbsp; <sup>3</sup> Tsinghua University

[![Project Page](https://img.shields.io/badge/Project-Page-176b86)](https://chenkangjie1123.github.io/VGGT-Diff/)
[![Paper](https://img.shields.io/badge/Paper-PDF-8f236f)](https://chenkangjie1123.github.io/VGGT-Diff/VGGT-Diff.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2609.33253-b31b1b)](https://arxiv.org/abs/2609.33253)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Coming_Soon-f1b928)

<img src="assets/vggtdiff_demo_20cases.gif" alt="VGGT-Diff results on twenty evaluation trajectories" width="100%">

</div>

## Updates

- **September 29, 2026:** The VGGT-Diff paper is now available on [arXiv](https://arxiv.org/abs/2609.33253). We also released the pose-free inference pipeline for generating camera-controlled videos directly from six RGB images.
- **September 27, 2026:** We released the VGGT-Diff [paper](https://chenkangjie1123.github.io/VGGT-Diff/VGGT-Diff.pdf), [project page](https://chenkangjie1123.github.io/VGGT-Diff/), and code.

## TODO

- [ ] Release half-resolution and full-resolution VGGT-Diff checkpoints.
- [ ] Release the full-resolution checkpoint for continuous camera-trajectory generation.
- [x] Release pose-free image-to-novel-view video inference from six RGB images.

## Overview

[VGGT-Diff](https://chenkangjie1123.github.io/VGGT-Diff/) (also searchable as **VGGT Diff** or **VGGTDiff**) is a geometry-routed multi-view diffusion model for sparse-view novel view synthesis. It combines visual evidence from VGGT-Omega with a pretrained video diffusion prior. Given six sparse-view images, it jointly synthesizes a camera-controlled novel-view sequence while preserving observed structure and completing unseen content.

This release contains the main training and inference paths, six-view visual conditioning, per-frame camera-trajectory conditioning, and a ready-to-run garden example. Model checkpoints are not stored in this repository and will be released separately.

## Installation

```bash
git clone https://github.com/chenkangjie1123/VGGT-Diff.git
cd VGGT-Diff

conda create -n vggtdiff python=3.10 -y
conda activate vggtdiff
pip install -e .
pip install git+https://github.com/facebookresearch/vggt-omega.git@399d4d62935deb71cedb1e1c35b7a90413a6bee4
```

Request access to the [VGGT-Omega checkpoint](https://huggingface.co/facebook/VGGT-Omega) and download `vggt_omega_1b_512.pt`. Wan2.1 weights are downloaded automatically on first use, or can be supplied through `--base-model-dir`.

## Inference

The default example includes six source views and an 80-frame camera trajectory:

```bash
python scripts/infer.py \
  --checkpoint /path/to/vggtdiff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt
```

The generated frames and video are written to `outputs/garden/`. Full-resolution inference is enabled with `--height 480 --width 832`; the lower default resolution is convenient for a first run.
The default inference path uses CPU offloading to minimize CUDA memory use. On a GPU with ample memory, set `--vram-limit-gib 64` to keep more weights resident, or `--vram-limit-gib -1` to disable offloading.

### Inference from six RGB images without input poses

The pose-free entry point estimates the six source cameras with VGGT-Omega, then generates 80 novel-view frames with VGGT-Diff. It defaults to 480p (832 × 480), 12 fps, and a route through source views 0 → 1 → 2 → 3 → 4 → 5. Name the six input images in their physical walking order. The route interpolates camera centers and orientations; it does not check for collisions with scene geometry.

```bash
python scripts/infer_pose_free.py \
  --checkpoint /path/to/full_resolution_vggtdiff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --source-dir /path/to/six_rgb_images \
  --output outputs/pose_free
```

The output contains `prediction.mp4`, `camera_trajectory.mp4`, a synchronized `prediction_with_trajectory.mp4`, all 80 PNG frames, and `cameras.json` with the estimated source cameras and the target path actually used. The camera view uses small antialiased frustums. No ground-truth poses or frames are read. You can reorder the default path with `--route-order 0 2 1 3 4 5`.

To provide your own 80 target cameras, pass `--trajectory-json /path/to/targets.json`. Give exactly one of `target_w2c` (OpenCV world-to-camera matrices in the VGGT-Omega-estimated world saved in `cameras.json`) or `target_c2w_relative_to_source0` (camera-to-world matrices in the first source camera's coordinate frame). Both are arrays of 80 homogeneous 4 × 4 matrices. Optional `target_intrinsics` is an array of 80 pixel-space 3 × 3 matrices; if omitted, the median of the six estimated source intrinsics is used. The custom path should avoid scene collisions and keep a sensible distance from the observed views.

### Custom scenes

Create a directory with this layout:

```text
my_scene/
├── source_views/
│   ├── 00.png
│   ├── 01.png
│   └── ...
└── trajectory.json
```

`trajectory.json` contains `source_w2c`, `target_w2c`, `source_intrinsics`, and `target_intrinsics`. Extrinsics use world-to-camera matrices in OpenCV convention; intrinsics are 3x3 pixel-space matrices at the source image resolution.

```bash
python scripts/infer.py \
  --checkpoint /path/to/vggtdiff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --example /path/to/my_scene \
  --output outputs/my_scene
```

## Training

Training scenes follow the folder-form DL3DV convention:

```text
dataset_root/
└── scene_id/
    ├── images_4/
    └── transforms.json
```

First cache the frozen VGGT-Omega source features:

```bash
python scripts/prepare_omega_cache.py \
  --dataset-root /path/to/dataset_root \
  --cache-root /path/to/omega_cache \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt
```

Then launch full-model fine-tuning with Accelerate:

```bash
accelerate launch scripts/train.py \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --resume /path/to/vggtdiff.safetensors \
  --output outputs/training_run
```

The default training protocol uses six source views, 80 contiguous targets, causal 4:1 target VAE compression, temporally packed per-frame Plucker conditioning, source-anchor camera normalization, and the point-track residual consistency loss.

## Citation

```bibtex
@article{chen2026vggtdiff,
  title={VGGT-Diff: Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis},
  author={Chen, Kangjie and Li, Xiangyu and Zhang, Dongbin and Zheng, Chaoda and Chen, Shijia and Deng, Jinhao and Lin, Hongbin and Choo, Sin Wai and Wang, Minqi and Yang, Minghao and Zhong, Dake and Song, Guorui and Zhang, Yu and Liu, Xianming and Wang, Boyang},
  journal={arXiv preprint arXiv:2609.33253},
  year={2026}
}
```

## Acknowledgements

This project builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), [FrameCrafter](https://github.com/szqwu/FrameCrafter), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We thank their authors for releasing their work.

## License

VGGT-Diff is released under the [VGGT-Diff Research License](LICENSE) for non-commercial research and educational use. Third-party components and dependencies remain subject to their respective licenses and terms; see [NOTICE](NOTICE).
