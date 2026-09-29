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

### Training recipes

VGGT-Diff uses the same frozen Wan2.1 Video VAE in both training recipes, but the latent layout is different. The corresponding checkpoints are separate model families and should not be treated as interchangeable.

| | Paper NVS model | Continuous-trajectory model |
| --- | --- | --- |
| Inputs | Six source views and a variable set of target views | Six source views and 80 ordered, contiguous target frames |
| Source VAE encoding | Each source image is encoded independently | Each source image is encoded independently |
| Target VAE encoding | Every target image is encoded independently | The complete target sequence is encoded through the causal temporal VAE |
| DiT view-time slots | `6 + N`, one slot per physical view | `6 + 21 = 27` slots for an 80-frame target trajectory |
| Camera conditioning | One camera and Plucker map per target slot | Four ordered target Plucker maps are packed into each compressed temporal slot |
| Intended use | Quantitative NVS and joint multi-target prediction | Smooth, camera-controlled long-trajectory video generation |

#### Paper NVS model

The quantitative model in the [paper](https://arxiv.org/abs/2609.33253) uses an independent-view latent layout. Each source image and each ground-truth target image is passed through the frozen VAE separately, so six sources and `N` targets produce exactly `6 + N` latent slots. Target slots are jointly denoised by the DiT, but they are treated as an unordered set of views rather than a temporally compressed video. Each slot remains aligned one-to-one with its image, camera, Plucker map, and routed VGGT-Omega geometry condition.

The paper trains on the 980-scene DL3DV-1K subset with a 6-to-variable-`N` curriculum. The 147-epoch half-resolution stage at `192x336` progresses through `{1, 2, 4}` -> `{2, 4, 8}` -> `{4, 8, 12}` -> `{4, 8, 12, 16}` targets. The 60-epoch full-resolution stage at `480x832` follows `{4, 8}` -> `{4, 8, 12}` -> `{4, 8, 12, 16}`. The Wan DiT learning rate is `1e-5` at half resolution and `5e-6` at full resolution, while newly introduced conditioning modules use `1e-4`. The VAE and VGGT-Omega remain frozen. Point-track residual consistency is applied in latent space with weight `0.1`, alongside geometry-condition dropout.

This independent-view protocol is the one used for all quantitative results reported in the paper. It does not use temporal VAE compression.

#### Continuous 80-frame trajectories

The long-trajectory recipe preserves independent encoding for the six source images, but jointly encodes the ordered target sequence through the frozen VAE's causal temporal path. An 80-frame target is padded by repeating its final frame once, producing an 81-frame sequence and 21 causal target latent slots. The repeated decoded frame is discarded. This reduces the DiT sequence from 86 physical views to 27 latent slots.

Camera conditions do not pass through the VAE. To preserve every requested pose, target Plucker maps are grouped according to the causal windows `[0]`, `[1, 2, 3, 4]`, ..., `[77, 78, 79, 79]`. The four ordered maps in each window are channel-packed and projected by the temporal Plucker adapter before entering the DiT.

Launch continuous-trajectory fine-tuning with Accelerate:

```bash
accelerate launch scripts/train.py \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --resume /path/to/vggtdiff_trajectory.safetensors \
  --target-frames 80 \
  --height 480 \
  --width 832 \
  --output outputs/trajectory_training_run
```

The original continuous-trajectory adaptation described in the paper uses `192x336`; the command above selects the full-resolution `480x832` continuation. Both variants use source-anchor camera normalization, temporally packed per-frame Plucker conditioning, and point-track residual consistency.

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
