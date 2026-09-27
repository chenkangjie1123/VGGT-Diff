<div align="center">

# VGGT-Diff

### Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis

Kangjie Chen · Xiangyu Li · Dongbin Zhang · Chaodao Zheng · Shijia Chen · Jinhao Deng · Hongbin Lin · Choo Sin Wai · Minqi Wang · Minghao Yang · Dake Zhong · Guorui Song · Yu Zhang · Xianming Liu · Boyang Wang

[![Project](https://img.shields.io/badge/Project-VGGT--Diff-176b86)](https://github.com/chenkangjie1123/VGGT-Diff)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-8f236f)](#todo)
[![License](https://img.shields.io/badge/License-Apache--2.0-555555)](LICENSE)

<video src="assets/vggtdiff_demo.mp4" poster="assets/demo_poster.jpg" controls muted loop playsinline width="100%"></video>

[Download the lightweight project demo](assets/vggtdiff_demo.mp4)

</div>

## Overview

VGGT-Diff combines geometry-routed visual evidence from VGGT-Omega with a pretrained video diffusion prior. Given six sparse-view images, it jointly synthesizes a camera-controlled novel-view sequence while preserving observed structure and completing unseen content.

This release contains the main training and inference path, the ordered four-pose Plucker conditioning used for long sequences, and a ready-to-run garden example. Model checkpoints are not stored in this repository and will be released separately.

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

`trajectory.json` contains `source_w2c`, `target_w2c`, `source_intrinsics`, and `target_intrinsics`. Extrinsics use world-to-camera matrices in OpenCV convention; intrinsics are 3×3 pixel-space matrices at the source image resolution.

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

The default training protocol uses six source views, 80 contiguous targets, causal 4:1 target VAE compression, ordered four-pose Plucker packing, source-anchor camera normalization, and the point-track residual consistency loss.

## Checkpoint format

VGGT-Diff checkpoints are safetensors files containing the complete fine-tuned DiT, including:

- `patch_embedding.*`
- `omega_adapter.*`
- `temporal_plucker_adapter.weight`

The loader validates these architecture-defining tensors before allocating the base model, so incompatible checkpoints fail early with a clear error.

## TODO

- [ ] Release half-resolution and full-resolution VGGT-Diff checkpoints.
- [ ] Publish Hugging Face model cards and direct download commands.
- [ ] Add quantitative evaluation scripts and benchmark manifests.

## Acknowledgements

This project builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), [FrameCrafter](https://github.com/szqwu/FrameCrafter), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We thank their authors for releasing their work.

## License

The code is released under the [Apache License 2.0](LICENSE). Model weights and third-party assets may be subject to their own terms.
