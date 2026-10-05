<div align="left">

<h3><strong>VGGT-Diff:</strong> <small>Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis</small></h3>

<sub>
<strong><a href="https://github.com/chenkangjie1123">Kangjie Chen</a></strong><sup>1*</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Xiangyu Li</a></strong><sup>1*</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=U1cdnYUAAAAJ&amp;hl=zh-CN">Dongbin Zhang</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=3YuWG1QAAAAJ&amp;hl=en">Chaoda Zheng</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Shijia Chen</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=4lD_AkgAAAAJ&amp;hl=en">Jinhao Deng</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=LqX1k5QAAAAJ&amp;hl=en">Hongbin Lin</a></strong><sup>2</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?hl=zh-CN&amp;user=XM2n3scAAAAJ">Choo Sin Wai</a></strong><sup>3</sup>&nbsp;
<strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Minqi Wang</a></strong><sup>2</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Minghao Yang</a></strong><sup>3</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Dake Zhong</a></strong><sup>3</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=qOOnZAoAAAAJ&amp;hl=en">Guorui Song</a></strong><sup>3</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Yu Zhang</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://scholar.google.com/citations?user=697UEEIAAAAJ&amp;hl=en">Xianming Liu</a></strong><sup>1</sup>,&nbsp; <strong><a href="https://github.com/chenkangjie1123/VGGT-Diff">Boyang Wang</a></strong><sup>1&dagger;</sup>
</sub>

<sub><strong><sup>1</sup> XPeng Motors&nbsp;&nbsp;&nbsp; <sup>2</sup> The Chinese University of Hong Kong&nbsp;&nbsp;&nbsp; <sup>3</sup> Tsinghua University</strong></sub>

[![Project Page](https://img.shields.io/badge/Project-Page-385d6e?style=flat-square)](https://chenkangjie1123.github.io/VGGT-Diff/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-385d6e?style=flat-square)](https://chenkangjie1123.github.io/VGGT-Diff/VGGT-Diff.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2609.33253-385d6e?style=flat-square)](https://arxiv.org/abs/2609.33253)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-385d6e?style=flat-square)](https://huggingface.co/papers/2609.33253)
[![Model Weights](https://img.shields.io/badge/Hugging_Face-Weights-385d6e?style=flat-square)](https://huggingface.co/chenkangjie1123/models?search=VGGT-Diff)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-385d6e?style=flat-square)](https://youtu.be/XkzSSh-ysKs)

<img src="assets/vggtdiff_demo_showcase.gif" alt="VGGT-Diff full-resolution novel-view synthesis across four scenes" width="100%">

<sub>VGGT-Diff delivers strong novel-view synthesis across challenging in-domain and cross-domain viewpoints, outperforming existing generative NVS methods. Core model weights are now available on Hugging Face.</sub>

</div>

## News

- 🎉 **Oct. 5, 2026: Model weights released!** All three core checkpoints are now on [Hugging Face](https://huggingface.co/chenkangjie1123/models?search=VGGT-Diff). 🎉
- **Sep. 29, 2026:** [arXiv paper](https://arxiv.org/abs/2609.33253) and pose-free inference released.
- **Sep. 27, 2026:** [Paper](https://chenkangjie1123.github.io/VGGT-Diff/VGGT-Diff.pdf), [project page](https://chenkangjie1123.github.io/VGGT-Diff/), and code released.

## Model Zoo

| <small>Model</small> | <small>Task</small> | <small>Release setting</small> | <small>Download</small> |
| --- | --- | --- | --- |
| <small>**VGGT-Diff NVS 192x336 (100K)**</small> | <small>Independent-view NVS</small> | <small>DL3DV clean-10K; 100K effective steps</small> | <small>[Model card and weights](https://huggingface.co/chenkangjie1123/VGGT-Diff-NVS-192x336-100K)</small> |
| <small>**VGGT-Diff NVS 480x832**</small> | <small>Full-resolution NVS used for the paper's baseline comparison</small> | <small>DL3DV clean-1K; ~18K half-resolution steps + 7,380 full-resolution steps</small> | <small>[Model card and weights](https://huggingface.co/chenkangjie1123/VGGT-Diff-NVS-480x832)</small> |
| <small>**VGGT-Diff Trajectory 480x832 (36K)**</small> | <small>80-frame continuous camera trajectories</small> | <small>Full-resolution trajectory adaptation</small> | <small>[Model card and weights](https://huggingface.co/chenkangjie1123/VGGT-Diff-Trajectory-480x832)</small> |

The repositories contain the trained VGGT-Diff DiT and conditioning weights. Wan2.1, VGGT-Omega, and other external model components remain separate dependencies and are not redistributed with these checkpoints.

Download a checkpoint with the Hugging Face CLI, for example:

```bash
hf download chenkangjie1123/VGGT-Diff-Trajectory-480x832 \
  vggt-diff.safetensors config.json LICENSE README.md \
  --local-dir checkpoints/VGGT-Diff-Trajectory-480x832
```

Use `--training-mode nvs` with either NVS checkpoint. The trajectory checkpoint is the default model family for `scripts/infer.py` and `scripts/infer_pose_free.py`.

## Installation

```bash
git clone https://github.com/chenkangjie1123/VGGT-Diff.git
cd VGGT-Diff

conda create -n vggtdiff python=3.10 -y
conda activate vggtdiff
pip install -e .
pip install -U huggingface_hub
pip install git+https://github.com/facebookresearch/vggt-omega.git@399d4d62935deb71cedb1e1c35b7a90413a6bee4
```

Request access to the [VGGT-Omega checkpoint](https://huggingface.co/facebook/VGGT-Omega) and download `vggt_omega_1b_512.pt`. Wan2.1 weights are downloaded automatically on first use, or can be supplied through `--base-model-dir`.

## Inference

The bundled example uses six source views, an 80-frame camera trajectory, and the released full-resolution trajectory checkpoint:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --height 480 \
  --width 832
```

Outputs are saved to `outputs/garden/`. Remove `--height 480 --width 832` for the lower, faster default resolution. CPU offloading is enabled by default; use `--vram-limit-gib 64` to retain more weights on a high-memory GPU or `--vram-limit-gib -1` to disable offloading.

### Pose-free inference from six RGB images

VGGT-Omega first estimates the six source cameras, then VGGT-Diff generates 80 novel views. Defaults are 832 × 480, 12 fps, and route `0 → 1 → 2 → 3 → 4 → 5`; name inputs in physical walking order. The route interpolates camera centers and orientations without scene-collision checks.

```bash
python scripts/infer_pose_free.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --source-dir /path/to/six_rgb_images \
  --output outputs/pose_free
```

No ground-truth poses or frames are read. Outputs are `prediction.mp4`, `camera_trajectory.mp4`, synchronized `prediction_with_trajectory.mp4`, 80 PNG frames, and `cameras.json` containing the estimated source cameras and target path used. Camera views use small antialiased frustums. Reorder the default path with `--route-order 0 2 1 3 4 5`.

For a custom 80-camera path, pass `--trajectory-json /path/to/targets.json` with exactly one of:

- `target_w2c`: 80 homogeneous 4 × 4 OpenCV world-to-camera matrices in the VGGT-Omega world saved in `cameras.json`.
- `target_c2w_relative_to_source0`: 80 homogeneous 4 × 4 camera-to-world matrices relative to source camera 0.

Optional `target_intrinsics` supplies 80 pixel-space 3 × 3 matrices; otherwise, the median of the six estimated source intrinsics is used. Keep custom paths collision-free and at a sensible distance from observed views.

### Custom scenes

Use this layout:

```text
my_scene/
├── source_views/
│   ├── 00.png
│   ├── 01.png
│   └── ...
└── trajectory.json
```

`trajectory.json` contains `source_w2c`, `target_w2c`, `source_intrinsics`, and `target_intrinsics`. Extrinsics are OpenCV world-to-camera matrices; intrinsics are 3 × 3 pixel-space matrices at source-image resolution.

```bash
python scripts/infer.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
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

Both recipes freeze the Wan2.1 VAE and VGGT-Omega while training the DiT and conditioning modules. Their latent layouts differ, so their checkpoints are separate model families.

| <small>Recipe</small> | <small>Targets</small> | <small>Latent layout</small> | <small>Plucker poses</small> |
| --- | --- | --- | --- |
| <small>**Paper NVS**</small> | <small>Variable `N`-view set</small> | <small>Independent source/target encoding; `6 + N` slots</small> | <small>1 per target slot</small> |
| <small>**Continuous trajectory**</small> | <small>80 ordered, contiguous frames</small> | <small>Independent sources; causal target VAE; `6 + 21` slots</small> | <small>4 ordered poses per compressed slot</small> |

#### Paper NVS model

This is the independent-view protocol used for the quantitative results in the [paper](https://arxiv.org/abs/2609.33253). Six sources and `N` targets produce `6 + N` one-to-one image, camera, Plucker, and geometry slots without temporal VAE compression.

```bash
accelerate launch scripts/train.py \
  --training-mode nvs \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --target-frames 1 2 4 \
  --height 192 --width 336 \
  --backbone-lr 1e-5 \
  --condition-lr 1e-4 \
  --output outputs/nvs_training_run
```

The paper progresses the half-resolution target curriculum through `{1,2,4}` -> `{2,4,8}` -> `{4,8,12}` -> `{4,8,12,16}`, then trains at `480x832` with `{4,8}` -> `{4,8,12}` -> `{4,8,12,16}` and a `5e-6` backbone learning rate. Resume the previous stage and change `--target-frames`, resolution, and learning rate accordingly.

#### Continuous 80-frame trajectories

An 80-frame target is padded to 81 frames and causally compressed into 21 target slots. Camera conditions bypass the VAE; the four ordered Plucker maps in each causal window are packed and projected by the temporal Plucker adapter.

```bash
accelerate launch scripts/train.py \
  --training-mode trajectory \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --resume /path/to/vggtdiff_trajectory.safetensors \
  --target-frames 80 \
  --height 480 \
  --width 832 \
  --output outputs/trajectory_training_run
```

The paper's original trajectory adaptation uses `192x336`; the command above selects the full-resolution continuation. Both recipes use source-anchor camera normalization and point-track residual consistency.

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
