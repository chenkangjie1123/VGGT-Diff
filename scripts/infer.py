#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from vggtdiff import VGGTDiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a VGGT-Diff camera trajectory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--omega-checkpoint", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--example", type=Path, default=Path("examples/garden"))
    parser.add_argument("--output", type=Path, default=Path("outputs/garden"))
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20263012)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--vram-limit-gib",
        type=float,
        default=0.0,
        help="Persistent CUDA weight budget; use -1 to disable CPU offload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.example / "source_views").glob("*.png"))
    if len(source_paths) != 6:
        raise ValueError(f"Expected six source images, found {len(source_paths)}")
    images = [Image.open(path).convert("RGB") for path in source_paths]
    trajectory = json.loads((args.example / "trajectory.json").read_text())
    intrinsics_image_size = trajectory.get("intrinsics_image_size")
    if intrinsics_image_size is not None:
        expected_size = tuple(int(value) for value in intrinsics_image_size)
        actual_size = images[0].size
        if expected_size != actual_size:
            raise ValueError(
                "trajectory intrinsics use image size "
                f"{expected_size}, but source images are {actual_size}"
            )
    w2c = np.asarray(
        trajectory["source_w2c"] + trajectory["target_w2c"], dtype=np.float32
    )
    intrinsics = np.asarray(
        trajectory["source_intrinsics"] + trajectory["target_intrinsics"],
        dtype=np.float32,
    )
    model = VGGTDiff(
        checkpoint=args.checkpoint,
        omega_checkpoint=args.omega_checkpoint,
        base_model_dir=args.base_model_dir,
        vram_limit_gib=None if args.vram_limit_gib < 0 else args.vram_limit_gib,
    )
    predictions = model.generate(
        images,
        w2c,
        intrinsics,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
    )
    frame_dir = args.output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(predictions):
        image.save(frame_dir / f"frame_{index:03d}.png")
    args.output.mkdir(parents=True, exist_ok=True)
    forward_frames = [np.asarray(image) for image in predictions]
    video_frames = forward_frames + forward_frames[::-1]
    imageio.mimsave(
        args.output / "vggtdiff.mp4",
        video_frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    print(args.output / "vggtdiff.mp4")


if __name__ == "__main__":
    main()
