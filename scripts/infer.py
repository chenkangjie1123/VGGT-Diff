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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.example / "source_views").glob("*.png"))
    if len(source_paths) != 6:
        raise ValueError(f"Expected six source images, found {len(source_paths)}")
    images = [Image.open(path).convert("RGB") for path in source_paths]
    trajectory = json.loads((args.example / "trajectory.json").read_text())
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
    imageio.mimsave(
        args.output / "vggtdiff.mp4",
        [np.asarray(image) for image in predictions],
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    print(args.output / "vggtdiff.mp4")


if __name__ == "__main__":
    main()
