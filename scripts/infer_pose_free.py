#!/usr/bin/env python3
"""Generate an 80-frame novel-view video from six RGB images, without source poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from vggtdiff import VGGTDiff
from vggtdiff.trajectory_visualization import trajectory_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="VGGT-Diff full-resolution checkpoint")
    parser.add_argument("--omega-checkpoint", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument(
        "--training-mode", choices=("nvs", "trajectory"), default="trajectory",
        help="Checkpoint family to load; use nvs for the NVS checkpoints",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("examples/garden/source_views")
    )
    parser.add_argument("--trajectory-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/pose_free"))
    parser.add_argument("--route-order", type=int, nargs=6, default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frustum-scale", type=float, default=0.25)
    parser.add_argument(
        "--vram-limit-gib",
        type=float,
        default=0.0,
        help="Persistent CUDA weight budget; -1 disables CPU offload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames != 80:
        raise ValueError("The released continuous-trajectory checkpoint expects 80 targets")
    if args.fps <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("fps, height and width must be positive")
    if args.height % 2 or args.width % 2:
        raise ValueError("height and width must be even for H.264 video")
    if args.trajectory_json is None and sorted(args.route_order) != list(range(6)):
        raise ValueError("--route-order must list source indices 0 through 5 exactly once")
    source_paths = sorted(
        path
        for path in args.source_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if len(source_paths) != 6:
        raise ValueError(
            f"Expected six source images in {args.source_dir}; found {len(source_paths)}"
        )
    images = []
    for path in source_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if len({image.size for image in images}) != 1:
        raise ValueError("All six source images must have the same size")

    model = VGGTDiff(
        checkpoint=args.checkpoint,
        omega_checkpoint=args.omega_checkpoint,
        base_model_dir=args.base_model_dir,
        vram_limit_gib=None if args.vram_limit_gib < 0 else args.vram_limit_gib,
        training_mode=args.training_mode,
    )
    predictions, cameras = model.generate_pose_free(
        images,
        trajectory_json=args.trajectory_json,
        frames=args.frames,
        route_order=tuple(args.route_order),
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
    )
    if len(predictions) != args.frames:
        raise ValueError(f"Expected {args.frames} generated frames; got {len(predictions)}")
    cameras["source_images"] = [path.name for path in source_paths]
    cameras["fps"] = args.fps
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    for index, image in enumerate(predictions):
        image.save(frame_dir / f"frame_{index:03d}.png")
    (args.output / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    video = args.output / "prediction.mp4"
    camera_video = args.output / "camera_trajectory.mp4"
    combined_video = args.output / "prediction_with_trajectory.mp4"
    writer_options = dict(
        fps=args.fps, codec="libx264", quality=8, macro_block_size=None
    )
    with (
        imageio.get_writer(video, **writer_options) as prediction_writer,
        imageio.get_writer(camera_video, **writer_options) as camera_writer,
        imageio.get_writer(combined_video, **writer_options) as combined_writer,
    ):
        for predicted, panel in zip(
            predictions,
            trajectory_frames(cameras, scale=args.frustum_scale),
            strict=True,
        ):
            frame = np.asarray(predicted.convert("RGB"))
            if frame.shape[:2] != (args.height, args.width):
                raise ValueError(f"Generated frame has unexpected size: {frame.shape[:2]}")
            panel_width = 2 * round((640 * args.height / 480) / 2)
            combined_panel = np.asarray(
                Image.fromarray(panel).resize(
                    (panel_width, args.height), Image.Resampling.LANCZOS
                )
            )
            prediction_writer.append_data(frame)
            camera_writer.append_data(panel)
            combined_writer.append_data(np.concatenate((frame, combined_panel), axis=1))
    print(video)
    print(camera_video)
    print(combined_video)


if __name__ == "__main__":
    main()
