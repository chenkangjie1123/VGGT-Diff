#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggtdiff.omega import OmegaRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VGGT-Omega source bundles")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--omega-checkpoint", required=True)
    parser.add_argument("--num-bundles", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260408500)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(scene: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sorted(
        path
        for path in (scene / "images_4").iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def select_sources(
    scene_key: str, frame_count: int, bundle: int, total: int, seed: int
) -> list[int]:
    digest = hashlib.sha256(f"{scene_key}:{bundle}:{seed}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    local = bundle >= int(np.ceil(0.8 * total)) and frame_count >= 24
    if local:
        size = min(frame_count, int(generator.integers(24, min(48, frame_count) + 1)))
        start = int(generator.integers(0, frame_count - size + 1))
        pool = np.arange(start, start + size)
    else:
        pool = np.arange(frame_count)
    segments = np.array_split(pool, 6)
    selected = [int(segment[int(generator.integers(len(segment)))]) for segment in segments]
    generator.shuffle(selected)
    return selected


def raw_w2c(transforms: dict, frame_ids: list[int]) -> torch.Tensor:
    c2w = torch.tensor(
        [transforms["frames"][index]["transform_matrix"] for index in frame_ids],
        dtype=torch.float32,
    )
    w2c = torch.linalg.inv(c2w)
    w2c[:, [1, 2], :] *= -1
    return w2c


def main() -> None:
    args = parse_args()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    runtime = OmegaRuntime(args.omega_checkpoint, torch.device("cuda"))
    complete = []
    for scene in scenes:
        transforms_path = scene / "transforms.json"
        transforms = json.loads(transforms_path.read_text())
        files = image_files(scene)
        frame_count = min(len(files), len(transforms["frames"]))
        if frame_count < 7:
            continue
        output_dir = args.cache_root / scene.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for bundle in range(args.num_bundles):
            output = output_dir / f"bundle_{bundle:02d}.pt"
            if output.is_file() and not args.overwrite:
                continue
            source_ids = select_sources(
                scene.name, frame_count, bundle, args.num_bundles, args.seed
            )
            images = []
            for frame_id in source_ids:
                with Image.open(files[frame_id]) as handle:
                    images.append(handle.convert("RGB"))
            cameras = raw_w2c(transforms, source_ids)
            intrinsic = torch.tensor(
                [
                    [transforms["fl_x"], 0.0, transforms["cx"]],
                    [0.0, transforms["fl_y"], transforms["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(6, 1, 1)
            features = runtime(images, cameras, intrinsic, num_output_frames=0)
            temporary = output.with_suffix(".tmp")
            torch.save(
                {
                    "source_frame_ids": torch.tensor(source_ids),
                    "omega_tokens": features["omega_tokens"].cpu().to(torch.bfloat16),
                    "omega_xyz": features["omega_xyz"].cpu().float(),
                    "omega_confidence": features["omega_confidence"].cpu().float(),
                },
                temporary,
            )
            temporary.replace(output)
        if all(
            (output_dir / f"bundle_{bundle:02d}.pt").is_file()
            for bundle in range(args.num_bundles)
        ):
            complete.append(scene.name)
    manifest = {
        "format": "vggtdiff_omega_source_bundles_v1",
        "scene_keys": complete,
        "num_bundles": args.num_bundles,
        "num_source_frames": 6,
        "omega_layer": 23,
    }
    (args.cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Prepared {len(complete)} scenes")


if __name__ == "__main__":
    main()
