from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .camera import normalize_source_anchor, plucker_rays


def _image_files(scene_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sorted(
        path
        for path in (scene_dir / "images_4").iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _raw_w2c(transforms: dict, frame_ids: list[int]) -> torch.Tensor:
    c2w = torch.tensor(
        [transforms["frames"][index]["transform_matrix"] for index in frame_ids],
        dtype=torch.float32,
    )
    w2c = torch.linalg.inv(c2w)
    w2c[:, [1, 2], :] *= -1
    return w2c


def _intrinsic(transforms: dict, height: int, width: int) -> torch.Tensor:
    source_width = float(transforms["w"])
    source_height = float(transforms["h"])
    scale = max(width / source_width, height / source_height)
    crop_x = (round(source_width * scale) - width) / 2.0
    crop_y = (round(source_height * scale) - height) / 2.0
    return torch.tensor(
        [
            [transforms["fl_x"] * scale, 0.0, transforms["cx"] * scale - crop_x],
            [0.0, transforms["fl_y"] * scale, transforms["cy"] * scale - crop_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _resize(image: Image.Image, height: int, width: int) -> Image.Image:
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resize_width = round(source_width * scale)
    resize_height = round(source_height * scale)
    image = image.resize((resize_width, resize_height), Image.Resampling.BILINEAR)
    left = (resize_width - width) // 2
    top = (resize_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _sample_nvs_targets(
    candidates: list[int], target_count: int, generator: torch.Generator
) -> list[int]:
    if len(candidates) < target_count:
        return []
    ordered = target_count >= 8 or bool(
        torch.rand((), generator=generator).item() < 0.5
    )
    if not ordered:
        indices = torch.randperm(len(candidates), generator=generator)[:target_count]
        return [candidates[int(index)] for index in indices]
    span_size = int(
        torch.randint(
            target_count, len(candidates) + 1, (1,), generator=generator
        ).item()
    )
    start = int(
        torch.randint(
            len(candidates) - span_size + 1, (1,), generator=generator
        ).item()
    )
    span = candidates[start : start + span_size]
    positions = torch.linspace(0, len(span) - 1, target_count).round().long()
    return [span[int(position)] for position in positions]


class SceneDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        omega_cache: str,
        height: int,
        width: int,
        target_frames: int | list[int] = 80,
        sampling_mode: str = "trajectory",
        repeat: int = 1,
        seed: int = 0,
        max_scenes: int = 0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.cache_root = Path(omega_cache)
        self.height = int(height)
        self.width = int(width)
        counts = [target_frames] if isinstance(target_frames, int) else target_frames
        self.target_frames = sorted({int(value) for value in counts})
        if not self.target_frames or self.target_frames[0] < 1:
            raise ValueError("target_frames must contain positive integers")
        if sampling_mode not in {"nvs", "trajectory"}:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
        if sampling_mode == "trajectory" and len(self.target_frames) != 1:
            raise ValueError("Trajectory sampling requires one target-frame count")
        self.sampling_mode = sampling_mode
        self.repeat = int(repeat)
        self.seed = int(seed)
        self.current_epoch = 0

        manifest = json.loads((self.cache_root / "manifest.json").read_text())
        supported_formats = {
            "vggtdiff_omega_source_bundles_v1",
            "framecrafter_omega_source_bundles_v1",
        }
        if manifest.get("format") not in supported_formats:
            raise ValueError("Unsupported VGGT-Diff Omega cache")
        self.num_bundles = int(manifest["num_bundles"])
        self.source_frames = int(
            manifest.get("num_source_frames", manifest.get("num_input_frames", 0))
        )
        if self.source_frames != 6:
            raise ValueError("The released training protocol expects six source views")
        available = set(manifest["scene_keys"])
        scenes = sorted(path for path in self.dataset_root.iterdir() if path.is_dir())
        self.scenes = [path for path in scenes if path.name in available]
        if max_scenes > 0:
            self.scenes = self.scenes[:max_scenes]
        if not self.scenes:
            raise ValueError("No dataset scenes have complete Omega caches")

    def __len__(self) -> int:
        return len(self.scenes) * self.repeat

    def _bundle_index(self, scene_key: str) -> int:
        digest = hashlib.sha256(scene_key.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "little") % self.num_bundles
        return (offset + self.current_epoch) % self.num_bundles

    def __getitem__(self, index: int) -> dict:
        for offset in range(len(self.scenes)):
            sample_index = (index + offset) % len(self.scenes)
            scene = self.scenes[sample_index]
            bundle_index = self._bundle_index(scene.name)
            package = torch.load(
                self.cache_root / scene.name / f"bundle_{bundle_index:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            transforms = json.loads((scene / "transforms.json").read_text())
            files = _image_files(scene)
            frame_count = min(len(files), len(transforms["frames"]))
            source_ids = [
                int(value) for value in package["source_frame_ids"].tolist()
            ]
            source_set = set(source_ids)

            generator = torch.Generator().manual_seed(
                self.seed + self.current_epoch * len(self) + sample_index
            )
            target_count = self.target_frames[
                int(
                    torch.randint(
                        len(self.target_frames), (1,), generator=generator
                    ).item()
                )
            ]

            if self.sampling_mode == "nvs":
                candidates = [
                    frame_id
                    for frame_id in range(frame_count)
                    if frame_id not in source_set
                ]
                if bundle_index >= math.ceil(0.8 * self.num_bundles):
                    local_candidates = [
                        frame_id
                        for frame_id in candidates
                        if min(source_ids) <= frame_id <= max(source_ids)
                    ]
                    if len(local_candidates) >= target_count:
                        candidates = local_candidates
                target_ids = _sample_nvs_targets(
                    candidates, target_count, generator
                )
                if not target_ids:
                    continue
            else:
                runs = []
                run = []
                for frame_id in range(frame_count):
                    if frame_id in source_set:
                        if run:
                            runs.append(run)
                            run = []
                        continue
                    run.append(frame_id)
                if run:
                    runs.append(run)
                valid_runs = [run for run in runs if len(run) >= target_count]
                if not valid_runs:
                    continue
                selected_run = valid_runs[
                    int(
                        torch.randint(
                            len(valid_runs), (1,), generator=generator
                        ).item()
                    )
                ]
                start = int(
                    torch.randint(
                        len(selected_run) - target_count + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
                target_ids = selected_run[start : start + target_count]
            frame_ids = source_ids + target_ids
            images = []
            for frame_id in frame_ids:
                with Image.open(files[frame_id]) as handle:
                    images.append(
                        _resize(handle.convert("RGB"), self.height, self.width)
                    )

            raw_w2c = _raw_w2c(transforms, frame_ids)
            intrinsic = _intrinsic(transforms, self.height, self.width)
            intrinsics = intrinsic.unsqueeze(0).repeat(len(frame_ids), 1, 1)
            _, normalized_c2w, _ = normalize_source_anchor(
                raw_w2c, self.source_frames
            )
            raymap = plucker_rays(
                normalized_c2w, intrinsics, height=self.height, width=self.width
            )
            return {
                "input_images": images[: self.source_frames],
                "target_images": images,
                "raymap": raymap,
                "omega_tokens": package["omega_tokens"],
                "omega_xyz": package["omega_xyz"],
                "omega_confidence": package["omega_confidence"],
                "omega_target_w2c": raw_w2c[self.source_frames :],
                "omega_target_intrinsics": intrinsics[self.source_frames :],
                "scene_key": scene.name,
                "source_frame_ids": torch.tensor(source_ids),
                "target_frame_ids": torch.tensor(target_ids),
                "prompt": "",
            }
        raise ValueError(
            f"No scene supports target counts {self.target_frames}"
        )
