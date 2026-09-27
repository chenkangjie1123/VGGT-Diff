from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_images_and_intrinsics(
    images: list[Image.Image],
    intrinsics: np.ndarray,
    height: int,
    width: int,
) -> tuple[list[Image.Image], np.ndarray]:
    resized = []
    adjusted = np.asarray(intrinsics, dtype=np.float32).copy()
    source_size = images[0].size
    if any(image.size != source_size for image in images):
        raise ValueError("All source images must have the same resolution")
    source_width, source_height = source_size
    scale = max(width / source_width, height / source_height)
    resize_width = round(source_width * scale)
    resize_height = round(source_height * scale)
    crop_x = (resize_width - width) / 2.0
    crop_y = (resize_height - height) / 2.0
    for image in images:
        image = image.convert("RGB").resize(
            (resize_width, resize_height), Image.Resampling.BICUBIC
        )
        left = round(crop_x)
        top = round(crop_y)
        resized.append(image.crop((left, top, left + width, top + height)))
    adjusted[:, 0, 0] *= scale
    adjusted[:, 1, 1] *= scale
    adjusted[:, 0, 2] = adjusted[:, 0, 2] * scale - crop_x
    adjusted[:, 1, 2] = adjusted[:, 1, 2] * scale - crop_y
    return resized, adjusted


def normalize_source_anchor(
    w2c: torch.Tensor, source_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if w2c.ndim != 3 or w2c.shape[-2:] != (4, 4):
        raise ValueError("w2c must have shape [N, 4, 4]")
    if not 0 < source_count <= len(w2c):
        raise ValueError("source_count must select a non-empty camera prefix")
    c2w = torch.linalg.inv(w2c)
    rotations = c2w[:, :3, :3]
    translations = c2w[:, :3, 3]
    source_translations = translations[:source_count]
    centroid = source_translations.mean(dim=0)
    anchor = int(
        torch.linalg.vector_norm(source_translations - centroid, dim=-1).argmin()
    )
    alignment = rotations[anchor].transpose(0, 1)
    translations = torch.einsum(
        "ij,vj->vi", alignment, translations - translations[anchor]
    )
    rotations = torch.einsum("ij,vjk->vik", alignment, rotations)
    scale = torch.linalg.vector_norm(
        translations[:source_count], dim=-1
    ).mean().clamp_min(1e-12)
    normalized = torch.zeros_like(c2w)
    normalized[:, :3, :3] = rotations
    normalized[:, :3, 3] = translations / scale
    normalized[:, 3, 3] = 1
    return torch.linalg.inv(normalized), normalized, scale


def plucker_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    downsample_factor: int = 8,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=c2w.device, dtype=c2w.dtype),
        torch.arange(width, device=c2w.device, dtype=c2w.dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        [x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1
    )
    directions = torch.einsum(
        "vij,hwj->vhwi", intrinsics.float().inverse().to(c2w), pixels
    )
    directions = torch.einsum(
        "vij,vhwj->vhwi", c2w[:, :3, :3], directions
    )
    directions = F.normalize(directions, dim=-1)
    origins = c2w[:, None, None, :3, 3].expand_as(directions)
    moments = torch.cross(origins, directions, dim=-1)
    rays = torch.cat([directions, moments], dim=-1).permute(0, 3, 1, 2)
    return torch.nn.PixelUnshuffle(downsample_factor)(rays)
