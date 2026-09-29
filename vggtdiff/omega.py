from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _unwrap_state_dict(package):
    if isinstance(package, dict):
        for key in ("model", "state_dict"):
            if key in package and isinstance(package[key], dict):
                return package[key]
    return package


def _camera_centers(w2c: np.ndarray) -> np.ndarray:
    return -np.einsum("...ji,...j->...i", w2c[..., :3, :3], w2c[..., :3, 3])


def _similarity_transform(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right
    variance = np.square(source_centered).sum() / len(source)
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid VGGT-Omega camera alignment scale: {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _unproject(depth, extrinsic, intrinsic):
    depth = depth[..., 0]
    views, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x[None].expand(views, -1, -1)
    y = y[None].expand(views, -1, -1)
    fx = intrinsic[:, 0, 0, None, None]
    fy = intrinsic[:, 1, 1, None, None]
    cx = intrinsic[:, 0, 2, None, None]
    cy = intrinsic[:, 1, 2, None, None]
    camera = torch.stack(
        ((x - cx) / fx * depth, (y - cy) / fy * depth, depth), dim=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return torch.einsum(
        "vij,vhwj->vhwi",
        rotation.transpose(1, 2),
        camera - translation[:, None, None],
    )


def _pool_patch_geometry(xyz, confidence, patch_size=16):
    views, height, width, _ = xyz.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    xyz = xyz[:, : patch_height * patch_size, : patch_width * patch_size]
    confidence = confidence[:, : patch_height * patch_size, : patch_width * patch_size]
    xyz = xyz.view(views, patch_height, patch_size, patch_width, patch_size, 3)
    confidence = confidence.view(
        views, patch_height, patch_size, patch_width, patch_size
    )
    valid = torch.isfinite(xyz).all(dim=-1) & torch.isfinite(confidence)
    weights = torch.where(valid, confidence.float().clamp_min(0), 0)
    denominator = weights.sum(dim=(2, 4)).clamp_min(1e-6)
    pooled_xyz = (torch.nan_to_num(xyz.float()) * weights[..., None]).sum(
        dim=(2, 4)
    ) / denominator[..., None]
    pooled_confidence = denominator / float(patch_size * patch_size)
    return pooled_xyz.flatten(1, 2), pooled_confidence.flatten(1, 2)


class OmegaRuntime(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        layer_index: int = 23,
        image_resolution: int = 512,
    ) -> None:
        super().__init__()
        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as error:
            raise ImportError(
                "Install VGGT-Omega from https://github.com/facebookresearch/vggt-omega"
            ) from error

        self.layer_index = int(layer_index)
        self.image_resolution = int(image_resolution)
        self.model = VGGTOmega(
            enable_camera=True, enable_depth=True, enable_alignment=False
        )
        package = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(_unwrap_state_dict(package), strict=True)
        self.model.eval().requires_grad_(False).to(device)
        if self.layer_index not in self.model.aggregator.cached_layer_indices:
            raise ValueError(f"VGGT-Omega layer {self.layer_index} is unavailable")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        from vggt_omega.utils.load_fn import (
            _balanced_target_shape,
            _crop_to_supported_aspect_ratio,
            _pad_images_to_common_size,
        )

        tensors = []
        shapes = set()
        for source in images:
            image = _crop_to_supported_aspect_ratio(source.convert("RGB"))
            width, height = image.size
            target_height, target_width = _balanced_target_shape(
                height / max(width, 1), self.image_resolution, 16
            )
            image = image.resize(
                (target_width, target_height), Image.Resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.uint8).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
            tensors.append(tensor)
            shapes.add(tuple(tensor.shape[-2:]))
        if len(shapes) > 1:
            tensors = _pad_images_to_common_size(tensors, shapes)
        return torch.stack(tensors).unsqueeze(0).to(self.device, non_blocking=True)

    @torch.inference_mode()
    def estimate_cameras(
        self, images: list[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate OpenCV world-to-camera poses and original-pixel intrinsics from RGB."""
        from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if len(images) != 6:
            raise ValueError("Exactly six source images are required")
        if len({image.size for image in images}) != 1:
            raise ValueError("All source images must have the same resolution")
        tensors = self._preprocess(images)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            layers, patch_start = self.model.aggregator(tensors)
        with torch.autocast(device_type="cuda", enabled=False):
            pose = self.model.camera_head(
                [None if layer is None else layer.detach() for layer in layers],
                patch_token_start=patch_start,
            )
            predicted_w2c, predicted_intrinsics = encoding_to_camera(
                pose, tensors.shape[-2:]
            )

        w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 6, axis=0)
        w2c[:, :3, :] = predicted_w2c[0].float().cpu().numpy()
        omega_intrinsics = predicted_intrinsics[0].float().cpu().numpy()
        omega_height, omega_width = tensors.shape[-2:]
        intrinsics = []
        for index, image in enumerate(images):
            width, height = image.size
            cropped_width, cropped_height = _crop_to_supported_aspect_ratio(
                image.convert("RGB")
            ).size
            left = (width - cropped_width) // 2
            top = (height - cropped_height) // 2
            to_original_pixels = np.array(
                [
                    [cropped_width / omega_width, 0, left],
                    [0, cropped_height / omega_height, top],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            intrinsics.append(to_original_pixels @ omega_intrinsics[index])
        intrinsics = np.stack(intrinsics)
        centers = np.linalg.inv(w2c)[:, :3, 3]
        baseline = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        if not np.isfinite(w2c).all() or not np.isfinite(intrinsics).all():
            raise ValueError("VGGT-Omega returned non-finite cameras")
        if np.median(baseline[np.triu_indices(6, 1)]) <= 1e-5:
            raise ValueError("VGGT-Omega returned coincident camera centers")
        if np.max(np.abs(np.linalg.det(w2c[:, :3, :3]) - 1)) > 1e-3:
            raise ValueError("VGGT-Omega returned non-rigid camera rotations")
        return w2c, intrinsics

    @torch.inference_mode()
    def forward(
        self,
        images: list[Image.Image],
        raw_w2c: torch.Tensor,
        raw_intrinsics: torch.Tensor,
        num_output_frames: int,
    ) -> dict[str, torch.Tensor]:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        tensors = self._preprocess(images)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            layers, patch_start = self.model.aggregator(tensors)
        tokens = layers[self.layer_index][:, :, patch_start:].contiguous()
        geometry_layers = [None if layer is None else layer.detach() for layer in layers]
        with torch.autocast(device_type="cuda", enabled=False):
            pose = self.model.camera_head(
                geometry_layers, patch_token_start=patch_start
            )
            depth, confidence = self.model.dense_head(
                geometry_layers, images=tensors, patch_token_start=patch_start
            )
            predicted_w2c, predicted_intrinsics = encoding_to_camera(
                pose, tensors.shape[-2:]
            )
            world = _unproject(
                depth[0].float(),
                predicted_w2c[0].float(),
                predicted_intrinsics[0].float(),
            )
            if confidence.ndim == 5 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            xyz, pooled_confidence = _pool_patch_geometry(
                world, confidence[0].float()
            )

        source_count = len(images)
        predicted_centers = _camera_centers(predicted_w2c[0].float().cpu().numpy())
        raw_centers = _camera_centers(raw_w2c[:source_count].float().cpu().numpy())
        scale, rotation, translation = _similarity_transform(
            predicted_centers, raw_centers
        )
        rotation = torch.from_numpy(rotation).to(xyz)
        translation = torch.from_numpy(translation).to(xyz)
        xyz = scale * torch.einsum("ij,vpj->vpi", rotation, xyz) + translation
        target_slice = slice(len(raw_w2c) - num_output_frames, len(raw_w2c))
        return {
            "omega_tokens": tokens[0].detach(),
            "omega_xyz": xyz.detach(),
            "omega_confidence": pooled_confidence.detach(),
            "omega_target_w2c": raw_w2c[target_slice].to(self.device),
            "omega_target_intrinsics": raw_intrinsics[target_slice].to(self.device),
        }
