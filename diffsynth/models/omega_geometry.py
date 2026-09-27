from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batch(tensor: torch.Tensor, dimensions: int) -> torch.Tensor:
    if tensor.ndim == dimensions - 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != dimensions:
        raise ValueError(f"expected {dimensions - 1}D or {dimensions}D tensor")
    return tensor


def _prepare_inputs(
    features: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    features = _as_batch(features, 4)
    xyz = _as_batch(xyz, 4)
    confidence = _as_batch(confidence, 3)
    target_w2c = _as_batch(target_w2c, 4)
    target_intrinsics = _as_batch(target_intrinsics, 4)
    if features.shape[:3] != xyz.shape[:3] or xyz.shape[:3] != confidence.shape:
        raise ValueError("Omega features, xyz, and confidence must share [B, V, P]")
    if target_w2c.shape[:2] != target_intrinsics.shape[:2]:
        raise ValueError("target camera batches do not match")
    if features.shape[0] != target_w2c.shape[0]:
        raise ValueError("source and target camera batches do not match")
    return features, xyz, confidence, target_w2c, target_intrinsics


def _infer_grid(point_count: int, aspect_ratio: float) -> tuple[int, int]:
    candidates = []
    for height in range(1, int(point_count**0.5) + 1):
        if point_count % height:
            continue
        width = point_count // height
        candidates.extend(((height, width), (width, height)))
    return min(
        candidates,
        key=lambda shape: abs(shape[1] / max(shape[0], 1) - aspect_ratio),
    )


def route_source_latent_layers(
    source_latents: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    output_hw: tuple[int, int] | None = None,
    image_downsample: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Associate source VAE latents with Omega points and route them to targets."""
    if source_latents.ndim != 5:
        raise ValueError("source_latents must have shape [B, C, V, H, W]")
    xyz = _as_batch(xyz, 4)
    if source_latents.shape[0] != xyz.shape[0]:
        raise ValueError("source latent and Omega batches do not match")
    if source_latents.shape[2] != xyz.shape[1]:
        raise ValueError("source latent and Omega view counts do not match")
    batch, channels, views, height, width = source_latents.shape
    point_count = xyz.shape[2]
    token_height, token_width = _infer_grid(point_count, width / max(height, 1))
    resized = F.interpolate(
        source_latents.permute(0, 2, 1, 3, 4).reshape(
            batch * views, channels, height, width
        ),
        size=(token_height, token_width),
        mode="bilinear",
        align_corners=False,
    )
    point_features = resized.reshape(
        batch, views, channels, point_count
    ).permute(0, 1, 3, 2).contiguous()
    return route_omega_feature_layers(
        point_features,
        xyz,
        confidence,
        target_w2c,
        target_intrinsics,
        output_hw=output_hw or (height, width),
        image_downsample=image_downsample,
    )


def route_omega_features(
    features: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    output_hw: tuple[int, int],
    image_downsample: int = 8,
    depth_tolerance: float = 0.03,
) -> torch.Tensor:
    """Z-buffer VGGT-Omega patch features into target latent views."""
    features, xyz, confidence, target_w2c, target_intrinsics = _prepare_inputs(
        features, xyz, confidence, target_w2c, target_intrinsics
    )
    batch, _, _, channels = features.shape
    target_views = target_w2c.shape[1]
    height, width = map(int, output_hw)
    image_height = height * int(image_downsample)
    image_width = width * int(image_downsample)
    routed = features.new_zeros(batch, channels, target_views, height, width)

    flat_features = features.flatten(1, 2).float()
    flat_xyz = xyz.flatten(1, 2).float()
    flat_confidence = confidence.flatten(1, 2).float()
    flat_confidence = torch.nan_to_num(flat_confidence, nan=0.0).clamp_min(0.0)
    confidence_scale = flat_confidence.amax(dim=1, keepdim=True).clamp_min(1e-6)
    flat_confidence = (flat_confidence / confidence_scale).clamp(0.0, 1.0)

    for batch_index in range(batch):
        points = flat_xyz[batch_index]
        point_features = flat_features[batch_index]
        point_confidence = flat_confidence[batch_index]
        finite_points = torch.isfinite(points).all(dim=-1)
        for view_index in range(target_views):
            camera = target_w2c[batch_index, view_index].float()
            intrinsic = target_intrinsics[batch_index, view_index].float()
            camera_xyz = points @ camera[:3, :3].transpose(0, 1) + camera[:3, 3]
            depth = camera_xyz[:, 2]
            valid = finite_points & torch.isfinite(depth) & (depth > 1e-4)
            safe_depth = depth.clamp_min(1e-4)
            pixel_x = intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2]
            pixel_y = intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2]
            latent_x = torch.round((pixel_x + 0.5) * width / image_width - 0.5).long()
            latent_y = torch.round((pixel_y + 0.5) * height / image_height - 0.5).long()
            valid &= (latent_x >= 0) & (latent_x < width)
            valid &= (latent_y >= 0) & (latent_y < height)
            if not torch.any(valid):
                continue

            point_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            pixel_indices = latent_y[point_indices] * width + latent_x[point_indices]
            visible_depth = depth[point_indices]
            zbuffer = visible_depth.new_full((height * width,), torch.inf)
            zbuffer.scatter_reduce_(0, pixel_indices, visible_depth, reduce="amin")
            front = visible_depth <= zbuffer[pixel_indices] * (1.0 + depth_tolerance)
            point_indices = point_indices[front]
            pixel_indices = pixel_indices[front]
            weights = point_confidence[point_indices]

            accumulation = point_features.new_zeros(height * width, channels)
            normalizer = point_features.new_zeros(height * width, 1)
            accumulation.index_add_(
                0, pixel_indices, point_features[point_indices] * weights[:, None]
            )
            normalizer.index_add_(0, pixel_indices, weights[:, None])
            accumulation = accumulation / normalizer.clamp_min(1e-6)
            routed[batch_index, :, view_index] = accumulation.view(
                height, width, channels
            ).permute(2, 0, 1).to(routed.dtype)
    return routed


def route_omega_feature_layers(
    features: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    output_hw: tuple[int, int],
    image_downsample: int = 8,
    second_layer_gap: float = 0.04,
    visibility_temperature: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bilinearly splat soft front/secondary layers and visibility statistics."""
    features, xyz, confidence, target_w2c, target_intrinsics = _prepare_inputs(
        features, xyz, confidence, target_w2c, target_intrinsics
    )
    batch, _, _, channels = features.shape
    target_views = target_w2c.shape[1]
    height, width = map(int, output_hw)
    pixels = height * width
    image_height = height * int(image_downsample)
    image_width = width * int(image_downsample)
    front_output = features.new_zeros(batch, channels, target_views, height, width)
    back_output = torch.zeros_like(front_output)
    statistics = features.new_zeros(batch, 4, target_views, height, width)

    flat_features = features.flatten(1, 2).float()
    flat_xyz = xyz.flatten(1, 2).float()
    flat_confidence = torch.nan_to_num(
        confidence.flatten(1, 2).float(), nan=0.0
    ).clamp_min(0.0)
    confidence_scale = flat_confidence.amax(dim=1, keepdim=True).clamp_min(1e-6)
    flat_confidence = (flat_confidence / confidence_scale).clamp(0.0, 1.0)

    for batch_index in range(batch):
        points = flat_xyz[batch_index]
        point_features = flat_features[batch_index]
        point_confidence = flat_confidence[batch_index]
        finite_points = torch.isfinite(points).all(dim=-1)
        for view_index in range(target_views):
            camera = target_w2c[batch_index, view_index].float()
            intrinsic = target_intrinsics[batch_index, view_index].float()
            camera_xyz = points @ camera[:3, :3].transpose(0, 1) + camera[:3, 3]
            depth = camera_xyz[:, 2]
            valid = finite_points & torch.isfinite(depth) & (depth > 1e-4)
            safe_depth = depth.clamp_min(1e-4)
            latent_x = (
                (intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2] + 0.5)
                * width
                / image_width
                - 0.5
            )
            latent_y = (
                (intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2] + 0.5)
                * height
                / image_height
                - 0.5
            )

            x0 = torch.floor(latent_x)
            y0 = torch.floor(latent_y)
            point_ids = torch.arange(len(points), device=points.device)
            neighbor_points = []
            neighbor_pixels = []
            neighbor_bilinear = []
            for x, y, weight in (
                (x0, y0, (1.0 - (latent_x - x0)) * (1.0 - (latent_y - y0))),
                (x0 + 1, y0, (latent_x - x0) * (1.0 - (latent_y - y0))),
                (x0, y0 + 1, (1.0 - (latent_x - x0)) * (latent_y - y0)),
                (x0 + 1, y0 + 1, (latent_x - x0) * (latent_y - y0)),
            ):
                x_long = x.long()
                y_long = y.long()
                inside = valid & (x_long >= 0) & (x_long < width) & (y_long >= 0) & (y_long < height)
                if torch.any(inside):
                    neighbor_points.append(point_ids[inside])
                    neighbor_pixels.append(y_long[inside] * width + x_long[inside])
                    neighbor_bilinear.append(weight[inside].clamp_min(0.0))
            if not neighbor_points:
                continue

            point_ids = torch.cat(neighbor_points)
            pixel_ids = torch.cat(neighbor_pixels)
            bilinear = torch.cat(neighbor_bilinear)
            entry_depth = depth[point_ids]
            entry_confidence = point_confidence[point_ids]
            base_weight = bilinear * entry_confidence

            front_depth = entry_depth.new_full((pixels,), torch.inf)
            front_depth.scatter_reduce_(0, pixel_ids, entry_depth, reduce="amin")
            front_relative = (entry_depth / front_depth[pixel_ids].clamp_min(1e-6) - 1.0).clamp_min(0.0)
            front_weight = base_weight * torch.exp(-front_relative / visibility_temperature)

            behind_front = entry_depth > front_depth[pixel_ids] * (1.0 + second_layer_gap)
            back_depth = entry_depth.new_full((pixels,), torch.inf)
            if torch.any(behind_front):
                back_depth.scatter_reduce_(
                    0, pixel_ids[behind_front], entry_depth[behind_front], reduce="amin"
                )
            finite_back = torch.isfinite(back_depth[pixel_ids])
            back_relative = torch.zeros_like(entry_depth)
            back_relative[finite_back] = (
                entry_depth[finite_back]
                / back_depth[pixel_ids[finite_back]].clamp_min(1e-6)
                - 1.0
            ).clamp_min(0.0)
            back_weight = base_weight * torch.exp(-back_relative / visibility_temperature)
            back_weight = torch.where(behind_front & finite_back, back_weight, 0.0)

            def aggregate(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                values = point_features.new_zeros(pixels, channels)
                normalizer = point_features.new_zeros(pixels, 1)
                values.index_add_(0, pixel_ids, point_features[point_ids] * weights[:, None])
                normalizer.index_add_(0, pixel_ids, weights[:, None])
                return values / normalizer.clamp_min(1e-6), normalizer[:, 0]

            front_features, front_support = aggregate(front_weight)
            back_features, back_support = aggregate(back_weight)
            confidence_sum = point_features.new_zeros(pixels)
            bilinear_sum = point_features.new_zeros(pixels)
            confidence_sum.index_add_(0, pixel_ids, bilinear * entry_confidence)
            bilinear_sum.index_add_(0, pixel_ids, bilinear)
            mean_confidence = confidence_sum / bilinear_sum.clamp_min(1e-6)
            depth_gap = torch.zeros_like(front_depth)
            has_two_layers = torch.isfinite(front_depth) & torch.isfinite(back_depth)
            depth_gap[has_two_layers] = (
                back_depth[has_two_layers] / front_depth[has_two_layers].clamp_min(1e-6) - 1.0
            ).clamp(0.0, 1.0)

            front_output[batch_index, :, view_index] = front_features.view(
                height, width, channels
            ).permute(2, 0, 1).to(front_output.dtype)
            back_output[batch_index, :, view_index] = back_features.view(
                height, width, channels
            ).permute(2, 0, 1).to(back_output.dtype)
            stats = torch.stack(
                [
                    front_support.clamp(0.0, 1.0),
                    back_support.clamp(0.0, 1.0),
                    mean_confidence.clamp(0.0, 1.0),
                    depth_gap,
                ]
            )
            statistics[batch_index, :, view_index] = stats.view(4, height, width).to(statistics.dtype)
    return front_output, back_output, statistics


def route_target_state_consensus(
    target_states: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    output_hw: tuple[int, int],
    image_downsample: int = 8,
    depth_tolerance: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport current noisy target latents through shared Omega points.

    Each target view contributes its visible latent state to the underlying 3D
    points. A view receives the confidence-weighted consensus from the other
    target views at pixels connected by those points. No clean target signal is
    used; this only couples the jointly sampled diffusion states.
    """
    if target_states.ndim != 5:
        raise ValueError("target_states must have shape [B, C, T, H, W]")
    xyz = _as_batch(xyz, 4)
    confidence = _as_batch(confidence, 3)
    target_w2c = _as_batch(target_w2c, 4)
    target_intrinsics = _as_batch(target_intrinsics, 4)
    batch, channels, target_views, height, width = target_states.shape
    if (height, width) != tuple(map(int, output_hw)):
        raise ValueError("target state resolution does not match output_hw")
    if xyz.shape[:3] != confidence.shape:
        raise ValueError("Omega xyz and confidence must share [B, V, P]")
    if target_w2c.shape[:2] != (batch, target_views):
        raise ValueError("target camera count does not match target states")

    pixels = height * width
    image_height = height * int(image_downsample)
    image_width = width * int(image_downsample)
    points = xyz.flatten(1, 2).float()
    point_confidence = torch.nan_to_num(
        confidence.flatten(1, 2).float(), nan=0.0
    ).clamp_min(0.0)
    confidence_scale = point_confidence.amax(dim=1, keepdim=True).clamp_min(1e-6)
    point_confidence = (point_confidence / confidence_scale).clamp(0.0, 1.0)
    output = target_states.new_zeros(batch, channels, target_views, height, width)
    support = target_states.new_zeros(batch, 1, target_views, height, width)

    for batch_index in range(batch):
        scene_points = points[batch_index]
        finite_points = torch.isfinite(scene_points).all(dim=-1)
        num_points = scene_points.shape[0]
        sampled_states = target_states.new_zeros(
            target_views, num_points, channels, dtype=torch.float32
        )
        point_weights = target_states.new_zeros(
            target_views, num_points, dtype=torch.float32
        )
        view_pixel_indices: list[torch.Tensor] = []
        view_visible: list[torch.Tensor] = []

        for view_index in range(target_views):
            camera = target_w2c[batch_index, view_index].float()
            intrinsic = target_intrinsics[batch_index, view_index].float()
            camera_xyz = (
                scene_points @ camera[:3, :3].transpose(0, 1) + camera[:3, 3]
            )
            depth = camera_xyz[:, 2]
            valid = finite_points & torch.isfinite(depth) & (depth > 1e-4)
            safe_depth = depth.clamp_min(1e-4)
            latent_x = torch.round(
                (intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2] + 0.5)
                * width / image_width - 0.5
            ).long()
            latent_y = torch.round(
                (intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2] + 0.5)
                * height / image_height - 0.5
            ).long()
            valid &= (latent_x >= 0) & (latent_x < width)
            valid &= (latent_y >= 0) & (latent_y < height)
            safe_x = latent_x.clamp(0, width - 1)
            safe_y = latent_y.clamp(0, height - 1)
            pixel_indices = safe_y * width + safe_x

            zbuffer = depth.new_full((pixels,), torch.inf)
            if torch.any(valid):
                zbuffer.scatter_reduce_(0, pixel_indices[valid], depth[valid], reduce="amin")
            visible = valid & (
                depth <= zbuffer[pixel_indices].clamp_min(1e-6) * (1.0 + depth_tolerance)
            )
            state_flat = target_states[
                batch_index, :, view_index
            ].float().flatten(1).transpose(0, 1)
            sampled_states[view_index, visible] = state_flat[pixel_indices[visible]]
            point_weights[view_index, visible] = point_confidence[batch_index, visible]
            view_pixel_indices.append(pixel_indices)
            view_visible.append(visible)

        weighted_sum = (sampled_states * point_weights[..., None]).sum(dim=0)
        weight_sum = point_weights.sum(dim=0)
        for view_index in range(target_views):
            other_weight = weight_sum - point_weights[view_index]
            other_sum = weighted_sum - (
                sampled_states[view_index] * point_weights[view_index, :, None]
            )
            point_consensus = other_sum / other_weight[:, None].clamp_min(1e-6)
            valid = view_visible[view_index] & (other_weight > 0)
            pixel_indices = view_pixel_indices[view_index][valid]
            weights = point_confidence[batch_index, valid] * other_weight[valid]
            accumulation = target_states.new_zeros(
                pixels, channels, dtype=torch.float32
            )
            normalizer = target_states.new_zeros(pixels, 1, dtype=torch.float32)
            accumulation.index_add_(
                0, pixel_indices, point_consensus[valid] * weights[:, None]
            )
            normalizer.index_add_(0, pixel_indices, weights[:, None])
            consensus = accumulation / normalizer.clamp_min(1e-6)
            output[batch_index, :, view_index] = consensus.view(
                height, width, channels
            ).permute(2, 0, 1).to(output.dtype)
            support[batch_index, :, view_index] = (
                normalizer[:, 0].view(height, width).clamp(0.0, 1.0).to(support.dtype)
            )
    return output, support


def point_track_consistency_loss(
    predicted_clean: torch.Tensor,
    clean_target: torch.Tensor,
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    image_downsample: int = 8,
    depth_tolerance: float = 0.03,
    huber_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match clean-latent errors along visible VGGT point correspondences.

    The pairwise target difference is retained, so view-dependent appearance is
    not forced to be identical. The loss only asks the denoiser error for the
    same 3D point to agree across target views.
    """
    if predicted_clean.shape != clean_target.shape or predicted_clean.ndim != 5:
        raise ValueError("clean latent tensors must share [B, C, T, H, W]")
    xyz = _as_batch(xyz, 4)
    confidence = _as_batch(confidence, 3)
    target_w2c = _as_batch(target_w2c, 4)
    target_intrinsics = _as_batch(target_intrinsics, 4)
    batch, channels, target_views, height, width = predicted_clean.shape
    if target_w2c.shape[:2] != (batch, target_views):
        raise ValueError("target camera count does not match clean latents")

    image_height = height * int(image_downsample)
    image_width = width * int(image_downsample)
    pixels = height * width
    points = xyz.flatten(1, 2).float()
    point_confidence = torch.nan_to_num(
        confidence.flatten(1, 2).float(), nan=0.0
    ).clamp_min(0.0)
    confidence_scale = point_confidence.amax(dim=1, keepdim=True).clamp_min(1e-6)
    point_confidence = (point_confidence / confidence_scale).clamp(0.0, 1.0)
    residual = (predicted_clean - clean_target).float()
    weighted_loss = residual.new_zeros(())
    weight_sum = residual.new_zeros(())
    track_count = 0
    pair_count = 0

    for batch_index in range(batch):
        scene_points = points[batch_index]
        finite_points = torch.isfinite(scene_points).all(dim=-1)
        view_residuals: list[torch.Tensor] = []
        view_visible: list[torch.Tensor] = []
        for view_index in range(target_views):
            camera = target_w2c[batch_index, view_index].float()
            intrinsic = target_intrinsics[batch_index, view_index].float()
            camera_xyz = (
                scene_points @ camera[:3, :3].transpose(0, 1) + camera[:3, 3]
            )
            depth = camera_xyz[:, 2]
            valid = finite_points & torch.isfinite(depth) & (depth > 1e-4)
            safe_depth = depth.clamp_min(1e-4)
            latent_x = torch.round(
                (intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2] + 0.5)
                * width / image_width - 0.5
            ).long()
            latent_y = torch.round(
                (intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2] + 0.5)
                * height / image_height - 0.5
            ).long()
            valid &= (latent_x >= 0) & (latent_x < width)
            valid &= (latent_y >= 0) & (latent_y < height)
            safe_x = latent_x.clamp(0, width - 1)
            safe_y = latent_y.clamp(0, height - 1)
            pixel_indices = safe_y * width + safe_x
            zbuffer = depth.new_full((pixels,), torch.inf)
            if torch.any(valid):
                zbuffer.scatter_reduce_(0, pixel_indices[valid], depth[valid], reduce="amin")
            visible = valid & (
                depth <= zbuffer[pixel_indices].clamp_min(1e-6) * (1.0 + depth_tolerance)
            )
            residual_flat = residual[
                batch_index, :, view_index
            ].flatten(1).transpose(0, 1)
            view_residuals.append(residual_flat[pixel_indices])
            view_visible.append(visible)

        for left in range(target_views):
            for right in range(left + 1, target_views):
                shared = view_visible[left] & view_visible[right]
                if not torch.any(shared):
                    continue
                differences = view_residuals[left][shared] - view_residuals[right][shared]
                per_track = F.smooth_l1_loss(
                    differences,
                    torch.zeros_like(differences),
                    beta=float(huber_beta),
                    reduction="none",
                ).mean(dim=-1)
                weights = point_confidence[batch_index, shared]
                weighted_loss = weighted_loss + (per_track * weights).sum()
                weight_sum = weight_sum + weights.sum()
                track_count += int(shared.sum().item())
                pair_count += 1

    loss = weighted_loss / weight_sum.clamp_min(1e-6)
    statistics = {
        "track_count": residual.new_tensor(float(track_count)),
        "pair_count": residual.new_tensor(float(pair_count)),
        "weight_sum": weight_sum.detach(),
    }
    return loss, statistics
