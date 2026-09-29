"""Antialiased, synchronized camera-path visualization for pose-free inference."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _frustum(c2w: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int], depth: float):
    center = c2w[:3, 3]
    right, down, forward = c2w[:3, :3].T
    half_width = size[0] / (2 * intrinsic[0, 0]) * depth
    half_height = size[1] / (2 * intrinsic[1, 1]) * depth
    plane = center + forward * depth
    corners = [
        plane + sx * half_width * right + sy * half_height * down
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return np.asarray([center, *corners])


def trajectory_frames(camera_data: dict, scale: float = 0.25):
    """Yield a 640x480 RGB camera panel for every predicted frame."""
    if scale <= 0:
        raise ValueError("Frustum scale must be positive")
    source = np.linalg.inv(np.asarray(camera_data["source_w2c"], dtype=np.float64))
    target = np.linalg.inv(np.asarray(camera_data["target_w2c"], dtype=np.float64))
    source_k = np.asarray(camera_data["source_intrinsics"], dtype=np.float64)
    target_k = np.asarray(camera_data["target_intrinsics"], dtype=np.float64)
    image_size = tuple(camera_data["source_image_size"])
    centers = np.concatenate((source[:, :3, 3], target[:, :3, 3]))
    up = _unit(np.mean(-source[:, :3, 1], axis=0))
    horizontal = centers - centers.mean(axis=0)
    horizontal -= np.outer(horizontal @ up, up)
    _, _, basis = np.linalg.svd(horizontal, full_matrices=False)
    major_candidate = basis[0] - np.dot(basis[0], up) * up
    if np.linalg.norm(major_candidate) < 1e-8:
        source_right = source[0, :3, 0]
        major_candidate = source_right - np.dot(source_right, up) * up
    major = _unit(major_candidate)
    if np.dot(source[-1, :3, 3] - source[0, :3, 3], major) < 0:
        major *= -1
    minor = _unit(np.cross(up, major))
    eye = _unit(0.68 * minor + 0.48 * major + 0.56 * up)
    screen_right = _unit(np.cross(up, eye))
    screen_up = _unit(np.cross(eye, screen_right))
    baseline = np.linalg.norm(
        source[:, None, :3, 3] - source[None, :, :3, 3], axis=-1
    )
    spread = float(np.median(baseline[np.triu_indices(len(source), 1)]))
    depth = max(0.11 * spread, 1e-3) * scale
    source_frustums = [
        _frustum(pose, intrinsic, image_size, depth)
        for pose, intrinsic in zip(source, source_k)
    ]
    target_frustums = [
        _frustum(pose, intrinsic, image_size, depth)
        for pose, intrinsic in zip(target, target_k)
    ]
    origin = centers.mean(axis=0)
    vertices = np.concatenate(source_frustums + target_frustums)
    coordinates = np.stack(
        ((vertices - origin) @ screen_right, (vertices - origin) @ screen_up), axis=1
    )
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    span = np.maximum(high - low, 1e-3)
    factor = min(530 / span[0], 350 / span[1])
    midpoint = (low + high) / 2
    aa = 3

    def project(points):
        points = np.asarray(points)
        x = (points - origin) @ screen_right
        y = (points - origin) @ screen_up
        return [
            (round((320 + factor * (px - midpoint[0])) * aa),
             round((245 - factor * (py - midpoint[1])) * aa))
            for px, py in zip(np.atleast_1d(x), np.atleast_1d(y))
        ]

    source_geometry = [project(vertices) for vertices in source_frustums]
    target_geometry = [project(vertices) for vertices in target_frustums]
    path = project(target[:, :3, 3])
    blue, magenta = (44, 178, 235), (247, 103, 206)
    base = Image.new("RGB", (640 * aa, 480 * aa), (14, 23, 37))
    background = ImageDraw.Draw(base)
    for x in range(40, 640, 40):
        background.line([(x * aa, 30 * aa), (x * aa, 450 * aa)], fill=(25, 42, 59))
    for y in range(40, 480, 40):
        background.line([(20 * aa, y * aa), (620 * aa, y * aa)], fill=(25, 42, 59))

    def draw_camera(draw, geometry, color, fill):
        center, corners = geometry[0], geometry[1:]
        draw.polygon(corners, fill=fill)
        for corner in corners:
            draw.line([center, corner], fill=color, width=2 * aa)
        draw.line(corners + [corners[0]], fill=color, width=2 * aa, joint="curve")
        radius = 3 * aa
        draw.ellipse(
            (center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius),
            fill=color,
        )

    for index, geometry in enumerate(target_geometry):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.line(path, fill=(58, 76, 101), width=3 * aa, joint="curve")
        if index:
            draw.line(path[: index + 1], fill=magenta, width=3 * aa, joint="curve")
        for source_camera in source_geometry:
            draw_camera(draw, source_camera, blue, (19, 43, 58))
        draw_camera(draw, geometry, magenta, (63, 30, 62))
        image = image.resize((640, 480), Image.Resampling.LANCZOS)
        yield np.asarray(image)
