"""Camera trajectory helpers for six RGB inputs without supplied source poses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _check_poses(poses: np.ndarray, count: int, name: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{name} must have shape [{count}, 4, 4] and finite values")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4):
        raise ValueError(f"{name} must contain homogeneous rigid transforms")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError(f"{name} must contain orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError(f"{name} must contain right-handed rotations")
    return poses


def _check_intrinsics(values: np.ndarray, count: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (count, 3, 3) or not np.isfinite(values).all():
        raise ValueError(f"{name} must have shape [{count}, 3, 3] and finite values")
    if np.any(values[:, 0, 0] <= 0) or np.any(values[:, 1, 1] <= 0):
        raise ValueError(f"{name} must have positive focal lengths")
    return values


def _interpolate_rotation(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return start.copy()
    if amount >= 1:
        return end.copy()
    relative = start.T @ end
    angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    if angle < 1e-7:
        return start.copy()
    if np.pi - angle < 1e-4:
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))])
    else:
        axis = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        ) / (2 * np.sin(angle))
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + np.sin(amount * angle) * skew
    rotation += (1 - np.cos(amount * angle)) * (skew @ skew)
    return start @ rotation


def default_viewpoint_trajectory(
    source_w2c: np.ndarray,
    source_intrinsics: np.ndarray,
    frames: int = 80,
    route_order: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Move between the six observed camera centers in the supplied walking order."""
    source_w2c = _check_poses(source_w2c, 6, "source_w2c")
    source_intrinsics = _check_intrinsics(source_intrinsics, 6, "source_intrinsics")
    if frames < 6 or frames % 4:
        raise ValueError("frames must be at least six and divisible by four")
    if sorted(route_order) != list(range(6)):
        raise ValueError("route_order must contain each source index from 0 to 5 once")
    ordered = np.linalg.inv(source_w2c)[list(route_order)]
    lengths = np.linalg.norm(np.diff(ordered[:, :3, 3], axis=0), axis=1)
    if np.any(lengths <= 1e-5):
        raise ValueError("Adjacent source viewpoints have coincident centers")
    knots = np.concatenate(([0], np.cumsum(lengths)))
    samples = np.linspace(0, knots[-1], frames)
    target_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    for frame, distance in enumerate(samples):
        segment = min(int(np.searchsorted(knots, distance, side="right") - 1), 4)
        amount = float(np.clip((distance - knots[segment]) / lengths[segment], 0, 1))
        target_c2w[frame, :3, 3] = (
            (1 - amount) * ordered[segment, :3, 3]
            + amount * ordered[segment + 1, :3, 3]
        )
        target_c2w[frame, :3, :3] = _interpolate_rotation(
            ordered[segment, :3, :3], ordered[segment + 1, :3, :3], amount
        )
    target_w2c = np.linalg.inv(target_c2w).astype(np.float32)
    target_intrinsics = np.repeat(
        np.median(source_intrinsics, axis=0)[None], frames, axis=0
    ).astype(np.float32)
    return target_w2c, target_intrinsics


def load_target_trajectory(
    path: str | Path,
    source_w2c: np.ndarray,
    source_intrinsics: np.ndarray,
    frames: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Load optional target cameras in Omega's world or relative to source view 0."""
    data = json.loads(Path(path).read_text())
    source_w2c = _check_poses(source_w2c, 6, "source_w2c")
    source_intrinsics = _check_intrinsics(source_intrinsics, 6, "source_intrinsics")
    has_world = "target_w2c" in data
    has_relative = "target_c2w_relative_to_source0" in data
    if has_world == has_relative:
        raise ValueError(
            "Provide exactly one of target_w2c or target_c2w_relative_to_source0"
        )
    if has_world:
        target_w2c = _check_poses(data["target_w2c"], frames, "target_w2c")
    else:
        relative = _check_poses(
            data["target_c2w_relative_to_source0"],
            frames,
            "target_c2w_relative_to_source0",
        )
        target_c2w = np.linalg.inv(source_w2c[0])[None] @ relative
        target_w2c = np.linalg.inv(target_c2w)
    if "target_intrinsics" in data:
        intrinsics = _check_intrinsics(
            data["target_intrinsics"], frames, "target_intrinsics"
        )
    else:
        intrinsics = np.repeat(
            np.median(source_intrinsics, axis=0)[None], frames, axis=0
        )
    return target_w2c.astype(np.float32), intrinsics.astype(np.float32)
