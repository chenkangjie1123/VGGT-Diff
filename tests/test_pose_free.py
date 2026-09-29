import json

import numpy as np

from vggtdiff.pose_free import default_viewpoint_trajectory, load_target_trajectory
from vggtdiff.trajectory_visualization import trajectory_frames


def _six_cameras():
    c2w = np.repeat(np.eye(4)[None], 6, axis=0)
    c2w[:, 0, 3] = np.arange(6)
    intrinsics = np.repeat(np.eye(3)[None], 6, axis=0)
    intrinsics[:, 0, 0] = 700
    intrinsics[:, 1, 1] = 700
    intrinsics[:, 0, 2] = 416
    intrinsics[:, 1, 2] = 240
    return np.linalg.inv(c2w), intrinsics


def test_default_route_visits_ordered_source_viewpoints():
    source_w2c, source_k = _six_cameras()
    order = (0, 2, 1, 3, 4, 5)
    target_w2c, target_k = default_viewpoint_trajectory(
        source_w2c, source_k, frames=80, route_order=order
    )
    centers = np.linalg.inv(target_w2c)[:, :3, 3]
    assert target_w2c.shape == (80, 4, 4)
    assert target_k.shape == (80, 3, 3)
    np.testing.assert_allclose(centers[0], [0, 0, 0])
    np.testing.assert_allclose(centers[-1], [5, 0, 0])
    assert np.isfinite(target_w2c).all()


def test_relative_target_poses_are_composed_with_estimated_source0(tmp_path):
    source_w2c, source_k = _six_cameras()
    source_w2c[0, 0, 3] = -10
    relative = np.repeat(np.eye(4)[None], 80, axis=0)
    relative[:, 1, 3] = 2
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"target_c2w_relative_to_source0": relative.tolist()}))
    target_w2c, target_k = load_target_trajectory(path, source_w2c, source_k)
    np.testing.assert_allclose(np.linalg.inv(target_w2c[0])[:3, 3], [10, 2, 0])
    np.testing.assert_allclose(target_k[0], source_k[0])


def test_trajectory_preview_has_one_antialiased_panel_per_target():
    source_w2c, source_k = _six_cameras()
    target_w2c, target_k = default_viewpoint_trajectory(
        source_w2c, source_k, frames=80
    )
    camera_data = {
        "source_w2c": source_w2c.tolist(),
        "source_intrinsics": source_k.tolist(),
        "target_w2c": target_w2c.tolist(),
        "target_intrinsics": target_k.tolist(),
        "source_image_size": [832, 480],
    }
    panels = trajectory_frames(camera_data)
    first = next(panels)
    assert first.shape == (480, 640, 3)
    assert first.dtype == np.uint8
    assert sum(1 for _ in panels) == 79
