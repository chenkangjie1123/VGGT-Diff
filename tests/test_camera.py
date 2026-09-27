import torch

from vggtdiff.camera import normalize_source_anchor, plucker_rays


def test_source_anchor_is_target_set_invariant():
    w2c = torch.eye(4).repeat(9, 1, 1)
    w2c[:, 0, 3] = torch.linspace(-2, 2, 9)
    _, first, _ = normalize_source_anchor(w2c[:8], source_count=6)
    _, second, _ = normalize_source_anchor(w2c, source_count=6)
    torch.testing.assert_close(first[:6], second[:6])


def test_plucker_shape():
    c2w = torch.eye(4).repeat(4, 1, 1)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 84.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]
    ).repeat(4, 1, 1)
    rays = plucker_rays(c2w, intrinsics, height=96, width=168)
    assert rays.shape == (4, 384, 12, 21)
