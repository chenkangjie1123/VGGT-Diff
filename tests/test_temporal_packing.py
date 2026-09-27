import torch

from diffsynth.pipelines.wan_video import _pack_temporal_raymaps


def test_ordered_temporal_packing_preserves_all_four_poses():
    source_frames = 6
    target_frames = 80
    raymap = torch.arange(
        (source_frames + target_frames) * 3, dtype=torch.float32
    ).view(source_frames + target_frames, 3, 1, 1)
    packed = _pack_temporal_raymaps(raymap, source_frames, target_frames, 4)
    assert packed.shape == (27, 12, 1, 1)
    first_group = packed[source_frames]
    expected = raymap[source_frames].repeat(4, 1, 1)
    torch.testing.assert_close(first_group, expected)
