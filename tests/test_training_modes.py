from types import SimpleNamespace

import torch
from PIL import Image

from vggtdiff.data import _sample_nvs_targets
from vggtdiff.training import VGGTDiffTrainingModule


def _module(training_mode: str) -> VGGTDiffTrainingModule:
    module = VGGTDiffTrainingModule.__new__(VGGTDiffTrainingModule)
    torch.nn.Module.__init__(module)
    module.training_mode = training_mode
    module.pipe = SimpleNamespace(device=torch.device("cpu"))
    return module


def _batch(target_frames: int) -> dict:
    source_frames = 6
    total_frames = source_frames + target_frames
    images = [Image.new("RGB", (16, 16)) for _ in range(total_frames)]
    return {
        "input_images": images[:source_frames],
        "target_images": images,
        "raymap": torch.randn(total_frames, 384, 2, 2),
        "omega_tokens": torch.randn(1, 2, 8),
        "omega_xyz": torch.randn(1, 2, 3),
        "omega_confidence": torch.ones(1, 2),
        "omega_target_w2c": torch.eye(4).repeat(target_frames, 1, 1),
        "omega_target_intrinsics": torch.eye(3).repeat(target_frames, 1, 1),
    }


def test_nvs_targets_keep_independent_view_slots():
    shared, _, _ = _module("nvs")._pipeline_inputs(_batch(12))
    assert shared["num_output_frames"] == 12
    assert shared["num_latent_frames"] == 18
    assert shared["raymap"].shape == (18, 384, 2, 2)
    assert shared["omega_target_w2c"].shape[0] == 12


def test_trajectory_targets_use_causal_slots():
    shared, _, _ = _module("trajectory")._pipeline_inputs(_batch(80))
    assert shared["num_output_frames"] == 21
    assert shared["num_latent_frames"] == 27
    assert shared["raymap"].shape == (27, 1536, 2, 2)
    assert shared["omega_target_w2c"].shape[0] == 21


def test_nvs_sampling_excludes_unavailable_views():
    candidates = [index for index in range(30) if index not in {2, 8, 14}]
    targets = _sample_nvs_targets(
        candidates, 12, torch.Generator().manual_seed(7)
    )
    assert len(targets) == 12
    assert len(set(targets)) == 12
    assert set(targets).issubset(candidates)
