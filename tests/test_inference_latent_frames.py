from types import SimpleNamespace

import torch
from PIL import Image

from diffsynth.diffusion.base_pipeline import PipelineUnit, PipelineUnitRunner
from diffsynth.pipelines.wan_video import WanVideoPipeline


def _dit(temporal_target_compression: int) -> SimpleNamespace:
    return SimpleNamespace(
        individual_encoding=True,
        temporal_target_compression=temporal_target_compression,
        temporal_plucker_packing=temporal_target_compression > 1,
    )


def _run_pipeline(dit: SimpleNamespace, target_frames: int) -> dict:
    pipe = WanVideoPipeline.__new__(WanVideoPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.scheduler = SimpleNamespace(
        set_timesteps=lambda *args, **kwargs: None, timesteps=[]
    )
    pipe.unit_runner = PipelineUnitRunner()
    pipe.dit = dit
    pipe.device = torch.device("cpu")
    pipe.device_type = "cpu"
    pipe.torch_dtype = torch.float32
    pipe.vram_management_enabled = False
    pipe.in_iteration_models = ("dit",)
    pipe.model_fn = lambda **kwargs: None
    pipe.vae = SimpleNamespace(
        decode=lambda latents, **kwargs: torch.zeros(
            1, 3, 1, latents.shape[3], latents.shape[4]
        )
    )
    captured = {}

    class CaptureUnit(PipelineUnit):
        def __init__(self):
            super().__init__(
                input_params=("num_latent_frames", "raymap"),
                output_params=("latents",),
            )

        def process(self, pipe, num_latent_frames=None, raymap=None):
            captured["num_latent_frames"] = num_latent_frames
            captured["raymap"] = raymap
            length = num_latent_frames if num_latent_frames is not None else 1
            return {"latents": torch.zeros(1, 16, length, 8, 8)}

    pipe.units = [CaptureUnit()]

    source_frames = 6
    height, width = 64, 64
    total_frames = source_frames + target_frames
    raymap = torch.randn(total_frames, 384, height // 8, width // 8)
    pipe(
        prompt="",
        negative_prompt="",
        input_image=[Image.new("RGB", (width, height))] * source_frames,
        raymap=raymap,
        height=height,
        width=width,
        num_frames=total_frames,
        num_latent_frames=None,
        num_output_frames=target_frames,
        num_physical_output_frames=target_frames,
        cfg_scale=1.0,
        omega_cfg_scale=1.0,
        num_inference_steps=1,
        seed=0,
        rand_device="cpu",
        tiled=False,
        progress_bar_cmd=lambda timesteps: timesteps,
    )
    return captured


def test_nvs_inference_keeps_one_latent_slot_per_frame():
    captured = _run_pipeline(_dit(temporal_target_compression=1), target_frames=12)
    assert captured["num_latent_frames"] == 18
    assert captured["raymap"].shape[0] == 18


def test_trajectory_inference_keeps_compressed_slots():
    captured = _run_pipeline(_dit(temporal_target_compression=4), target_frames=80)
    assert captured["num_latent_frames"] == 27
    assert captured["raymap"].shape == (27, 1536, 8, 8)
