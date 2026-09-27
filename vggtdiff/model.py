from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.models.omega_condition import OmegaGridAdapter, expand_patch_embedding
from diffsynth.pipelines.wan_video import WanVideoPipeline

from .camera import normalize_source_anchor, plucker_rays, resize_images_and_intrinsics
from .checkpoint import validate_checkpoint
from .omega import OmegaRuntime


def _model_configs(base_model_dir: str | None):
    if base_model_dir is None:
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P"
        configs = [
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern=(
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                ),
            ),
        ]
        tokenizer = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        )
        return configs, tokenizer

    root = Path(base_model_dir)
    if list(root.glob("diffusion_pytorch_model*.safetensors")):
        model_root = root
    else:
        model_root = root / "Wan-AI" / "Wan2.1-I2V-14B-480P"
    dit_files = sorted(glob.glob(str(model_root / "diffusion_pytorch_model*.safetensors")))
    if not dit_files:
        raise FileNotFoundError(f"Wan2.1 DiT shards not found under {model_root}")

    converted = root / "DiffSynth-Studio" / "Wan-Series-Converted-Safetensors"

    def resolve(converted_name: str, original_name: str) -> str:
        candidates = (converted / converted_name, model_root / original_name)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"Missing Wan2.1 component: {original_name}")

    configs = [
        ModelConfig(path=dit_files),
        ModelConfig(
            path=resolve(
                "models_t5_umt5-xxl-enc-bf16.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
        ),
        ModelConfig(path=resolve("Wan2.1_VAE.safetensors", "Wan2.1_VAE.pth")),
        ModelConfig(
            path=resolve(
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
        ),
    ]
    tokenizer_candidates = (
        root / "tokenizer",
        root.parent / "tokenizer" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl",
        root / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl",
    )
    tokenizer_path = next(
        (path for path in tokenizer_candidates if path.is_dir()), None
    )
    if tokenizer_path is None:
        raise FileNotFoundError("Wan2.1 tokenizer directory was not found")
    tokenizer = ModelConfig(path=str(tokenizer_path))
    return configs, tokenizer


def _replace_patch_embedding(model, input_channels: int) -> None:
    source = model.patch_embedding
    replacement = nn.Conv3d(
        input_channels,
        model.dim,
        kernel_size=model.patch_size,
        stride=model.patch_size,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.zero_()
        channels = min(source.in_channels, input_channels)
        replacement.weight[:, :channels].copy_(source.weight[:, :channels])
        if source.bias is not None:
            replacement.bias.copy_(source.bias)
    model.patch_embedding = replacement
    model.in_dim = input_channels


def configure_dit(model) -> None:
    model.individual_encoding = True
    model.temporal_target_compression = 4
    model.temporal_plucker_packing = True
    model.temporal_plucker_adapter = nn.Conv3d(
        1536,
        384,
        kernel_size=1,
        bias=False,
        device=model.patch_embedding.weight.device,
        dtype=model.patch_embedding.weight.dtype,
    )
    _replace_patch_embedding(model, 420)
    model.omega_adapter = OmegaGridAdapter(
        token_dim=2048,
        output_channels=32,
        router_mode="soft_multilayer",
        num_feature_layers=1,
    ).to(model.patch_embedding.weight)
    model.patch_embedding = expand_patch_embedding(model.patch_embedding, 32)
    model.in_dim += 32


class VGGTDiff:
    def __init__(
        self,
        checkpoint: str,
        omega_checkpoint: str,
        base_model_dir: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("VGGT-Diff inference requires a CUDA-compatible device")
        validate_checkpoint(checkpoint)
        configs, tokenizer = _model_configs(base_model_dir)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype,
            device=device,
            model_configs=configs,
            tokenizer_config=tokenizer,
            redirect_common_files=False,
        )
        configure_dit(self.pipe.dit)
        state = load_state_dict(checkpoint, torch_dtype=dtype, device="cpu")
        result = self.pipe.dit.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [
            key
            for key in result.missing_keys
            if not key.endswith("freqs") and "rope" not in key
        ]
        if unexpected or missing:
            raise ValueError(
                "Checkpoint architecture mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.omega = OmegaRuntime(
            checkpoint_path=omega_checkpoint,
            device=torch.device(device),
            layer_index=23,
            image_resolution=512,
        )
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def generate(
        self,
        source_images: list[Image.Image],
        w2c: np.ndarray,
        intrinsics: np.ndarray,
        height: int = 192,
        width: int = 336,
        steps: int = 50,
        seed: int = 42,
    ) -> list[Image.Image]:
        if len(source_images) != 6:
            raise ValueError("The released checkpoint expects exactly six source images")
        w2c = np.asarray(w2c, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if w2c.ndim != 3 or w2c.shape[-2:] != (4, 4):
            raise ValueError("w2c must have shape [6 + N, 4, 4]")
        if intrinsics.shape != (len(w2c), 3, 3):
            raise ValueError("intrinsics must have shape [6 + N, 3, 3]")
        if len(w2c) <= len(source_images):
            raise ValueError("At least one target camera is required")

        omega_images = [image.convert("RGB") for image in source_images]
        source_images, intrinsics = resize_images_and_intrinsics(
            omega_images, intrinsics, height, width
        )
        raw_w2c = torch.from_numpy(w2c)
        raw_intrinsics = torch.from_numpy(intrinsics)
        _, normalized_c2w, _ = normalize_source_anchor(raw_w2c, len(source_images))
        raymap = plucker_rays(
            normalized_c2w, raw_intrinsics, height=height, width=width
        )
        target_count = len(w2c) - len(source_images)
        omega = self.omega(
            images=omega_images,
            raw_w2c=raw_w2c,
            raw_intrinsics=raw_intrinsics,
            num_output_frames=target_count,
        )

        def move(value):
            return value.to(self.device, dtype=self.dtype, non_blocking=True)

        generated = self.pipe(
            prompt="",
            negative_prompt="",
            input_image=source_images,
            raymap=move(raymap),
            height=height,
            width=width,
            num_frames=len(source_images) + target_count,
            num_latent_frames=None,
            num_output_frames=target_count,
            num_physical_output_frames=target_count,
            cfg_scale=1.0,
            omega_cfg_scale=1.0,
            num_inference_steps=steps,
            seed=seed,
            rand_device="cuda",
            tiled=False,
            omega_tokens=move(omega["omega_tokens"]),
            omega_xyz=move(omega["omega_xyz"]),
            omega_confidence=move(omega["omega_confidence"]),
            omega_target_w2c=move(omega["omega_target_w2c"]),
            omega_target_intrinsics=move(omega["omega_target_intrinsics"]),
        )
        return generated[-target_count:]
