import os
import torch, types
import numpy as np
from PIL import Image
from einops import rearrange
from typing import Optional, Union
from tqdm import tqdm
from typing_extensions import Literal

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, modulate, rope_apply, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.omega_geometry import (
    route_omega_feature_layers,
    route_source_latent_layers,
)


def _pack_temporal_raymaps(
    raymap: torch.Tensor,
    source_frames: int,
    physical_targets: int,
    compression: int,
) -> torch.Tensor:
    """Pack every physical target ray map into its causal VAE time group.

    Sources are repeated across the subframe-channel axis so an endpoint-
    initialized projection reproduces the original source conditioning. Target
    groups follow the Wan causal layout: [0], [1..4], [5..8], ... .
    """
    if int(compression) != 4:
        raise ValueError("Ordered temporal ray-map packing currently requires 4:1 compression")
    source_frames = int(source_frames)
    physical_targets = int(physical_targets)
    if raymap.ndim == 5:
        frame_major = raymap.permute(0, 2, 1, 3, 4)
        packed = [
            _pack_temporal_raymaps(item, source_frames, physical_targets, compression)
            for item in frame_major
        ]
        return torch.stack(packed, dim=0).permute(0, 2, 1, 3, 4)
    if raymap.ndim != 4:
        raise ValueError(f"Expected [F,C,H,W] or [B,C,F,H,W] ray maps, got {raymap.shape}")
    if raymap.shape[0] != source_frames + physical_targets:
        raise ValueError(
            "Physical ray-map frame count does not match sources + targets: "
            f"{raymap.shape[0]} vs {source_frames + physical_targets}"
        )

    sources = raymap[:source_frames]
    targets = raymap[source_frames:]
    channels, height, width = targets.shape[1:]
    packed_sources = torch.cat([sources] * compression, dim=1)

    groups = [targets[0:1].expand(compression, -1, -1, -1)]
    for start in range(1, physical_targets, compression):
        group = targets[start:start + compression]
        if group.shape[0] < compression:
            group = torch.cat(
                [group, group[-1:].expand(compression - group.shape[0], -1, -1, -1)],
                dim=0,
            )
        groups.append(group)
    expected = _temporal_latent_frame_count(physical_targets, compression)
    if len(groups) != expected:
        raise ValueError(f"Packed {len(groups)} target groups, expected {expected}")
    packed_targets = torch.stack(
        [group.reshape(compression * channels, height, width) for group in groups]
    )
    return torch.cat([packed_sources, packed_targets], dim=0)


def _omega_cfg_time_weight(progress: float, schedule: str) -> float:
    """Return a smooth, training-free Omega guidance weight over denoising."""
    progress = float(np.clip(progress, 0.0, 1.0))
    if schedule == "constant":
        return 1.0
    if schedule == "early":
        phase = min(progress / 0.6, 1.0)
        return float(np.cos(0.5 * np.pi * phase) ** 2)
    if schedule == "middle":
        return float(np.sin(np.pi * progress) ** 2)
    if schedule == "late":
        phase = max((progress - 0.4) / 0.6, 0.0)
        return float(np.sin(0.5 * np.pi * phase) ** 2)
    raise ValueError(f"Unsupported Omega CFG schedule: {schedule}")


def _target_guidance_map(
    reference: torch.Tensor,
    target_views: int,
    target_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Keep source predictions conditioned while scaling target guidance."""
    if not 0 < int(target_views) <= reference.shape[2]:
        raise ValueError(
            f"Invalid target view count {target_views} for {reference.shape[2]} frames"
        )
    scale = torch.as_tensor(target_scale, device=reference.device, dtype=reference.dtype)
    if scale.ndim == 0:
        scale = scale.reshape(1, 1, 1, 1, 1)
    if scale.ndim != 5:
        raise ValueError("target_scale must be scalar or [B, 1, T, H, W]")
    if scale.shape[2] not in (1, int(target_views)):
        raise ValueError("target_scale frame count does not match target views")
    result = reference.new_ones(
        reference.shape[0], 1, reference.shape[2], reference.shape[3], reference.shape[4]
    )
    result[:, :, -int(target_views):] = scale
    return result


class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedderMultiple(),
            WanVideoUnit_ImageEmbedderVAE_IndividualEncoding(),
            WanVideoUnit_VisibilityDecomposedAnchor(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name
            if dist.is_available() and dist.is_initialized():
                device = get_device_name()

        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')

        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        depth_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Reference
        vace_reference_image: Optional[Image.Image] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        omega_cfg_scale: Optional[float] = 1.0,
        omega_cfg_mode: str = "full",
        omega_cfg_schedule: str = "constant",
        omega_cfg_confidence_power: float = 1.0,
        omega_cfg_weak_scale: float = 0.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        raymap: Optional[torch.Tensor] = None,
        num_latent_frames: Optional[int] = None,
        num_output_frames: int = 1,
        num_physical_output_frames: Optional[int] = None,
        omega_tokens: Optional[torch.Tensor] = None,
        omega_xyz: Optional[torch.Tensor] = None,
        omega_confidence: Optional[torch.Tensor] = None,
        omega_target_w2c: Optional[torch.Tensor] = None,
        omega_target_intrinsics: Optional[torch.Tensor] = None,
        return_depth: bool = False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        temporal_target_compression = int(
            getattr(self.dit, "temporal_target_compression", 1)
        )
        if temporal_target_compression > 1:
            physical_targets = int(
                num_output_frames
                if num_physical_output_frames is None
                else num_physical_output_frames
            )
            latent_targets = _temporal_latent_frame_count(
                physical_targets, temporal_target_compression
            )
            condition_indices = list(
                range(0, physical_targets, temporal_target_compression)
            )
            if condition_indices[-1] != physical_targets - 1:
                condition_indices.append(physical_targets - 1)
            source_frames = len(input_image) if isinstance(input_image, list) else 1
            physical_raymaps = (
                raymap is not None
                and (
                    (raymap.ndim == 4 and raymap.shape[0] == source_frames + physical_targets)
                    or (raymap.ndim == 5 and raymap.shape[2] == source_frames + physical_targets)
                )
            )
            if physical_raymaps and getattr(self.dit, "temporal_plucker_packing", False):
                raymap = _pack_temporal_raymaps(
                    raymap, source_frames, physical_targets, temporal_target_compression
                )
            elif physical_raymaps and raymap.ndim == 4:
                raymap_indices = list(range(source_frames)) + [
                    source_frames + index for index in condition_indices
                ]
                raymap = raymap[raymap_indices]
            elif physical_raymaps and raymap.ndim == 5:
                raymap_indices = list(range(source_frames)) + [
                    source_frames + index for index in condition_indices
                ]
                raymap = raymap[:, :, raymap_indices]
            if (
                omega_target_w2c is not None
                and omega_target_w2c.shape[-3] == physical_targets
            ):
                omega_target_w2c = omega_target_w2c[..., condition_indices, :, :]
            if (
                omega_target_intrinsics is not None
                and omega_target_intrinsics.shape[-3] == physical_targets
            ):
                omega_target_intrinsics = omega_target_intrinsics[
                    ..., condition_indices, :, :
                ]
            num_output_frames = latent_targets
            num_physical_output_frames = physical_targets
            if num_latent_frames is None:
                num_latent_frames = source_frames + latent_targets
        elif num_latent_frames is None and getattr(self.dit, "individual_encoding", False):
            # NVS layout: individual encoding keeps one latent slot per physical
            # frame, so the latent length is the full source + target frame count.
            source_frames = len(input_image) if isinstance(input_image, list) else 1
            physical_targets = int(
                num_output_frames
                if num_physical_output_frames is None
                else num_physical_output_frames
            )
            num_latent_frames = source_frames + physical_targets

        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "depth_video": depth_video,
            "denoising_strength": denoising_strength,
            "vace_reference_image": vace_reference_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "omega_condition_scale": 1.0,
            "omega_target_condition_scale": 1.0,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "raymap": raymap,
            "num_latent_frames": num_latent_frames,
            "num_output_frames": num_output_frames,
            "num_physical_output_frames": num_physical_output_frames,
            "omega_tokens": omega_tokens,
            "omega_xyz": omega_xyz,
            "omega_confidence": omega_confidence,
            "omega_target_w2c": omega_target_w2c,
            "omega_target_intrinsics": omega_target_intrinsics,
            "generate_depth": bool(return_depth),
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        if omega_cfg_mode not in {"full", "weak_full", "target_only", "target_confidence"}:
            raise ValueError(f"Unsupported Omega CFG mode: {omega_cfg_mode}")
        if omega_cfg_schedule not in {"constant", "early", "middle", "late"}:
            raise ValueError(f"Unsupported Omega CFG schedule: {omega_cfg_schedule}")
        if float(omega_cfg_confidence_power) <= 0:
            raise ValueError("omega_cfg_confidence_power must be positive")
        if not 0.0 <= float(omega_cfg_weak_scale) < 1.0:
            raise ValueError("omega_cfg_weak_scale must be in [0, 1)")

        routed_confidence = None
        if omega_cfg_scale != 1.0 and omega_cfg_mode == "target_confidence":
            geometry = (
                omega_xyz,
                omega_confidence,
                omega_target_w2c,
                omega_target_intrinsics,
            )
            if not all(value is not None for value in geometry):
                raise ValueError("Confidence-adaptive Omega CFG requires full geometry")
            dummy_features = omega_xyz.new_zeros(
                (*omega_confidence.shape, 1), dtype=torch.float32
            )
            _, _, routed_statistics = route_omega_feature_layers(
                dummy_features,
                omega_xyz,
                omega_confidence,
                omega_target_w2c,
                omega_target_intrinsics,
                output_hw=inputs_shared["latents"].shape[-2:],
            )
            routed_confidence = routed_statistics[:, 2:3].to(
                device=self.device, dtype=self.torch_dtype
            ).pow(float(omega_cfg_confidence_power))

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if omega_cfg_scale != 1.0:
                if omega_tokens is None:
                    raise ValueError("Omega CFG requires omega_tokens")
                if cfg_scale != 1.0:
                    raise ValueError(
                        "Omega CFG currently requires text cfg_scale=1 to avoid a "
                        "four-branch guidance ambiguity"
                    )
                inputs_without_omega = dict(inputs_shared)
                if omega_cfg_mode == "full":
                    inputs_without_omega["omega_condition_scale"] = 0.0
                elif omega_cfg_mode == "weak_full":
                    inputs_without_omega["omega_condition_scale"] = float(omega_cfg_weak_scale)
                else:
                    inputs_without_omega["omega_target_condition_scale"] = 0.0
                noise_pred_without_omega = self.model_fn(
                    **models,
                    **inputs_without_omega,
                    **inputs_posi,
                    timestep=timestep,
                )
                denominator = max(len(self.scheduler.timesteps) - 1, 1)
                progress = progress_id / denominator
                time_weight = _omega_cfg_time_weight(progress, omega_cfg_schedule)
                effective_scale = 1.0 + time_weight * (float(omega_cfg_scale) - 1.0)
                if omega_cfg_mode == "weak_full":
                    # Preserve omega_cfg_scale's zero-condition interpretation,
                    # but estimate the guidance direction from a weak condition
                    # strength that the model observed during training.
                    guidance_scale = 1.0 + time_weight * (
                        float(omega_cfg_scale) - 1.0
                    ) / (1.0 - float(omega_cfg_weak_scale))
                elif omega_cfg_mode == "full":
                    guidance_scale = effective_scale
                else:
                    target_scale = effective_scale
                    if routed_confidence is not None:
                        target_scale = 1.0 + time_weight * (
                            float(omega_cfg_scale) - 1.0
                        ) * routed_confidence
                    guidance_scale = _target_guidance_map(
                        noise_pred_posi,
                        int(num_output_frames),
                        target_scale,
                    )
                conditioned_dtype = noise_pred_posi.dtype
                guidance_scale = torch.as_tensor(
                    guidance_scale,
                    device=noise_pred_posi.device,
                    dtype=torch.float32,
                )
                noise_pred_posi = (
                    noise_pred_without_omega.float()
                    + guidance_scale
                    * (noise_pred_posi.float() - noise_pred_without_omega.float())
                ).to(conditioned_dtype)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Strip reference frames if present
        if vace_reference_image is not None:
            if isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

        # Decode. RGB and depth share the frozen VAE but remain separate 16D
        # state streams throughout the sampler.
        self.load_models_to_device(['vae'])
        depth_latents = None
        if (
            getattr(self.dit, "joint_depth", False)
            and inputs_shared["latents"].shape[1] > int(self.dit.rgb_latent_channels)
        ):
            channels = int(self.dit.rgb_latent_channels)
            rgb_latents = inputs_shared["latents"][:, :channels]
            depth_latents = inputs_shared["latents"][:, channels:]
        else:
            rgb_latents = inputs_shared["latents"]
        temporal_target_compression = int(
            getattr(self.dit, "temporal_target_compression", 1)
        )
        if self.dit.individual_encoding and temporal_target_compression > 1:
            target_latent_frames = int(num_output_frames)
            if not 0 < target_latent_frames < rgb_latents.shape[2]:
                raise ValueError("Invalid temporally compressed target latent count")
            source_latent_frames = rgb_latents.shape[2] - target_latent_frames
            source_video = [
                self.vae.decode(
                    rgb_latents[:, :, i:i + 1],
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                for i in range(source_latent_frames)
            ]
            target_video = self.vae.decode(
                rgb_latents[:, :, source_latent_frames:],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if num_physical_output_frames is not None:
                target_video = target_video[:, :, :int(num_physical_output_frames)]
            video = torch.concat(source_video + [target_video], dim=2)
            if depth_latents is not None:
                raise ValueError("Temporal target compression does not support joint depth")
            depth = None
        elif self.dit.individual_encoding:
            video = []
            depth = [] if depth_latents is not None else None
            for i in range(rgb_latents.shape[2]):
                video.append(self.vae.decode(rgb_latents[:, :, i:i+1], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
                if depth_latents is not None:
                    depth.append(self.vae.decode(depth_latents[:, :, i:i+1], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            video = torch.concat(video, dim=2)
            if depth is not None:
                depth = torch.concat(depth, dim=2)
        else:
            video = self.vae.decode(rgb_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            depth = self.vae.decode(depth_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) if depth_latents is not None else None

        if output_type == "quantized":
            video = self.vae_output_to_video(video)
            if depth is not None:
                depth = self.vae_output_to_video(depth)
        self.last_depth_video = depth
        self.load_models_to_device([])
        return (video, depth) if return_depth else video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "num_latent_frames", "generate_depth"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, num_latent_frames=None, generate_depth=False):
        if num_latent_frames is not None:
            length = num_latent_frames
        else:
            length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        channels = pipe.vae.model.z_dim
        joint_mode = getattr(pipe.dit, "joint_depth_mode", "bidirectional")
        if getattr(pipe.dit, "joint_depth", False) and (
            joint_mode == "bidirectional" or generate_depth
        ):
            channels *= 2
        shape = (1, channels, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_InputVideoEmbedderMultiple(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video", "depth_video", "noise", "tiled", "tile_size",
                "tile_stride", "num_physical_output_frames",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(
        self, pipe: WanVideoPipeline, input_video, depth_video, noise, tiled,
        tile_size, tile_stride, num_physical_output_frames=None,
    ):
        if input_video is None or not pipe.dit.individual_encoding:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        temporal_target_compression = int(
            getattr(pipe.dit, "temporal_target_compression", 1)
        )
        physical_targets = int(num_physical_output_frames or 0)
        if temporal_target_compression > 1:
            if depth_video is not None or getattr(pipe.dit, "joint_depth", False):
                raise ValueError("Temporal target compression does not support joint depth")
            if not 0 < physical_targets < input_video.shape[2]:
                raise ValueError("Physical target count must split source and target video")
            source_count = input_video.shape[2] - physical_targets
        else:
            source_count = input_video.shape[2]

        input_latents = []
        for i in range(source_count):
            input_latents.append(pipe.vae.encode(input_video[:, :, i:i+1], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))
        if temporal_target_compression > 1:
            target_video = input_video[:, :, source_count:]
            padded_frames = _temporal_padded_frame_count(
                physical_targets, temporal_target_compression
            )
            if padded_frames > physical_targets:
                target_video = torch.cat(
                    [
                        target_video,
                        target_video[:, :, -1:].expand(
                            -1, -1, padded_frames - physical_targets, -1, -1
                        ),
                    ],
                    dim=2,
                )
            target_latents = pipe.vae.encode(
                target_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            expected_targets = _temporal_latent_frame_count(
                physical_targets, temporal_target_compression
            )
            if target_latents.shape[2] != expected_targets:
                raise ValueError(
                    "Unexpected target VAE length: "
                    f"{target_latents.shape[2]} vs {expected_targets}"
                )
            input_latents.append(target_latents)
        input_latents = torch.concat(input_latents, dim=2)
        if getattr(pipe.dit, "joint_depth", False):
            if depth_video is None or len(depth_video) != input_video.shape[2]:
                raise ValueError("joint depth training requires one depth image per RGB frame")
            depth_video = pipe.preprocess_video(depth_video)
            depth_latents = []
            for i in range(depth_video.shape[2]):
                depth_latents.append(pipe.vae.encode(depth_video[:, :, i:i+1], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device))
            depth_latents = torch.concat(depth_latents, dim=2)
            input_latents = torch.cat([input_latents, depth_latents], dim=1)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "mask_loss": True}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_ImageEmbedderVAE_IndividualEncoding(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image", "end_image", "num_frames", "height", "width",
                "tiled", "tile_size", "tile_stride", "num_latent_frames",
                "num_physical_output_frames",
            ),
            output_params=("y", "source_latents"),
            onload_model_names=("vae",)
        )

    def process(
        self, pipe: WanVideoPipeline, input_image, end_image, num_frames,
        height, width, tiled, tile_size, tile_stride, num_latent_frames=None,
        num_physical_output_frames=None,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding or not pipe.dit.individual_encoding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = []
        for i in range(len(input_image)):
            image.append(pipe.preprocess_image(input_image[i].resize((width, height))).to(pipe.device))
        image = torch.concat(image, dim=0)

        num_input = len(input_image)
        num_total = num_latent_frames if num_latent_frames is not None else (num_input + 1)
        num_output = num_total - num_input

        msk = torch.ones(1, num_total, height//8, width//8, device=pipe.device)
        msk[:, num_input:] = 0

        temporal_target_compression = int(
            getattr(pipe.dit, "temporal_target_compression", 1)
        )
        if end_image is not None and temporal_target_compression > 1:
            raise ValueError("End-image conditioning is unsupported in temporal target mode")
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            target_zeros = torch.zeros(3, num_output, height, width).to(image.device)
            vae_input = torch.concat([image.transpose(0, 1), target_zeros], dim=1)

        msk = torch.repeat_interleave(msk, repeats=4, dim=1)
        msk = msk.view(1, num_total, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]  # [4, num_total, h//8, w//8]

        ys = []
        for i in range(num_input):
            ys.append(pipe.vae.encode([vae_input[:, i:i+1].to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0])
        if temporal_target_compression > 1:
            physical_targets = int(num_physical_output_frames or 0)
            expected_targets = _temporal_latent_frame_count(
                physical_targets, temporal_target_compression
            )
            if expected_targets != num_output:
                raise ValueError("Target context length does not match compressed target latents")
            padded_frames = _temporal_padded_frame_count(
                physical_targets, temporal_target_compression
            )
            target_zeros = torch.zeros(
                1, 3, padded_frames, height, width,
                device=pipe.device, dtype=pipe.torch_dtype,
            )
            target_y = pipe.vae.encode(
                target_zeros,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            if target_y.shape[1] != expected_targets:
                raise ValueError("Unexpected target context VAE length")
            ys.append(target_y)
        else:
            for i in range(num_input, num_total):
                ys.append(pipe.vae.encode([vae_input[:, i:i+1].to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0])
        y = torch.concat(ys, dim=1)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        source_latents = torch.concat(ys[:num_input], dim=1).unsqueeze(0)
        source_latents = source_latents.to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"y": y, "source_latents": source_latents}


def _temporal_latent_frame_count(frame_count: int, factor: int) -> int:
    if frame_count <= 0 or factor <= 0:
        raise ValueError("Temporal frame count and compression factor must be positive")
    return 1 + (int(frame_count) - 1 + int(factor) - 1) // int(factor)


def _temporal_padded_frame_count(frame_count: int, factor: int) -> int:
    latent_count = _temporal_latent_frame_count(frame_count, factor)
    return 1 + int(factor) * (latent_count - 1)


class WanVideoUnit_VisibilityDecomposedAnchor(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "source_latents",
                "omega_xyz",
                "omega_confidence",
                "omega_target_w2c",
                "omega_target_intrinsics",
                "num_output_frames",
            ),
            output_params=("vdf_anchor", "vdf_statistics"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        source_latents,
        omega_xyz,
        omega_confidence,
        omega_target_w2c,
        omega_target_intrinsics,
        num_output_frames,
    ):
        if not hasattr(pipe.dit, "vdf_adapter"):
            return {}
        required = (
            source_latents,
            omega_xyz,
            omega_confidence,
            omega_target_w2c,
            omega_target_intrinsics,
        )
        if any(value is None for value in required):
            raise ValueError("VDF requires source latents and projected Omega geometry")
        if omega_target_w2c.shape[-3] != int(num_output_frames):
            raise ValueError("VDF target camera count does not match output frames")
        front, _, statistics = route_source_latent_layers(
            source_latents,
            omega_xyz,
            omega_confidence,
            omega_target_w2c,
            omega_target_intrinsics,
            output_hw=source_latents.shape[-2:],
        )
        return {
            "vdf_anchor": front.to(source_latents),
            "vdf_statistics": statistics.to(source_latents),
        }



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        if pipe.dit.individual_encoding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


def _block_modulation(block, t_mod):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    values = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        values = tuple(value.squeeze(2) for value in values)
    return values


def _depth_reads_rgb_block(block, rgb, depth, context, t_mod, freqs):
    """Update depth from RGB+depth keys while keeping RGB depth-independent."""
    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
    ) = _block_modulation(block, t_mod)
    depth_input = modulate(block.norm1(depth), shift_msa, scale_msa)
    rgb_input = modulate(block.norm1(rgb), shift_msa, scale_msa)
    attention = block.self_attn
    query = attention.norm_q(attention.q(depth_input))
    rgb_key = attention.norm_k(attention.k(rgb_input))
    depth_key = attention.norm_k(attention.k(depth_input))
    rgb_value = attention.v(rgb_input)
    depth_value = attention.v(depth_input)
    query = rope_apply(query, freqs, attention.num_heads)
    keys = rope_apply(
        torch.cat([rgb_key, depth_key], dim=1),
        torch.cat([freqs, freqs], dim=0),
        attention.num_heads,
    )
    values = torch.cat([rgb_value, depth_value], dim=1)
    residual = attention.o(attention.attn(query, keys, values))
    depth = block.gate(depth, gate_msa, residual)
    depth = depth + block.cross_attn(block.norm3(depth), context)
    depth_input = modulate(block.norm2(depth), shift_mlp, scale_mlp)
    return block.gate(depth, gate_mlp, block.ffn(depth_input))


def _rgb_to_depth_block(block, rgb, depth, context, t_mod, freqs):
    rgb = block(rgb, context, t_mod, freqs)
    depth = _depth_reads_rgb_block(block, rgb, depth, context, t_mod, freqs)
    return rgb, depth


def _inject_query_pose_tokens(
    dit,
    hidden: torch.Tensor,
    raymap: torch.Tensor,
    num_output_frames: int,
) -> torch.Tensor:
    """Add a dedicated dense Plucker embedding to target-frame tokens."""
    if raymap.dim() == 4:
        raymap = raymap.permute(1, 0, 2, 3).unsqueeze(0)
    if raymap.shape[0] != hidden.shape[0]:
        if hidden.shape[0] % raymap.shape[0]:
            raise ValueError("Query-pose batch cannot match the DiT batch")
        raymap = raymap.repeat_interleave(
            hidden.shape[0] // raymap.shape[0], dim=0
        )

    output_frames = int(num_output_frames)
    if output_frames <= 0 or output_frames > raymap.shape[2]:
        raise ValueError(
            f"Invalid target-frame count {output_frames} for raymap "
            f"with {raymap.shape[2]} frames"
        )
    target_raymap = torch.zeros_like(raymap)
    target_raymap[:, :, -output_frames:] = raymap[:, :, -output_frames:]
    query_tokens = dit.query_pose_patch_embedding(target_raymap).to(
        device=hidden.device, dtype=hidden.dtype
    )
    if query_tokens.shape != hidden.shape:
        raise ValueError(
            "Query-pose and DiT token grids differ: "
            f"{tuple(query_tokens.shape)} vs {tuple(hidden.shape)}"
        )
    return hidden + query_tokens



def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    raymap: Optional[torch.Tensor] = None,
    omega_tokens: Optional[torch.Tensor] = None,
    omega_xyz: Optional[torch.Tensor] = None,
    omega_confidence: Optional[torch.Tensor] = None,
    omega_target_w2c: Optional[torch.Tensor] = None,
    omega_target_intrinsics: Optional[torch.Tensor] = None,
    omega_condition_scale: Optional[torch.Tensor] = None,
    omega_target_condition_scale: Optional[torch.Tensor] = None,
    vdf_anchor: Optional[torch.Tensor] = None,
    vdf_statistics: Optional[torch.Tensor] = None,
    num_output_frames: int = 1,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    # Timestep
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    joint_depth = getattr(dit, "joint_depth", False)
    joint_depth_mode = getattr(dit, "joint_depth_mode", "bidirectional")
    if joint_depth:
        rgb_channels = int(dit.rgb_latent_channels)
        rgb_latents = latents[:, :rgb_channels]
        depth_latents = (
            latents[:, rgb_channels:]
            if latents.shape[1] > rgb_channels
            else None
        )
        x = rgb_latents
        noisy_latents = rgb_latents
    else:
        x = latents
        depth_latents = None
        noisy_latents = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if depth_latents is not None and depth_latents.shape[0] != context.shape[0]:
        depth_latents = torch.concat(
            [depth_latents] * context.shape[0], dim=0
        )
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    if raymap is not None:
        if raymap.dim() == 4:
            raymap = raymap.permute(1, 0, 2, 3).unsqueeze(0)
        if hasattr(dit, "temporal_plucker_adapter"):
            expected = int(dit.temporal_plucker_adapter.in_channels)
            if raymap.shape[1] != expected:
                raise ValueError(
                    "Packed Plucker channels do not match the temporal adapter: "
                    f"{raymap.shape[1]} vs {expected}"
                )
            raymap = dit.temporal_plucker_adapter(raymap)
        query_pose_raymap = raymap
        x = torch.cat([x, raymap], dim=1)
    else:
        query_pose_raymap = None
    # Append depth after all pretrained FrameCrafter channels so RGB/y/raymap
    # retain their original patch-embedding indices.
    if depth_latents is not None and joint_depth_mode == "bidirectional":
        x = torch.cat([x, depth_latents], dim=1)
    use_omega_renderer = omega_tokens is not None and hasattr(
        dit, "omega_renderer_adapter"
    )
    condition_scale = torch.as_tensor(
        1.0 if omega_condition_scale is None else omega_condition_scale,
        device=x.device,
        dtype=x.dtype,
    )
    if omega_tokens is not None and hasattr(dit, "omega_global_adapter"):
        omega_context = dit.omega_global_adapter(omega_tokens).to(
            device=context.device, dtype=context.dtype
        )
        omega_context = omega_context * condition_scale.to(omega_context)
        if omega_context.shape[0] != context.shape[0]:
            if context.shape[0] % omega_context.shape[0]:
                raise ValueError("Omega global context batch cannot match CFG batch")
            omega_context = omega_context.repeat_interleave(
                context.shape[0] // omega_context.shape[0], dim=0
            )
        context = torch.cat([context, omega_context], dim=1)
    elif omega_tokens is not None and not use_omega_renderer:
        if not hasattr(dit, "omega_adapter"):
            raise RuntimeError(
                "omega_tokens were provided but the DiT has no Omega adapter"
            )
        omega_condition = dit.omega_adapter(
            omega_tokens,
            output_hw=x.shape[-2:],
            num_output_frames=int(num_output_frames),
            omega_xyz=omega_xyz,
            omega_confidence=omega_confidence,
            omega_target_w2c=omega_target_w2c,
            omega_target_intrinsics=omega_target_intrinsics,
            noisy_latents=noisy_latents,
            timestep=timestep,
        ).to(device=x.device, dtype=x.dtype)
        omega_condition = omega_condition * condition_scale
        target_condition_scale = torch.as_tensor(
            1.0 if omega_target_condition_scale is None else omega_target_condition_scale,
            device=x.device,
            dtype=x.dtype,
        )
        if int(num_output_frames):
            frame_scale = omega_condition.new_ones(
                omega_condition.shape[0], 1, omega_condition.shape[2], 1, 1
            )
            frame_scale[:, :, -int(num_output_frames):] = target_condition_scale
            omega_condition = omega_condition * frame_scale
        if omega_condition.shape[2:] != x.shape[2:]:
            raise ValueError(
                "Omega condition and DiT input must have the same frame/spatial shape: "
                f"{tuple(omega_condition.shape)} vs {tuple(x.shape)}"
            )
        x = torch.cat([x, omega_condition], dim=1)

    # Patchify
    x = dit.patchify(x)
    if hasattr(dit, "query_pose_patch_embedding"):
        if query_pose_raymap is None:
            raise ValueError(
                "The LVSM-style query-pose adapter requires a dense raymap"
            )
        x = _inject_query_pose_tokens(
            dit,
            x,
            query_pose_raymap,
            num_output_frames=int(num_output_frames),
        )
    depth_stream = None
    if depth_latents is not None and joint_depth_mode == "rgb_to_depth":
        depth_stream = dit.depth_patch_embedding(depth_latents)
        if depth_stream.shape[2:] != x.shape[2:]:
            raise ValueError(
                "RGB and depth patch grids differ: "
                f"{tuple(x.shape)} vs {tuple(depth_stream.shape)}"
            )
    if use_omega_renderer:
        x = dit.omega_renderer_adapter(
            x,
            omega_tokens * condition_scale.to(omega_tokens),
            t,
            num_output_frames=int(num_output_frames),
        )

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    if depth_stream is not None:
        depth_stream = rearrange(
            depth_stream, 'b c f h w -> b (f h w) c'
        ).contiguous()

    # Zero-temporal RoPE
    f_freqs = torch.ones_like(dit.freqs[0][:f])
    freqs = torch.cat([
        f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    checkpoint_every_n = max(
        1, int(os.environ.get("DIT_CHECKPOINT_EVERY_N", "1"))
    )
    uncheckpoint_last_n = max(
        0, int(os.environ.get("DIT_UNCHECKPOINT_LAST_N", "0"))
    )
    uncheckpoint_start = max(0, len(dit.blocks) - uncheckpoint_last_n)
    checkpointing_enabled = (
        use_gradient_checkpointing or use_gradient_checkpointing_offload
    )

    for block_index, block in enumerate(dit.blocks):
        checkpoint_this_block = checkpointing_enabled and (
            checkpoint_every_n == 1
            or block_index % checkpoint_every_n == 0
        )
        if uncheckpoint_last_n and block_index >= uncheckpoint_start:
            checkpoint_this_block = False
        if (
            depth_stream is not None
            and use_gradient_checkpointing_offload
            and checkpoint_this_block
        ):
            with torch.autograd.graph.save_on_cpu():
                x, depth_stream = torch.utils.checkpoint.checkpoint(
                    lambda rgb, depth, ctx, mod, pos, block=block: _rgb_to_depth_block(
                        block, rgb, depth, ctx, mod, pos
                    ),
                    x, depth_stream, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif depth_stream is not None and checkpoint_this_block:
            x, depth_stream = torch.utils.checkpoint.checkpoint(
                lambda rgb, depth, ctx, mod, pos, block=block: _rgb_to_depth_block(
                    block, rgb, depth, ctx, mod, pos
                ),
                x, depth_stream, context, t_mod, freqs,
                use_reentrant=False,
            )
        elif depth_stream is not None:
            x, depth_stream = _rgb_to_depth_block(
                block, x, depth_stream, context, t_mod, freqs
            )
        elif use_gradient_checkpointing_offload and checkpoint_this_block:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif checkpoint_this_block:
            x = torch.utils.checkpoint.checkpoint(
                block,
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    if hasattr(dit, "vdf_adapter"):
        if vdf_anchor is None or vdf_statistics is None:
            raise ValueError("VDF adapter is enabled but its routed anchor is missing")
        x = dit.vdf_adapter(
            velocity=x,
            noisy_latents=noisy_latents,
            timestep=timestep,
            anchor=vdf_anchor,
            statistics=vdf_statistics,
            num_output_frames=int(num_output_frames),
        )
    if depth_stream is not None:
        depth_stream = dit.depth_head(depth_stream, t)
        depth_stream = dit.unpatchify(depth_stream, (f, h, w))
        x = torch.cat([x, depth_stream], dim=1)
    return x
