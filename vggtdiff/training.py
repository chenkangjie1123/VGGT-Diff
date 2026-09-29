from __future__ import annotations

import torch

from diffsynth.core import load_state_dict
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import WanVideoPipeline, _pack_temporal_raymaps

from .loss import flow_point_track_loss
from .model import _model_configs, configure_dit


class VGGTDiffTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        checkpoint: str | None,
        base_model_dir: str | None,
        device: torch.device,
        training_mode: str = "trajectory",
        condition_drop_probability: float = 0.10,
        condition_weak_probability: float = 0.20,
        condition_weak_min: float = 0.20,
        condition_weak_max: float = 0.80,
        point_track_weight: float = 0.10,
        point_track_huber_beta: float = 0.10,
        point_track_max_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        configs, tokenizer = _model_configs(base_model_dir)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=configs,
            tokenizer_config=tokenizer,
            redirect_common_files=False,
        )
        configure_dit(self.pipe.dit, training_mode=training_mode)
        self.switch_pipe_to_training_mode(self.pipe, trainable_models="dit")
        if checkpoint is not None:
            state = load_state_dict(
                checkpoint, torch_dtype=torch.bfloat16, device="cpu"
            )
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
        self.condition_drop_probability = float(condition_drop_probability)
        self.training_mode = training_mode
        self.condition_weak_probability = float(condition_weak_probability)
        self.condition_weak_min = float(condition_weak_min)
        self.condition_weak_max = float(condition_weak_max)
        self.point_track_weight = float(point_track_weight)
        self.point_track_huber_beta = float(point_track_huber_beta)
        self.point_track_max_sigma = float(point_track_max_sigma)
        self.use_gradient_checkpointing = True
        self.use_gradient_checkpointing_offload = False

    def optimizer_parameter_groups(
        self,
        backbone_learning_rate: float,
        condition_learning_rate: float,
        plucker_learning_rate: float,
    ) -> list[dict]:
        backbone = []
        condition = []
        plucker = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".temporal_plucker_adapter." in name:
                plucker.append(parameter)
            elif ".patch_embedding." in name or ".omega_adapter." in name:
                condition.append(parameter)
            else:
                backbone.append(parameter)
        groups = [
            {"params": backbone, "lr": float(backbone_learning_rate)},
            {"params": condition, "lr": float(condition_learning_rate)},
            {"params": plucker, "lr": float(plucker_learning_rate)},
        ]
        return [group for group in groups if group["params"]]

    def _pipeline_inputs(self, data: dict):
        source_frames = len(data["input_images"])
        physical_targets = len(data["target_images"]) - source_frames
        if self.training_mode == "trajectory":
            latent_targets = 1 + (physical_targets - 1 + 3) // 4
            condition_indices = list(range(0, physical_targets, 4))
            if condition_indices[-1] != physical_targets - 1:
                condition_indices.append(physical_targets - 1)
            raymap = _pack_temporal_raymaps(
                data["raymap"], source_frames, physical_targets, 4
            )
        else:
            latent_targets = physical_targets
            condition_indices = list(range(physical_targets))
            raymap = data["raymap"]
        shared = {
            "input_image": data["input_images"],
            "input_video": data["target_images"],
            "depth_video": None,
            "raymap": raymap,
            "height": data["input_images"][0].height,
            "width": data["input_images"][0].width,
            "num_frames": len(data["target_images"]),
            "num_output_frames": latent_targets,
            "num_physical_output_frames": physical_targets,
            "num_latent_frames": source_frames + latent_targets,
            "generate_depth": False,
            "cfg_scale": 1.0,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
            "omega_tokens": data["omega_tokens"],
            "omega_xyz": data["omega_xyz"],
            "omega_confidence": data["omega_confidence"],
            "omega_target_w2c": data["omega_target_w2c"][condition_indices],
            "omega_target_intrinsics": data["omega_target_intrinsics"][condition_indices],
        }
        return shared, {"prompt": ""}, {}

    def _condition_scale(self) -> torch.Tensor:
        draw = torch.rand((), device=self.pipe.device)
        if draw < self.condition_drop_probability:
            return draw.new_zeros(())
        if draw < self.condition_drop_probability + self.condition_weak_probability:
            return torch.empty((), device=self.pipe.device).uniform_(
                self.condition_weak_min, self.condition_weak_max
            )
        return draw.new_ones(())

    def forward(self, data: dict):
        inputs = self._pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _ = inputs
        shared["omega_condition_scale"] = self._condition_scale()
        return flow_point_track_loss(
            self.pipe,
            point_track_weight=self.point_track_weight,
            huber_beta=self.point_track_huber_beta,
            max_sigma=self.point_track_max_sigma,
            **shared,
            **positive,
        )
