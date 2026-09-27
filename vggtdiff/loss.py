from __future__ import annotations

import torch

from diffsynth.models.omega_geometry import point_track_consistency_loss


def flow_point_track_loss(
    pipe,
    point_track_weight: float,
    huber_beta: float,
    max_sigma: float,
    **inputs,
):
    maximum = int(inputs.get("max_timestep_boundary", 1.0) * len(pipe.scheduler.timesteps))
    minimum = int(inputs.get("min_timestep_boundary", 0.0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(minimum, maximum, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    sigma = pipe.scheduler.sigmas[timestep_id].to(device=pipe.device, dtype=torch.float32)
    clean = inputs["input_latents"]
    noise = torch.randn_like(clean)
    noisy = pipe.scheduler.add_noise(clean, noise, timestep)
    inputs["latents"] = noisy
    target = pipe.scheduler.training_target(clean, noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    velocity = pipe.model_fn(**models, **inputs, timestep=timestep)
    target_frames = int(inputs["num_output_frames"])
    velocity = velocity[:, :, -target_frames:]
    target = target[:, :, -target_frames:]
    flow_loss = torch.nn.functional.mse_loss(velocity.float(), target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)

    predicted_clean = noisy[:, :, -target_frames:].float() - sigma * velocity.float()
    consistency, statistics = point_track_consistency_loss(
        predicted_clean=predicted_clean,
        clean_target=clean[:, :, -target_frames:].float(),
        xyz=inputs["omega_xyz"],
        confidence=inputs["omega_confidence"],
        target_w2c=inputs["omega_target_w2c"],
        target_intrinsics=inputs["omega_target_intrinsics"],
        image_downsample=8,
        huber_beta=float(huber_beta),
    )
    gate = torch.cos(
        (sigma / float(max_sigma)).clamp(0.0, 1.0) * torch.pi / 2.0
    ).square()
    gate = torch.where(sigma < float(max_sigma), gate, torch.zeros_like(gate)).mean()
    pipe.last_point_track_statistics = {
        "flow_loss": flow_loss.detach(),
        "point_track_loss": consistency.detach(),
        "point_track_gate": gate.detach(),
        **statistics,
    }
    return flow_loss + float(point_track_weight) * gate * consistency
