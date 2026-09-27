from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .omega_geometry import (
    route_omega_feature_layers,
    route_omega_features,
    route_target_state_consensus,
)


def infer_token_grid(num_tokens: int, target_aspect: float) -> tuple[int, int]:
    candidates = []
    for height in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % height:
            continue
        width = num_tokens // height
        candidates.append((abs(width / height - target_aspect), height, width))
        candidates.append((abs(height / width - target_aspect), width, height))
    if not candidates:
        raise ValueError(f"cannot factor {num_tokens} Omega tokens into a 2D grid")
    _, height, width = min(candidates)
    return height, width


class OmegaGlobalTokenAdapter(nn.Module):
    """Expose every frozen Omega token to the DiT cross-attention context."""

    def __init__(
        self,
        token_dim: int = 2048,
        output_dim: int = 5120,
        pool_height: int = 0,
        pool_width: int = 0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.pool_height = int(pool_height)
        self.pool_width = int(pool_width)
        if (self.pool_height > 0) != (self.pool_width > 0):
            raise ValueError("global token pooling requires both height and width")
        self.norm = nn.LayerNorm(self.token_dim, bias=False)
        self.proj = nn.Linear(self.token_dim, self.output_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.002)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 4:
            raise ValueError(
                "omega_tokens must have shape [B, V, P, C] or [V, P, C]"
            )
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected Omega token dim {self.token_dim}, got {tokens.shape[-1]}"
            )
        context = self.norm(tokens)
        if self.pool_height > 0:
            batch, views, num_tokens, channels = context.shape
            grid_height, grid_width = infer_token_grid(
                num_tokens, target_aspect=self.pool_width / self.pool_height
            )
            context = context.view(
                batch, views, grid_height, grid_width, channels
            ).permute(0, 1, 4, 2, 3)
            context = F.adaptive_avg_pool2d(
                context.flatten(0, 1),
                (self.pool_height, self.pool_width),
            ).view(
                batch,
                views,
                channels,
                self.pool_height,
                self.pool_width,
            ).permute(0, 1, 3, 4, 2)
            context = context.flatten(2, 3)
        context = self.proj(context)
        return context.flatten(1, 2).contiguous()


class _MultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, cross_attention: bool) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"attention dim {dim} must be divisible by {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.kv = (
            nn.Linear(self.dim, self.dim * 2, bias=False)
            if cross_attention
            else None
        )
        self.qkv = (
            None
            if cross_attention
            else nn.Linear(self.dim, self.dim * 3, bias=False)
        )
        self.out = nn.Linear(self.dim, self.dim, bias=False)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.qkv is not None:
            query, key, value = self.qkv(query).chunk(3, dim=-1)
        else:
            if memory is None:
                raise ValueError("cross-attention requires memory tokens")
            query = self.q(query)
            key, value = self.kv(memory).chunk(2, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(query),
            self._heads(key),
            self._heads(value),
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).flatten(2)
        return self.out(attended)


class _OmegaRendererBlock(nn.Module):
    """LagerNVS-style target self-attention followed by scene cross-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, bias=False)
        self.self_attention = _MultiheadAttention(dim, num_heads, False)
        self.cross_query_norm = nn.LayerNorm(dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim, bias=False)
        self.cross_attention = _MultiheadAttention(dim, num_heads, True)
        self.mlp_norm = nn.LayerNorm(dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = query + self.self_attention(self.query_norm(query))
        query = query + self.cross_attention(
            self.cross_query_norm(query), self.memory_norm(memory)
        )
        return query + self.mlp(self.mlp_norm(query))


class _OmegaRegisterReader(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim, bias=False)
        self.cross_attention = _MultiheadAttention(dim, num_heads, True)
        self.mlp_norm = nn.LayerNorm(dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attention(
            self.query_norm(query), self.memory_norm(memory)
        )
        return query + self.mlp(self.mlp_norm(query))


class OmegaLagerConditionAdapter(nn.Module):
    """Read all Omega scene tokens with target-aligned lightweight queries.

    With no registers, every block follows LagerNVS and attends the complete
    scene memory. With registers enabled, most blocks use a compact scene
    summary and periodic blocks recover patch-level detail, following the
    alternating global/register design used to scale VGGT-Omega.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        dit_dim: int = 5120,
        renderer_dim: int = 768,
        depth: int = 2,
        num_heads: int = 12,
        num_registers: int = 0,
        full_attention_every: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("renderer depth must be positive")
        if full_attention_every < 0:
            raise ValueError("full_attention_every cannot be negative")
        if num_registers == 0 and full_attention_every:
            raise ValueError("full_attention_every requires scene registers")
        self.token_dim = int(token_dim)
        self.dit_dim = int(dit_dim)
        self.renderer_dim = int(renderer_dim)
        self.depth = int(depth)
        self.num_registers = int(num_registers)
        self.full_attention_every = int(full_attention_every)

        # The connector dimensions intentionally match LagerNVS's lightweight
        # renderer interface, while accepting Omega's concatenated 2048D state.
        self.scene_norm = nn.LayerNorm(self.token_dim, bias=False)
        self.scene_connector = nn.Linear(
            self.token_dim, self.renderer_dim, bias=False
        )
        self.query_norm = nn.LayerNorm(self.dit_dim, bias=False)
        self.query_connector = nn.Linear(
            self.dit_dim, self.renderer_dim, bias=False
        )
        self.time_connector = nn.Linear(
            self.dit_dim, self.renderer_dim, bias=False
        )
        self.blocks = nn.ModuleList(
            [
                _OmegaRendererBlock(self.renderer_dim, num_heads)
                for _ in range(self.depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.renderer_dim, bias=False)
        self.output_connector = nn.Linear(
            self.renderer_dim, self.dit_dim, bias=False
        )
        nn.init.zeros_(self.output_connector.weight)

        if self.num_registers > 0:
            self.scene_registers = nn.Parameter(
                torch.empty(1, self.num_registers, self.renderer_dim)
            )
            nn.init.normal_(self.scene_registers, std=0.02)
            self.register_reader = _OmegaRegisterReader(
                self.renderer_dim, num_heads
            )
        else:
            self.scene_registers = None
            self.register_reader = None

    def _repeat_batch(
        self, tensor: torch.Tensor, target_batch: int, name: str
    ) -> torch.Tensor:
        if tensor.shape[0] == target_batch:
            return tensor
        if target_batch % tensor.shape[0]:
            raise ValueError(f"{name} batch cannot match DiT batch")
        return tensor.repeat_interleave(target_batch // tensor.shape[0], dim=0)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        omega_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        num_output_frames: int,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 5:
            raise ValueError("DiT patch tokens must have shape [B, C, F, H, W]")
        if omega_tokens.ndim == 3:
            omega_tokens = omega_tokens.unsqueeze(0)
        if omega_tokens.ndim != 4 or omega_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "omega_tokens must have shape [B, V, P, "
                f"{self.token_dim}]"
            )

        batch, channels, frames, height, width = patch_tokens.shape
        target_views = int(num_output_frames)
        if not 0 < target_views <= frames:
            raise ValueError(
                f"num_output_frames={target_views} is invalid for {frames} frames"
            )
        omega_tokens = self._repeat_batch(omega_tokens, batch, "Omega token")
        time_embedding = self._repeat_batch(
            time_embedding, batch, "time embedding"
        )

        scene_memory = self.scene_connector(self.scene_norm(omega_tokens))
        scene_memory = scene_memory.flatten(1, 2)
        target = patch_tokens[:, :, frames - target_views :]
        target = target.permute(0, 2, 3, 4, 1).reshape(
            batch * target_views, height * width, channels
        )
        time_condition = self.time_connector(time_embedding)
        time_condition = time_condition[:, None, None, :].expand(
            batch, target_views, 1, self.renderer_dim
        ).reshape(batch * target_views, 1, self.renderer_dim)
        query = self.query_connector(self.query_norm(target)) + time_condition

        full_memory = scene_memory[:, None].expand(
            batch, target_views, scene_memory.shape[1], self.renderer_dim
        ).reshape(batch * target_views, scene_memory.shape[1], self.renderer_dim)
        register_memory = None
        if self.scene_registers is not None:
            registers = self.scene_registers.expand(batch, -1, -1)
            registers = self.register_reader(registers, scene_memory)
            register_memory = registers[:, None].expand(
                batch, target_views, self.num_registers, self.renderer_dim
            ).reshape(
                batch * target_views, self.num_registers, self.renderer_dim
            )

        for index, block in enumerate(self.blocks):
            use_full_memory = register_memory is None or (
                self.full_attention_every > 0
                and (index + 1) % self.full_attention_every == 0
            )
            query = block(
                query, full_memory if use_full_memory else register_memory
            )

        delta = self.output_connector(self.output_norm(query))
        delta = delta.view(
            batch, target_views, height, width, channels
        ).permute(0, 4, 1, 2, 3)
        source = patch_tokens[:, :, : frames - target_views]
        target = patch_tokens[:, :, frames - target_views :] + delta
        return torch.cat([source, target], dim=2)


class OmegaGridAdapter(nn.Module):
    """Project Omega tokens into source and geometry-routed target views."""

    def __init__(
        self,
        token_dim: int = 2048,
        output_channels: int = 32,
        router_mode: str = "hard_zbuffer",
        num_feature_layers: int = 1,
    ) -> None:
        super().__init__()
        if router_mode not in {
            "hard_zbuffer",
            "soft_multilayer",
            "state_adaptive_multilayer",
            "state_consensus_multilayer",
            "layered_channels",
            "layered_uncertainty",
        }:
            raise ValueError(f"unsupported Omega router mode: {router_mode}")
        self.token_dim = int(token_dim)
        self.output_channels = int(output_channels)
        self.router_mode = router_mode
        self.num_feature_layers = int(num_feature_layers)
        if self.num_feature_layers < 1:
            raise ValueError("num_feature_layers must be positive")
        if self.num_feature_layers == 1:
            # Preserve the established checkpoint key names for every existing run.
            self.norm = nn.LayerNorm(self.token_dim, bias=False)
            self.proj = nn.Linear(self.token_dim, self.output_channels, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
            self.norms = None
            self.projs = None
            self.feature_layer_logits = None
        else:
            self.norm = None
            self.proj = None
            self.norms = nn.ModuleList(
                nn.LayerNorm(self.token_dim, bias=False)
                for _ in range(self.num_feature_layers)
            )
            self.projs = nn.ModuleList(
                nn.Linear(self.token_dim, self.output_channels, bias=False)
                for _ in range(self.num_feature_layers)
            )
            for projection in self.projs:
                nn.init.normal_(projection.weight, std=0.02)
            # Uniform initialization lets training, rather than layer order,
            # determine the contribution of every cached representation.
            self.feature_layer_logits = nn.Parameter(
                torch.zeros(self.num_feature_layers)
            )

        if self.router_mode in {
            "soft_multilayer",
            "state_adaptive_multilayer",
            "state_consensus_multilayer",
        }:
            hidden_channels = max(self.output_channels, 32)
            self.router_refiner = nn.Sequential(
                nn.Conv2d(self.output_channels * 2 + 4, hidden_channels, 1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, self.output_channels, 1),
            )
            nn.init.zeros_(self.router_refiner[-1].weight)
            nn.init.zeros_(self.router_refiner[-1].bias)

        if self.router_mode in {
            "state_adaptive_multilayer",
            "state_consensus_multilayer",
        }:
            # Flow state and time select among geometry hypotheses. The final
            # layer starts at zero so a Soft 2-Layer checkpoint is preserved
            # exactly before this branch learns anything.
            state_channels = 16
            consensus_channels = state_channels * 2 + 1 if (
                self.router_mode == "state_consensus_multilayer"
            ) else 0
            state_input_channels = (
                self.output_channels * 3 + 4 + state_channels + 1 + consensus_channels
            )
            hidden_channels = max(self.output_channels * 2, 64)
            self.state_router = nn.Sequential(
                nn.Conv2d(state_input_channels, hidden_channels, 1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 3, 1),
            )
            nn.init.zeros_(self.state_router[-1].weight)
            nn.init.zeros_(self.state_router[-1].bias)

    @property
    def total_output_channels(self) -> int:
        """Return the full DiT condition width produced by this router."""
        if self.router_mode == "layered_channels":
            return self.output_channels * 2 + 4
        if self.router_mode == "layered_uncertainty":
            return self.output_channels * 3 + 4
        return self.output_channels

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.num_feature_layers == 1:
            if tokens.ndim == 3:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 4:
                raise ValueError(
                    "single-layer omega_tokens must have shape [B,V,P,C] "
                    "or [V,P,C]"
                )
            return self.proj(self.norm(tokens))

        if tokens.ndim != 5:
            raise ValueError(
                "multi-layer omega_tokens must have shape [B,L,V,P,C]"
            )
        if tokens.shape[1] != self.num_feature_layers:
            raise ValueError(
                f"expected {self.num_feature_layers} Omega layers, got "
                f"{tokens.shape[1]}"
            )
        projected = torch.stack(
            [
                projection(norm(tokens[:, index]))
                for index, (norm, projection) in enumerate(
                    zip(self.norms, self.projs)
                )
            ],
            dim=1,
        )
        weights = self.feature_layer_logits.softmax(dim=0).to(projected)
        return (projected * weights.view(1, -1, 1, 1, 1)).sum(dim=1)

    def _pad_source_features(self, features: torch.Tensor) -> torch.Tensor:
        extra_channels = self.total_output_channels - self.output_channels
        if extra_channels <= 0:
            return features
        padding = features.new_zeros(
            features.shape[0],
            features.shape[1],
            extra_channels,
            features.shape[3],
            features.shape[4],
        )
        return torch.cat([features, padding], dim=2)

    def _resize_source_features(
        self,
        features: torch.Tensor,
        batch: int,
        source_views: int,
        grid_height: int,
        grid_width: int,
        out_height: int,
        out_width: int,
    ) -> torch.Tensor:
        features = features.view(
            batch, source_views, grid_height, grid_width, self.output_channels
        )
        features = features.permute(0, 1, 4, 2, 3).flatten(0, 1)
        source_aspect = grid_width / grid_height
        target_aspect = out_width / out_height
        scale_x = min(target_aspect / source_aspect, 1.0)
        scale_y = min(source_aspect / target_aspect, 1.0)
        theta = features.new_zeros((features.shape[0], 2, 3), dtype=torch.float32)
        theta[:, 0, 0] = scale_x
        theta[:, 1, 1] = scale_y
        grid = F.affine_grid(
            theta,
            size=(features.shape[0], self.output_channels, out_height, out_width),
            align_corners=False,
        )
        features = F.grid_sample(
            features.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(features.dtype)
        return features.view(
            batch, source_views, self.output_channels, out_height, out_width
        )

    def forward(
        self,
        tokens: torch.Tensor,
        output_hw: tuple[int, int],
        num_output_frames: int,
        omega_xyz: torch.Tensor | None = None,
        omega_confidence: torch.Tensor | None = None,
        omega_target_w2c: torch.Tensor | None = None,
        omega_target_intrinsics: torch.Tensor | None = None,
        noisy_latents: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"expected Omega token dim {self.token_dim}, got {tokens.shape[-1]}")
        projected_features = self._project_tokens(tokens)
        batch, source_views, num_tokens, _ = projected_features.shape
        out_height, out_width = map(int, output_hw)
        grid_height, grid_width = infer_token_grid(
            num_tokens, target_aspect=out_width / out_height
        )
        features = self._resize_source_features(
            projected_features,
            batch,
            source_views,
            grid_height,
            grid_width,
            out_height,
            out_width,
        )
        features = self._pad_source_features(features)

        if num_output_frames < 0:
            raise ValueError("num_output_frames must be non-negative")
        if num_output_frames:
            geometry = (
                omega_xyz,
                omega_confidence,
                omega_target_w2c,
                omega_target_intrinsics,
            )
            if all(value is not None for value in geometry):
                hard_features = route_omega_features(
                    projected_features,
                    omega_xyz,
                    omega_confidence,
                    omega_target_w2c,
                    omega_target_intrinsics,
                    output_hw=(out_height, out_width),
                )
                if self.router_mode in {
                    "soft_multilayer",
                    "state_adaptive_multilayer",
                    "state_consensus_multilayer",
                    "layered_channels",
                    "layered_uncertainty",
                }:
                    front, back, statistics = route_omega_feature_layers(
                        projected_features,
                        omega_xyz,
                        omega_confidence,
                        omega_target_w2c,
                        omega_target_intrinsics,
                        output_hw=(out_height, out_width),
                    )
                    if self.router_mode in {
                        "soft_multilayer",
                        "state_adaptive_multilayer",
                        "state_consensus_multilayer",
                    }:
                        target_views = hard_features.shape[2]
                        router_input = torch.cat([front, back, statistics], dim=1)
                        router_input = router_input.permute(0, 2, 1, 3, 4).reshape(
                            batch * target_views,
                            self.output_channels * 2 + 4,
                            out_height,
                            out_width,
                        )
                        correction = self.router_refiner(router_input).view(
                            batch,
                            target_views,
                            self.output_channels,
                            out_height,
                            out_width,
                        ).permute(0, 2, 1, 3, 4)
                        target_features = hard_features + correction
                        if self.router_mode in {
                            "state_adaptive_multilayer",
                            "state_consensus_multilayer",
                        }:
                            if noisy_latents is None or timestep is None:
                                raise ValueError(
                                    "state-adaptive routing requires noisy latents and timestep"
                                )
                            if noisy_latents.ndim != 5 or noisy_latents.shape[1] < 16:
                                raise ValueError(
                                    "noisy_latents must have shape [B, >=16, F, H, W]"
                                )
                            if noisy_latents.shape[0] != batch:
                                if noisy_latents.shape[0] % batch:
                                    raise ValueError(
                                        "noisy latent batch cannot match Omega batch"
                                    )
                                noisy_latents = noisy_latents[:batch]
                            state = noisy_latents[
                                :, :16, -target_views:, :, :
                            ].float()
                            if state.shape[-2:] != (out_height, out_width):
                                state = F.interpolate(
                                    state.permute(0, 2, 1, 3, 4).flatten(0, 1),
                                    size=(out_height, out_width),
                                    mode="bilinear",
                                    align_corners=False,
                                ).view(
                                    batch,
                                    target_views,
                                    16,
                                    out_height,
                                    out_width,
                                ).permute(0, 2, 1, 3, 4)
                            time = timestep.float().reshape(-1)
                            if time.numel() == 1:
                                time = time.expand(batch)
                            elif time.numel() != batch:
                                if time.numel() % batch:
                                    raise ValueError(
                                        "timestep batch cannot match Omega batch"
                                    )
                                time = time[:batch]
                            time = (time / 1000.0).clamp(0.0, 1.0)
                            time = time[:, None, None, None, None].expand(
                                batch, 1, target_views, out_height, out_width
                            )
                            state_parts = [
                                hard_features, front, back, statistics, state, time
                            ]
                            if self.router_mode == "state_consensus_multilayer":
                                consensus, consensus_support = route_target_state_consensus(
                                    state,
                                    omega_xyz,
                                    omega_confidence,
                                    omega_target_w2c,
                                    omega_target_intrinsics,
                                    output_hw=(out_height, out_width),
                                )
                                state_parts.extend(
                                    [consensus, consensus - state, consensus_support]
                                )
                            state_input = torch.cat(state_parts, dim=1).permute(
                                0, 2, 1, 3, 4
                            ).reshape(
                                batch * target_views,
                                self.state_router[0].in_channels,
                                out_height,
                                out_width,
                            )
                            selection = torch.tanh(
                                self.state_router(state_input.to(router_input.dtype))
                            ).view(
                                batch, target_views, 3, out_height, out_width
                            ).permute(0, 2, 1, 3, 4)
                            # Three interpretable residual choices: replace the
                            # hard splat by the soft front, expose the secondary
                            # surface, or reject the static Soft correction.
                            target_features = target_features + 0.5 * (
                                selection[:, 0:1] * (front - hard_features)
                                + selection[:, 1:2] * (back - front)
                                - selection[:, 2:3] * correction
                            )
                    elif self.router_mode == "layered_channels":
                        target_features = torch.cat(
                            [hard_features, back, statistics], dim=1
                        )
                    else:
                        disagreement = (front - back).abs()
                        target_features = torch.cat(
                            [hard_features, back, disagreement, statistics], dim=1
                        )
                else:
                    target_features = hard_features
                target_features = target_features.permute(0, 2, 1, 3, 4)
            elif any(value is not None for value in geometry):
                raise ValueError("all Omega geometry tensors must be provided together")
            else:
                target_features = features.new_zeros(
                    batch,
                    int(num_output_frames),
                    self.total_output_channels,
                    out_height,
                    out_width,
                )
            if target_features.shape[1] != int(num_output_frames):
                raise ValueError("Omega target camera count does not match output frames")
            features = torch.cat([features, target_features], dim=1)
        return features.permute(0, 2, 1, 3, 4).contiguous()


def expand_patch_embedding(
    patch_embedding: nn.Conv3d,
    extra_channels: int,
) -> nn.Conv3d:
    if extra_channels <= 0:
        return patch_embedding
    expanded = nn.Conv3d(
        patch_embedding.in_channels + int(extra_channels),
        patch_embedding.out_channels,
        kernel_size=patch_embedding.kernel_size,
        stride=patch_embedding.stride,
        padding=patch_embedding.padding,
        dilation=patch_embedding.dilation,
        groups=patch_embedding.groups,
        bias=patch_embedding.bias is not None,
        padding_mode=patch_embedding.padding_mode,
        device=patch_embedding.weight.device,
        dtype=patch_embedding.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : patch_embedding.in_channels].copy_(patch_embedding.weight)
        if patch_embedding.bias is not None:
            expanded.bias.copy_(patch_embedding.bias)
    return expanded
