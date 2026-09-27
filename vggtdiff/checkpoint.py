from __future__ import annotations

from pathlib import Path

from safetensors import safe_open


EXPECTED_TENSORS = {
    "patch_embedding.weight": (5120, 452, 1, 2, 2),
    "omega_adapter.proj.weight": (32, 2048),
    "omega_adapter.norm.weight": (2048,),
    "temporal_plucker_adapter.weight": (384, 1536, 1, 1, 1),
}


def validate_checkpoint(path: str | Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"VGGT-Diff checkpoint not found: {path}. "
            "Model weights will be published separately."
        )
    with safe_open(path, framework="numpy") as handle:
        keys = set(handle.keys())
        for name, expected_shape in EXPECTED_TENSORS.items():
            if name not in keys:
                raise ValueError(f"Checkpoint is missing required tensor: {name}")
            shape = tuple(handle.get_slice(name).get_shape())
            if shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {name}: {shape}, expected {expected_shape}"
                )
