from __future__ import annotations

import torch


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def to(self, *args, **kwargs):
        for model in self.children():
            model.to(*args, **kwargs)
        return self

    def trainable_param_names(self) -> set[str]:
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def export_trainable_state_dict(
        self, state_dict: dict, remove_prefix: str | None = None
    ) -> dict:
        names = self.trainable_param_names()
        exported = {name: value for name, value in state_dict.items() if name in names}
        if remove_prefix is None:
            return exported
        return {
            name[len(remove_prefix):] if name.startswith(remove_prefix) else name: value
            for name, value in exported.items()
        }

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.is_floating_point():
                data = data.to(torch_float_dtype)
            return data
        if isinstance(data, tuple):
            return tuple(
                self.transfer_data_to_device(value, device, torch_float_dtype)
                for value in data
            )
        if isinstance(data, list):
            return [
                self.transfer_data_to_device(value, device, torch_float_dtype)
                for value in data
            ]
        if isinstance(data, dict):
            return {
                key: self.transfer_data_to_device(value, device, torch_float_dtype)
                for key, value in data.items()
            }
        return data

    def switch_pipe_to_training_mode(self, pipe, trainable_models: str) -> None:
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except(trainable_models.split(","))
