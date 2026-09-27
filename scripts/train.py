#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffsynth.diffusion import ModelLogger
from vggtdiff.checkpoint import validate_checkpoint
from vggtdiff.data import SceneDataset
from vggtdiff.training import VGGTDiffTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGGT-Diff")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--omega-cache", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target-frames", type=int, default=80)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--condition-lr", type=float, default=1e-5)
    parser.add_argument("--plucker-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260408501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None:
        validate_checkpoint(args.resume)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation
    )
    set_seed(args.seed, device_specific=True)
    dataset = SceneDataset(
        dataset_root=args.dataset_root,
        omega_cache=args.omega_cache,
        height=args.height,
        width=args.width,
        target_frames=args.target_frames,
        repeat=args.repeat,
        seed=args.seed,
        max_scenes=args.max_scenes,
    )
    loader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda items: items[0],
        num_workers=args.workers,
    )
    model = VGGTDiffTrainingModule(
        checkpoint=args.resume,
        base_model_dir=args.base_model_dir,
        device=accelerator.device,
    )
    groups = model.optimizer_parameter_groups(
        args.backbone_lr, args.condition_lr, args.plucker_lr
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return 0.1 + 0.9 * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            args.steps - args.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    logger = ModelLogger(str(args.output), remove_prefix_in_ckpt="pipe.dit.")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    step = 0
    epoch = 0
    progress = tqdm(total=args.steps, disable=not accelerator.is_main_process)
    while step < args.steps:
        dataset.current_epoch = epoch
        for batch in loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}")
            logger.on_step_end(
                accelerator, model, save_steps=args.save_every, loss=loss
            )
            if step >= args.steps:
                break
        epoch += 1
    logger.on_training_end(accelerator, model, save_steps=args.save_every)
    progress.close()


if __name__ == "__main__":
    main()
