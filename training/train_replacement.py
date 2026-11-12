"""Train a replacement ReducedVGG on CIFAR-10 and save a reproducible checkpoint.

The original learned weights cannot be reconstructed from zeroed integer files.
Uses a held-out validation split for checkpoint selection, and evaluates the
untouched official test split only after training.
"""
import argparse
import json
import random
import time
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "weights"))
from model_reference import ReducedVGG
sys.path.insert(0, str(ROOT / "scripts"))
from classification_metrics import CIFAR10_CLASSES, classification_metrics


def ste_quantize(x, activation=False):
    scaled = torch.floor(x * 16) if activation else torch.round(x * 16)
    quantized = torch.clamp(scaled, -32768, 32767) / 16
    return x + (quantized - x).detach()


def enable_qat(model):
    """Train with Q12.4 parameters, activation casts and frozen BN statistics.

    Accumulation and BN arithmetic still use floating point, so this is a
    training approximation, not a replacement for vendor fixed-point simulation.
    """
    def conv_forward(module, x):
        y = F.conv2d(x, ste_quantize(module.weight), ste_quantize(module.bias),
                     module.stride, module.padding, module.dilation, module.groups)
        return ste_quantize(y, True)

    def linear_forward(module, x):
        return ste_quantize(F.linear(x, ste_quantize(module.weight),
                                     ste_quantize(module.bias)), True)

    def bn_forward(module, x):
        return ste_quantize(F.batch_norm(
            x.float(), ste_quantize(module.running_mean),
            ste_quantize(module.running_var), ste_quantize(module.weight),
            ste_quantize(module.bias), False, 0.0, module.eps), True)

    model.qat_enabled = True
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.forward = types.MethodType(conv_forward, module)
        elif isinstance(module, nn.Linear):
            module.forward = types.MethodType(linear_forward, module)
        elif isinstance(module, nn.BatchNorm2d):
            module.forward = types.MethodType(bn_forward, module)
    return model


def prepare_images(raw, mean, std, augment=False):
    x = raw.permute(0, 3, 1, 2).float() / 255
    if augment:
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        batch = x.shape[0]
        r = torch.randint(9, (batch, 1, 1), device=x.device)
        c = torch.randint(9, (batch, 1, 1), device=x.device)
        rows = r + torch.arange(32, device=x.device)[None, :, None]
        cols = c + torch.arange(32, device=x.device)[None, None, :]
        x = x.permute(0, 2, 3, 1)[torch.arange(batch, device=x.device)[:, None, None], rows, cols]
        x = x.permute(0, 3, 1, 2)
        flip = torch.rand(batch, 1, 1, 1, device=x.device) < .5
        x = torch.where(flip, x.flip(-1), x)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, raw, labels, indices, batch_size, mean, std, return_metrics=False):
    if batch_size < 1 or len(indices) == 0:
        raise ValueError("Evaluation needs a positive batch size and nonempty indices")
    model.eval()
    correct = 0
    outputs, targets = [], []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        inputs = prepare_images(raw[idx], mean, std)
        if getattr(model, "qat_enabled", False):
            inputs = ste_quantize(inputs, True)
        y = model(inputs)
        if not torch.isfinite(y).all():
            raise ValueError("Nonfinite inference during evaluation")
        correct += (y.argmax(1) == labels[idx]).sum().item()
        if return_metrics:
            outputs.append(y.cpu().numpy())
            targets.append(labels[idx].cpu().numpy())
    if return_metrics:
        return classification_metrics(np.concatenate(targets), np.concatenate(outputs), CIFAR10_CLASSES)
    return correct / len(indices)


def validate_training_args(args):
    if args.epochs < 0 or args.qat_epochs < 0 or args.epochs + args.qat_epochs == 0:
        raise ValueError("Choose nonnegative epoch counts with at least one training epoch")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch_size and threads must be positive")
    if args.epochs == 0 and not args.resume:
        raise ValueError("QAT-only fine-tuning requires --resume with a trained checkpoint")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)


def train(args):
    validate_training_args(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use the tf-gpu environment")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}; torch: {torch.__version__}", flush=True)
    train_data = CIFAR10(str(args.data), train=True, download=True)
    test_data = CIFAR10(str(args.data), train=False, download=True)
    raw = torch.from_numpy(train_data.data).to(device)
    labels = torch.tensor(train_data.targets, device=device)
    test_raw = torch.from_numpy(test_data.data).to(device)
    test_labels = torch.tensor(test_data.targets, device=device)
    split = torch.randperm(len(raw), generator=torch.Generator().manual_seed(args.seed))
    train_idx, val_idx = split[:45000].to(device), split[45000:].to(device)
    mean = torch.tensor([.4914, .4822, .4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.2023, .1994, .2010], device=device).view(1, 3, 1, 1)
    model = ReducedVGG().to(device, memory_format=torch.channels_last)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_states = {}
    for phase, epochs, lr in [("fp32", args.epochs, .001), ("qat", args.qat_epochs, .0001)]:
        if not epochs:
            continue
        if phase == "qat":
            model = enable_qat(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and phase == "fp32")
        best_val = -1
        for epoch in range(epochs):
            start = time.perf_counter()
            model.train()
            order = train_idx[torch.randperm(len(train_idx), device=device)]
            correct = 0
            total_loss = 0
            for offset in range(0, len(order), args.batch_size):
                idx = order[offset:offset + args.batch_size]
                x = prepare_images(raw[idx], mean, std, True)
                if phase == "qat":
                    x = ste_quantize(x, True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda" and phase == "fp32"):
                    logits = model(x)
                    loss = F.cross_entropy(logits, labels[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                correct += (logits.argmax(1) == labels[idx]).sum().item()
                total_loss += loss.item() * len(idx)
            scheduler.step()
            val_metrics = evaluate(model, raw, labels, val_idx, args.batch_size, mean, std, return_metrics=True)
            val_acc = val_metrics["accuracy"]
            record = dict(phase=phase, epoch=epoch + 1, loss=total_loss / len(order),
                          train_accuracy=correct / len(order), validation_accuracy=val_acc,
                          validation_macro_recall=val_metrics["macro_recall"],
                          validation_macro_precision=val_metrics["macro_precision"],
                          validation_macro_f1=val_metrics["macro_f1"],
                          seconds=time.perf_counter() - start)
            history.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if val_acc > best_val:
                best_val = val_acc
                state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                checkpoint = {"model_state": state, "seed": args.seed, "phase": phase,
                              "epoch": epoch + 1, "validation_accuracy": val_acc,
                              "validation_macro_recall": val_metrics["macro_recall"],
                              "validation_macro_f1": val_metrics["macro_f1"],
                              "architecture": "ReducedVGG", "normalization_mean": [.4914,.4822,.4465],
                              "normalization_std": [.2023,.1994,.2010],
                              "train_images": 45000, "validation_images": 5000}
                torch.save(checkpoint, args.output / f"best_{phase}.pth")
                best_states[phase] = state
        model.load_state_dict(best_states[phase])
    results = {}
    for phase in best_states:
        candidate = ReducedVGG().to(device, memory_format=torch.channels_last)
        candidate.load_state_dict(best_states[phase])
        if phase == "qat":
            enable_qat(candidate)
        metrics = evaluate(
            candidate, test_raw, test_labels, torch.arange(len(test_raw), device=device),
            args.batch_size, mean, std, return_metrics=True)
        results[phase + "_test_accuracy"] = metrics["accuracy"]
        results[phase + "_test_macro_recall"] = metrics["macro_recall"]
        results[phase + "_test_metrics"] = metrics
    results.update(seed=args.seed, test_images=10000, device=str(device), torch_version=str(torch.__version__))
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps({key: value for key, value in results.items() if not isinstance(value, dict)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".verification/data")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/training_run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--resume", type=Path, help="Warm-start model weights; optimizer and phase schedules restart")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())
