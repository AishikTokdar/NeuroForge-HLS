"""Observe Q12.4 casts and accumulator bounds on labeled CIFAR-10 images.

This uses floating-point operators with Q12.4 cast hooks. It does not certify
vendor fixed-point arithmetic or unobserved inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "weights"))
from model_reference import load_model


@torch.no_grad()
def audit(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = CIFAR10(str(args.data), train=False, download=False)
    model = load_model(args.params, "fixed16").eval().to(args.device)
    stats = {}
    handles = []

    def cast(name, x):
        if not torch.isfinite(x).all():
            raise ValueError(f"Nonfinite values in {name}")
        raw = torch.floor(x * 16)
        row = stats.setdefault(name, dict(minimum=None, maximum=None, values_checked=0,
                                          q12_4_overflow_values=0))
        low, high = x.min().item(), x.max().item()
        row["minimum"] = min(row["minimum"], low) if row["minimum"] is not None else low
        row["maximum"] = max(row["maximum"], high) if row["maximum"] is not None else high
        row["values_checked"] += x.numel()
        row["q12_4_overflow_values"] += int(((raw < -32768) | (raw > 32767)).sum().item())
        return raw.clamp(-32768, 32767) / 16

    def output_hook(name):
        def hook(module, inputs, output):
            return cast(name, output)
        return hook

    def bound_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            if isinstance(module, nn.Conv2d):
                bounds = F.conv2d(x.abs(), module.weight.abs(), module.bias.abs(),
                                  module.stride, module.padding, module.dilation, module.groups)
            else:
                bounds = F.linear(x.abs(), module.weight.abs(), module.bias.abs())
            bound = bounds.max().item()
            row = stats.setdefault(name, dict(minimum=None, maximum=None, values_checked=0,
                                              q12_4_overflow_values=0))
            row["maximum_absolute_partial_sum_bound"] = max(
                bound, row.get("maximum_absolute_partial_sum_bound", 0))
        return hook

    def input_hook(module, inputs):
        return (cast("input", inputs[0]),)

    handles.append(model.register_forward_pre_hook(input_hook))
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_pre_hook(bound_hook(name)))
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handles.append(module.register_forward_hook(output_hook(name)))
    mean = torch.tensor([.4914, .4822, .4465], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([.2023, .1994, .2010], device=args.device).view(1, 3, 1, 1)
    correct = 0
    try:
        for start in range(0, len(data), args.batch_size):
            raw = torch.from_numpy(data.data[start:start + args.batch_size]).to(args.device)
            x = raw.permute(0, 3, 1, 2).float() / 255
            output = model(((x - mean) / std).contiguous())
            labels = torch.tensor(data.targets[start:start + args.batch_size], device=args.device)
            correct += int((output.argmax(1) == labels).sum().item())
    finally:
        for handle in handles:
            handle.remove()
    maximum_bound = max(row.get("maximum_absolute_partial_sum_bound", 0) for row in stats.values())
    result = dict(samples=len(data), approximate_q12_4_accuracy=correct / len(data),
                  arithmetic="FP32 operators with Q12.4 parameter/input/output casts; TF32 disabled",
                  cast_overflow_values=sum(row["q12_4_overflow_values"] for row in stats.values()),
                  cast_values_checked=sum(row["values_checked"] for row in stats.values()),
                  accumulator_limit=8388608,
                  maximum_observed_absolute_partial_sum_bound=maximum_bound,
                  observed_accumulator_bound_exceeds_limit=maximum_bound >= 8388608,
                  layers=stats, vendor_bit_accurate_simulation=False,
                  parameter_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(args.params.glob("*.bin"))
                                    if not path.name.endswith("num_batches_tracked.bin") and
                                    path.name not in {"input_fp32.bin", "output_golden_fp32.bin"}},
                  scope="Observed test inputs only; no guarantee for future inputs. No calibration was fitted to the test set.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".verification/data")
    parser.add_argument("--params", type=Path, default=ROOT / "hls/params")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/numeric_range_audit.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    audit(parser.parse_args())
