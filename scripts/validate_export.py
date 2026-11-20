"""Validate exported learned parameters on the official CIFAR-10 test split.

NO_HLS C++ uses floating point with Q12.4 weights. This does not certify RTL
or fixed-point activation arithmetic. Test labels never select a checkpoint.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "weights"))
from create_test_files import create_test_files
from model_reference import load_model
if __package__:
    from .classification_metrics import classification_metrics
else:
    from classification_metrics import classification_metrics


def parameter_arrays(model):
    state = model.state_dict()
    names = []
    for block in range(4):
        for conv, bn in [(0, 1), (3, 4)]:
            names.extend([f"features.{block}.block.{conv}.weight", f"features.{block}.block.{conv}.bias"])
            names.extend(f"features.{block}.block.{bn}.{suffix}" for suffix in
                         ["weight", "bias", "running_mean", "running_var"])
    names.extend(["classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"])
    return [np.ascontiguousarray(state[name].numpy(), dtype="f4") for name in names]


@torch.no_grad()
def torch_logits(model, images, device):
    model = model.to(device).eval()
    outputs = []
    for offset in range(0, len(images), 128):
        outputs.append(model(images[offset:offset + 128].to(device)).cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite PyTorch logits")
    model.cpu()
    return result


def validate(args):
    torch.set_num_threads(4)
    if args.cpp_samples < 1 or args.workers < 1:
        raise ValueError("cpp_samples and workers must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = CIFAR10(str(args.data), train=False, download=False)
    images = torch.from_numpy(data.data).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)
    std = torch.tensor([.2023, .1994, .2010]).view(1, 3, 1, 1)
    images = ((images - mean) / std).contiguous()
    labels = np.asarray(data.targets)
    fp32 = load_model(args.weights / "params_fp32")
    q16 = load_model(args.weights / "params_int32", "q16")
    converted = load_model(args.params, "fixed16")
    for name, tensor in fp32.state_dict().items():
        if not name.endswith("num_batches_tracked"):
            torch.testing.assert_close(q16.state_dict()[name], tensor, atol=1 / 65536, rtol=0)
    floating_logits = torch_logits(fp32, images, args.device)
    converted_logits = torch_logits(converted, images, args.device)
    fp32_accuracy = float(np.mean(floating_logits.argmax(1) == labels))
    converted_accuracy = float(np.mean(converted_logits.argmax(1) == labels))
    print(f"FP32 test accuracy {fp32_accuracy:.4%}; Q12.4-weight host accuracy {converted_accuracy:.4%}", flush=True)
    if converted_accuracy < .35 or len(np.unique(converted_logits.argmax(1))) < 8:
        raise ValueError("Converted classifier failed the minimum usefulness check")
    build = ROOT / ".verification/export_validation"
    build.mkdir(parents=True, exist_ok=True)
    dll = build / ("probe.dll" if os.name == "nt" else "probe.so")
    command = ["g++", "-std=c++11", "-O3", "-shared", "-DNO_HLS",
               "-Wno-unknown-pragmas", "-I" + str(ROOT / "hls"),
               str(ROOT / "tests/native_probe.cpp"), str(ROOT / "hls/tiled_conv.cpp"),
               str(ROOT / "hls/utils.cpp"), "-o", str(dll)]
    command += ["-static-libgcc", "-static-libstdc++"] if os.name == "nt" else ["-fPIC"]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(dll))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.probe_infer.argtypes = [ctypes.POINTER(ptr)]
    arrays = parameter_arrays(converted)
    count = min(args.cpp_samples, len(images))
    actual = np.empty((count, 10), dtype="f4")
    inputs = images.numpy()
    start = time.perf_counter()
    def infer(i):
        buffers = [inputs[i]] + arrays + [actual[i]]
        pointers = (ptr * len(buffers))(*(a.ctypes.data_as(ptr) for a in buffers))
        lib.probe_infer(pointers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, _ in enumerate(pool.map(infer, range(count)), start=1):
            if completed % 500 == 0:
                print(f"C++ verified {completed}/{count} labeled images", flush=True)
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(actual, converted_logits[:count], atol=.01, rtol=.001)
    agreement = float(np.mean(actual.argmax(1) == converted_logits[:count].argmax(1)))
    if agreement < .999:
        raise ValueError("C++ and PyTorch predictions disagree")
    for output in [args.params, args.weights / "params"]:
        create_test_files(args.params, output, "fixed16", input_data=inputs[0])
    tensor_stats = {name: {"elements": tensor.numel(), "nonzero": int(torch.count_nonzero(tensor))}
                    for name, tensor in converted.state_dict().items() if tensor.ndim > 1}
    cpp_metrics = classification_metrics(labels[:count], actual, data.classes)
    results = {
        "dataset": "official CIFAR-10 test split", "test_images": len(labels),
        "dataset_archive_md5": "c58f30108f718f92721af3b95e74349a",
        "fp32_test_accuracy": fp32_accuracy,
        "q12_4_weight_host_test_accuracy": converted_accuracy,
        "cpp_images": count, "cpp_accuracy": float(np.mean(actual.argmax(1) == labels[:count])),
        "cpp_macro_recall": cpp_metrics["macro_recall"],
        "cpp_weighted_recall": cpp_metrics["weighted_recall"],
        "cpp_macro_precision": cpp_metrics["macro_precision"],
        "cpp_macro_f1": cpp_metrics["macro_f1"],
        "cpp_classification": cpp_metrics,
        "fp32_classification": classification_metrics(labels, floating_logits, data.classes),
        "q12_4_weight_host_classification": classification_metrics(labels, converted_logits, data.classes),
        "cpp_pytorch_prediction_agreement": agreement,
        "cpp_max_abs_logit_error": float(np.max(np.abs(actual - converted_logits[:count]))),
        "cpp_total_seconds": elapsed, "cpp_workers": args.workers,
        "cpp_arithmetic": "NO_HLS floating point with Q12.4 parameters",
        "fixed_point_rtl_verified": False, "converted_learned_tensors": tensor_stats,
        "golden_vector": {"dataset_index": 0, "label": int(labels[0]), "class": data.classes[labels[0]],
                          "prediction": int(actual[0].argmax()), "logits": actual[0].tolist()},
        "torch_version": str(torch.__version__), "device": args.device,
    }
    for folder in ["params_fp32", "params_int32", "params"]:
        results[folder + "_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.weights / folder).glob("*.bin"))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(build / "test_predictions.npz", labels=labels[:count], cpp_logits=actual,
                        pytorch_logits=converted_logits[:count])
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions_output, labels=labels[:count], cpp_logits=actual,
                        pytorch_logits=converted_logits[:count])
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".verification/data")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--params", type=Path, default=ROOT / "hls/params")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/weight_validation.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpp-samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4, help="Parallel independent C++ inference calls")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "docs/test_predictions.npz")
    validate(parser.parse_args())
