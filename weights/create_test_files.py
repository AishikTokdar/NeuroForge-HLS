"""Generate deterministic input and genuine PyTorch golden logits."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model_reference import load_model


def create_test_files(source, output, encoding="fp32", seed=42, input_data=None):
    model = load_model(source, encoding)
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(1, 3, 32, 32, generator=generator) if input_data is None else torch.as_tensor(input_data, dtype=torch.float32).reshape(1, 3, 32, 32)
    if not torch.isfinite(inputs).all():
        raise ValueError("Nonfinite input vector")
    with torch.no_grad():
        logits = model(inputs)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite reference inference")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs.numpy().astype("<f4").tofile(output / "input_fp32.bin")
    logits.numpy().astype("<f4").tofile(output / "output_golden_fp32.bin")
    print(f"PyTorch logits: {logits.numpy().ravel()}")
    print(f"Saved test vectors to {output}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root.parent / "hls" / "params")
    parser.add_argument("--output", type=Path, default=root.parent / "hls" / "params")
    parser.add_argument("--format", choices=("fp32", "q16", "fixed16"), default="fixed16",
                        help="Default fixed16 isolates C++ implementation from parameter quantization error")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-file", type=Path, help="3072 normalized CHW little-endian FP32 values")
    args = parser.parse_args()
    inputs = np.fromfile(args.input_file, dtype="<f4") if args.input_file else None
    create_test_files(args.source, args.output, args.format, args.seed, inputs)
