"""Convert explicit FP32 or legacy Q16.16 parameters to little-endian Q12.4."""
import argparse
from pathlib import Path
import numpy as np
from model_reference import load_model


def quantize_fixed16(data):
    if not np.isfinite(data).all():
        raise ValueError("Cannot quantize nonfinite parameters")
    return np.clip(np.rint(data * 16.0), -32768, 32767).astype("<i2")


def convert_int32_to_fixed16_12(data):
    return quantize_fixed16(data.astype(np.float64) / 65536.0)


def convert(source, output, encoding="fp32"):
    model = load_model(source, encoding)
    converted = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        values = tensor.numpy().astype(np.float64)
        raw = np.rint(values * 16.0)
        if np.any((raw < -32768) | (raw > 32767)):
            raise ValueError(f"Q12.4 overflow in {name}; calibrate the numeric format")
        quantized = quantize_fixed16(values)
        if tensor.ndim > 1 and not np.any(quantized):
            raise ValueError(f"Q12.4 erased every weight in {name}; use more fractional bits")
        converted[name.replace(".", "_") + ".bin"] = quantized
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in converted.items():
        values.tofile(output / filename)
    print(f"Converted {len(converted)} inference tensors to {output}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "params_fp32")
    parser.add_argument("--output", type=Path, default=root.parent / "hls" / "params")
    parser.add_argument("--format", choices=("fp32", "q16"), default="fp32")
    args = parser.parse_args()
    convert(args.source, args.output, args.format)
