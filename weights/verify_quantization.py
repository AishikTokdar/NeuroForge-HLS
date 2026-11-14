"""Measure parameter MSE/SQNR and verify each Q12.4 binary against its source."""
import argparse
from pathlib import Path
import numpy as np
from convert_weights import quantize_fixed16
from model_reference import load_model


def verify_all_weights(source, output, encoding="fp32"):
    model = load_model(source, encoding)
    records = []
    for name, tensor in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        original = tensor.numpy().astype(np.float64).ravel()
        path = Path(output) / (name.replace(".", "_") + ".bin")
        if path.stat().st_size != original.size * 2:
            raise ValueError(f"Incorrect converted size: {path}")
        raw = np.fromfile(path, dtype="<i2")
        if not np.array_equal(raw, quantize_fixed16(original)):
            raise ValueError(f"Converted values do not match source: {path}")
        scaled = np.rint(original * 16.0)
        overflow = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
        if overflow:
            raise ValueError(f"{name}: {overflow} saturated values")
        if tensor.ndim > 1 and not np.any(raw):
            raise ValueError(f"Quantization erased all weights: {name}")
        mse = float(np.mean((original - raw.astype(np.float64) / 16.0) ** 2))
        power = float(np.mean(original ** 2))
        sqnr = 10 * np.log10(power / mse) if mse and power else (float("inf") if not mse else -float("inf"))
        records.append((name, mse, sqnr))
        print(f"{name}: MSE={mse:.8g}, SQNR={sqnr:.3f} dB")
    print(f"Verified {len(records)} tensors; dataset accuracy requires labeled inference.")
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "params_fp32")
    parser.add_argument("--output", type=Path, default=root.parent / "hls" / "params")
    parser.add_argument("--format", choices=("fp32", "q16"), default="fp32")
    args = parser.parse_args()
    verify_all_weights(args.source, args.output, args.format)
