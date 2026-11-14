"""Export a validated PyTorch ReducedVGG checkpoint as FP32 and Q16.16 bins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model_reference import ReducedVGG, load_model


def export_checkpoint(checkpoint_path, output_root):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model = ReducedVGG().eval()
    model.load_state_dict(state, strict=True)
    arrays = {}
    for name, tensor in model.state_dict().items():
        values = tensor.detach().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite tensor: {name}")
        if values.ndim > 1 and not np.any(values):
            raise ValueError(f"All-zero learned tensor: {name}")
        if name.endswith("running_var") and np.any(values < 0):
            raise ValueError(f"Negative BatchNorm variance: {name}")
        scaled = values if name.endswith("num_batches_tracked") else np.rint(values.astype(np.float64) * 65536)
        if np.any(scaled < -2147483648) or np.any(scaled > 2147483647):
            raise ValueError(f"Q16.16 overflow: {name}")
        if values.ndim > 1 and not np.any(scaled):
            raise ValueError(f"Q16.16 erased every learned weight: {name}")
        arrays[name.replace(".", "_") + ".bin"] = (
            values.astype("<f4"), scaled.astype("<i4"))
    output_root = Path(output_root)
    manifest = {"checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "architecture": "ReducedVGG", "source": str(checkpoint_path.name),
                "fp32_encoding": "little-endian IEEE float32",
                "int32_encoding": "little-endian signed Q16.16; counters are unscaled integers",
                "files": {}}
    for folder, column in [("params_fp32", 0), ("params_int32", 1)]:
        output = output_root / folder
        output.mkdir(parents=True, exist_ok=True)
        for name, pair in arrays.items():
            pair[column].tofile(output / name)
        manifest["files"][folder] = {name: hashlib.sha256(pair[column].tobytes()).hexdigest()
                                       for name, pair in arrays.items()}
    fp32 = load_model(output_root / "params_fp32")
    q16 = load_model(output_root / "params_int32", "q16")
    for name, tensor in fp32.state_dict().items():
        if not name.endswith("num_batches_tracked"):
            torch.testing.assert_close(q16.state_dict()[name], tensor, atol=1 / 65536, rtol=0)
    (output_root / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(arrays)} tensors in FP32 and genuine Q16.16 format")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output_root)
