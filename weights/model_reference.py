"""Shared model with explicit little-endian binary parameter formats."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ReducedVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 32), VGGBlock(32, 64), VGGBlock(64, 128), VGGBlock(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1024, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def load_model(source, encoding="fp32", allow_degenerate=False):
    model = ReducedVGG().eval()
    state = model.state_dict()
    dtypes = {"fp32": "<f4", "q16": "<i4", "fixed16": "<i2"}
    for name, tensor in state.items():
        if name.endswith("num_batches_tracked"):
            state[name] = torch.zeros_like(tensor)
            continue
        path = Path(source) / (name.replace(".", "_") + ".bin")
        dtype = np.dtype(dtypes[encoding])
        if path.stat().st_size != tensor.numel() * dtype.itemsize:
            raise ValueError(f"Incorrect binary size: {path}")
        values = np.fromfile(path, dtype=dtype).astype(np.float32)
        if encoding != "fp32":
            values /= 65536.0 if encoding == "q16" else 16.0
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values: {path}")
        if name.endswith("running_var") and np.any(values < 0):
            raise ValueError(f"Negative BatchNorm variance: {path}")
        if not allow_degenerate and tensor.ndim > 1 and not np.any(values):
            raise ValueError(f"All-zero convolution/classifier weights: {path}; re-export trained weights")
        state[name] = torch.from_numpy(values.reshape(tensor.shape))
    model.load_state_dict(state)
    return model
