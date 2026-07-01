#!/usr/bin/env python3
"""
Create test input and golden output files for HLS testbench using actual PyTorch model inference
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# VGG Block definition exactly as in vgg_training.ipynb
class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ReducedVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 32, num_convs=2),
            VGGBlock(32, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Construct state_dict from binary files
state_dict = {}
params_dir = Path("params_int32")

# We read raw INT32 parameters and scale them back to float
def load_param(name, shape):
    path = params_dir / f"{name}.bin"
    arr = np.fromfile(path, dtype=np.int32).astype(np.float32) / 65536.0
    return torch.from_numpy(arr.reshape(shape))

print("Loading PyTorch model parameters from params_int32/...")
try:
    # VGG Blocks
    for b in range(4):
        in_ch = 3 if b == 0 else (32 * (2**(b-1)))
        out_ch = 32 * (2**b)
        
        # Conv 1
        state_dict[f"features.{b}.block.0.weight"] = load_param(f"features_{b}_block_0_weight", (out_ch, in_ch, 3, 3))
        state_dict[f"features.{b}.block.0.bias"] = load_param(f"features_{b}_block_0_bias", (out_ch,))
        
        # BN 1
        state_dict[f"features.{b}.block.1.weight"] = load_param(f"features_{b}_block_1_weight", (out_ch,))
        state_dict[f"features.{b}.block.1.bias"] = load_param(f"features_{b}_block_1_bias", (out_ch,))
        state_dict[f"features.{b}.block.1.running_mean"] = load_param(f"features_{b}_block_1_running_mean", (out_ch,))
        state_dict[f"features.{b}.block.1.running_var"] = load_param(f"features_{b}_block_1_running_var", (out_ch,))
        state_dict[f"features.{b}.block.1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
        
        # Conv 2
        state_dict[f"features.{b}.block.3.weight"] = load_param(f"features_{b}_block_3_weight", (out_ch, out_ch, 3, 3))
        state_dict[f"features.{b}.block.3.bias"] = load_param(f"features_{b}_block_3_bias", (out_ch,))
        
        # BN 2
        state_dict[f"features.{b}.block.4.weight"] = load_param(f"features_{b}_block_4_weight", (out_ch,))
        state_dict[f"features.{b}.block.4.bias"] = load_param(f"features_{b}_block_4_bias", (out_ch,))
        state_dict[f"features.{b}.block.4.running_mean"] = load_param(f"features_{b}_block_4_running_mean", (out_ch,))
        state_dict[f"features.{b}.block.4.running_var"] = load_param(f"features_{b}_block_4_running_var", (out_ch,))
        state_dict[f"features.{b}.block.4.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # Classifier
    state_dict["classifier.1.weight"] = load_param("classifier_1_weight", (256, 1024))
    state_dict["classifier.1.bias"] = load_param("classifier_1_bias", (256,))
    state_dict["classifier.4.weight"] = load_param("classifier_4_weight", (10, 256))
    state_dict["classifier.4.bias"] = load_param("classifier_4_bias", (10,))
    print("Successfully built state_dict.")
except Exception as e:
    print(f"Error loading parameters: {e}")
    exit(1)

# Instantiate model and load state_dict
model = ReducedVGG(num_classes=10)
model.load_state_dict(state_dict)
model.eval()

# Generate random input (32x32x3) normalized
torch.manual_seed(42)
input_tensor = torch.randn(1, 3, 32, 32)

# Run inference
with torch.no_grad():
    output_tensor = model(input_tensor)

print("Inference output prediction logits:")
print(output_tensor.numpy()[0])

# Flatten and save to binary files
input_data = input_tensor.numpy().flatten().astype(np.float32)
output_data = output_tensor.numpy().flatten().astype(np.float32)

os.makedirs('../hls/params', exist_ok=True)

with open('../hls/params/input_fp32.bin', 'wb') as f:
    f.write(input_data.tobytes())

with open('../hls/params/output_golden_fp32.bin', 'wb') as f:
    f.write(output_data.tobytes())

print("\nSuccessfully created real inference validation files:")
print("  hls/params/input_fp32.bin")
print("  hls/params/output_golden_fp32.bin")
