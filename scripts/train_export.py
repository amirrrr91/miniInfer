import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import struct

# ── Model ──────────────────────────────────────────────────────────────────
# Two linear layers: 784 → 128 → 10
# This is the exact network our C engine will run.

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # weights: 128×784, bias: 128
        self.fc2 = nn.Linear(128, 10)    # weights: 10×128,  bias: 10

    def forward(self, x):
        x = x.view(-1, 784)              # flatten 28×28 image to 784
        x = torch.relu(self.fc1(x))        # linear → ReLU
        x = self.fc2(x)                    # linear (softmax applied in loss)
        return x

# ── Training ───────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=1000)

model     = TinyMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(5):
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        optimizer.step()
    print(f"  epoch {epoch+1}/5 done")

# Quick accuracy check
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
print(f"Test accuracy: {correct / len(test_data) * 100:.1f}%")

# ── Weight export ──────────────────────────────────────────────────────────
# Binary format we designed for our C loader:
#
#   [4 bytes] magic:      "MINF"
#   [4 bytes] version:    1  (uint32)
#   [4 bytes] num_layers: 2  (uint32)
#
#   for each layer:
#     [4 bytes] rows      (uint32)  — output size
#     [4 bytes] cols      (uint32)  — input size
#     [rows*cols * 4 bytes] weights (float32, row-major)
#     [4 bytes] bias_len  (uint32)
#     [bias_len * 4 bytes] bias    (float32)

layers = [model.fc1, model.fc2]

with open('weights/model.bin', 'wb') as f:
    f.write(b'MINF')                                    # magic
    f.write(struct.pack('I', 1))                        # version
    f.write(struct.pack('I', len(layers)))               # num layers

    for layer in layers:
        W = layer.weight.detach().numpy().astype(np.float32)  # shape: (rows, cols)
        b = layer.bias.detach().numpy().astype(np.float32)    # shape: (rows,)

        rows, cols = W.shape
        f.write(struct.pack('II', rows, cols))          # dimensions
        f.write(W.tobytes())                               # weight matrix (row-major)
        f.write(struct.pack('I', len(b)))               # bias length
        f.write(b.tobytes())                               # bias vector

print("Weights exported → weights/model.bin")

# ── Sanity check ───────────────────────────────────────────────────────────
# Print what C should expect to read back
for i, layer in enumerate(layers):
    W = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    print(f"  Layer {i}: W={W.shape}, b={b.shape}, W[0][0]={W[0][0]:.6f}")
