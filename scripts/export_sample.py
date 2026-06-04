import torch
import struct
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST('./data', train=False, download=False, transform=transform)

for i in range(100):
    image, label = test_data[i * 88]  # spread samples across dataset

    pixels = image.view(-1).numpy()

    filename = f'weights/sample_{i}.bin'

    with open(filename, 'wb') as f:
        f.write(struct.pack('I', label))
        f.write(pixels.tobytes())

    print(f"Exported: {filename} — true label: {label}")
    print(f"First pixel value: {pixels[0]:.6f}")
