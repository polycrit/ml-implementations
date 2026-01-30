import matplotlib.pyplot as plt
import torch
from resnet import FSD50KDataset


def visualize_samples(n=6):
    dataset = FSD50KDataset(split="validation")
    labels = {v: k for k, v in dataset.label_to_idx.items()}

    plt.figure(figsize=(12, 8))
    for i in range(n):
        spec, label_idx = dataset[i]

        plt.subplot(2, (n + 1) // 2, i + 1)
        plt.imshow(spec.squeeze(), aspect="auto", origin="lower", cmap="magma")
        plt.title(labels.get(label_idx, "Unknown"))
        plt.axis("off")

    plt.tight_layout()
    plt.show()


visualize_samples(6)
