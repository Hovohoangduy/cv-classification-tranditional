"""Render genuine RGB training samples, one per CIFAR-10 class."""
from pathlib import Path
import os
import pickle
os.environ.setdefault('MPLCONFIGDIR', 'outputs/.matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def dataset_samples():
    with (ROOT / 'data/cifar-10-batches-py/data_batch_1').open('rb') as stream:
        batch = pickle.load(stream, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b'labels'])
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.1))
    for label, (name, axis) in enumerate(zip(CLASSES, axes.flat)):
        index = int(np.flatnonzero(labels == label)[0])
        axis.imshow(images[index], interpolation='nearest')
        axis.set_title(name, fontsize=12, color='#24364b', pad=6)
        axis.set(xticks=[], yticks=[])
        for spine in axis.spines.values():
            spine.set_color('#8392a5')
            spine.set_linewidth(1.0)
    fig.tight_layout(h_pad=1.4, w_pad=0.8)
    destination = ROOT / 'report/assets/figures/cifar10_samples.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


if __name__ == '__main__':
    dataset_samples()
