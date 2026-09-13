"""Histogram of Oriented Gradients implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing import rgb_to_gray, to_float


@dataclass
class HOG:
    cell_size: int = 4
    block_size: int = 2
    num_bins: int = 9
    clip: float = 0.2
    epsilon: float = 1e-6

    def _gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return rgb_to_gray(image[None])[0]
        if image.ndim != 2:
            raise ValueError("HOG expects a grayscale or RGB image")
        return to_float(image)

    def transform_one(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(np.asarray(image))
        height = (gray.shape[0] // self.cell_size) * self.cell_size
        width = (gray.shape[1] // self.cell_size) * self.cell_size
        gray = gray[:height, :width]
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gx[:, 0], gx[:, -1] = gray[:, 1] - gray[:, 0], gray[:, -1] - gray[:, -2]
        gy[1:-1] = gray[2:] - gray[:-2]
        gy[0], gy[-1] = gray[1] - gray[0], gray[-1] - gray[-2]
        magnitude = np.hypot(gx, gy)
        orientation = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

        cells_y, cells_x = height // self.cell_size, width // self.cell_size
        histograms = np.zeros((cells_y, cells_x, self.num_bins), dtype=np.float32)
        bin_position = orientation / (180.0 / self.num_bins)
        lower = np.floor(bin_position).astype(np.int64) % self.num_bins
        upper = (lower + 1) % self.num_bins
        upper_weight = bin_position - np.floor(bin_position)
        lower_weight = 1.0 - upper_weight
        for cy in range(cells_y):
            for cx in range(cells_x):
                ys = slice(cy * self.cell_size, (cy + 1) * self.cell_size)
                xs = slice(cx * self.cell_size, (cx + 1) * self.cell_size)
                for bins, weights in ((lower, lower_weight), (upper, upper_weight)):
                    np.add.at(
                        histograms[cy, cx],
                        bins[ys, xs].ravel(),
                        (magnitude[ys, xs] * weights[ys, xs]).ravel(),
                    )

        blocks: list[np.ndarray] = []
        for by in range(cells_y - self.block_size + 1):
            for bx in range(cells_x - self.block_size + 1):
                block = histograms[by : by + self.block_size, bx : bx + self.block_size].ravel()
                block = block / np.sqrt(np.dot(block, block) + self.epsilon**2)
                block = np.minimum(block, self.clip)
                block = block / np.sqrt(np.dot(block, block) + self.epsilon**2)
                blocks.append(block)
        if not blocks:
            raise ValueError("Image is too small for the selected cell/block size")
        return np.concatenate(blocks).astype(np.float32)

    def transform(self, images: np.ndarray, verbose: bool = False) -> np.ndarray:
        features = []
        for index, image in enumerate(images):
            features.append(self.transform_one(image))
            if verbose and (index + 1) % 1000 == 0:
                print(f"HOG: {index + 1}/{len(images)} images")
        return np.stack(features)

