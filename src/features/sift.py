"""Educational SIFT-style detector and 128-D descriptor using NumPy only.

This is intentionally compact: it uses one octave by default, which is suitable
for the tiny CIFAR-10 images after optional 2x nearest-neighbour upscaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing import rgb_to_gray, to_float


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(axis**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padding = ((radius, radius), (0, 0)) if axis == 0 else ((0, 0), (radius, radius))
    padded = np.pad(image, padding, mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        if axis == 0:
            output += weight * padded[offset : offset + image.shape[0], :]
        else:
            output += weight * padded[:, offset : offset + image.shape[1]]
    return output


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    return _convolve_axis(_convolve_axis(image, kernel, 1), kernel, 0)


@dataclass
class Keypoint:
    y: int
    x: int
    scale_index: int
    response: float
    orientation: float = 0.0


@dataclass
class SIFT:
    num_scales: int = 5
    sigma: float = 1.0
    contrast_threshold: float = 0.02
    edge_threshold: float = 10.0
    max_keypoints: int = 64
    upscale: int = 2

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = rgb_to_gray(image[None])[0] if image.ndim == 3 else to_float(image)
        if self.upscale > 1:
            gray = np.repeat(np.repeat(gray, self.upscale, axis=0), self.upscale, axis=1)
        return gray.astype(np.float32)

    def _pyramid(self, image: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        factor = 2 ** (1.0 / max(1, self.num_scales - 2))
        blurred = [gaussian_blur(image, self.sigma * factor**index) for index in range(self.num_scales)]
        dogs = [blurred[index + 1] - blurred[index] for index in range(len(blurred) - 1)]
        return blurred, dogs

    def _is_not_edge(self, dog: np.ndarray, y: int, x: int) -> bool:
        dxx = dog[y, x + 1] + dog[y, x - 1] - 2 * dog[y, x]
        dyy = dog[y + 1, x] + dog[y - 1, x] - 2 * dog[y, x]
        dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1] - dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4
        determinant = dxx * dyy - dxy * dxy
        if determinant <= 1e-12:
            return False
        ratio = ((dxx + dyy) ** 2) / determinant
        limit = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        return bool(ratio < limit)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, list[Keypoint]]:
        gray = self._prepare(image)
        blurred, dogs = self._pyramid(gray)
        points: list[Keypoint] = []
        for scale in range(1, len(dogs) - 1):
            current = dogs[scale]
            for y in range(1, current.shape[0] - 1):
                for x in range(1, current.shape[1] - 1):
                    value = current[y, x]
                    if abs(value) < self.contrast_threshold:
                        continue
                    cube = np.stack([dog[y - 1 : y + 2, x - 1 : x + 2] for dog in dogs[scale - 1 : scale + 2]])
                    is_extreme = value >= cube.max() or value <= cube.min()
                    if is_extreme and self._is_not_edge(current, y, x):
                        points.append(Keypoint(y, x, scale, float(abs(value))))
        points.sort(key=lambda point: point.response, reverse=True)
        points = points[: self.max_keypoints]
        for point in points:
            point.orientation = self._orientation(blurred[point.scale_index + 1], point.y, point.x)
        return gray, points

    @staticmethod
    def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1] = image[2:] - image[:-2]
        magnitude = np.hypot(gx, gy)
        angle = np.mod(np.arctan2(gy, gx), 2 * np.pi)
        return magnitude, angle

    def _orientation(self, image: np.ndarray, y: int, x: int) -> float:
        magnitude, angle = self._gradients(image)
        radius = 6
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        histogram = np.zeros(36, dtype=np.float32)
        bins = np.floor(angle[y0:y1, x0:x1] * 36 / (2 * np.pi)).astype(int) % 36
        np.add.at(histogram, bins.ravel(), magnitude[y0:y1, x0:x1].ravel())
        return float((np.argmax(histogram) + 0.5) * 2 * np.pi / 36)

    def _descriptor(self, image: np.ndarray, point: Keypoint) -> np.ndarray | None:
        radius = 8
        if point.y - radius < 1 or point.y + radius >= image.shape[0] - 1:
            return None
        if point.x - radius < 1 or point.x + radius >= image.shape[1] - 1:
            return None
        magnitude, angle = self._gradients(image)
        patch_magnitude = magnitude[point.y - radius : point.y + radius, point.x - radius : point.x + radius]
        patch_angle = np.mod(
            angle[point.y - radius : point.y + radius, point.x - radius : point.x + radius] - point.orientation,
            2 * np.pi,
        )
        descriptor = []
        for cy in range(4):
            for cx in range(4):
                magnitudes = patch_magnitude[cy * 4 : (cy + 1) * 4, cx * 4 : (cx + 1) * 4]
                angles = patch_angle[cy * 4 : (cy + 1) * 4, cx * 4 : (cx + 1) * 4]
                histogram = np.zeros(8, dtype=np.float32)
                bins = np.floor(angles * 8 / (2 * np.pi)).astype(int) % 8
                np.add.at(histogram, bins.ravel(), magnitudes.ravel())
                descriptor.append(histogram)
        vector = np.concatenate(descriptor)
        vector /= np.linalg.norm(vector) + 1e-8
        vector = np.minimum(vector, 0.2)
        vector /= np.linalg.norm(vector) + 1e-8
        return vector.astype(np.float32)

    def detect_and_describe(self, image: np.ndarray) -> tuple[list[Keypoint], np.ndarray]:
        gray, points = self.detect(np.asarray(image))
        valid_points: list[Keypoint] = []
        descriptors: list[np.ndarray] = []
        for point in points:
            descriptor = self._descriptor(gray, point)
            if descriptor is not None:
                valid_points.append(point)
                descriptors.append(descriptor)
        matrix = np.stack(descriptors) if descriptors else np.empty((0, 128), dtype=np.float32)
        return valid_points, matrix

