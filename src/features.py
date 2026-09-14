"""HOG, SIFT, K-means và Bag of Visual Words cài đặt bằng NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import rgb_to_gray, to_float


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
            raise ValueError("HOG chỉ nhận ảnh grayscale hoặc RGB")
        return to_float(image)

    def transform_one(self, image: np.ndarray) -> np.ndarray:
        gray = self._gray(np.asarray(image))
        height = (gray.shape[0] // self.cell_size) * self.cell_size
        width = (gray.shape[1] // self.cell_size) * self.cell_size
        gray = gray[:height, :width]
        gx, gy = np.zeros_like(gray), np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gx[:, 0], gx[:, -1] = gray[:, 1] - gray[:, 0], gray[:, -1] - gray[:, -2]
        gy[1:-1] = gray[2:] - gray[:-2]
        gy[0], gy[-1] = gray[1] - gray[0], gray[-1] - gray[-2]
        magnitude = np.hypot(gx, gy)
        orientation = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

        cells_y, cells_x = height // self.cell_size, width // self.cell_size
        histograms = np.zeros((cells_y, cells_x, self.num_bins), dtype=np.float32)
        position = orientation / (180.0 / self.num_bins)
        lower = np.floor(position).astype(np.int64) % self.num_bins
        upper = (lower + 1) % self.num_bins
        upper_weight = position - np.floor(position)
        for cy in range(cells_y):
            for cx in range(cells_x):
                ys = slice(cy * self.cell_size, (cy + 1) * self.cell_size)
                xs = slice(cx * self.cell_size, (cx + 1) * self.cell_size)
                for bins, weights in ((lower, 1 - upper_weight), (upper, upper_weight)):
                    np.add.at(
                        histograms[cy, cx],
                        bins[ys, xs].ravel(),
                        (magnitude[ys, xs] * weights[ys, xs]).ravel(),
                    )

        blocks = []
        for by in range(cells_y - self.block_size + 1):
            for bx in range(cells_x - self.block_size + 1):
                block = histograms[by : by + self.block_size, bx : bx + self.block_size].ravel()
                block /= np.sqrt(np.dot(block, block) + self.epsilon**2)
                block = np.minimum(block, self.clip)
                block /= np.sqrt(np.dot(block, block) + self.epsilon**2)
                blocks.append(block)
        if not blocks:
            raise ValueError("Ảnh quá nhỏ so với cell_size/block_size")
        return np.concatenate(blocks).astype(np.float32)

    def transform(self, images: np.ndarray, verbose: bool = False) -> np.ndarray:
        features = []
        for index, image in enumerate(images):
            features.append(self.transform_one(image))
            if verbose and (index + 1) % 1000 == 0:
                print(f"HOG: {index + 1}/{len(images)}")
        return np.stack(features)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, round(3 * sigma))
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
    """SIFT giáo dục một octave, phù hợp cho CIFAR-10 sau khi upscale."""

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
        factor = 2 ** (1 / max(1, self.num_scales - 2))
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
        return bool(((dxx + dyy) ** 2) / determinant < ((self.edge_threshold + 1) ** 2) / self.edge_threshold)

    @staticmethod
    def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx, gy = np.zeros_like(image), np.zeros_like(image)
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1] = image[2:] - image[:-2]
        return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), 2 * np.pi)

    def _orientation(self, image: np.ndarray, y: int, x: int) -> float:
        magnitude, angle = self._gradients(image)
        radius = 6
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        histogram = np.zeros(36, dtype=np.float32)
        bins = np.floor(angle[y0:y1, x0:x1] * 36 / (2 * np.pi)).astype(int) % 36
        np.add.at(histogram, bins.ravel(), magnitude[y0:y1, x0:x1].ravel())
        return float((np.argmax(histogram) + 0.5) * 2 * np.pi / 36)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, list[Keypoint]]:
        gray = self._prepare(image)
        blurred, dogs = self._pyramid(gray)
        points = []
        for scale in range(1, len(dogs) - 1):
            current = dogs[scale]
            for y in range(1, current.shape[0] - 1):
                for x in range(1, current.shape[1] - 1):
                    value = current[y, x]
                    if abs(value) < self.contrast_threshold:
                        continue
                    cube = np.stack([dog[y - 1 : y + 2, x - 1 : x + 2] for dog in dogs[scale - 1 : scale + 2]])
                    if (value >= cube.max() or value <= cube.min()) and self._is_not_edge(current, y, x):
                        points.append(Keypoint(y, x, scale, float(abs(value))))
        points.sort(key=lambda point: point.response, reverse=True)
        points = points[: self.max_keypoints]
        for point in points:
            point.orientation = self._orientation(blurred[point.scale_index + 1], point.y, point.x)
        return gray, points

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
        valid_points, descriptors = [], []
        for point in points:
            descriptor = self._descriptor(gray, point)
            if descriptor is not None:
                valid_points.append(point)
                descriptors.append(descriptor)
        matrix = np.stack(descriptors) if descriptors else np.empty((0, 128), dtype=np.float32)
        return valid_points, matrix


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None]
        - 2 * x @ centers.T,
        0,
    )


@dataclass
class KMeans:
    num_clusters: int = 128
    max_iterations: int = 50
    tolerance: float = 1e-4
    seed: int = 42
    centers_: np.ndarray | None = None

    def fit(self, samples: np.ndarray) -> "KMeans":
        x = np.asarray(samples, dtype=np.float32)
        if len(x) < self.num_clusters:
            raise ValueError("Số descriptor phải lớn hơn hoặc bằng số cluster")
        rng = np.random.default_rng(self.seed)
        centers = x[rng.choice(len(x), self.num_clusters, replace=False)].copy()
        for _ in range(self.max_iterations):
            labels = np.argmin(_squared_distances(x, centers), axis=1)
            updated = centers.copy()
            for cluster in range(self.num_clusters):
                members = x[labels == cluster]
                updated[cluster] = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
            shift = float(np.linalg.norm(updated - centers))
            centers = updated
            if shift <= self.tolerance:
                break
        self.centers_ = centers
        return self

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("KMeans chưa được fit")
        return np.argmin(_squared_distances(np.asarray(samples, dtype=np.float32), self.centers_), axis=1)


@dataclass
class BagOfVisualWords:
    vocabulary_size: int = 128
    max_descriptors: int = 50000
    max_iterations: int = 50
    seed: int = 42
    kmeans_: KMeans | None = None

    def fit(self, descriptor_sets: list[np.ndarray]) -> "BagOfVisualWords":
        nonempty = [descriptors for descriptors in descriptor_sets if len(descriptors)]
        if not nonempty:
            raise ValueError("Không có SIFT descriptor")
        descriptors = np.concatenate(nonempty)
        if len(descriptors) > self.max_descriptors:
            rng = np.random.default_rng(self.seed)
            descriptors = descriptors[rng.choice(len(descriptors), self.max_descriptors, replace=False)]
        self.kmeans_ = KMeans(self.vocabulary_size, self.max_iterations, seed=self.seed).fit(descriptors)
        return self

    def transform(self, descriptor_sets: list[np.ndarray]) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("BagOfVisualWords chưa được fit")
        output = np.zeros((len(descriptor_sets), self.vocabulary_size), dtype=np.float32)
        for index, descriptors in enumerate(descriptor_sets):
            if len(descriptors):
                words = self.kmeans_.predict(descriptors)
                output[index] = np.bincount(words, minlength=self.vocabulary_size)
                output[index] /= np.linalg.norm(output[index]) + 1e-8
        return output

    def fit_transform(self, descriptor_sets: list[np.ndarray]) -> np.ndarray:
        return self.fit(descriptor_sets).transform(descriptor_sets)

