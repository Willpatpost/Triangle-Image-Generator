from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from core.config import ColorMode


@dataclass(frozen=True, slots=True)
class EvolutionGuide:
    """Read-only target and baseline used to place useful new shapes.

    ``error_stride`` subsamples the reference/baseline difference into an
    error map that drives guided placement and mutation targeting; a stride
    of 1 keeps the original uniform random probing.
    """

    reference: np.ndarray
    baseline: np.ndarray
    error_stride: int = 4
    _error_cells: np.ndarray | None = field(default=None, repr=False, compare=False)
    _error_cumulative: np.ndarray | None = field(default=None, repr=False, compare=False)
    _error_total: float = field(default=0.0, repr=False, compare=False)
    _error_integral: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.reference.shape != self.baseline.shape:
            raise ValueError("reference and baseline must have matching shapes")
        if self.error_stride < 1:
            raise ValueError("error_stride must be at least 1")
        if self.error_stride == 1:
            return
        cells = self._build_error_cells()
        object.__setattr__(self, "_error_cells", cells)
        if cells.size == 0:
            return
        cumulative = np.cumsum(cells.ravel())
        object.__setattr__(self, "_error_cumulative", cumulative)
        object.__setattr__(self, "_error_total", float(cumulative[-1]))
        integral = np.zeros((cells.shape[0] + 1, cells.shape[1] + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(cells, axis=0), axis=1)
        object.__setattr__(self, "_error_integral", integral)

    def _build_error_cells(self) -> np.ndarray:
        reference = np.asarray(self.reference, dtype=np.float32)
        baseline = np.asarray(self.baseline, dtype=np.float32)
        diff = reference - baseline
        if diff.ndim == 3:
            diff = np.einsum("yxc,yxc->yx", diff, diff)
        else:
            diff = diff * diff
        return np.ascontiguousarray(
            diff[:: self.error_stride, :: self.error_stride], dtype=np.float64
        )

    @classmethod
    def from_background(
        cls,
        reference: np.ndarray,
        background: int | tuple[int, int, int],
        error_stride: int = 4,
    ) -> EvolutionGuide:
        value = np.asarray(background, dtype=np.float32)
        baseline = np.broadcast_to(value, reference.shape)
        return cls(reference=reference, baseline=baseline, error_stride=error_stride)

    def sample_high_error_point(self, candidates: int, *, rng: random.Random) -> tuple[int, int]:
        height, width = self.reference.shape[:2]
        if self._error_cumulative is not None and self._error_total > 0.0:
            threshold = rng.random() * self._error_total
            index = int(np.searchsorted(self._error_cumulative, threshold, side="right"))
            index = min(index, self._error_cumulative.size - 1)
            cells = self._error_cells
            assert cells is not None
            cell_y, cell_x = divmod(index, cells.shape[1])
            stride = self.error_stride
            span_x = max(1, min(stride, width - cell_x * stride))
            span_y = max(1, min(stride, height - cell_y * stride))
            x = cell_x * stride + rng.randrange(span_x)
            y = cell_y * stride + rng.randrange(span_y)
            return x, y

        best_x = rng.randrange(width)
        best_y = rng.randrange(height)
        best_error = -1.0

        if self.reference.ndim == 2:
            for _ in range(candidates):
                x = rng.randrange(width)
                y = rng.randrange(height)
                delta = float(self.reference[y, x]) - float(self.baseline[y, x])
                error = delta * delta
                if error > best_error:
                    best_x, best_y, best_error = x, y, error
        else:
            for _ in range(candidates):
                x = rng.randrange(width)
                y = rng.randrange(height)
                target = self.reference[y, x]
                current = self.baseline[y, x]
                d0 = float(target[0]) - float(current[0])
                d1 = float(target[1]) - float(current[1])
                d2 = float(target[2]) - float(current[2])
                error = d0 * d0 + d1 * d1 + d2 * d2
                if error > best_error:
                    best_x, best_y, best_error = x, y, error

        return best_x, best_y

    def shape_error_mass(self, min_x: int, min_y: int, max_x: int, max_y: int) -> float:
        """Summed error over a pixel bounding box, from the strided error cells."""
        if self._error_integral is None or self._error_total <= 0.0:
            return 0.0
        cells = self._error_cells
        assert cells is not None
        stride = self.error_stride
        x0 = min(min_x // stride, cells.shape[1] - 1)
        x1 = min(max_x // stride, cells.shape[1] - 1)
        y0 = min(min_y // stride, cells.shape[0] - 1)
        y1 = min(max_y // stride, cells.shape[0] - 1)
        if x1 < x0 or y1 < y0:
            return 0.0
        integral = self._error_integral
        assert integral is not None
        return float(
            integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
        )

    def optimal_color(
        self,
        x: int,
        y: int,
        alpha: float,
        mode: ColorMode,
    ) -> int | tuple[int, int, int]:
        alpha = max(alpha, 1.0 / 255.0)
        target = self.reference[y, x]
        current = self.baseline[y, x]

        if mode == "grayscale":
            value = (float(target) - (1.0 - alpha) * float(current)) / alpha
            return int(np.clip(round(value), 0, 255))

        values = (
            float(target[channel]) - (1.0 - alpha) * float(current[channel])
            for channel in range(3)
        )
        return tuple(int(np.clip(round(value / alpha), 0, 255)) for value in values)
