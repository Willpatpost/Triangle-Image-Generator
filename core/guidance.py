from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from core.config import ColorMode


@dataclass(frozen=True, slots=True)
class EvolutionGuide:
    """Read-only target and baseline used to place useful new shapes."""

    reference: np.ndarray
    baseline: np.ndarray

    def __post_init__(self) -> None:
        if self.reference.shape != self.baseline.shape:
            raise ValueError("reference and baseline must have matching shapes")

    @classmethod
    def from_background(
        cls,
        reference: np.ndarray,
        background: int | tuple[int, int, int],
    ) -> EvolutionGuide:
        value = np.asarray(background, dtype=np.float32)
        baseline = np.broadcast_to(value, reference.shape)
        return cls(reference=reference, baseline=baseline)

    def sample_high_error_point(self, candidates: int) -> tuple[int, int]:
        height, width = self.reference.shape[:2]
        best_x = random.randrange(width)
        best_y = random.randrange(height)
        best_error = -1.0

        if self.reference.ndim == 2:
            for _ in range(candidates):
                x = random.randrange(width)
                y = random.randrange(height)
                delta = float(self.reference[y, x]) - float(self.baseline[y, x])
                error = delta * delta
                if error > best_error:
                    best_x, best_y, best_error = x, y, error
        else:
            for _ in range(candidates):
                x = random.randrange(width)
                y = random.randrange(height)
                target = self.reference[y, x]
                current = self.baseline[y, x]
                d0 = float(target[0]) - float(current[0])
                d1 = float(target[1]) - float(current[1])
                d2 = float(target[2]) - float(current[2])
                error = d0 * d0 + d1 * d1 + d2 * d2
                if error > best_error:
                    best_x, best_y, best_error = x, y, error

        return best_x, best_y

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
