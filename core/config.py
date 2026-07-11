from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColorMode = Literal["grayscale", "color"]
ShapeMode = Literal["triangle", "circle", "square", "voronoi", "mixed"]
RendererBackend = Literal["auto", "numpy", "numba", "cuda"]

RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "enable_logging",
        "log_interval_sec",
        "log_level",
        "max_workers",
        "parallel_min_pixels",
        "renderer_backend",
        "use_dirty_regions",
        "numba_min_pixels",
        "cuda_min_pixels",
        "use_compositing_cache",
        "compositing_cache_stride",
        "compositing_cache_max_mb",
    }
)


@dataclass
class Config:
    mode: ColorMode = "color"
    shape_mode: ShapeMode = "triangle"

    # Population
    pop_size: int = 30
    nb_elite: int = 6
    nb_elements_initial: int = 15
    nb_elements_max: int = 150
    min_shapes: int = 5

    # Mutation
    mutation_rate: float = 1.0
    mutation_operations: int = 1
    prob_structural: float = 0.15
    prob_add_vs_del: float = 0.92
    prob_reorder: float = 0.03
    prob_mutate_geometry: float = 0.65
    mutation_sigma_points: float = 10.0
    mutation_sigma_color: float = 12.0
    mutation_sigma_alpha: float = 10.0
    mutate_all_color_channels: bool = False

    # Crossover
    crossover_rate: float = 0.0
    prob_uniform_crossover: float = 0.5
    prob_blend_crossover: float = 0.5

    # Target guidance
    target_guided_shapes: bool = True
    target_guided_probability: float = 0.9
    target_guidance_candidates: int = 24

    # Rendering
    fixed_alpha: float = -1.0  # <0 uses each shape's alpha
    alpha_min: int = 10
    alpha_max: int = 255
    gaussian_blur_sigma: float = 0.0
    renderer_backend: RendererBackend = "auto"
    use_dirty_regions: bool = True
    numba_min_pixels: int = 4_096
    cuda_min_pixels: int = 262_144
    use_compositing_cache: bool = True
    compositing_cache_stride: int = 8
    compositing_cache_max_mb: float = 32.0
    new_shape_size_divisor: float = 4.0

    # Fitness
    shape_penalty_weight: float = 0.0
    fitness_goal: float = 0.001

    # Evolution
    stagnation_threshold: int = 5000
    adaptive_mutation_boost: float = 1.5
    hill_climb_interval: int = 100
    hill_climb_attempts: int = 3

    # Parallelism
    max_workers: int = 0  # 0 = auto, capped at four cache-preserving threads
    parallel_min_pixels: int = 500_000  # counts channel values, not just width * height

    # Logging
    enable_logging: bool = True
    log_interval_sec: float = 10.0
    log_level: str = "INFO"

    # Runtime (set during execution)
    background: int | tuple[int, int, int] = field(default=255, repr=False)

    def validate(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.nb_elite < 1:
            raise ValueError("nb_elite must be at least 1")
        if self.nb_elite >= self.pop_size:
            raise ValueError("nb_elite must be smaller than pop_size")
        if self.nb_elements_initial < 1:
            raise ValueError("nb_elements_initial must be at least 1")
        if self.nb_elements_max < self.nb_elements_initial:
            raise ValueError("nb_elements_max must be greater than or equal to nb_elements_initial")
        if self.min_shapes < 1:
            raise ValueError("min_shapes must be at least 1")
        if self.nb_elements_max < self.min_shapes:
            raise ValueError("nb_elements_max must be greater than or equal to min_shapes")
        if self.max_workers < 0:
            raise ValueError("max_workers must be 0 or greater")
        if self.parallel_min_pixels < 1:
            raise ValueError("parallel_min_pixels must be at least 1")
        if self.mutation_rate <= 0:
            raise ValueError("mutation_rate must be greater than 0")
        if self.mutation_operations < 1:
            raise ValueError("mutation_operations must be at least 1")
        probabilities = {
            "prob_structural": self.prob_structural,
            "prob_add_vs_del": self.prob_add_vs_del,
            "prob_reorder": self.prob_reorder,
            "prob_mutate_geometry": self.prob_mutate_geometry,
            "crossover_rate": self.crossover_rate,
            "target_guided_probability": self.target_guided_probability,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.prob_structural + self.prob_reorder > 1.0:
            raise ValueError("prob_structural + prob_reorder must not exceed 1.0")
        if self.prob_uniform_crossover < 0 or self.prob_blend_crossover < 0:
            raise ValueError("crossover weights must be non-negative")
        if self.crossover_rate > 0 and self.prob_uniform_crossover + self.prob_blend_crossover <= 0:
            raise ValueError("at least one crossover weight must be positive")
        if self.target_guidance_candidates < 1:
            raise ValueError("target_guidance_candidates must be at least 1")
        if min(self.mutation_sigma_points, self.mutation_sigma_color, self.mutation_sigma_alpha) < 0:
            raise ValueError("mutation sigmas must be non-negative")
        if not 0 <= self.alpha_min <= 255:
            raise ValueError("alpha_min must be between 0 and 255")
        if not 0 <= self.alpha_max <= 255:
            raise ValueError("alpha_max must be between 0 and 255")
        if self.alpha_min > self.alpha_max:
            raise ValueError("alpha_min must be less than or equal to alpha_max")
        if self.fixed_alpha > 1.0:
            raise ValueError("fixed_alpha must be negative or between 0.0 and 1.0")
        if self.gaussian_blur_sigma < 0:
            raise ValueError("gaussian_blur_sigma must be non-negative")
        if self.renderer_backend not in {"auto", "numpy", "numba", "cuda"}:
            raise ValueError("renderer_backend must be auto, numpy, numba, or cuda")
        if self.numba_min_pixels < 1 or self.cuda_min_pixels < 1:
            raise ValueError("accelerator pixel thresholds must be positive")
        if self.compositing_cache_stride < 1:
            raise ValueError("compositing_cache_stride must be at least 1")
        if self.compositing_cache_max_mb <= 0:
            raise ValueError("compositing_cache_max_mb must be greater than 0")
        if self.new_shape_size_divisor < 1.0:
            raise ValueError("new_shape_size_divisor must be at least 1.0")
        if self.shape_penalty_weight < 0:
            raise ValueError("shape_penalty_weight must be non-negative")
        if self.fitness_goal < 0:
            raise ValueError("fitness_goal must be non-negative")
        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be at least 1")
        if self.adaptive_mutation_boost < 1.0:
            raise ValueError("adaptive_mutation_boost must be at least 1.0")
        if self.hill_climb_interval < 0 or self.hill_climb_attempts < 0:
            raise ValueError("hill climb settings must be non-negative")

    def elite_fraction(self) -> float:
        return self.nb_elite / max(self.pop_size, 1)
