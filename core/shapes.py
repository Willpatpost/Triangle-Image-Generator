from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.config import Config, ColorMode, ShapeMode


def clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


@dataclass
class Triangle:
    points: np.ndarray  # shape (3, 2), int
    color: int | tuple[int, int, int]
    alpha: int

    def copy(self) -> Triangle:
        return Triangle(
            points=self.points.copy(),
            color=copy.deepcopy(self.color),
            alpha=self.alpha,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        xs = self.points[:, 0]
        ys = self.points[:, 1]
        min_x = max(0, int(xs.min()))
        max_x = min(width - 1, int(xs.max()))
        min_y = max(0, int(ys.min()))
        max_y = min(height - 1, int(ys.max()))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        pass  # kept for API symmetry with compositing layer


@dataclass
class Circle:
    center: np.ndarray  # shape (2,), int
    radius: int
    color: int | tuple[int, int, int]
    alpha: int

    def copy(self) -> Circle:
        return Circle(
            center=self.center.copy(),
            radius=self.radius,
            color=copy.deepcopy(self.color),
            alpha=self.alpha,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y = self.center
        min_x = max(0, int(x - self.radius))
        max_x = min(width - 1, int(x + self.radius))
        min_y = max(0, int(y - self.radius))
        max_y = min(height - 1, int(y + self.radius))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        pass


@dataclass
class Square:
    top_left: np.ndarray  # shape (2,), int
    side: int
    color: int | tuple[int, int, int]
    alpha: int

    def copy(self) -> Square:
        return Square(
            top_left=self.top_left.copy(),
            side=self.side,
            color=copy.deepcopy(self.color),
            alpha=self.alpha,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y = self.top_left
        min_x = max(0, int(x))
        max_x = min(width - 1, int(x + self.side))
        min_y = max(0, int(y))
        max_y = min(height - 1, int(y + self.side))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        pass


@dataclass
class VoronoiSite:
    point: np.ndarray  # shape (2,), int
    color: int | tuple[int, int, int]
    alpha: int = 255

    def copy(self) -> VoronoiSite:
        return VoronoiSite(
            point=self.point.copy(),
            color=copy.deepcopy(self.color),
            alpha=self.alpha,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        return 0, 0, width - 1, height - 1

    def invalidate_caches(self) -> None:
        pass


Shape = Triangle | Circle | Square | VoronoiSite


def shape_kind(shape: Shape) -> str:
    if isinstance(shape, Triangle):
        return "triangle"
    if isinstance(shape, Circle):
        return "circle"
    if isinstance(shape, VoronoiSite):
        return "voronoi"
    return "square"


def _random_color(mode: ColorMode) -> int | tuple[int, int, int]:
    if mode == "grayscale":
        return random.randint(0, 255)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def _mutate_color(
    color: int | tuple[int, int, int],
    mode: ColorMode,
    config: Config,
    scale: float,
    channel: int | None = None,
) -> int | tuple[int, int, int]:
    if mode == "grayscale":
        assert isinstance(color, int)
        return clamp_int(
            color + random.gauss(0, 1) * config.mutation_sigma_color * scale,
            0,
            255,
        )

    assert isinstance(color, tuple)
    channels = list(color)
    if config.mutate_all_color_channels:
        return tuple(
            clamp_int(c + random.gauss(0, 1) * config.mutation_sigma_color * scale, 0, 255)
            for c in channels
        )

    if channel is None:
        channel = random.randint(0, 2)
    channels[channel] = clamp_int(
        channels[channel] + random.gauss(0, 1) * config.mutation_sigma_color * scale,
        0,
        255,
    )
    return tuple(channels)


def _mutate_appearance(shape: Shape, mode: ColorMode, config: Config, scale: float) -> None:
    if mode == "grayscale":
        if random.random() < 0.5:
            shape.color = _mutate_color(shape.color, mode, config, scale)
        else:
            shape.alpha = clamp_int(
                shape.alpha + random.gauss(0, 1) * config.mutation_sigma_alpha * scale,
                config.alpha_min,
                config.alpha_max,
            )
        return

    slot = random.randint(0, 3)
    if slot == 3:
        shape.alpha = clamp_int(
            shape.alpha + random.gauss(0, 1) * config.mutation_sigma_alpha * scale,
            config.alpha_min,
            config.alpha_max,
        )
    else:
        shape.color = _mutate_color(shape.color, mode, config, scale, channel=slot)


def random_triangle(width: int, height: int, mode: ColorMode, config: Config) -> Triangle:
    points = np.array(
        [
            [random.randint(0, width - 1), random.randint(0, height - 1)]
            for _ in range(3)
        ],
        dtype=np.int32,
    )
    color = _random_color(mode)
    alpha = random.randint(config.alpha_min, config.alpha_max)
    return Triangle(points=points, color=color, alpha=alpha)


def random_circle(width: int, height: int, mode: ColorMode, config: Config) -> Circle:
    max_radius = max(1, min(width, height) // 3)
    center = np.array(
        [random.randint(0, width - 1), random.randint(0, height - 1)],
        dtype=np.int32,
    )
    radius = random.randint(1, max_radius)
    alpha = random.randint(config.alpha_min, config.alpha_max)
    return Circle(center=center, radius=radius, color=_random_color(mode), alpha=alpha)


def random_square(width: int, height: int, mode: ColorMode, config: Config) -> Square:
    max_side = max(1, min(width, height) // 2)
    side = random.randint(1, max_side)
    max_x = max(0, width - side)
    max_y = max(0, height - side)
    top_left = np.array(
        [random.randint(0, max_x), random.randint(0, max_y)],
        dtype=np.int32,
    )
    alpha = random.randint(config.alpha_min, config.alpha_max)
    return Square(top_left=top_left, side=side, color=_random_color(mode), alpha=alpha)


def random_voronoi_site(width: int, height: int, mode: ColorMode, config: Config) -> VoronoiSite:
    point = np.array(
        [random.randint(0, width - 1), random.randint(0, height - 1)],
        dtype=np.int32,
    )
    return VoronoiSite(point=point, color=_random_color(mode))


def random_shape(width: int, height: int, mode: ColorMode, shape_mode: ShapeMode, config: Config) -> Shape:
    kind = shape_mode
    if kind == "mixed":
        kind = random.choice(("triangle", "circle", "square"))
    if kind == "voronoi":
        return random_voronoi_site(width, height, mode, config)
    if kind == "circle":
        return random_circle(width, height, mode, config)
    if kind == "square":
        return random_square(width, height, mode, config)
    return random_triangle(width, height, mode, config)


def mutate_triangle(tri: Triangle, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    scale = config.mutation_rate * rate
    for i in range(3):
        for j in range(2):
            limit = width - 1 if j == 0 else height - 1
            delta = random.gauss(0, 1) * config.mutation_sigma_points * scale
            tri.points[i, j] = clamp_int(tri.points[i, j] + delta, 0, limit)

    _mutate_appearance(tri, mode, config, scale)


def mutate_circle(circle: Circle, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    scale = config.mutation_rate * rate
    circle.center[0] = clamp_int(
        circle.center[0] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        0,
        width - 1,
    )
    circle.center[1] = clamp_int(
        circle.center[1] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        0,
        height - 1,
    )
    circle.radius = clamp_int(
        circle.radius + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        1,
        max(1, min(width, height) // 2),
    )
    _mutate_appearance(circle, mode, config, scale)


def mutate_square(square: Square, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    scale = config.mutation_rate * rate
    square.top_left[0] = clamp_int(
        square.top_left[0] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        0,
        width - 1,
    )
    square.top_left[1] = clamp_int(
        square.top_left[1] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        0,
        height - 1,
    )
    square.side = clamp_int(
        square.side + random.gauss(0, 1) * config.mutation_sigma_points * scale,
        1,
        max(1, min(width - int(square.top_left[0]), height - int(square.top_left[1]))),
    )
    _mutate_appearance(square, mode, config, scale)


def mutate_voronoi_site(site: VoronoiSite, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    scale = config.mutation_rate * rate
    if random.random() < 0.2:
        site.point[0] = clamp_int(
            site.point[0] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
            0,
            width - 1,
        )
        site.point[1] = clamp_int(
            site.point[1] + random.gauss(0, 1) * config.mutation_sigma_points * scale,
            0,
            height - 1,
        )
    else:
        site.color = _mutate_color(site.color, mode, config, scale * 1.5)


def mutate_shape(shape: Shape, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    if isinstance(shape, Triangle):
        mutate_triangle(shape, width, height, mode, config, rate)
    elif isinstance(shape, Circle):
        mutate_circle(shape, width, height, mode, config, rate)
    elif isinstance(shape, VoronoiSite):
        mutate_voronoi_site(shape, width, height, mode, config, rate)
    else:
        mutate_square(shape, width, height, mode, config, rate)


def blend_triangles(a: Triangle, b: Triangle, width: int, height: int, mode: ColorMode, config: Config) -> Triangle:
    t = random.uniform(0.25, 0.75)
    points = np.round(a.points * t + b.points * (1 - t)).astype(np.int32)
    for i in range(3):
        points[i, 0] = clamp_int(points[i, 0], 0, width - 1)
        points[i, 1] = clamp_int(points[i, 1], 0, height - 1)

    if mode == "grayscale":
        assert isinstance(a.color, int) and isinstance(b.color, int)
        color = clamp_int(a.color * t + b.color * (1 - t), 0, 255)
    else:
        assert isinstance(a.color, tuple) and isinstance(b.color, tuple)
        color = tuple(
            clamp_int(ac * t + bc * (1 - t), 0, 255)
            for ac, bc in zip(a.color, b.color)
        )

    alpha = clamp_int(a.alpha * t + b.alpha * (1 - t), config.alpha_min, config.alpha_max)
    return Triangle(points=points, color=color, alpha=alpha)


def blend_circles(a: Circle, b: Circle, width: int, height: int, mode: ColorMode, config: Config) -> Circle:
    t = random.uniform(0.25, 0.75)
    center = np.round(a.center * t + b.center * (1 - t)).astype(np.int32)
    center[0] = clamp_int(center[0], 0, width - 1)
    center[1] = clamp_int(center[1], 0, height - 1)
    radius = clamp_int(a.radius * t + b.radius * (1 - t), 1, max(1, min(width, height) // 2))
    color = _blend_color(a.color, b.color, mode, t)
    alpha = clamp_int(a.alpha * t + b.alpha * (1 - t), config.alpha_min, config.alpha_max)
    return Circle(center=center, radius=radius, color=color, alpha=alpha)


def blend_squares(a: Square, b: Square, width: int, height: int, mode: ColorMode, config: Config) -> Square:
    t = random.uniform(0.25, 0.75)
    top_left = np.round(a.top_left * t + b.top_left * (1 - t)).astype(np.int32)
    top_left[0] = clamp_int(top_left[0], 0, width - 1)
    top_left[1] = clamp_int(top_left[1], 0, height - 1)
    max_side = max(1, min(width - int(top_left[0]), height - int(top_left[1])))
    side = clamp_int(a.side * t + b.side * (1 - t), 1, max_side)
    color = _blend_color(a.color, b.color, mode, t)
    alpha = clamp_int(a.alpha * t + b.alpha * (1 - t), config.alpha_min, config.alpha_max)
    return Square(top_left=top_left, side=side, color=color, alpha=alpha)


def blend_voronoi_sites(a: VoronoiSite, b: VoronoiSite, width: int, height: int, mode: ColorMode, config: Config) -> VoronoiSite:
    t = random.uniform(0.25, 0.75)
    point = np.round(a.point * t + b.point * (1 - t)).astype(np.int32)
    point[0] = clamp_int(point[0], 0, width - 1)
    point[1] = clamp_int(point[1], 0, height - 1)
    color = _blend_color(a.color, b.color, mode, t)
    return VoronoiSite(point=point, color=color)


def _blend_color(
    a: int | tuple[int, int, int],
    b: int | tuple[int, int, int],
    mode: ColorMode,
    t: float,
) -> int | tuple[int, int, int]:
    if mode == "grayscale":
        assert isinstance(a, int) and isinstance(b, int)
        return clamp_int(a * t + b * (1 - t), 0, 255)
    assert isinstance(a, tuple) and isinstance(b, tuple)
    return tuple(clamp_int(ac * t + bc * (1 - t), 0, 255) for ac, bc in zip(a, b))


def blend_shapes(a: Shape, b: Shape, width: int, height: int, mode: ColorMode, config: Config) -> Shape:
    if isinstance(a, Triangle) and isinstance(b, Triangle):
        return blend_triangles(a, b, width, height, mode, config)
    if isinstance(a, Circle) and isinstance(b, Circle):
        return blend_circles(a, b, width, height, mode, config)
    if isinstance(a, Square) and isinstance(b, Square):
        return blend_squares(a, b, width, height, mode, config)
    if isinstance(a, VoronoiSite) and isinstance(b, VoronoiSite):
        return blend_voronoi_sites(a, b, width, height, mode, config)
    return a.copy() if random.random() < 0.5 else b.copy()


@dataclass
class Individual:
    width: int
    height: int
    mode: ColorMode
    triangles: list[Shape] = field(default_factory=list)
    fitness: float = float("inf")
    _compositing_cache: list[np.ndarray | None] = field(default_factory=list, repr=False)

    def copy(self) -> Individual:
        return Individual(
            width=self.width,
            height=self.height,
            mode=self.mode,
            triangles=[t.copy() for t in self.triangles],
            fitness=self.fitness,
        )

    def invalidate_cache_from(self, index: int) -> None:
        if not self._compositing_cache:
            return
        for i in range(index, len(self._compositing_cache)):
            self._compositing_cache[i] = None

    def ensure_min_triangles(self, config: Config) -> None:
        while len(self.triangles) < config.min_triangles:
            self.triangles.append(random_shape(self.width, self.height, self.mode, config.shape_mode, config))
        self._compositing_cache = []

    def add_random_triangle(self, config: Config) -> None:
        if len(self.triangles) >= config.nb_elements_max:
            return
        self.triangles.append(random_shape(self.width, self.height, self.mode, config.shape_mode, config))
        self._compositing_cache = []
        self.fitness = float("inf")

    def remove_random_triangle(self, config: Config) -> None:
        if len(self.triangles) <= config.min_triangles:
            return
        idx = random.randint(0, len(self.triangles) - 1)
        del self.triangles[idx]
        self._compositing_cache = []
        self.fitness = float("inf")

    def mutate(self, config: Config, rate: float = 1.0) -> None:
        self.ensure_min_triangles(config)
        operations = max(1, int(round(2 * rate)))

        for _ in range(operations):
            roll = random.random()
            if roll < config.prob_structural:
                if random.random() < config.prob_add_vs_del:
                    self.add_random_triangle(config)
                else:
                    self.remove_random_triangle(config)
            elif roll < config.prob_structural + config.prob_reorder and len(self.triangles) >= 2:
                i, j = random.sample(range(len(self.triangles)), 2)
                self.triangles[i], self.triangles[j] = self.triangles[j], self.triangles[i]
                self.invalidate_cache_from(min(i, j))
                self.fitness = float("inf")
            elif self.triangles:
                idx = random.randint(0, len(self.triangles) - 1)
                mutate_shape(self.triangles[idx], self.width, self.height, self.mode, config, rate)
                self.invalidate_cache_from(idx)
                self.fitness = float("inf")

    @classmethod
    def random(cls, width: int, height: int, mode: ColorMode, config: Config, count: int) -> Individual:
        ind = cls(width=width, height=height, mode=mode)
        for _ in range(count):
            ind.triangles.append(random_shape(width, height, mode, config.shape_mode, config))
        return ind

    @classmethod
    def crossover_uniform(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
    ) -> Individual:
        child_count = random.randint(
            min(len(parent_a.triangles), len(parent_b.triangles)),
            min(config.nb_elements_max, max(len(parent_a.triangles), len(parent_b.triangles))),
        )
        child_count = max(config.min_triangles, child_count)
        child = cls(width=parent_a.width, height=parent_a.height, mode=parent_a.mode)
        for i in range(child_count):
            if i < len(parent_a.triangles) and i < len(parent_b.triangles):
                source = parent_a if random.random() < 0.5 else parent_b
                child.triangles.append(source.triangles[i].copy())
            elif i < len(parent_a.triangles):
                child.triangles.append(parent_a.triangles[i].copy())
            elif i < len(parent_b.triangles):
                child.triangles.append(parent_b.triangles[i].copy())
            else:
                child.triangles.append(random_shape(parent_a.width, parent_a.height, parent_a.mode, config.shape_mode, config))
        child.ensure_min_triangles(config)
        return child

    @classmethod
    def crossover_blend(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
    ) -> Individual:
        count = random.randint(
            min(len(parent_a.triangles), len(parent_b.triangles)),
            min(config.nb_elements_max, max(len(parent_a.triangles), len(parent_b.triangles))),
        )
        count = max(config.min_triangles, count)
        child = cls(width=parent_a.width, height=parent_a.height, mode=parent_a.mode)
        for i in range(count):
            if i < len(parent_a.triangles) and i < len(parent_b.triangles):
                child.triangles.append(
                    blend_shapes(
                        parent_a.triangles[i],
                        parent_b.triangles[i],
                        parent_a.width,
                        parent_a.height,
                        parent_a.mode,
                        config,
                    )
                )
            elif i < len(parent_a.triangles):
                child.triangles.append(parent_a.triangles[i].copy())
            else:
                child.triangles.append(parent_b.triangles[i].copy())
        child.ensure_min_triangles(config)
        return child
