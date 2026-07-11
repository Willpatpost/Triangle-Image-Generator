from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.config import Config, ColorMode, ShapeMode
    from core.guidance import EvolutionGuide


MaskCache = tuple[int, int, int, int, int, int, np.ndarray, bool]
BBox = tuple[int, int, int, int]


def clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


@dataclass(slots=True)
class Triangle:
    points: np.ndarray  # shape (3, 2), int
    color: int | tuple[int, int, int]
    alpha: int
    _mask_cache: MaskCache | None = field(default=None, repr=False, compare=False)

    def copy(self) -> Triangle:
        return Triangle(
            points=self.points.copy(),
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def fork(self) -> Triangle:
        return Triangle(
            points=self.points,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        p0, p1, p2 = self.points
        min_x = max(0, min(int(p0[0]), int(p1[0]), int(p2[0])))
        max_x = min(width - 1, max(int(p0[0]), int(p1[0]), int(p2[0])))
        min_y = max(0, min(int(p0[1]), int(p1[1]), int(p2[1])))
        max_y = min(height - 1, max(int(p0[1]), int(p1[1]), int(p2[1])))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        self._mask_cache = None


@dataclass(slots=True)
class Circle:
    center: np.ndarray  # shape (2,), int
    radius: int
    color: int | tuple[int, int, int]
    alpha: int
    _mask_cache: MaskCache | None = field(default=None, repr=False, compare=False)

    def copy(self) -> Circle:
        return Circle(
            center=self.center.copy(),
            radius=self.radius,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def fork(self) -> Circle:
        return Circle(
            center=self.center,
            radius=self.radius,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y = self.center
        min_x = max(0, int(x - self.radius))
        max_x = min(width - 1, int(x + self.radius))
        min_y = max(0, int(y - self.radius))
        max_y = min(height - 1, int(y + self.radius))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        self._mask_cache = None


@dataclass(slots=True)
class Square:
    top_left: np.ndarray  # shape (2,), int
    side: int
    color: int | tuple[int, int, int]
    alpha: int
    _mask_cache: MaskCache | None = field(default=None, repr=False, compare=False)

    def copy(self) -> Square:
        return Square(
            top_left=self.top_left.copy(),
            side=self.side,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def fork(self) -> Square:
        return Square(
            top_left=self.top_left,
            side=self.side,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y = self.top_left
        min_x = max(0, int(x))
        max_x = min(width - 1, int(x + self.side))
        min_y = max(0, int(y))
        max_y = min(height - 1, int(y + self.side))
        return min_x, min_y, max_x, max_y

    def invalidate_caches(self) -> None:
        self._mask_cache = None


@dataclass(slots=True)
class VoronoiSite:
    point: np.ndarray  # shape (2,), int
    color: int | tuple[int, int, int]
    alpha: int = 255
    _mask_cache: MaskCache | None = field(default=None, repr=False, compare=False)

    def copy(self) -> VoronoiSite:
        return VoronoiSite(
            point=self.point.copy(),
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def fork(self) -> VoronoiSite:
        return VoronoiSite(
            point=self.point,
            color=self.color,
            alpha=self.alpha,
            _mask_cache=self._mask_cache,
        )

    def bounding_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        return 0, 0, width - 1, height - 1

    def invalidate_caches(self) -> None:
        self._mask_cache = None


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


def _choose_anchor(
    width: int,
    height: int,
    config: Config,
    guide: EvolutionGuide | None,
) -> tuple[int, int, bool]:
    guided = (
        guide is not None
        and config.target_guided_shapes
        and random.random() < config.target_guided_probability
    )
    if guided:
        x, y = guide.sample_high_error_point(config.target_guidance_candidates)
        return x, y, True
    return random.randint(0, width - 1), random.randint(0, height - 1), False


def _new_shape_color(
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None,
    x: int,
    y: int,
    alpha: int,
    guided: bool,
) -> int | tuple[int, int, int]:
    if guide is None or not guided:
        return _random_color(mode)
    alpha_factor = config.fixed_alpha if 0.0 <= config.fixed_alpha <= 1.0 else alpha / 255.0
    return guide.optimal_color(x, y, alpha_factor, mode)


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


def random_triangle(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
) -> Triangle:
    max_span = max(1, int(min(width, height) / config.new_shape_size_divisor))
    center_x, center_y, guided = _choose_anchor(width, height, config, guide)
    points = []
    for _ in range(3):
        points.append(
            [
                clamp_int(center_x + random.randint(-max_span, max_span), 0, width - 1),
                clamp_int(center_y + random.randint(-max_span, max_span), 0, height - 1),
            ]
        )
    points = np.array(points, dtype=np.int32)
    alpha = random.randint(config.alpha_min, config.alpha_max)
    color = _new_shape_color(mode, config, guide, center_x, center_y, alpha, guided)
    return Triangle(points=points, color=color, alpha=alpha)


def random_circle(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
) -> Circle:
    max_radius = max(1, int(min(width, height) / config.new_shape_size_divisor))
    center_x, center_y, guided = _choose_anchor(width, height, config, guide)
    center = np.array([center_x, center_y], dtype=np.int32)
    radius = random.randint(1, max_radius)
    alpha = random.randint(config.alpha_min, config.alpha_max)
    color = _new_shape_color(mode, config, guide, center_x, center_y, alpha, guided)
    return Circle(center=center, radius=radius, color=color, alpha=alpha)


def random_square(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
) -> Square:
    max_side = max(1, int(min(width, height) / config.new_shape_size_divisor))
    side = random.randint(1, max_side)
    max_x = max(0, width - side)
    max_y = max(0, height - side)
    center_x, center_y, guided = _choose_anchor(width, height, config, guide)
    top_left = np.array([
        clamp_int(center_x - side / 2, 0, max_x),
        clamp_int(center_y - side / 2, 0, max_y),
    ], dtype=np.int32)
    alpha = random.randint(config.alpha_min, config.alpha_max)
    color = _new_shape_color(mode, config, guide, center_x, center_y, alpha, guided)
    return Square(top_left=top_left, side=side, color=color, alpha=alpha)


def random_voronoi_site(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
) -> VoronoiSite:
    x, y, guided = _choose_anchor(width, height, config, guide)
    point = np.array([x, y], dtype=np.int32)
    color = _new_shape_color(mode, config, guide, x, y, 255, guided)
    return VoronoiSite(point=point, color=color)


def random_shape(
    width: int,
    height: int,
    mode: ColorMode,
    shape_mode: ShapeMode,
    config: Config,
    guide: EvolutionGuide | None = None,
) -> Shape:
    kind = shape_mode
    if kind == "mixed":
        kind = random.choice(("triangle", "circle", "square"))
    if kind == "voronoi":
        return random_voronoi_site(width, height, mode, config, guide)
    if kind == "circle":
        return random_circle(width, height, mode, config, guide)
    if kind == "square":
        return random_square(width, height, mode, config, guide)
    return random_triangle(width, height, mode, config, guide)


def mutate_triangle(tri: Triangle, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    if random.random() < config.prob_mutate_geometry:
        tri.points = tri.points.copy()
        point = random.randrange(3)
        axis = random.randrange(2)
        limit = width - 1 if axis == 0 else height - 1
        delta = random.gauss(0, 1) * config.mutation_sigma_points * rate
        tri.points[point, axis] = clamp_int(tri.points[point, axis] + delta, 0, limit)
        tri.invalidate_caches()
    else:
        _mutate_appearance(tri, mode, config, rate)


def mutate_circle(circle: Circle, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    if random.random() < config.prob_mutate_geometry:
        circle.center = circle.center.copy()
        slot = random.randrange(3)
        delta = random.gauss(0, 1) * config.mutation_sigma_points * rate
        if slot < 2:
            limit = width - 1 if slot == 0 else height - 1
            circle.center[slot] = clamp_int(circle.center[slot] + delta, 0, limit)
        else:
            circle.radius = clamp_int(circle.radius + delta, 1, max(1, min(width, height) // 2))
        circle.invalidate_caches()
    else:
        _mutate_appearance(circle, mode, config, rate)


def mutate_square(square: Square, width: int, height: int, mode: ColorMode, config: Config, rate: float) -> None:
    if random.random() < config.prob_mutate_geometry:
        square.top_left = square.top_left.copy()
        slot = random.randrange(3)
        delta = random.gauss(0, 1) * config.mutation_sigma_points * rate
        if slot == 0:
            square.top_left[0] = clamp_int(square.top_left[0] + delta, 0, max(0, width - square.side))
        elif slot == 1:
            square.top_left[1] = clamp_int(square.top_left[1] + delta, 0, max(0, height - square.side))
        else:
            max_side = max(1, min(width - int(square.top_left[0]), height - int(square.top_left[1])))
            square.side = clamp_int(square.side + delta, 1, max_side)
        square.invalidate_caches()
    else:
        _mutate_appearance(square, mode, config, rate)


def mutate_voronoi_site(
    site: VoronoiSite,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
) -> None:
    if random.random() < 0.2:
        site.point = site.point.copy()
        axis = random.randrange(2)
        limit = width - 1 if axis == 0 else height - 1
        delta = random.gauss(0, 1) * config.mutation_sigma_points * rate
        site.point[axis] = clamp_int(site.point[axis] + delta, 0, limit)
        site.invalidate_caches()
    else:
        site.color = _mutate_color(site.color, mode, config, rate * 1.5)


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


def blend_voronoi_sites(
    a: VoronoiSite,
    b: VoronoiSite,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
) -> VoronoiSite:
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


@dataclass(slots=True)
class Individual:
    width: int
    height: int
    mode: ColorMode
    triangles: list[Shape] = field(default_factory=list)
    fitness: float = float("inf")
    _compositing_cache: list[np.ndarray | None] = field(default_factory=list, repr=False)
    _compositing_cache_signature: tuple[object, ...] | None = field(default=None, repr=False)
    _pixel_error_sum: float | None = field(default=None, repr=False)
    _dirty_from: int | None = field(default=None, repr=False)
    _dirty_bbox: BBox | None = field(default=None, repr=False)
    _base_render: np.ndarray | None = field(default=None, repr=False)
    _base_error_sum: float | None = field(default=None, repr=False)

    def copy(self) -> Individual:
        return Individual(
            width=self.width,
            height=self.height,
            mode=self.mode,
            triangles=[t.copy() for t in self.triangles],
            fitness=self.fitness,
            _compositing_cache=list(self._compositing_cache),
            _compositing_cache_signature=self._compositing_cache_signature,
            _pixel_error_sum=self._pixel_error_sum,
            _dirty_from=self._dirty_from,
            _dirty_bbox=self._dirty_bbox,
            _base_render=self._base_render,
            _base_error_sum=self._base_error_sum,
        )

    def fork(self) -> Individual:
        """Create an offspring using copy-on-write geometry arrays."""
        return Individual(
            width=self.width,
            height=self.height,
            mode=self.mode,
            triangles=[shape.fork() for shape in self.triangles],
            fitness=self.fitness,
            _compositing_cache=list(self._compositing_cache),
            _compositing_cache_signature=self._compositing_cache_signature,
            _pixel_error_sum=self._pixel_error_sum,
        )

    def _cached_final_render(self) -> np.ndarray | None:
        if len(self._compositing_cache) != len(self.triangles) or not self._compositing_cache:
            return None
        return self._compositing_cache[-1]

    def _mark_dirty(self, index: int, bbox: BBox) -> None:
        if self._dirty_from is None:
            self._base_render = self._cached_final_render()
            self._base_error_sum = self._pixel_error_sum
            self._dirty_from = index
            self._dirty_bbox = bbox
        else:
            self._dirty_from = min(self._dirty_from, index)
            assert self._dirty_bbox is not None
            self._dirty_bbox = (
                min(self._dirty_bbox[0], bbox[0]),
                min(self._dirty_bbox[1], bbox[1]),
                max(self._dirty_bbox[2], bbox[2]),
                max(self._dirty_bbox[3], bbox[3]),
            )
        self._pixel_error_sum = None

    def _clear_dirty_tracking(self) -> None:
        self._dirty_from = None
        self._dirty_bbox = None
        self._base_render = None
        self._base_error_sum = None

    def invalidate_cache_from(self, index: int) -> None:
        self._pixel_error_sum = None
        if not self._compositing_cache:
            return
        for i in range(index, len(self._compositing_cache)):
            self._compositing_cache[i] = None

    def clear_caches(self, *, include_shape_masks: bool = True) -> None:
        self._compositing_cache = []
        self._compositing_cache_signature = None
        self._pixel_error_sum = None
        self._clear_dirty_tracking()
        if include_shape_masks:
            for shape in self.triangles:
                shape.invalidate_caches()

    def ensure_min_triangles(self, config: Config, guide: EvolutionGuide | None = None) -> bool:
        added = False
        while len(self.triangles) < config.min_triangles:
            shape = random_shape(
                self.width,
                self.height,
                self.mode,
                config.shape_mode,
                config,
                guide,
            )
            self._mark_dirty(len(self.triangles), shape.bounding_box(self.width, self.height))
            self.triangles.append(shape)
            added = True
        if added:
            self.fitness = float("inf")
        return added

    def add_random_triangle(self, config: Config, guide: EvolutionGuide | None = None) -> bool:
        if len(self.triangles) >= config.nb_elements_max:
            return False
        shape = random_shape(
            self.width,
            self.height,
            self.mode,
            config.shape_mode,
            config,
            guide,
        )
        self._mark_dirty(len(self.triangles), shape.bounding_box(self.width, self.height))
        self.triangles.append(shape)
        self.fitness = float("inf")
        return True

    def remove_random_triangle(self, config: Config) -> bool:
        if len(self.triangles) <= config.min_triangles:
            return False
        idx = random.randint(0, len(self.triangles) - 1)
        self._mark_dirty(idx, self.triangles[idx].bounding_box(self.width, self.height))
        del self.triangles[idx]
        self.invalidate_cache_from(idx)
        self.fitness = float("inf")
        return True

    def _mutate_random_shape(self, config: Config, rate: float) -> bool:
        if not self.triangles:
            return False
        idx = random.randint(0, len(self.triangles) - 1)
        old_bbox = self.triangles[idx].bounding_box(self.width, self.height)
        self._mark_dirty(idx, old_bbox)
        mutate_shape(
            self.triangles[idx],
            self.width,
            self.height,
            self.mode,
            config,
            rate,
        )
        self._mark_dirty(idx, self.triangles[idx].bounding_box(self.width, self.height))
        self.invalidate_cache_from(idx)
        self.fitness = float("inf")
        return True

    def mutate(
        self,
        config: Config,
        rate: float | None = None,
        guide: EvolutionGuide | None = None,
    ) -> None:
        effective_rate = config.mutation_rate if rate is None else rate
        self.ensure_min_triangles(config, guide)
        operations = max(1, int(round(config.mutation_operations * effective_rate)))

        for _ in range(operations):
            roll = random.random()
            if roll < config.prob_structural:
                if random.random() < config.prob_add_vs_del:
                    changed = self.add_random_triangle(config, guide)
                else:
                    changed = self.remove_random_triangle(config)
                if not changed:
                    self._mutate_random_shape(config, effective_rate)
            elif roll < config.prob_structural + config.prob_reorder and len(self.triangles) >= 2:
                i, j = random.sample(range(len(self.triangles)), 2)
                self._mark_dirty(i, self.triangles[i].bounding_box(self.width, self.height))
                self._mark_dirty(j, self.triangles[j].bounding_box(self.width, self.height))
                self.triangles[i], self.triangles[j] = self.triangles[j], self.triangles[i]
                self.invalidate_cache_from(min(i, j))
                self.fitness = float("inf")
            elif self.triangles:
                self._mutate_random_shape(config, effective_rate)

    @classmethod
    def random(
        cls,
        width: int,
        height: int,
        mode: ColorMode,
        config: Config,
        count: int,
        guide: EvolutionGuide | None = None,
    ) -> Individual:
        ind = cls(width=width, height=height, mode=mode)
        for _ in range(count):
            ind.triangles.append(random_shape(width, height, mode, config.shape_mode, config, guide))
        return ind

    @classmethod
    def crossover_uniform(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
        guide: EvolutionGuide | None = None,
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
                child.triangles.append(
                    random_shape(
                        parent_a.width,
                        parent_a.height,
                        parent_a.mode,
                        config.shape_mode,
                        config,
                        guide,
                    )
                )
        child.ensure_min_triangles(config, guide)
        return child

    @classmethod
    def crossover_blend(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
        guide: EvolutionGuide | None = None,
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
        child.ensure_min_triangles(config, guide)
        return child


def _scale_coordinate(value: int, old_size: int, new_size: int) -> int:
    if old_size <= 1 or new_size <= 1:
        return 0
    return clamp_int(value * (new_size - 1) / (old_size - 1), 0, new_size - 1)


def scale_individual(individual: Individual, width: int, height: int) -> Individual:
    """Scale a shape genome to a new raster size for coarse-to-fine evolution."""
    if width < 1 or height < 1:
        raise ValueError("scaled dimensions must be positive")

    radius_scale = min(width / individual.width, height / individual.height)
    shapes: list[Shape] = []
    for shape in individual.triangles:
        if isinstance(shape, Triangle):
            points = np.empty_like(shape.points)
            for index, point in enumerate(shape.points):
                points[index, 0] = _scale_coordinate(int(point[0]), individual.width, width)
                points[index, 1] = _scale_coordinate(int(point[1]), individual.height, height)
            shapes.append(Triangle(points=points, color=shape.color, alpha=shape.alpha))
        elif isinstance(shape, Circle):
            center = np.array(
                [
                    _scale_coordinate(int(shape.center[0]), individual.width, width),
                    _scale_coordinate(int(shape.center[1]), individual.height, height),
                ],
                dtype=np.int32,
            )
            radius = clamp_int(shape.radius * radius_scale, 1, max(1, min(width, height) // 2))
            shapes.append(Circle(center=center, radius=radius, color=shape.color, alpha=shape.alpha))
        elif isinstance(shape, Square):
            top_left = np.array(
                [
                    _scale_coordinate(int(shape.top_left[0]), individual.width, width),
                    _scale_coordinate(int(shape.top_left[1]), individual.height, height),
                ],
                dtype=np.int32,
            )
            max_side = max(1, min(width - int(top_left[0]), height - int(top_left[1])))
            side = clamp_int(shape.side * radius_scale, 1, max_side)
            shapes.append(Square(top_left=top_left, side=side, color=shape.color, alpha=shape.alpha))
        else:
            point = np.array(
                [
                    _scale_coordinate(int(shape.point[0]), individual.width, width),
                    _scale_coordinate(int(shape.point[1]), individual.height, height),
                ],
                dtype=np.int32,
            )
            shapes.append(VoronoiSite(point=point, color=shape.color, alpha=shape.alpha))

    return Individual(
        width=width,
        height=height,
        mode=individual.mode,
        triangles=shapes,
        fitness=float("inf"),
    )
