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
    _mask_cache: MaskCache | None = field(default=None, repr=False, compare=False)

    def copy(self) -> VoronoiSite:
        return VoronoiSite(
            point=self.point.copy(),
            color=self.color,
            _mask_cache=self._mask_cache,
        )

    def fork(self) -> VoronoiSite:
        return VoronoiSite(
            point=self.point,
            color=self.color,
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


def _random_color(mode: ColorMode, rng: random.Random) -> int | tuple[int, int, int]:
    if mode == "grayscale":
        return rng.randint(0, 255)
    return (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )


def _choose_anchor(
    width: int,
    height: int,
    config: Config,
    guide: EvolutionGuide | None,
    *,
    rng: random.Random,
) -> tuple[int, int, bool]:
    guided = (
        guide is not None
        and config.target_guided_shapes
        and rng.random() < config.target_guided_probability
    )
    if guided:
        x, y = guide.sample_high_error_point(config.target_guidance_candidates, rng=rng)
        return x, y, True
    return rng.randint(0, width - 1), rng.randint(0, height - 1), False


def _guided_alpha_and_color(
    config: Config,
    guide: EvolutionGuide,
    x: int,
    y: int,
    mode: ColorMode,
) -> tuple[int, int | tuple[int, int, int]]:
    """Pick the guided alpha/color pair with the lowest error on a small patch."""
    height, width = guide.reference.shape[:2]
    patch = 8
    x0 = max(0, x - patch // 2)
    y0 = max(0, y - patch // 2)
    x1 = min(width, x0 + patch)
    y1 = min(height, y0 + patch)
    target = np.asarray(guide.reference[y0:y1, x0:x1], dtype=np.float32)
    baseline = np.asarray(guide.baseline[y0:y1, x0:x1], dtype=np.float32)

    steps = 5
    span = max(config.alpha_max - config.alpha_min, 1)
    best_alpha = config.alpha_max
    best_color: int | tuple[int, int, int] = 0 if mode == "grayscale" else (0, 0, 0)
    best_error = float("inf")
    for step in range(steps):
        alpha = config.alpha_min + (span * step) // max(steps - 1, 1)
        factor = alpha / 255.0
        color = guide.optimal_color(x, y, factor, mode)
        color_array = np.asarray(color, dtype=np.float32)
        predicted = (1.0 - factor) * baseline + factor * color_array
        error = float(np.sum((target - predicted) ** 2, dtype=np.float64))
        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_color = color
    return best_alpha, best_color


def _choose_alpha_and_color(
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None,
    x: int,
    y: int,
    guided: bool,
    *,
    rng: random.Random,
) -> tuple[int, int | tuple[int, int, int]]:
    if guide is None or not guided:
        return rng.randint(config.alpha_min, config.alpha_max), _random_color(mode, rng)
    if config.guided_alpha_search:
        return _guided_alpha_and_color(config, guide, x, y, mode)
    alpha = rng.randint(config.alpha_min, config.alpha_max)
    alpha_factor = config.fixed_alpha if 0.0 <= config.fixed_alpha <= 1.0 else alpha / 255.0
    return alpha, guide.optimal_color(x, y, alpha_factor, mode)


def _mutate_color(
    color: int | tuple[int, int, int],
    mode: ColorMode,
    config: Config,
    scale: float,
    *,
    rng: random.Random,
    channel: int | None = None,
) -> int | tuple[int, int, int]:
    if mode == "grayscale":
        if not isinstance(color, int):
            raise TypeError(f"grayscale mode requires int colors, got {type(color).__name__}")
        return clamp_int(
            color + rng.gauss(0, 1) * config.mutation_sigma_color * scale,
            0,
            255,
        )

    if not isinstance(color, tuple):
        raise TypeError(f"color mode requires (r, g, b) tuple colors, got {type(color).__name__}")
    channels = list(color)
    if config.mutate_all_color_channels:
        return tuple(
            clamp_int(c + rng.gauss(0, 1) * config.mutation_sigma_color * scale, 0, 255)
            for c in channels
        )

    if channel is None:
        channel = rng.randint(0, 2)
    channels[channel] = clamp_int(
        channels[channel] + rng.gauss(0, 1) * config.mutation_sigma_color * scale,
        0,
        255,
    )
    return tuple(channels)


def _mutate_appearance(
    shape: Shape,
    mode: ColorMode,
    config: Config,
    scale: float,
    *,
    rng: random.Random,
) -> None:
    if mode == "grayscale":
        if rng.random() < 0.5:
            shape.color = _mutate_color(shape.color, mode, config, scale, rng=rng)
        else:
            shape.alpha = clamp_int(
                shape.alpha + rng.gauss(0, 1) * config.mutation_sigma_alpha * scale,
                config.alpha_min,
                config.alpha_max,
            )
        return

    slot = rng.randint(0, 3)
    if slot == 3:
        shape.alpha = clamp_int(
            shape.alpha + rng.gauss(0, 1) * config.mutation_sigma_alpha * scale,
            config.alpha_min,
            config.alpha_max,
        )
    else:
        shape.color = _mutate_color(shape.color, mode, config, scale, rng=rng, channel=slot)


def random_triangle(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> Triangle:
    max_span = max(1, int(min(width, height) / config.new_shape_size_divisor))
    center_x, center_y, guided = _choose_anchor(width, height, config, guide, rng=rng)
    points = []
    for _ in range(3):
        points.append(
            [
                clamp_int(center_x + rng.randint(-max_span, max_span), 0, width - 1),
                clamp_int(center_y + rng.randint(-max_span, max_span), 0, height - 1),
            ]
        )
    points = np.array(points, dtype=np.int32)
    alpha, color = _choose_alpha_and_color(mode, config, guide, center_x, center_y, guided, rng=rng)
    return Triangle(points=points, color=color, alpha=alpha)


def random_circle(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> Circle:
    max_radius = max(1, int(min(width, height) / config.new_shape_size_divisor))
    center_x, center_y, guided = _choose_anchor(width, height, config, guide, rng=rng)
    center = np.array([center_x, center_y], dtype=np.int32)
    radius = rng.randint(1, max_radius)
    alpha, color = _choose_alpha_and_color(mode, config, guide, center_x, center_y, guided, rng=rng)
    return Circle(center=center, radius=radius, color=color, alpha=alpha)


def random_square(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> Square:
    max_side = max(1, int(min(width, height) / config.new_shape_size_divisor))
    side = rng.randint(1, max_side)
    max_x = max(0, width - side)
    max_y = max(0, height - side)
    center_x, center_y, guided = _choose_anchor(width, height, config, guide, rng=rng)
    top_left = np.array([
        clamp_int(center_x - side / 2, 0, max_x),
        clamp_int(center_y - side / 2, 0, max_y),
    ], dtype=np.int32)
    alpha, color = _choose_alpha_and_color(mode, config, guide, center_x, center_y, guided, rng=rng)
    return Square(top_left=top_left, side=side, color=color, alpha=alpha)


def random_voronoi_site(
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> VoronoiSite:
    x, y, guided = _choose_anchor(width, height, config, guide, rng=rng)
    point = np.array([x, y], dtype=np.int32)
    if guided:
        alpha_factor = config.fixed_alpha if 0.0 <= config.fixed_alpha <= 1.0 else 1.0
        color = guide.optimal_color(x, y, alpha_factor, mode)
    else:
        color = _random_color(mode, rng)
    return VoronoiSite(point=point, color=color)


def random_shape(
    width: int,
    height: int,
    mode: ColorMode,
    shape_mode: ShapeMode,
    config: Config,
    guide: EvolutionGuide | None = None,
    *,
    rng: random.Random,
) -> Shape:
    kind = shape_mode
    if kind == "mixed":
        # Voronoi genomes render as one full-canvas partition, so they cannot
        # combine with layered shapes inside a single genome.
        kind = rng.choice(("triangle", "circle", "square"))
    if kind == "voronoi":
        return random_voronoi_site(width, height, mode, config, guide, rng=rng)
    if kind == "circle":
        return random_circle(width, height, mode, config, guide, rng=rng)
    if kind == "square":
        return random_square(width, height, mode, config, guide, rng=rng)
    return random_triangle(width, height, mode, config, guide, rng=rng)


def mutate_triangle(
    tri: Triangle,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
    *,
    rng: random.Random,
) -> None:
    if rng.random() < config.prob_mutate_geometry:
        tri.points = tri.points.copy()
        point = rng.randrange(3)
        axis = rng.randrange(2)
        limit = width - 1 if axis == 0 else height - 1
        delta = rng.gauss(0, 1) * config.mutation_sigma_points * rate
        tri.points[point, axis] = clamp_int(tri.points[point, axis] + delta, 0, limit)
        tri.invalidate_caches()
    else:
        _mutate_appearance(tri, mode, config, rate, rng=rng)


def mutate_circle(
    circle: Circle,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
    *,
    rng: random.Random,
) -> None:
    if rng.random() < config.prob_mutate_geometry:
        circle.center = circle.center.copy()
        slot = rng.randrange(3)
        delta = rng.gauss(0, 1) * config.mutation_sigma_points * rate
        if slot < 2:
            limit = width - 1 if slot == 0 else height - 1
            circle.center[slot] = clamp_int(circle.center[slot] + delta, 0, limit)
        else:
            circle.radius = clamp_int(circle.radius + delta, 1, max(1, min(width, height) // 2))
        circle.invalidate_caches()
    else:
        _mutate_appearance(circle, mode, config, rate, rng=rng)


def mutate_square(
    square: Square,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
    *,
    rng: random.Random,
) -> None:
    if rng.random() < config.prob_mutate_geometry:
        square.top_left = square.top_left.copy()
        slot = rng.randrange(3)
        delta = rng.gauss(0, 1) * config.mutation_sigma_points * rate
        if slot == 0:
            square.top_left[0] = clamp_int(square.top_left[0] + delta, 0, max(0, width - square.side))
        elif slot == 1:
            square.top_left[1] = clamp_int(square.top_left[1] + delta, 0, max(0, height - square.side))
        else:
            max_side = max(1, min(width - int(square.top_left[0]), height - int(square.top_left[1])))
            square.side = clamp_int(square.side + delta, 1, max_side)
        square.invalidate_caches()
    else:
        _mutate_appearance(square, mode, config, rate, rng=rng)


def _voronoi_cell_centroids(
    sites: list[VoronoiSite],
    width: int,
    height: int,
    stride: int,
) -> np.ndarray:
    """Approximate cell centroids from a strided nearest-site grid."""
    xs = np.arange(0, width, stride, dtype=np.float32)
    ys = np.arange(0, height, stride, dtype=np.float32)
    grid_x = xs[None, :]
    grid_y = ys[:, None]
    best_index = np.zeros((ys.size, xs.size), dtype=np.int64)
    best_dist = np.full((ys.size, xs.size), np.inf, dtype=np.float32)
    for index, site in enumerate(sites):
        dist = (grid_x - float(site.point[0])) ** 2 + (grid_y - float(site.point[1])) ** 2
        closer = dist < best_dist
        best_index[closer] = index
        np.minimum(best_dist, dist, out=best_dist)
    flat = best_index.ravel()
    count = np.bincount(flat, minlength=len(sites))
    sum_x = np.bincount(flat, weights=np.tile(xs, ys.size), minlength=len(sites))
    sum_y = np.bincount(flat, weights=np.repeat(ys, xs.size), minlength=len(sites))
    centroids = np.empty((len(sites), 2), dtype=np.float64)
    nonzero = count > 0
    centroids[nonzero, 0] = sum_x[nonzero] / count[nonzero]
    centroids[nonzero, 1] = sum_y[nonzero] / count[nonzero]
    for index, site in enumerate(sites):
        if not nonzero[index]:
            centroids[index, 0] = float(site.point[0])
            centroids[index, 1] = float(site.point[1])
    return centroids


def mutate_voronoi_site(
    site: VoronoiSite,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
    *,
    rng: random.Random,
    sites: list[VoronoiSite] | None = None,
) -> None:
    if rng.random() < config.voronoi_geometry_probability:
        if (
            sites is not None
            and len(sites) > 1
            and rng.random() < config.voronoi_lloyd_probability
        ):
            centroids = _voronoi_cell_centroids(sites, width, height, config.voronoi_lloyd_stride)
            index = next(position for position, sibling in enumerate(sites) if sibling is site)
            target_x, target_y = centroids[index]
            site.point = np.array(
                [
                    clamp_int((int(site.point[0]) + target_x) / 2.0, 0, width - 1),
                    clamp_int((int(site.point[1]) + target_y) / 2.0, 0, height - 1),
                ],
                dtype=np.int32,
            )
            site.invalidate_caches()
            return
        site.point = site.point.copy()
        axis = rng.randrange(2)
        limit = width - 1 if axis == 0 else height - 1
        delta = rng.gauss(0, 1) * config.mutation_sigma_points * rate
        site.point[axis] = clamp_int(site.point[axis] + delta, 0, limit)
        site.invalidate_caches()
    else:
        site.color = _mutate_color(
            site.color,
            mode,
            config,
            rate * config.voronoi_color_sigma_scale,
            rng=rng,
        )


def mutate_shape(
    shape: Shape,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    rate: float,
    *,
    rng: random.Random,
    voronoi_sites: list[VoronoiSite] | None = None,
) -> None:
    if isinstance(shape, Triangle):
        mutate_triangle(shape, width, height, mode, config, rate, rng=rng)
    elif isinstance(shape, Circle):
        mutate_circle(shape, width, height, mode, config, rate, rng=rng)
    elif isinstance(shape, VoronoiSite):
        mutate_voronoi_site(
            shape, width, height, mode, config, rate, rng=rng, sites=voronoi_sites
        )
    else:
        mutate_square(shape, width, height, mode, config, rate, rng=rng)


def blend_triangles(
    a: Triangle,
    b: Triangle,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    *,
    rng: random.Random,
) -> Triangle:
    t = rng.uniform(0.25, 0.75)
    points = np.round(a.points * t + b.points * (1 - t)).astype(np.int32)
    for i in range(3):
        points[i, 0] = clamp_int(points[i, 0], 0, width - 1)
        points[i, 1] = clamp_int(points[i, 1], 0, height - 1)

    if mode == "grayscale":
        if not (isinstance(a.color, int) and isinstance(b.color, int)):
            raise TypeError("grayscale mode requires int colors")
        color = clamp_int(a.color * t + b.color * (1 - t), 0, 255)
    else:
        if not (isinstance(a.color, tuple) and isinstance(b.color, tuple)):
            raise TypeError("color mode requires (r, g, b) tuple colors")
        color = tuple(
            clamp_int(ac * t + bc * (1 - t), 0, 255)
            for ac, bc in zip(a.color, b.color)
        )

    alpha = clamp_int(a.alpha * t + b.alpha * (1 - t), config.alpha_min, config.alpha_max)
    return Triangle(points=points, color=color, alpha=alpha)


def blend_circles(
    a: Circle,
    b: Circle,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    *,
    rng: random.Random,
) -> Circle:
    t = rng.uniform(0.25, 0.75)
    center = np.round(a.center * t + b.center * (1 - t)).astype(np.int32)
    center[0] = clamp_int(center[0], 0, width - 1)
    center[1] = clamp_int(center[1], 0, height - 1)
    radius = clamp_int(a.radius * t + b.radius * (1 - t), 1, max(1, min(width, height) // 2))
    color = _blend_color(a.color, b.color, mode, t)
    alpha = clamp_int(a.alpha * t + b.alpha * (1 - t), config.alpha_min, config.alpha_max)
    return Circle(center=center, radius=radius, color=color, alpha=alpha)


def blend_squares(
    a: Square,
    b: Square,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    *,
    rng: random.Random,
) -> Square:
    t = rng.uniform(0.25, 0.75)
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
    *,
    rng: random.Random,
) -> VoronoiSite:
    t = rng.uniform(0.25, 0.75)
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
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError("grayscale mode requires int colors")
        return clamp_int(a * t + b * (1 - t), 0, 255)
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        raise TypeError("color mode requires (r, g, b) tuple colors")
    return tuple(clamp_int(ac * t + bc * (1 - t), 0, 255) for ac, bc in zip(a, b))


def blend_shapes(
    a: Shape,
    b: Shape,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
    *,
    rng: random.Random,
) -> Shape:
    if isinstance(a, Triangle) and isinstance(b, Triangle):
        return blend_triangles(a, b, width, height, mode, config, rng=rng)
    if isinstance(a, Circle) and isinstance(b, Circle):
        return blend_circles(a, b, width, height, mode, config, rng=rng)
    if isinstance(a, Square) and isinstance(b, Square):
        return blend_squares(a, b, width, height, mode, config, rng=rng)
    if isinstance(a, VoronoiSite) and isinstance(b, VoronoiSite):
        return blend_voronoi_sites(a, b, width, height, mode, config, rng=rng)
    return a.copy() if rng.random() < 0.5 else b.copy()


@dataclass(slots=True)
class Individual:
    width: int
    height: int
    mode: ColorMode
    shapes: list[Shape] = field(default_factory=list)
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
            shapes=[shape.copy() for shape in self.shapes],
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
            shapes=[shape.fork() for shape in self.shapes],
            fitness=self.fitness,
            _compositing_cache=list(self._compositing_cache),
            _compositing_cache_signature=self._compositing_cache_signature,
            _pixel_error_sum=self._pixel_error_sum,
        )

    def _cached_final_render(self) -> np.ndarray | None:
        if len(self._compositing_cache) != len(self.shapes) or not self._compositing_cache:
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
            for shape in self.shapes:
                shape.invalidate_caches()

    def ensure_min_shapes(
        self,
        config: Config,
        guide: EvolutionGuide | None = None,
        *,
        rng: random.Random,
    ) -> bool:
        added = False
        while len(self.shapes) < config.min_shapes:
            shape = random_shape(
                self.width,
                self.height,
                self.mode,
                config.shape_mode,
                config,
                guide,
                rng=rng,
            )
            self._mark_dirty(len(self.shapes), shape.bounding_box(self.width, self.height))
            self.shapes.append(shape)
            added = True
        if added:
            self.fitness = float("inf")
        return added

    def add_random_shape(
        self,
        config: Config,
        guide: EvolutionGuide | None = None,
        *,
        rng: random.Random,
    ) -> bool:
        if len(self.shapes) >= config.nb_elements_max:
            return False
        shape = random_shape(
            self.width,
            self.height,
            self.mode,
            config.shape_mode,
            config,
            guide,
            rng=rng,
        )
        self._mark_dirty(len(self.shapes), shape.bounding_box(self.width, self.height))
        self.shapes.append(shape)
        self.fitness = float("inf")
        return True

    def remove_random_shape(self, config: Config, *, rng: random.Random) -> bool:
        if len(self.shapes) <= config.min_shapes:
            return False
        idx = rng.randint(0, len(self.shapes) - 1)
        self._mark_dirty(idx, self.shapes[idx].bounding_box(self.width, self.height))
        del self.shapes[idx]
        self.invalidate_cache_from(idx)
        self.fitness = float("inf")
        return True

    def _pick_mutation_target(
        self,
        config: Config,
        guide: EvolutionGuide | None,
        *,
        rng: random.Random,
    ) -> int:
        if guide is None or not config.weighted_mutation_targets:
            return rng.randint(0, len(self.shapes) - 1)
        total = 0.0
        weights = []
        for shape in self.shapes:
            weight = guide.shape_error_mass(*shape.bounding_box(self.width, self.height))
            weights.append(weight)
            total += weight
        if total <= 0.0:
            return rng.randint(0, len(self.shapes) - 1)
        threshold = rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                return index
        return len(self.shapes) - 1

    def _mutate_random_shape(
        self,
        config: Config,
        rate: float,
        guide: EvolutionGuide | None,
        *,
        rng: random.Random,
    ) -> bool:
        if not self.shapes:
            return False
        idx = self._pick_mutation_target(config, guide, rng=rng)
        target = self.shapes[idx]
        voronoi_sites = None
        if isinstance(target, VoronoiSite):
            voronoi_sites = [shape for shape in self.shapes if isinstance(shape, VoronoiSite)]
        old_bbox = target.bounding_box(self.width, self.height)
        self._mark_dirty(idx, old_bbox)
        mutate_shape(
            target,
            self.width,
            self.height,
            self.mode,
            config,
            rate,
            rng=rng,
            voronoi_sites=voronoi_sites,
        )
        self._mark_dirty(idx, target.bounding_box(self.width, self.height))
        self.invalidate_cache_from(idx)
        self.fitness = float("inf")
        return True

    def mutate(
        self,
        config: Config,
        *,
        rate: float | None = None,
        guide: EvolutionGuide | None = None,
        rng: random.Random,
    ) -> None:
        effective_rate = config.mutation_rate if rate is None else rate
        self.ensure_min_shapes(config, guide, rng=rng)
        operations = max(1, int(round(config.mutation_operations * effective_rate)))

        for _ in range(operations):
            roll = rng.random()
            if roll < config.prob_structural:
                if rng.random() < config.prob_add_vs_del:
                    changed = self.add_random_shape(config, guide, rng=rng)
                else:
                    changed = self.remove_random_shape(config, rng=rng)
                if not changed:
                    self._mutate_random_shape(config, effective_rate, guide, rng=rng)
            elif roll < config.prob_structural + config.prob_reorder and len(self.shapes) >= 2:
                i, j = rng.sample(range(len(self.shapes)), 2)
                self._mark_dirty(i, self.shapes[i].bounding_box(self.width, self.height))
                self._mark_dirty(j, self.shapes[j].bounding_box(self.width, self.height))
                self.shapes[i], self.shapes[j] = self.shapes[j], self.shapes[i]
                self.invalidate_cache_from(min(i, j))
                self.fitness = float("inf")
            elif self.shapes:
                self._mutate_random_shape(config, effective_rate, guide, rng=rng)

    @classmethod
    def random(
        cls,
        width: int,
        height: int,
        mode: ColorMode,
        config: Config,
        count: int,
        guide: EvolutionGuide | None = None,
        *,
        rng: random.Random,
    ) -> Individual:
        ind = cls(width=width, height=height, mode=mode)
        for _ in range(count):
            ind.shapes.append(
                random_shape(width, height, mode, config.shape_mode, config, guide, rng=rng)
            )
        return ind

    @classmethod
    def crossover_uniform(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
        guide: EvolutionGuide | None = None,
        *,
        rng: random.Random,
    ) -> Individual:
        child_count = rng.randint(
            min(len(parent_a.shapes), len(parent_b.shapes)),
            min(config.nb_elements_max, max(len(parent_a.shapes), len(parent_b.shapes))),
        )
        child_count = max(config.min_shapes, child_count)
        child = cls(width=parent_a.width, height=parent_a.height, mode=parent_a.mode)
        for i in range(child_count):
            if i < len(parent_a.shapes) and i < len(parent_b.shapes):
                source = parent_a if rng.random() < 0.5 else parent_b
                child.shapes.append(source.shapes[i].copy())
            elif i < len(parent_a.shapes):
                child.shapes.append(parent_a.shapes[i].copy())
            elif i < len(parent_b.shapes):
                child.shapes.append(parent_b.shapes[i].copy())
            else:
                child.shapes.append(
                    random_shape(
                        parent_a.width,
                        parent_a.height,
                        parent_a.mode,
                        config.shape_mode,
                        config,
                        guide,
                        rng=rng,
                    )
                )
        child.ensure_min_shapes(config, guide, rng=rng)
        return child

    @classmethod
    def crossover_blend(
        cls,
        parent_a: Individual,
        parent_b: Individual,
        config: Config,
        guide: EvolutionGuide | None = None,
        *,
        rng: random.Random,
    ) -> Individual:
        count = rng.randint(
            min(len(parent_a.shapes), len(parent_b.shapes)),
            min(config.nb_elements_max, max(len(parent_a.shapes), len(parent_b.shapes))),
        )
        count = max(config.min_shapes, count)
        child = cls(width=parent_a.width, height=parent_a.height, mode=parent_a.mode)
        for i in range(count):
            if i < len(parent_a.shapes) and i < len(parent_b.shapes):
                child.shapes.append(
                    blend_shapes(
                        parent_a.shapes[i],
                        parent_b.shapes[i],
                        parent_a.width,
                        parent_a.height,
                        parent_a.mode,
                        config,
                        rng=rng,
                    )
                )
            elif i < len(parent_a.shapes):
                child.shapes.append(parent_a.shapes[i].copy())
            else:
                child.shapes.append(parent_b.shapes[i].copy())
        child.ensure_min_shapes(config, guide, rng=rng)
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
    for shape in individual.shapes:
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
            shapes.append(VoronoiSite(point=point, color=shape.color))

    return Individual(
        width=width,
        height=height,
        mode=individual.mode,
        shapes=shapes,
        fitness=float("inf"),
    )
