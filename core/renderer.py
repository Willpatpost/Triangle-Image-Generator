from __future__ import annotations

import numpy as np

from core.config import ColorMode, Config
from core.shapes import Circle, Individual, Shape, Square, Triangle, VoronoiSite


def _alpha_factor(config: Config, shape: Shape) -> float:
    if 0.0 <= config.fixed_alpha <= 1.0:
        return config.fixed_alpha
    return shape.alpha / 255.0


def _triangle_mask_local(tri: Triangle, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    x = np.arange(min_x, max_x + 1)
    y = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x, y)

    pt0, pt1, pt2 = tri.points
    det = (pt1[1] - pt2[1]) * (pt0[0] - pt2[0]) + (pt2[0] - pt1[0]) * (pt0[1] - pt2[1])
    if det == 0:
        return np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)

    a = ((pt1[1] - pt2[1]) * (xv - pt2[0]) + (pt2[0] - pt1[0]) * (yv - pt2[1])) / det
    b = ((pt2[1] - pt0[1]) * (xv - pt2[0]) + (pt0[0] - pt2[0]) * (yv - pt2[1])) / det
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)


def _circle_mask_local(circle: Circle, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    x = np.arange(min_x, max_x + 1)
    y = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x, y)
    cx, cy = circle.center
    return (xv - cx) ** 2 + (yv - cy) ** 2 <= circle.radius ** 2


def _square_mask_local(square: Square, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    x = np.arange(min_x, max_x + 1)
    y = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x, y)
    sx, sy = square.top_left
    return (xv >= sx) & (xv <= sx + square.side) & (yv >= sy) & (yv <= sy + square.side)


def _shape_mask_local(shape: Shape, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    if isinstance(shape, Triangle):
        return _triangle_mask_local(shape, min_x, min_y, max_x, max_y)
    if isinstance(shape, Circle):
        return _circle_mask_local(shape, min_x, min_y, max_x, max_y)
    if isinstance(shape, VoronoiSite):
        return np.ones((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)
    return _square_mask_local(shape, min_x, min_y, max_x, max_y)


def _blend_region_grayscale(
    canvas: np.ndarray,
    shape: Shape,
    min_x: int,
    min_y: int,
    mask: np.ndarray,
    alpha: float,
) -> None:
    region = canvas[min_y : min_y + mask.shape[0], min_x : min_x + mask.shape[1]]
    assert isinstance(shape.color, int)
    blended = np.where(mask, alpha * shape.color + (1.0 - alpha) * region, region)
    canvas[min_y : min_y + mask.shape[0], min_x : min_x + mask.shape[1]] = blended


def _blend_region_color(
    canvas: np.ndarray,
    shape: Shape,
    min_x: int,
    min_y: int,
    mask: np.ndarray,
    alpha: float,
) -> None:
    region = canvas[min_y : min_y + mask.shape[0], min_x : min_x + mask.shape[1]]
    assert isinstance(shape.color, tuple)
    for channel, value in enumerate(shape.color):
        channel_region = region[..., channel]
        blended = np.where(mask, alpha * value + (1.0 - alpha) * channel_region, channel_region)
        canvas[min_y : min_y + mask.shape[0], min_x : min_x + mask.shape[1], channel] = blended


def composite_shape(
    canvas: np.ndarray,
    shape: Shape,
    width: int,
    height: int,
    mode: ColorMode,
    config: Config,
) -> None:
    min_x, min_y, max_x, max_y = shape.bounding_box(width, height)
    if min_x > max_x or min_y > max_y:
        return

    mask = _shape_mask_local(shape, min_x, min_y, max_x, max_y)
    if not mask.any():
        return

    alpha = _alpha_factor(config, shape)
    if mode == "grayscale":
        _blend_region_grayscale(canvas, shape, min_x, min_y, mask, alpha)
    else:
        _blend_region_color(canvas, shape, min_x, min_y, mask, alpha)


def new_canvas(width: int, height: int, mode: ColorMode, background: int | tuple[int, int, int]) -> np.ndarray:
    if mode == "grayscale":
        return np.full((height, width), background, dtype=np.float32)
    assert isinstance(background, tuple)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for c, value in enumerate(background):
        canvas[..., c] = value
    return canvas


def render_voronoi(individual: Individual, background: int | tuple[int, int, int]) -> np.ndarray:
    width, height = individual.width, individual.height
    mode = individual.mode
    sites = [shape for shape in individual.triangles if isinstance(shape, VoronoiSite)]
    if not sites:
        return np.clip(new_canvas(width, height, mode, background), 0, 255).astype(np.uint8)

    yy, xx = np.indices((height, width))
    best_dist = np.full((height, width), np.inf, dtype=np.float32)
    if mode == "grayscale":
        canvas = new_canvas(width, height, mode, background)
        for site in sites:
            sx, sy = site.point
            dist = (xx - sx) ** 2 + (yy - sy) ** 2
            mask = dist < best_dist
            assert isinstance(site.color, int)
            canvas[mask] = site.color
            best_dist[mask] = dist[mask]
    else:
        canvas = new_canvas(width, height, mode, background)
        for site in sites:
            sx, sy = site.point
            dist = (xx - sx) ** 2 + (yy - sy) ** 2
            mask = dist < best_dist
            assert isinstance(site.color, tuple)
            canvas[mask] = site.color
            best_dist[mask] = dist[mask]

    return np.clip(canvas, 0, 255).astype(np.uint8)


def render_individual(individual: Individual, config: Config, background: int | tuple[int, int, int]) -> np.ndarray:
    width, height = individual.width, individual.height
    mode = individual.mode

    if config.shape_mode == "voronoi" or any(isinstance(shape, VoronoiSite) for shape in individual.triangles):
        return render_voronoi(individual, background)

    if config.use_compositing_cache and individual._compositing_cache:
        for i, cached in enumerate(individual._compositing_cache):
            if cached is not None and i == len(individual._compositing_cache) - 1:
                return np.clip(cached, 0, 255).astype(np.uint8)

    canvas = new_canvas(width, height, mode, background)
    cache: list[np.ndarray | None] = []

    for shape in individual.triangles:
        composite_shape(canvas, shape, width, height, mode, config)
        if config.use_compositing_cache:
            cache.append(canvas.copy())
        else:
            cache.append(None)

    if config.use_compositing_cache:
        individual._compositing_cache = cache

    return np.clip(canvas, 0, 255).astype(np.uint8)
