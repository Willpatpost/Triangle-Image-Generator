from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from core.config import ColorMode, Config
from core.paths import ITERATION_PREFIX
from core.shapes import Circle, Individual, Shape, Square, Triangle, VoronoiSite, shape_kind


def load_reference_image(
    path: str,
    mode: ColorMode,
    downsample: int,
    blur_sigma: float,
) -> tuple[np.ndarray, int | tuple[int, int, int]]:
    pil_mode = "L" if mode == "grayscale" else "RGB"
    image = Image.open(path).convert(pil_mode)
    if downsample > 1:
        width = max(1, image.width // downsample)
        height = max(1, image.height // downsample)
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    if blur_sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    array = np.array(image, dtype=np.uint8)

    if mode == "grayscale":
        background = int(np.mean(array))
    else:
        background = tuple(int(np.mean(array[..., c])) for c in range(3))

    return array, background


def save_image(array: np.ndarray, path: str, mode: ColorMode) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if mode == "grayscale":
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)
    logging.info("Saved %s", path)


def save_comparison(
    reference: np.ndarray,
    rendered: np.ndarray,
    path: str,
    mode: ColorMode,
) -> None:
    if mode == "grayscale":
        diff = np.abs(reference.astype(np.int16) - rendered.astype(np.int16)).astype(np.uint8)
        diff = np.stack([diff, diff, diff], axis=-1)
        ref_rgb = np.stack([reference, reference, reference], axis=-1)
        gen_rgb = np.stack([rendered, rendered, rendered], axis=-1)
    else:
        diff = np.abs(reference.astype(np.int16) - rendered.astype(np.int16)).astype(np.uint8)
        ref_rgb = reference
        gen_rgb = rendered

    combined = np.concatenate([ref_rgb, gen_rgb, diff], axis=1)
    save_image(combined, path, "color")


def create_gif(image_dir: str, gif_name: str, duration_ms: int) -> None:
    frames = sorted(
        [f for f in os.listdir(image_dir) if f.startswith(ITERATION_PREFIX) and f.endswith(".png")],
        key=lambda name: int(name.replace(ITERATION_PREFIX, "").replace(".png", "")),
    )
    if not frames:
        return

    images = [Image.open(os.path.join(image_dir, frame)).copy() for frame in frames]
    gif_path = os.path.join(image_dir, gif_name)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    logging.info("Created GIF %s", gif_path)


def _shape_to_dict(shape: Shape) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": shape_kind(shape),
        "color": shape.color if isinstance(shape.color, int) else list(shape.color),
        "alpha": shape.alpha,
    }
    if isinstance(shape, Triangle):
        data["points"] = shape.points.tolist()
    elif isinstance(shape, Circle):
        data["center"] = shape.center.tolist()
        data["radius"] = shape.radius
    elif isinstance(shape, VoronoiSite):
        data["point"] = shape.point.tolist()
    else:
        data["top_left"] = shape.top_left.tolist()
        data["side"] = shape.side
    return data


def _shape_from_dict(data: dict[str, Any]) -> Shape:
    color = data["color"]
    if isinstance(color, list):
        color = tuple(color)

    kind = data.get("kind", "triangle")
    if kind == "circle":
        return Circle(
            center=np.array(data["center"], dtype=np.int32),
            radius=int(data["radius"]),
            color=color,
            alpha=int(data["alpha"]),
        )
    if kind == "square":
        return Square(
            top_left=np.array(data["top_left"], dtype=np.int32),
            side=int(data["side"]),
            color=color,
            alpha=int(data["alpha"]),
        )
    if kind == "voronoi":
        return VoronoiSite(
            point=np.array(data["point"], dtype=np.int32),
            color=color,
            alpha=int(data.get("alpha", 255)),
        )
    return Triangle(points=np.array(data["points"], dtype=np.int32), color=color, alpha=int(data["alpha"]))


def _config_to_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    data.pop("background", None)
    return data


def save_state(
    path: str,
    iteration: int,
    best: Individual,
    best_fitness: float,
    config: Config,
    background: int | tuple[int, int, int],
    source_image: str | None = None,
    downsample: int | None = None,
) -> None:
    payload = {
        "iteration": iteration,
        "best_fitness": best_fitness,
        "width": best.width,
        "height": best.height,
        "mode": best.mode,
        "background": background if isinstance(background, int) else list(background),
        "config": _config_to_dict(config),
        "shapes": [_shape_to_dict(t) for t in best.triangles],
    }
    payload["triangles"] = payload["shapes"]
    if source_image is not None:
        payload["source_image"] = source_image
    if downsample is not None:
        payload["downsample"] = downsample
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved state %s", path)


def load_state(path: str) -> tuple[int, float, Individual, Config, int | tuple[int, int, int]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    config_fields = {f.name for f in Config.__dataclass_fields__.values()}
    config_data = {k: v for k, v in payload["config"].items() if k in config_fields and k != "background"}
    config = Config(**config_data)

    background = payload["background"]
    if isinstance(background, list):
        background = tuple(background)

    shape_payload = payload.get("shapes", payload.get("triangles", []))
    individual = Individual(
        width=int(payload["width"]),
        height=int(payload["height"]),
        mode=payload["mode"],
        triangles=[_shape_from_dict(item) for item in shape_payload],
        fitness=float(payload["best_fitness"]),
    )
    return int(payload["iteration"]), float(payload["best_fitness"]), individual, config, background
