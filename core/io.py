from __future__ import annotations

import json
import hashlib
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from core.config import ColorMode, Config
from core.shapes import Circle, Individual, Shape, Square, Triangle, VoronoiSite, shape_kind


STATE_VERSION = 2


@dataclass(slots=True)
class EvolutionState:
    iteration: int
    best_fitness: float
    best: Individual
    config: Config
    background: int | tuple[int, int, int]
    source_image: str | None = None
    source_sha256: str | None = None
    downsample: int = 1
    population: list[Individual] | None = None
    stagnation: int = 0
    mutation_rate: float | None = None
    python_random_state: tuple[int, tuple[int, ...], float | None] | None = None
    numpy_random_state: tuple[str, np.ndarray, int, int, float] | None = None
    version: int = STATE_VERSION


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


def _individual_to_dict(individual: Individual) -> dict[str, Any]:
    return {
        "width": individual.width,
        "height": individual.height,
        "mode": individual.mode,
        "fitness": individual.fitness,
        "shapes": [_shape_to_dict(shape) for shape in individual.shapes],
    }


def _individual_from_dict(data: dict[str, Any], fallback_fitness: float = float("inf")) -> Individual:
    shape_payload = data.get("shapes", data.get("triangles", []))
    return Individual(
        width=int(data["width"]),
        height=int(data["height"]),
        mode=data["mode"],
        shapes=[_shape_from_dict(item) for item in shape_payload],
        fitness=float(data.get("fitness", fallback_fitness)),
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_state_to_json(state: tuple[int, tuple[int, ...], float | None]) -> list[Any]:
    return [state[0], list(state[1]), state[2]]


def _python_state_from_json(data: list[Any]) -> tuple[int, tuple[int, ...], float | None]:
    return int(data[0]), tuple(int(value) for value in data[1]), data[2]


def _numpy_state_to_json(state: tuple[str, np.ndarray, int, int, float]) -> list[Any]:
    return [state[0], state[1].tolist(), state[2], state[3], state[4]]


def _numpy_state_from_json(data: list[Any]) -> tuple[str, np.ndarray, int, int, float]:
    return (
        str(data[0]),
        np.asarray(data[1], dtype=np.uint32),
        int(data[2]),
        int(data[3]),
        float(data[4]),
    )


def save_state(
    path: str,
    iteration: int,
    best: Individual,
    best_fitness: float,
    config: Config,
    background: int | tuple[int, int, int],
    source_image: str | None = None,
    downsample: int | None = None,
    *,
    population: Sequence[Individual] | None = None,
    stagnation: int = 0,
    mutation_rate: float | None = None,
    python_random_state: tuple[int, tuple[int, ...], float | None] | None = None,
    numpy_random_state: tuple[str, np.ndarray, int, int, float] | None = None,
) -> None:
    resolved_source: str | None = None
    source_hash: str | None = None
    if source_image is not None:
        source_path = Path(source_image).expanduser().resolve()
        resolved_source = str(source_path)
        if source_path.is_file():
            source_hash = file_sha256(source_path)

    payload = {
        "version": STATE_VERSION,
        "iteration": iteration,
        "best_fitness": best_fitness,
        "best": _individual_to_dict(best),
        "background": background if isinstance(background, int) else list(background),
        "config": _config_to_dict(config),
        "source_image": resolved_source,
        "source_sha256": source_hash,
        "downsample": max(1, downsample or 1),
        "stagnation": max(0, stagnation),
        "mutation_rate": mutation_rate,
    }
    if population is not None:
        payload["population"] = [_individual_to_dict(individual) for individual in population]
    if python_random_state is not None:
        payload["python_random_state"] = _python_state_to_json(python_random_state)
    if numpy_random_state is not None:
        payload["numpy_random_state"] = _numpy_state_to_json(numpy_random_state)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    logging.info("Saved state %s", path)


def read_state(path: str) -> EvolutionState:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    version = int(payload.get("version", 1))
    if version > STATE_VERSION:
        raise ValueError(
            f"Session version {version} is newer than the supported version {STATE_VERSION}."
        )

    config_fields = {f.name for f in Config.__dataclass_fields__.values()}
    raw_config = dict(payload["config"])
    if "min_shapes" not in raw_config and "min_triangles" in raw_config:
        raw_config["min_shapes"] = raw_config["min_triangles"]
    config_data = {k: v for k, v in raw_config.items() if k in config_fields and k != "background"}
    config = Config(**config_data)
    config.validate()

    background = payload["background"]
    if isinstance(background, list):
        background = tuple(background)

    best_fitness = float(payload["best_fitness"])
    if "best" in payload:
        best = _individual_from_dict(payload["best"], best_fitness)
    else:
        best = _individual_from_dict(payload, best_fitness)

    population_payload = payload.get("population")
    population = None
    if population_payload is not None:
        population = [_individual_from_dict(item) for item in population_payload]

    python_state_payload = payload.get("python_random_state")
    numpy_state_payload = payload.get("numpy_random_state")
    return EvolutionState(
        version=version,
        iteration=int(payload["iteration"]),
        best_fitness=best_fitness,
        best=best,
        config=config,
        background=background,
        source_image=payload.get("source_image"),
        source_sha256=payload.get("source_sha256"),
        downsample=max(1, int(payload.get("downsample", 1))),
        population=population,
        stagnation=max(0, int(payload.get("stagnation", 0))),
        mutation_rate=(
            float(payload["mutation_rate"])
            if payload.get("mutation_rate") is not None
            else None
        ),
        python_random_state=(
            _python_state_from_json(python_state_payload)
            if python_state_payload is not None
            else None
        ),
        numpy_random_state=(
            _numpy_state_from_json(numpy_state_payload)
            if numpy_state_payload is not None
            else None
        ),
    )


def load_state(path: str) -> tuple[int, float, Individual, Config, int | tuple[int, int, int]]:
    """Load a state using the original tuple API for compatibility."""
    state = read_state(path)
    return state.iteration, state.best_fitness, state.best, state.config, state.background
