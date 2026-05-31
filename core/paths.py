"""Project paths and filename conventions."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_SAMPLE_IMAGE = IMAGES_DIR / "Yinyang.png"
DEFAULT_STATE_FILE = OUTPUT_DIR / "best_state.json"

FINAL_IMAGE_NAME = "final.png"
COMPARISON_IMAGE_NAME = "comparison.png"
EVOLUTION_GIF_NAME = "evolution.gif"
STATE_FILE_NAME = "best_state.json"
ITERATION_PREFIX = "iteration"


def resolve_image_path(path: str | Path) -> Path:
    """Resolve an image path against the working directory, project root, and images/."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    search_paths = (
        Path.cwd() / candidate,
        PROJECT_ROOT / candidate,
        IMAGES_DIR / candidate.name,
    )
    for option in search_paths:
        if option.is_file():
            return option.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def resolve_output_dir(path: str | Path) -> Path:
    """Resolve an output directory relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a file or directory path against the working directory and project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    for option in (Path.cwd() / candidate, PROJECT_ROOT / candidate):
        if option.exists():
            return option.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def output_artifact(output_dir: str | Path, filename: str) -> Path:
    return Path(output_dir) / filename


def iteration_frame_path(output_dir: str | Path, iteration: int) -> Path:
    return output_artifact(output_dir, f"{ITERATION_PREFIX}{iteration}.png")
