#!/usr/bin/env python3
"""Measure evolution throughput per renderer backend on a synthetic target."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.acceleration import AcceleratorUnavailable
from core.config import Config
from core.evolver import EvolutionSession


def build_target(path: Path, width: int, height: int, mode: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if mode == "grayscale":
        array = rng.integers(0, 256, (height, width), dtype=np.uint8)
        Image.fromarray(array, mode="L").save(path)
    else:
        array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="numpy", help="comma-separated backends to measure")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--mode", choices=("color", "grayscale"), default="color")
    parser.add_argument(
        "--shape-mode",
        dest="shape_mode",
        choices=("triangle", "circle", "square", "voronoi", "mixed"),
        default="mixed",
    )
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=2, help="untimed steps before measuring")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target.png"
        build_target(target, args.width, args.height, args.mode, args.seed)

        for name in args.backends.split(","):
            name = name.strip()
            if not name:
                continue
            config = Config(
                mode=args.mode,
                shape_mode=args.shape_mode,
                pop_size=args.population,
                renderer_backend=name,
                max_workers=1,
                enable_logging=False,
            )
            try:
                config.validate()
                session = EvolutionSession(str(target), config, seed=args.seed)
                for _ in range(args.warmup):
                    session.step()
                started = time.monotonic()
                steps = 0
                deadline = started + args.seconds
                while time.monotonic() < deadline:
                    session.step()
                    steps += 1
                elapsed = time.monotonic() - started
                print(
                    f"{name:>6}: {steps / elapsed:9.1f} steps/s | "
                    f"best fitness {session.global_best.fitness:.6f} | "
                    f"shapes {len(session.global_best.shapes)}"
                )
            except AcceleratorUnavailable as exc:
                print(f"{name:>6}: unavailable ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
