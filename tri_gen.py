#!/usr/bin/env python3
"""Evolve semi-transparent shapes to approximate a target image."""

from __future__ import annotations

import argparse
import sys

from core.config import Config
from core.paths import resolve_image_path, resolve_output_dir, resolve_project_path
from core.pipeline import run_evolution, setup_logging

EPILOG = """
examples:
  python tri_gen.py
  python tri_gen.py -i Yinyang.png --downsample 2
  python tri_gen.py -i images/Yinyang.png --iterations 5000
  python tri_gen.py -i photo.jpg --mode color -o output/color-run
  python tri_gen.py -i photo.jpg --shape mixed --mode color
  python tri_gen.py -i photo.jpg --shape voronoi --shapes 80
  python tri_gen.py --resume
  python tri_gen.py --resume output/best_state.json --iterations 20000
"""


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tri_gen",
        description=(
            "Approximate an image with semi-transparent shapes using a genetic algorithm."
        ),
        formatter_class=HelpFormatter,
        epilog=EPILOG,
    )

    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--image",
        "-i",
        default="images/Yinyang.png",
        metavar="PATH",
        help="Target image (searches images/ if given as a filename).",
    )
    input_group.add_argument(
        "--mode",
        choices=("grayscale", "color"),
        default="color",
        help="Render shapes in grayscale or full color.",
    )
    input_group.add_argument(
        "--shape",
        choices=("triangle", "circle", "square", "voronoi", "mixed"),
        default="triangle",
        help="Primitive shape type to evolve.",
    )
    input_group.add_argument(
        "--downsample",
        type=int,
        default=1,
        metavar="N",
        help="Shrink the image by this factor before evolving (faster previews).",
    )

    evolution_group = parser.add_argument_group("Evolution")
    evolution_group.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        metavar="N",
        help="Maximum number of generations to run.",
    )
    evolution_group.add_argument(
        "--resume",
        nargs="?",
        const="output/best_state.json",
        default=None,
        metavar="PATH",
        help="Continue from a saved state (default: output/best_state.json).",
    )
    evolution_group.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducible runs.",
    )

    population_group = parser.add_argument_group("Population")
    population_group.add_argument(
        "--pop-size",
        type=int,
        default=30,
        metavar="N",
        help="Number of candidates evaluated each generation.",
    )
    population_group.add_argument(
        "--elite",
        type=int,
        default=6,
        metavar="N",
        help="Top candidates preserved unchanged each generation.",
    )
    population_group.add_argument(
        "--triangles",
        "--shapes",
        type=int,
        default=15,
        dest="triangles",
        metavar="N",
        help="Starting shape count per candidate.",
    )
    population_group.add_argument(
        "--max-triangles",
        "--max-shapes",
        type=int,
        default=150,
        dest="max_triangles",
        metavar="N",
        help="Maximum shapes allowed per candidate.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        "-o",
        default="output",
        metavar="DIR",
        help="Directory for results, GIF, and saved state.",
    )
    output_group.add_argument(
        "--no-comparison",
        action="store_true",
        help="Do not save the target/result/diff comparison image.",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs.",
    )
    output_group.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity when not using --quiet.",
    )

    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="Parallel fitness workers (0 = one worker per CPU core minus one).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.downsample < 1:
        parser.error("--downsample must be at least 1")

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    if args.pop_size < 2:
        parser.error("--pop-size must be at least 2")

    if args.elite < 1:
        parser.error("--elite must be at least 1")

    if args.elite >= args.pop_size:
        parser.error("--elite must be smaller than --pop-size")

    if args.triangles < 1:
        parser.error("--triangles must be at least 1")

    if args.max_triangles < args.triangles:
        parser.error("--max-triangles must be greater than or equal to --triangles")

    if args.max_triangles < Config().min_triangles:
        parser.error(f"--max-triangles must be at least {Config().min_triangles}")

    if args.workers < 0:
        parser.error("--workers must be 0 or greater")

    image_path = resolve_image_path(args.image)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    output_dir = resolve_output_dir(args.output)
    resume_path = None
    if args.resume is not None:
        resume_path = resolve_project_path(args.resume)
        if not resume_path.is_file():
            parser.error(f"Resume state not found: {resume_path}")

    config = Config(
        mode=args.mode,
        shape_mode=args.shape,
        pop_size=args.pop_size,
        nb_elite=args.elite,
        nb_elements_initial=args.triangles,
        nb_elements_max=args.max_triangles,
        max_workers=args.workers,
        save_directory=str(output_dir),
        save_comparison=not args.no_comparison,
        log_level=args.log_level,
        enable_logging=not args.quiet,
    )

    setup_logging(config.log_level, config.enable_logging)

    try:
        run_evolution(
            str(image_path),
            config,
            iterations=args.iterations,
            downsample=args.downsample,
            resume_path=str(resume_path) if resume_path else None,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
