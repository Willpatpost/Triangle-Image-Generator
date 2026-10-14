# Changelog

## Unreleased

- New `elite_candidates` (default 2) and `elite_candidate_rate` (default 0.5)
  configuration: every generation injects fork-and-mutate candidates of the
  incumbent best individual, scored through the incremental path and trimmed
  by rank. On a 225×225 structured target the default improves mean best
  fitness by about 6% over 5 s and 12% over 15 s versus disabling it, at
  roughly 8% fewer steps/s; k=4 and full-rate candidates showed no gain.
  Set `elite_candidates` to 0 to restore the previous behavior.
- `benchmarks/benchmark.py` accepts `--target` to benchmark against an image
  file and `--elite-candidates` for candidate sweeps.
- New opt-in guidance upgrades, off by default after measurement: an
  error-map guide (`guidance_map_stride`), error-weighted mutation targets
  (`weighted_mutation_targets`), and guided alpha/color search
  (`guided_alpha_search`). Combined, they regressed mean best fitness by
  16% at 5 s and 34% at 15 s on the structured benchmark (over-exploitation;
  one seed improved sharply), so random probing and uniform targets stay
  the defaults.
- New opt-in Voronoi Lloyd relaxation, off by default after measurement:
  `voronoi_lloyd_probability` (default 0) and `voronoi_lloyd_stride` move a
  site halfway toward its cell centroid instead of Gaussian jitter. On the
  voronoi-mode structured benchmark it regressed mean best fitness by 5%
  at 5 s and 9% at 15 s — centroid moves optimize partition geometry, not
  the target's color boundaries, and displace useful jitter.
- Stagnation now ends in a diversity restart: after
  `restart_stagnation_threshold` (default 2000) consecutive non-improving
  generations the population is reseeded around its elites (0 disables).
  Short-horizon benchmarks cannot reach the threshold, so the default
  rests on design grounds, not measurement.
- Optional mutation annealing (`mutation_anneal_iterations`, default 0):
  the per-step mutation rate scales linearly down to a 25% floor across
  the configured iteration span.

## 2.4.0 — 2026-10-14

- Evolution randomness flows through a per-session RNG instead of the
  module-global `random`; seeding one session no longer disturbs another,
  and session files keep restoring the exact stream
- `VoronoiSite.alpha` removed — it was serialized but never blended by any
  renderer; older session files still load (the stored value is ignored)
- Voronoi mutation tuning is config-driven (`voronoi_geometry_probability`,
  `voronoi_color_sigma_scale`) instead of hardcoded magic numbers
- Documented that "Mixed" genomes draw triangles/circles/squares; Voronoi is
  a full-canvas partition and stays its own shape family
- Renderer guards raise real exceptions instead of asserts (no longer
  stripped by `python -O`), and cached final renders are read-only so
  callers cannot corrupt the compositing cache through the returned array
- Removed the unused `setup_logging`; `run_evolution` logs through a
  private logger that honors `enable_logging`
- GUI: the worker/UI protocol and button state machine moved into
  `gui_state.py` and are unit-tested headlessly; session previews render
  off the UI thread; closing the window offers to save the active session
- Added `benchmarks/benchmark.py` for steps-per-second per renderer backend

## 2.3.0 — 2026-10-14

First release under the `genetic-shape-image-generator` package name, aligned
with the project's own 2.3 lineage.

- MIT license; package metadata now names the author and license.
- Evolution engine in `core` with triangle, circle, square, Voronoi, and mixed
  genomes.
- Tkinter desktop app (`app.py`) with start/stop/continue/refine, session
  save/open, and best-image export.
- Progressive coarse-to-fine evolution via `run_progressive_evolution`.
- Dirty-region incremental rendering with exact incremental fitness scoring.
- Optional Numba CPU and CUDA renderer backends with automatic fallback.
- Versioned JSON session files (format version 2) with atomic writes, source
  hash verification, and legacy version-1 support.
- `run_evolution` now delegates to `EvolutionSession` and raises
  `FileNotFoundError` when a resume file is missing.
- CI runs the test suite against the installed package instead of the source
  tree.
- Sample image `images/Spongebob LeBron.png` removed (third-party art that is
  not redistributable under the project license).
