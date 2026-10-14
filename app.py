#!/usr/bin/env python3
"""Desktop GUI for genetic shape image approximation."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core.acceleration import available_renderer_backends
from core.config import Config
from core.io import save_image
from gui_state import (
    EvolutionController,
    SessionNeedsSource,
    SessionPreview,
    SessionPreviewFailed,
    SessionSettings,
    WorkerError,
    WorkerFinished,
    WorkerSessionReady,
    WorkerUpdate,
    compute_control_states,
)


CANVAS_WIDTH = 560
CANVAS_HEIGHT = 440

SHAPES = {
    "Triangles": "triangle",
    "Circles": "circle",
    "Squares": "square",
    "Voronoi": "voronoi",
    "Mixed": "mixed",
}
MODES = {"RGB": "color", "B&W": "grayscale"}
SCALES = {"Full": 1, "1/2": 2, "1/3": 3, "1/4": 4, "1/5": 5, "1/6": 6}
BACKENDS = {
    "Automatic": "auto",
    "NumPy": "numpy",
    "Numba": "numba",
    "CUDA": "cuda",
}
NEXT_SCALE = {6: 4, 5: 3, 4: 2, 3: 2, 2: 1}


class GeneticImageApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Genetic Shape Image Generator")
        self.minsize(1180, 720)

        self.controller = EvolutionController()
        self.image_path: Path | None = None
        self.target_photo: ImageTk.PhotoImage | None = None
        self.evolution_photo: ImageTk.PhotoImage | None = None
        self.best_image: np.ndarray | None = None
        self._updating_settings = False

        self.shape_var = tk.StringVar(value="Triangles")
        self.mode_var = tk.StringVar(value="RGB")
        self.scale_var = tk.StringVar(value="1/2")
        self.population_var = tk.IntVar(value=24)
        self.initial_shapes_var = tk.IntVar(value=18)
        self.max_shapes_var = tk.IntVar(value=140)
        self.backend_var = tk.StringVar(value="Automatic")
        self.workers_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Choose an image or open a session to begin.")

        self._build_ui()
        self._watch_settings()
        self.after(80, self._poll_updates)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        controls = ttk.Frame(self, padding=(16, 12, 16, 8))
        controls.pack(fill=tk.X)

        self.choose_button = ttk.Button(controls, text="Choose Image", command=self._choose_image)
        self.choose_button.grid(row=0, column=0, padx=(0, 6), pady=3)
        self.open_session_button = ttk.Button(controls, text="Open Session", command=self._open_session)
        self.open_session_button.grid(row=0, column=1, padx=(0, 18), pady=3)

        ttk.Label(controls, text="Shape").grid(row=0, column=2, padx=(0, 6), pady=3)
        self.shape_box = ttk.Combobox(
            controls,
            textvariable=self.shape_var,
            values=tuple(SHAPES),
            width=11,
            state="readonly",
        )
        self.shape_box.grid(row=0, column=3, padx=(0, 14), pady=3)

        ttk.Label(controls, text="Color").grid(row=0, column=4, padx=(0, 6), pady=3)
        self.mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=tuple(MODES),
            width=7,
            state="readonly",
        )
        self.mode_box.grid(row=0, column=5, padx=(0, 14), pady=3)

        ttk.Label(controls, text="Working resolution").grid(row=0, column=6, padx=(0, 6), pady=3)
        self.scale_box = ttk.Combobox(
            controls,
            textvariable=self.scale_var,
            values=tuple(SCALES),
            width=7,
            state="readonly",
        )
        self.scale_box.grid(row=0, column=7, padx=(0, 18), pady=3)

        self.start_button = ttk.Button(controls, text="Start", command=self._start_evolving, state=tk.DISABLED)
        self.start_button.grid(row=0, column=8, padx=(0, 6), pady=3)
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop_evolving, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=9, padx=(0, 6), pady=3)
        self.refine_button = ttk.Button(controls, text="Refine", command=self._refine, state=tk.DISABLED)
        self.refine_button.grid(row=0, column=10, pady=3)

        ttk.Label(controls, text="Population").grid(row=1, column=0, sticky="e", padx=(0, 6), pady=3)
        self.population_box = ttk.Spinbox(
            controls, from_=4, to=128, textvariable=self.population_var, width=6, justify=tk.CENTER
        )
        self.population_box.grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(controls, text="Initial shapes").grid(row=1, column=2, padx=(0, 6), pady=3)
        self.initial_shapes_box = ttk.Spinbox(
            controls, from_=1, to=500, textvariable=self.initial_shapes_var, width=7, justify=tk.CENTER
        )
        self.initial_shapes_box.grid(row=1, column=3, sticky="w", pady=3)

        ttk.Label(controls, text="Maximum shapes").grid(row=1, column=4, padx=(0, 6), pady=3)
        self.max_shapes_box = ttk.Spinbox(
            controls, from_=1, to=2000, textvariable=self.max_shapes_var, width=7, justify=tk.CENTER
        )
        self.max_shapes_box.grid(row=1, column=5, sticky="w", pady=3)

        available = set(available_renderer_backends())
        backend_labels = tuple(
            label for label, value in BACKENDS.items() if value == "auto" or value in available
        )
        ttk.Label(controls, text="Renderer").grid(row=1, column=6, sticky="e", padx=(0, 6), pady=3)
        self.backend_box = ttk.Combobox(
            controls,
            textvariable=self.backend_var,
            values=backend_labels,
            width=11,
            state="readonly",
        )
        self.backend_box.grid(row=1, column=7, sticky="w", pady=3)

        ttk.Label(controls, text="Workers").grid(row=1, column=8, sticky="e", padx=(0, 6), pady=3)
        self.workers_box = ttk.Spinbox(
            controls, from_=0, to=8, textvariable=self.workers_var, width=5, justify=tk.CENTER
        )
        self.workers_box.grid(row=1, column=9, sticky="w", pady=3)

        actions = ttk.Frame(controls)
        actions.grid(row=1, column=10, sticky="e", pady=3)
        self.save_session_button = ttk.Button(
            actions, text="Save Session", command=self._save_session, state=tk.DISABLED
        )
        self.save_session_button.pack(side=tk.LEFT, padx=(0, 6))
        self.save_button = ttk.Button(actions, text="Save Best", command=self._save_best, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)

        controls.columnconfigure(10, weight=1)

        body = ttk.Frame(self, padding=(16, 8, 16, 12))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Evolution", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Target", style="Title.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.evolution_label = ttk.Label(body, anchor=tk.CENTER, background="#151515")
        self.evolution_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0), padx=(0, 6))

        self.target_label = ttk.Label(body, anchor=tk.CENTER, background="#151515")
        self.target_label.grid(row=1, column=1, sticky="nsew", pady=(8, 0), padx=(6, 0))

        ttk.Label(
            self,
            textvariable=self.status_var,
            style="Status.TLabel",
            padding=(16, 0, 16, 14),
        ).pack(fill=tk.X)

        self._setting_widgets: tuple[tuple[tk.Widget, str], ...] = (
            (self.shape_box, "readonly"),
            (self.mode_box, "readonly"),
            (self.scale_box, "readonly"),
            (self.population_box, "normal"),
            (self.initial_shapes_box, "normal"),
            (self.max_shapes_box, "normal"),
            (self.backend_box, "readonly"),
            (self.workers_box, "normal"),
        )

    def _watch_settings(self) -> None:
        for variable in (
            self.shape_var,
            self.mode_var,
            self.scale_var,
            self.population_var,
            self.initial_shapes_var,
            self.max_shapes_var,
            self.backend_var,
            self.workers_var,
        ):
            variable.trace_add("write", self._settings_changed)

    def _settings_changed(self, *_args: object) -> None:
        if self._updating_settings or self.controller.is_running():
            return
        if not self.controller.has_resumable_state():
            return
        self.controller.reset_for_settings_change()
        self.best_image = None
        self.evolution_label.configure(image="", text="")
        self._apply_control_states()
        self.status_var.set("Settings changed. Ready to start a new evolution.")

    def _apply_control_states(self) -> None:
        states = compute_control_states(
            running=self.controller.is_running(),
            image_chosen=self.image_path is not None,
            has_session=self.controller.session is not None,
            has_resume=self.controller.resume_path is not None,
            has_best=self.best_image is not None,
            session_downsample=(
                self.controller.session.downsample if self.controller.session is not None else None
            ),
        )

        self.choose_button.configure(state=tk.NORMAL if states.choose_image else tk.DISABLED)
        self.open_session_button.configure(state=tk.NORMAL if states.open_session else tk.DISABLED)
        for widget, enabled_state in self._setting_widgets:
            widget.configure(state=enabled_state if states.settings else tk.DISABLED)
        if states.start_text is not None:
            self.start_button.configure(text=states.start_text)
        self.start_button.configure(state=tk.NORMAL if states.start_enabled else tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL if states.stop else tk.DISABLED)
        self.refine_button.configure(state=tk.NORMAL if states.refine else tk.DISABLED)
        self.save_button.configure(state=tk.NORMAL if states.save_best else tk.DISABLED)
        self.save_session_button.configure(state=tk.NORMAL if states.save_session else tk.DISABLED)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=(
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        self.image_path = Path(path).resolve()
        self.controller.reset_for_settings_change()
        self.best_image = None
        self._show_target(self.image_path)
        self.evolution_label.configure(image="", text="")
        self._apply_control_states()
        self.status_var.set(f"Ready: {self.image_path.name}")

    def _open_session(self) -> None:
        if self.controller.is_running():
            return
        path = filedialog.askopenfilename(
            title="Open evolution session",
            filetypes=(("Evolution session", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return

        run_id = self.controller.begin_run()
        self.status_var.set("Loading session...")
        threading.Thread(
            target=self.controller.load_session_preview,
            args=(run_id, Path(path).resolve()),
            daemon=True,
            name="session-loader",
        ).start()

    def _session_backend_label(self, config: Config) -> str:
        label = next(
            (label for label, value in BACKENDS.items() if value == config.renderer_backend),
            "Automatic",
        )
        if label not in self.backend_box.cget("values"):
            label = "Automatic"
        return label

    def _apply_session_preview(self, message: SessionPreview) -> None:
        state = message.state
        shape_label = next(label for label, value in SHAPES.items() if value == state.config.shape_mode)
        mode_label = next(label for label, value in MODES.items() if value == state.config.mode)
        scale_label = next((label for label, value in SCALES.items() if value == state.downsample), "Full")
        backend_label = self._session_backend_label(state.config)

        self._updating_settings = True
        try:
            self.shape_var.set(shape_label)
            self.mode_var.set(mode_label)
            self.scale_var.set(scale_label)
            self.population_var.set(state.config.pop_size)
            self.initial_shapes_var.set(state.config.nb_elements_initial)
            self.max_shapes_var.set(state.config.nb_elements_max)
            self.backend_var.set(backend_label)
            self.workers_var.set(state.config.max_workers)
        finally:
            self._updating_settings = False

        self.image_path = message.source
        self.best_image = message.rendered
        self._show_target(message.source)
        self._show_evolution(message.rendered)
        self._apply_control_states()
        self.status_var.set(
            f"Session loaded at iteration {state.iteration:,} | Best {state.best_fitness:.6f}"
        )

    def _current_settings(self) -> SessionSettings:
        if self.image_path is None:
            raise ValueError("Choose an image before starting.")
        population = int(self.population_var.get())
        initial_shapes = int(self.initial_shapes_var.get())
        max_shapes = int(self.max_shapes_var.get())
        if initial_shapes > max_shapes:
            raise ValueError("Initial shapes cannot exceed maximum shapes.")
        return SessionSettings(
            image_path=self.image_path,
            mode=MODES[self.mode_var.get()],
            shape_mode=SHAPES[self.shape_var.get()],
            downsample=SCALES[self.scale_var.get()],
            population=population,
            initial_shapes=initial_shapes,
            max_shapes=max_shapes,
            renderer_backend=BACKENDS[self.backend_var.get()],
            max_workers=int(self.workers_var.get()),
        )

    @staticmethod
    def _config_for(settings: SessionSettings) -> Config:
        elite_count = max(1, min(settings.population - 1, round(settings.population * 0.2)))
        config = Config(
            mode=settings.mode,  # type: ignore[arg-type]
            shape_mode=settings.shape_mode,  # type: ignore[arg-type]
            pop_size=settings.population,
            nb_elite=elite_count,
            nb_elements_initial=settings.initial_shapes,
            nb_elements_max=settings.max_shapes,
            min_shapes=min(5, settings.initial_shapes),
            hill_climb_interval=75,
            hill_climb_attempts=1,
            max_workers=settings.max_workers,
            renderer_backend=settings.renderer_backend,  # type: ignore[arg-type]
            use_compositing_cache=True,
            enable_logging=False,
        )
        config.validate()
        return config

    def _start_evolving(self, *, refine_to: int | None = None) -> None:
        if self.controller.is_running():
            return
        try:
            settings = self._current_settings()
            config = self._config_for(settings)
        except (KeyError, tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.controller.start_worker(settings, config, refine_to)
        self._apply_control_states()
        self.status_var.set("Preparing evolution...")

    def _poll_updates(self) -> None:
        try:
            for message in self.controller.drain():
                if isinstance(message, WorkerSessionReady):
                    self._updating_settings = True
                    try:
                        scale_label = next(
                            label for label, value in SCALES.items() if value == message.session.downsample
                        )
                        self.scale_var.set(scale_label)
                    finally:
                        self._updating_settings = False
                    continue
                if isinstance(message, WorkerError):
                    self.controller.request_stop()
                    messagebox.showerror("Evolution failed", str(message.error))
                    self.status_var.set("Evolution stopped after an error.")
                    continue
                if isinstance(message, WorkerFinished):
                    self._apply_control_states()
                    if not message.failed and self.controller.session is not None:
                        self.status_var.set(
                            f"Paused at iteration {self.controller.session.iteration:,} | "
                            f"Best {self.controller.session.global_best.fitness:.6f}"
                        )
                    continue
                if isinstance(message, SessionPreview):
                    self._apply_session_preview(message)
                    continue
                if isinstance(message, SessionPreviewFailed):
                    messagebox.showerror("Could not open session", str(message.error))
                    self.status_var.set("Session open failed.")
                    continue
                if isinstance(message, SessionNeedsSource):
                    selected = filedialog.askopenfilename(
                        title="Choose the original source image",
                        filetypes=(
                            ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                            ("All files", "*.*"),
                        ),
                    )
                    if selected:
                        run_id = self.controller.begin_run()
                        threading.Thread(
                            target=self.controller.load_session_preview,
                            args=(run_id, message.session_path, Path(selected).resolve()),
                            daemon=True,
                            name="session-loader",
                        ).start()
                    else:
                        self.status_var.set("Session open cancelled.")
                    continue
                if isinstance(message, WorkerUpdate):
                    self.best_image = message.rendered
                    self._show_evolution(message.rendered)
                    self.save_button.configure(state=tk.NORMAL)
                    self.status_var.set(
                        "Iteration {0.iteration:,} | Best {0.best_fitness:.6f} | "
                        "Current {0.current_fitness:.6f} | Shapes {0.shape_count}".format(message.snapshot)
                    )
                    continue
        finally:
            self.after(80, self._poll_updates)

    def _stop_evolving(self) -> None:
        if not self.controller.is_running():
            return
        self.controller.request_stop()
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping evolution...")

    def _refine(self) -> None:
        session = self.controller.session
        if session is None or self.controller.is_running() or session.downsample <= 1:
            return
        next_scale = NEXT_SCALE.get(session.downsample, session.downsample - 1)
        self._start_evolving(refine_to=next_scale)

    def _save_best(self) -> None:
        if self.best_image is None:
            return
        mode = "color" if self.best_image.ndim == 3 else "grayscale"
        path = filedialog.asksaveasfilename(
            title="Save best image",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            save_image(self.best_image, path, mode)
            self.status_var.set(f"Saved {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Could not save image", str(exc))

    def _save_session(self) -> None:
        session = self.controller.session
        if session is None or self.controller.is_running():
            return
        path = filedialog.asksaveasfilename(
            title="Save evolution session",
            defaultextension=".json",
            filetypes=(("Evolution session", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            session.save(path)
            self.status_var.set(f"Saved session {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Could not save session", str(exc))

    def _show_target(self, path: Path) -> None:
        with Image.open(path) as source:
            image = source.convert("RGB")
        self.target_photo = self._photo_for_panel(image)
        self.target_label.configure(image=self.target_photo, text="")

    def _show_evolution(self, array: np.ndarray) -> None:
        if array.ndim == 2:
            image = Image.fromarray(array, mode="L").convert("RGB")
        else:
            image = Image.fromarray(array, mode="RGB")
        self.evolution_photo = self._photo_for_panel(image)
        self.evolution_label.configure(image=self.evolution_photo, text="")

    @staticmethod
    def _photo_for_panel(image: Image.Image) -> ImageTk.PhotoImage:
        image = image.copy()
        image.thumbnail((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _on_close(self) -> None:
        session = self.controller.session
        if session is not None:
            answer = messagebox.askyesnocancel(
                "Save session?",
                "Save this evolution session before closing?",
            )
            if answer is None:
                return
            if answer:
                self.controller.request_stop()
                self.controller.join_worker(timeout=10.0)
                if self.controller.is_running():
                    messagebox.showwarning(
                        "Still running",
                        "The evolution thread did not stop in time; closing without saving.",
                    )
                else:
                    path = filedialog.asksaveasfilename(
                        title="Save evolution session",
                        defaultextension=".json",
                        filetypes=(("Evolution session", "*.json"), ("All files", "*.*")),
                    )
                    if not path:
                        return
                    try:
                        session.save(path)
                    except Exception as exc:
                        messagebox.showerror("Could not save session", str(exc))
                        return
        self.controller.request_stop()
        self.destroy()


def main() -> int:
    app = GeneticImageApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
