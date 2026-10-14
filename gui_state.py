"""Display-independent state and messaging for the desktop app.

This module deliberately avoids tkinter so the worker/UI protocol and the
control-state logic can be unit-tested headlessly; app.py owns all widgets.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.config import Config
from core.evolver import EvolutionSession, EvolutionSnapshot
from core.io import EvolutionState, file_sha256, read_state
from core.renderer import render_individual


@dataclass(frozen=True)
class SessionSettings:
    image_path: Path
    mode: str
    shape_mode: str
    downsample: int
    population: int
    initial_shapes: int
    max_shapes: int
    renderer_backend: str
    max_workers: int


@dataclass
class WorkerSessionReady:
    run_id: int
    session: EvolutionSession
    settings: SessionSettings


@dataclass
class WorkerUpdate:
    run_id: int
    snapshot: EvolutionSnapshot
    rendered: np.ndarray


@dataclass
class WorkerFinished:
    run_id: int
    failed: bool


@dataclass
class WorkerError:
    run_id: int
    error: Exception


@dataclass
class SessionPreview:
    run_id: int
    state: EvolutionState
    source: Path
    session_path: Path
    rendered: np.ndarray


@dataclass
class SessionNeedsSource:
    run_id: int
    session_path: Path


@dataclass
class SessionPreviewFailed:
    run_id: int
    error: Exception


WorkerMessage = (
    WorkerUpdate
    | WorkerSessionReady
    | WorkerFinished
    | WorkerError
    | SessionPreview
    | SessionNeedsSource
    | SessionPreviewFailed
)


@dataclass(frozen=True)
class ControlStates:
    choose_image: bool
    open_session: bool
    settings: bool
    start_enabled: bool
    start_text: str | None
    stop: bool
    refine: bool
    save_best: bool
    save_session: bool


def compute_control_states(
    *,
    running: bool,
    image_chosen: bool,
    has_session: bool,
    has_resume: bool,
    has_best: bool,
    session_downsample: int | None,
) -> ControlStates:
    """The button state machine as a pure function of the app state."""
    if running:
        return ControlStates(
            choose_image=False,
            open_session=False,
            settings=False,
            start_enabled=False,
            start_text=None,
            stop=True,
            refine=False,
            save_best=False,
            save_session=False,
        )
    start_text = "Continue" if has_session else ("Resume" if has_resume else "Start")
    can_refine = has_session and session_downsample is not None and session_downsample > 1
    return ControlStates(
        choose_image=True,
        open_session=True,
        settings=True,
        start_enabled=image_chosen,
        start_text=start_text,
        stop=False,
        refine=can_refine,
        save_best=has_best,
        save_session=has_session,
    )


@dataclass
class SessionFactory:
    """Creates engine sessions for a run; stubbed out in tests."""

    def create(self, settings: SessionSettings, config: Config) -> EvolutionSession:
        return EvolutionSession(str(settings.image_path), config, downsample=settings.downsample)

    def resume(self, resume_path: str, settings: SessionSettings) -> EvolutionSession:
        return EvolutionSession.from_state(
            resume_path,
            image_path=str(settings.image_path),
            runtime_overrides={
                "renderer_backend": settings.renderer_backend,
                "max_workers": settings.max_workers,
                "enable_logging": False,
            },
        )


class EvolutionController:
    """Owns the run state shared between the UI thread and evolution workers."""

    def __init__(self) -> None:
        self.updates: queue.Queue[WorkerMessage] = queue.Queue()
        self.stop_event = threading.Event()
        self.run_id = 0
        self.worker: threading.Thread | None = None
        self.session: EvolutionSession | None = None
        self.session_settings: SessionSettings | None = None
        self.resume_path: Path | None = None

    def begin_run(self) -> int:
        self.run_id += 1
        self.stop_event.clear()
        return self.run_id

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def request_stop(self) -> None:
        self.stop_event.set()

    def join_worker(self, timeout: float) -> None:
        if self.worker is not None:
            self.worker.join(timeout=timeout)

    def start_worker(
        self,
        settings: SessionSettings,
        config: Config,
        refine_to: int | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        existing_session = self.session if self.session_settings == settings else None
        resume_path = self.resume_path if existing_session is None else None
        run_id = self.begin_run()
        self.worker = threading.Thread(
            target=self.run_worker,
            args=(run_id, settings, config, existing_session, resume_path, refine_to, session_factory),
            daemon=True,
            name="evolution-worker",
        )
        self.worker.start()

    def run_worker(
        self,
        run_id: int,
        settings: SessionSettings,
        config: Config,
        existing_session: EvolutionSession | None,
        resume_path: Path | None,
        refine_to: int | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        factory = session_factory or SessionFactory()
        failed = False
        try:
            if existing_session is not None:
                session = existing_session
            elif resume_path is not None:
                session = factory.resume(str(resume_path), settings)
            else:
                session = factory.create(settings, config)

            if refine_to is not None and refine_to < session.downsample:
                session.refine_resolution(refine_to)
                settings = replace(settings, downsample=refine_to)

            self.updates.put(WorkerSessionReady(run_id=run_id, session=session, settings=settings))
            last_render = 0.0
            latest_snapshot: EvolutionSnapshot | None = None
            while not self.stop_event.is_set():
                snapshot = session.step_many(4)
                latest_snapshot = snapshot
                now = time.monotonic()
                if snapshot.improved or now - last_render >= 0.25:
                    self.updates.put(
                        WorkerUpdate(run_id=run_id, snapshot=snapshot, rendered=session.render_best())
                    )
                    last_render = now

            if latest_snapshot is None:
                latest_snapshot = EvolutionSnapshot(
                    iteration=session.iteration,
                    best_fitness=session.global_best.fitness,
                    current_fitness=session.ranked[0][1],
                    shape_count=len(session.global_best.shapes),
                    improved=False,
                )
            self.updates.put(
                WorkerUpdate(
                    run_id=run_id,
                    snapshot=latest_snapshot,
                    rendered=session.render_best(),
                )
            )
        except Exception as exc:
            failed = True
            self.updates.put(WorkerError(run_id=run_id, error=exc))
        finally:
            self.updates.put(WorkerFinished(run_id=run_id, failed=failed))

    def load_session_preview(
        self,
        run_id: int,
        session_path: Path,
        source_override: Path | None = None,
        renderer_backend: str = "auto",
    ) -> None:
        try:
            state = read_state(str(session_path))
            if source_override is not None:
                source = source_override
            elif state.source_image:
                source = Path(state.source_image).expanduser()
            else:
                source = None
            if source is None or not source.is_file():
                self.updates.put(SessionNeedsSource(run_id=run_id, session_path=session_path))
                return
            source = source.resolve()
            if state.source_sha256 and file_sha256(source) != state.source_sha256:
                raise ValueError("That image does not match the source saved with this session.")
            preview_config = replace(state.config, renderer_backend=renderer_backend)
            rendered = render_individual(state.best, preview_config, state.background)
            self.updates.put(
                SessionPreview(
                    run_id=run_id,
                    state=state,
                    source=source,
                    session_path=session_path,
                    rendered=rendered,
                )
            )
        except Exception as exc:
            self.updates.put(SessionPreviewFailed(run_id=run_id, error=exc))

    def drain(self) -> list[WorkerMessage]:
        """Return current-run messages in order; stale messages are dropped."""
        messages: list[WorkerMessage] = []
        while True:
            try:
                message = self.updates.get_nowait()
            except queue.Empty:
                return messages
            if message.run_id != self.run_id:
                continue
            self._absorb(message)
            messages.append(message)

    def _absorb(self, message: WorkerMessage) -> None:
        if isinstance(message, WorkerSessionReady):
            self.session = message.session
            self.session_settings = message.settings
            self.resume_path = None
        elif isinstance(message, SessionPreview):
            self.session = None
            self.session_settings = None
            self.resume_path = message.session_path
        elif isinstance(message, WorkerFinished):
            self.worker = None

    def reset_for_settings_change(self) -> None:
        self.session = None
        self.session_settings = None
        self.resume_path = None

    def has_resumable_state(self) -> bool:
        return self.session is not None or self.resume_path is not None
