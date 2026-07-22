"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/progress_service.py

Run-progress helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from ..io.json_io import write_json
from ..workspaces.loader import WorkspaceContext

HEARTBEAT_INTERVAL_SECONDS = 5.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_run_id(*, kind: str, name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{kind}__{name}__{timestamp}_{uuid4().hex[:8]}"


@dataclass(slots=True)
class RunProgressRecorder:
    context: WorkspaceContext
    command: str
    run_id: str
    current_stage: str
    expected_steps: int = 0
    warnings: list[str] = field(default_factory=list)
    event_sink: Callable[[dict[str, object]], None] | None = None
    _started_at_value: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_at_value = _now()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._write_run(state="running", current_step=None)
        self._append_event("run_started", {"current_stage": self.current_stage})

    @property
    def run_dir(self) -> Path:
        return self.context.output_root / "runs" / self.run_id

    @property
    def staging_dir(self) -> Path:
        return self.run_dir / "staging"

    def step_started(self, *, current_step: str) -> None:
        self._write_run(state="running", current_step=current_step)
        self._append_event("step_started", {"current_step": current_step})

    def step_finished(self, *, current_step: str, status: str) -> None:
        self._write_run(state="running", current_step=current_step)
        self._append_event("step_finished", {"current_step": current_step, "status": status})

    def step_progress(self, *, current_step: str, message: str | None = None) -> None:
        self._write_run(state="running", current_step=current_step)
        payload: dict[str, object] = {"current_step": current_step}
        if message is not None:
            payload["message"] = message
        self._append_event("step_progress", payload)

    def heartbeat(self, *, current_step: str | None) -> None:
        self._write_run(state="running", current_step=current_step)
        self._append_event("heartbeat", {"current_step": current_step})

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        self._write_run(state="running", current_step=None)
        self._append_event("warning", {"message": message})

    def succeed(self) -> None:
        self._write_run(state="succeeded", current_step=None)
        self._append_event("run_succeeded", {})

    def fail(self, *, current_step: str | None, message: str) -> None:
        self.warnings.append(message)
        self._write_run(state="failed", current_step=current_step)
        self._append_event("run_failed", {"current_step": current_step, "message": message})

    def _write_run(self, *, state: str, current_step: str | None) -> None:
        write_json(
            self.run_dir / "run.json",
            {
                "run_id": self.run_id,
                "command": self.command,
                "workspace_id": self.context.workspace_id,
                "state": state,
                "started_at": self._started_at_value,
                "updated_at": _now(),
                "current_stage": self.current_stage,
                "current_step": current_step,
                "expected_steps": self.expected_steps,
                "warnings": list(self.warnings),
            },
        )

    def _append_event(self, event_type: str, payload: dict[str, object]) -> None:
        record = {
            "timestamp": _now(),
            "event_type": event_type,
            "run_id": self.run_id,
            "workspace_id": self.context.workspace_id,
            "command": self.command,
            "current_stage": self.current_stage,
            **payload,
        }
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=False))
            handle.write("\n")
        if self.event_sink is not None:
            self.event_sink(record)


def start_run_progress(
    context: WorkspaceContext,
    *,
    command: str,
    run_id: str,
    current_stage: str,
    expected_steps: int = 0,
    event_sink: Callable[[dict[str, object]], None] | None = None,
) -> RunProgressRecorder:
    return RunProgressRecorder(
        context=context,
        command=command,
        run_id=run_id,
        current_stage=current_stage,
        expected_steps=expected_steps,
        event_sink=event_sink,
    )


@contextmanager
def heartbeat_scope(progress: RunProgressRecorder, *, current_step: str):
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
            progress.heartbeat(current_step=current_step)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(HEARTBEAT_INTERVAL_SECONDS, 0.1))
