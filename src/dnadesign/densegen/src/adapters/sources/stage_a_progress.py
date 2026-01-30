"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_progress.py

Stage-A PWM mining progress rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, TextIO

from ...core.score_tiers import score_tier_counts
from ...utils import logging_utils
from ...utils.rich_style import make_panel, make_table

log = logging.getLogger(__name__)


@dataclass
class StageAProgressState:
    motif_id: str
    backend: str
    phase: Optional[str]
    generated: int
    target: int
    accepted: Optional[int]
    accepted_target: Optional[int]
    target_fraction: Optional[float]
    tier_fractions: Optional[list[float]]
    elapsed: float
    batch_index: Optional[int]
    batch_total: Optional[int]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if int(self.target) <= 0:
            raise ValueError("Stage-A progress target must be > 0.")
        if int(self.generated) < 0:
            raise ValueError("Stage-A progress generated must be >= 0.")
        if self.accepted is not None and int(self.accepted) < 0:
            raise ValueError("Stage-A progress accepted must be >= 0.")
        if self.accepted_target is not None and int(self.accepted_target) < 0:
            raise ValueError("Stage-A progress accepted_target must be >= 0.")
        if float(self.elapsed) < 0.0:
            raise ValueError("Stage-A progress elapsed must be >= 0.")
        if self.batch_index is not None and int(self.batch_index) < 0:
            raise ValueError("Stage-A progress batch_index must be >= 0.")
        if self.batch_total is not None and int(self.batch_total) < 0:
            raise ValueError("Stage-A progress batch_total must be >= 0.")
        if self.target_fraction is not None and not (0.0 < float(self.target_fraction) <= 1.0):
            raise ValueError("Stage-A target_fraction must be in (0, 1].")


def _format_rate(rate: float) -> str:
    if rate >= 1000.0:
        return f"{rate / 1000.0:.1f}k/s"
    return f"{rate:.1f}/s"


def _format_tier_fraction_label(fractions: Sequence[float] | None) -> str:
    if not fractions:
        return ""
    parts = []
    for frac in fractions:
        pct = float(frac) * 100.0
        if pct.is_integer():
            parts.append(str(int(pct)))
        else:
            parts.append(str(round(pct, 3)).rstrip("0").rstrip("."))
    return "/".join(parts)


def _format_tier_yield(eligible_unique: int, *, fractions: Sequence[float] | None = None) -> str:
    if eligible_unique <= 0:
        return "-"
    n0, n1, n2, _n3 = score_tier_counts(int(eligible_unique), fractions=fractions)
    return f"{n0}/{n1}/{n2}"


def _format_progress_bar(current: int, total: int, *, width: int = 12) -> str:
    if total <= 0:
        return "[?]"
    filled = int(width * (current / max(1, total)))
    filled = min(width, max(0, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_pwm_progress_line(
    *,
    motif_id: str,
    backend: str,
    phase: Optional[str],
    generated: int,
    target: int,
    accepted: Optional[int],
    accepted_target: Optional[int],
    batch_index: Optional[int],
    batch_total: Optional[int],
    elapsed: float,
    target_fraction: Optional[float],
    tier_fractions: Sequence[float] | None,
    tier_yield: Optional[str],
) -> str:
    safe_target = max(1, int(target))
    gen_pct = min(100, int(100 * generated / safe_target))
    gen_bar = _format_progress_bar(int(generated), int(safe_target))
    parts = [f"PWM {motif_id}", backend]
    if phase:
        parts.append(f"phase {phase}")
    parts.append(f"gen {gen_pct}% {gen_bar} ({generated}/{safe_target})")
    if accepted is not None:
        if accepted_target is not None and int(accepted_target) > 0:
            acc_pct = min(100, int(100 * int(accepted) / max(1, int(accepted_target))))
            parts.append(f"eligible_unique {acc_pct}% ({accepted}/{accepted_target})")
        else:
            parts.append(f"eligible_unique {accepted}")
    if target_fraction is not None:
        parts.append(f"tier {float(target_fraction) * 100:.3f}%")
    if tier_yield:
        tier_label = _format_tier_fraction_label(tier_fractions)
        if tier_label:
            parts.append(f"tiers {tier_label}={tier_yield}")
        else:
            parts.append(f"tiers {tier_yield}")
    if batch_index is not None:
        total_label = "-" if batch_total is None else str(int(batch_total))
        parts.append(f"batch {int(batch_index)}/{total_label}")
    elapsed_label = f"{max(0.0, float(elapsed)):.1f}s"
    rate = generated / elapsed if elapsed > 0 else 0.0
    parts.append(elapsed_label)
    parts.append(_format_rate(rate))
    return " | ".join(parts)


def _format_stage_a_milestone(
    *,
    motif_id: str,
    phase: str,
    detail: Optional[str] = None,
    elapsed: Optional[float] = None,
) -> str:
    message = f"Stage-A {phase} for {motif_id}"
    suffix = []
    if detail:
        suffix.append(str(detail))
    if elapsed is not None:
        suffix.append(f"elapsed={float(elapsed):.1f}s")
    if suffix:
        return message + " | " + " | ".join(suffix)
    return message


def _stage_a_live_render(state: dict[str, StageAProgressState]):
    table = make_table(show_header=True, pad_edge=False)
    table.add_column("motif")
    table.add_column("backend")
    table.add_column("phase")
    table.add_column("generated/target", no_wrap=True, overflow="ellipsis", min_width=14)
    table.add_column("gen %", no_wrap=True, overflow="ellipsis", min_width=12)
    table.add_column("eligible_unique/target", no_wrap=True, overflow="ellipsis", min_width=18)
    table.add_column("tier target")
    table.add_column("tier yield")
    table.add_column("batch")
    table.add_column("elapsed")
    table.add_column("rate")
    for key in sorted(state, key=lambda k: str(state[k].motif_id)):
        row = state[key]
        generated = int(row.generated)
        target = max(1, int(row.target))
        accepted = row.accepted
        accepted_target = row.accepted_target
        target_fraction = row.target_fraction
        elapsed = float(row.elapsed)
        batch_index = row.batch_index
        batch_total = row.batch_total
        gen_pct = min(100, int(100 * generated / target))
        gen_bar = _format_progress_bar(int(generated), int(target))
        gen_label = f"{generated}/{target}"
        progress_label = f"{gen_bar} {gen_pct}%"
        if accepted is None:
            eligible_label = "-"
        elif accepted_target is not None and int(accepted_target) > 0:
            acc_pct = min(100, int(100 * int(accepted) / max(1, int(accepted_target))))
            eligible_label = f"{accepted}/{accepted_target} ({acc_pct}%)"
        else:
            eligible_label = str(accepted)
        tier_target_label = "-"
        if target_fraction is not None:
            tier_target_label = f"{float(target_fraction) * 100:.3f}%"
        tier_fractions = row.tier_fractions
        tier_yield = _format_tier_yield(int(accepted), fractions=tier_fractions) if accepted is not None else "-"
        tier_label = _format_tier_fraction_label(tier_fractions)
        if tier_label and tier_yield != "-":
            tier_yield = f"{tier_label}={tier_yield}"
        elapsed_label = f"{max(0.0, float(elapsed)):.1f}s"
        rate = generated / elapsed if elapsed > 0 else 0.0
        rate_label = _format_rate(rate)
        batch_label = "-"
        if batch_index is not None:
            total_label = "-" if batch_total is None else str(int(batch_total))
            batch_label = f"{int(batch_index)}/{total_label}"
        table.add_row(
            str(row.motif_id),
            str(row.backend),
            str(row.phase or "-"),
            gen_label,
            progress_label,
            eligible_label,
            tier_target_label,
            tier_yield,
            batch_label,
            elapsed_label,
            rate_label,
        )
    return make_panel(table, title="Stage-A mining")


def _stage_a_live_start(stream: TextIO):
    import shutil

    from rich.console import Console
    from rich.live import Live

    tty = bool(getattr(stream, "isatty", lambda: False)())
    if not tty:
        return None, None
    width = shutil.get_terminal_size((120, 24)).columns
    console = Console(file=stream, force_terminal=tty, width=width)
    if not console.is_terminal:
        return None, None
    live = Live(console=console, refresh_per_second=4, transient=False)
    live.start()
    return live, console


class StageAProgressManager:
    def __init__(self, *, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._state: dict[str, StageAProgressState] = {}
        self._live = None
        self._console = None

    def _ensure_live(self) -> bool:
        if self._live is not None:
            return True
        self._live, self._console = _stage_a_live_start(self._stream)
        return self._live is not None

    def register(
        self,
        *,
        key: str,
        motif_id: str,
        backend: str,
        phase: Optional[str],
        target: int,
        accepted_target: Optional[int],
        target_fraction: Optional[float],
        tier_fractions: Sequence[float] | None,
    ) -> bool:
        with self._lock:
            if not self._ensure_live():
                return False
            self._state[key] = StageAProgressState(
                motif_id=str(motif_id),
                backend=str(backend),
                phase=phase,
                generated=0,
                target=int(target),
                accepted=None,
                accepted_target=int(accepted_target) if accepted_target is not None else None,
                target_fraction=float(target_fraction) if target_fraction is not None else None,
                tier_fractions=list(tier_fractions) if tier_fractions is not None else None,
                elapsed=0.0,
                batch_index=None,
                batch_total=None,
            )
            return True

    def update(
        self,
        *,
        key: str,
        generated: int,
        accepted: Optional[int],
        elapsed: float,
        target: Optional[int] = None,
        accepted_target: Optional[int] = None,
        batch_index: Optional[int] = None,
        batch_total: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> None:
        with self._lock:
            row = self._state.get(key)
            if row is None:
                return
            row.generated = int(generated)
            row.accepted = int(accepted) if accepted is not None else None
            row.elapsed = float(elapsed)
            if phase is not None:
                row.phase = phase
            if target is not None:
                row.target = int(target)
            if accepted_target is not None:
                row.accepted_target = int(accepted_target)
            if batch_index is not None:
                row.batch_index = int(batch_index)
            if batch_total is not None:
                row.batch_total = int(batch_total)
            row.validate()
            if self._live is None:
                return
            renderable = _stage_a_live_render(self._state)
            self._live.update(renderable, refresh=True)

    def set_phase(self, *, key: str, phase: Optional[str]) -> None:
        with self._lock:
            row = self._state.get(key)
            if row is None:
                return
            row.phase = phase
            if self._live is None:
                return
            renderable = _stage_a_live_render(self._state)
            self._live.update(renderable, refresh=True)

    def finish(self, *, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)
            if not self._state:
                if self._live is not None:
                    self._live.stop()
                self._live = None
                self._console = None


class _PwmSamplingProgress:
    def __init__(
        self,
        *,
        motif_id: str,
        backend: str,
        target: int,
        accepted_target: Optional[int],
        stream: TextIO,
        tier_fractions: Sequence[float] | None = None,
        manager: StageAProgressManager | None = None,
        target_fraction: Optional[float] = None,
        min_interval: float = 0.2,
    ) -> None:
        self.motif_id = motif_id
        self.backend = backend
        self.target = int(target)
        self.accepted_target = accepted_target
        self.stream = stream
        self._tier_fractions = list(tier_fractions) if tier_fractions is not None else None
        self._manager = manager
        self.target_fraction = target_fraction
        self.min_interval = float(min_interval)
        self._mode = str(logging_utils.get_progress_style())
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._enabled = bool(logging_utils.is_progress_enabled())
        interactive = tty
        self._use_live = self._enabled and self._mode == "screen" and interactive
        self._use_table = False
        self._allow_carriage = interactive
        self._live_key = f"{self.motif_id}:{id(self)}"
        self._phase: Optional[str] = "mining"
        self._start = time.monotonic()
        self._last_update = self._start
        self._last_len = 0
        self._last_state: (
            tuple[
                int,
                Optional[int],
                Optional[int],
                Optional[int],
                int,
                Optional[int],
                Optional[str],
            ]
            | None
        ) = None
        self._shown = False
        if self._enabled:
            logging_utils.set_progress_active(True)
        if self._use_live:
            if self._manager is None:
                self._manager = StageAProgressManager(stream=self.stream)
            self._use_live = self._manager.register(
                key=self._live_key,
                motif_id=self.motif_id,
                backend=self.backend,
                phase=self._phase,
                target=int(self.target),
                accepted_target=self.accepted_target,
                target_fraction=self.target_fraction,
                tier_fractions=self._tier_fractions,
            )

    def update(
        self,
        *,
        generated: int,
        accepted: Optional[int],
        batch_index: Optional[int] = None,
        batch_total: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and self._shown and (now - self._last_update) < float(self.min_interval):
            return
        state = (
            int(generated),
            int(accepted) if accepted is not None else None,
            batch_index,
            batch_total,
            int(self.target),
            int(self.accepted_target) if self.accepted_target is not None else None,
            self._phase,
        )
        if self._shown and state == self._last_state:
            if self._mode != "screen" or logging_utils.is_progress_line_visible():
                self._last_update = now
                return
        if self._use_live:
            if self._manager is None:
                raise RuntimeError("Stage-A progress manager missing for live update.")
            self._manager.update(
                key=self._live_key,
                generated=int(generated),
                accepted=accepted,
                elapsed=now - self._start,
                phase=self._phase,
                target=int(self.target),
                accepted_target=self.accepted_target,
                batch_index=batch_index,
                batch_total=batch_total,
            )
        else:
            line = _format_pwm_progress_line(
                motif_id=self.motif_id,
                backend=self.backend,
                phase=self._phase,
                generated=int(generated),
                target=int(self.target),
                accepted=accepted,
                accepted_target=self.accepted_target,
                batch_index=batch_index,
                batch_total=batch_total,
                elapsed=now - self._start,
                target_fraction=self.target_fraction,
                tier_fractions=self._tier_fractions,
                tier_yield=_format_tier_yield(
                    int(accepted),
                    fractions=self._tier_fractions,
                )
                if accepted is not None
                else None,
            )
            if self._allow_carriage:
                padded = line.ljust(self._last_len)
                self._last_len = max(self._last_len, len(line))
                self.stream.write(f"\r{padded}")
                self.stream.flush()
                logging_utils.mark_progress_line_visible()
            else:
                self.stream.write(f"{line}\n")
                self.stream.flush()
        self._last_update = now
        self._last_state = state
        self._shown = True

    def set_phase(self, phase: Optional[str]) -> None:
        self._phase = phase
        if not self._use_live:
            if self._last_state is None:
                self.update(generated=0, accepted=None, force=True)
                return
            generated, accepted, batch_index, batch_total, _target, _accepted_target, _phase = self._last_state
            self.update(
                generated=int(generated),
                accepted=accepted,
                batch_index=batch_index,
                batch_total=batch_total,
                force=True,
            )
            return
        if self._manager is None:
            raise RuntimeError("Stage-A progress manager missing for live update.")
        self._manager.set_phase(key=self._live_key, phase=phase)

    def finish(self) -> None:
        if self._enabled:
            logging_utils.set_progress_active(False)
        if self._use_live:
            if self._manager is None:
                raise RuntimeError("Stage-A progress manager missing for live finish.")
            self._manager.finish(key=self._live_key)
        if not self._shown:
            return
        if not self._use_live and self._allow_carriage:
            self.stream.write("\n")
            self.stream.flush()
