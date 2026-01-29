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
import os
import threading
import time
from typing import Optional, TextIO

from ...core.score_tiers import score_tier_counts
from ...utils import logging_utils
from ...utils.rich_style import make_panel, make_table

log = logging.getLogger(__name__)
_STAGE_A_LIVE = None
_STAGE_A_LIVE_MODE: str | None = None
_STAGE_A_LIVE_CONSOLE = None
_STAGE_A_LIVE_STATE: dict[str, dict[str, object]] = {}
_STAGE_A_LIVE_LOCK = threading.Lock()


def _format_rate(rate: float) -> str:
    if rate >= 1000.0:
        return f"{rate / 1000.0:.1f}k/s"
    return f"{rate:.1f}/s"


def _format_tier_yield(eligible_unique: int) -> str:
    if eligible_unique <= 0:
        return "-"
    n0, n1, n2, _n3 = score_tier_counts(int(eligible_unique))
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
    generated: int,
    target: int,
    accepted: Optional[int],
    accepted_target: Optional[int],
    batch_index: Optional[int],
    batch_total: Optional[int],
    elapsed: float,
    target_fraction: Optional[float],
    tier_yield: Optional[str],
) -> str:
    safe_target = max(1, int(target))
    gen_pct = min(100, int(100 * generated / safe_target))
    gen_bar = _format_progress_bar(int(generated), int(safe_target))
    parts = [f"PWM {motif_id}", backend, f"gen {gen_pct}% {gen_bar} ({generated}/{safe_target})"]
    if accepted is not None:
        if accepted_target is not None and int(accepted_target) > 0:
            acc_pct = min(100, int(100 * int(accepted) / max(1, int(accepted_target))))
            parts.append(f"eligible {acc_pct}% ({accepted}/{accepted_target})")
        else:
            parts.append(f"eligible {accepted}")
    if target_fraction is not None:
        parts.append(f"tier {float(target_fraction) * 100:.3f}%")
    if tier_yield:
        parts.append(f"tiers 0.1/1/9={tier_yield}")
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


def _stage_a_live_render(state: dict[str, dict[str, object]]):
    table = make_table(show_header=True, pad_edge=False)
    table.add_column("motif")
    table.add_column("generated", no_wrap=True, overflow="ellipsis", min_width=14)
    table.add_column("progress", no_wrap=True, overflow="ellipsis", min_width=12)
    table.add_column("eligible", no_wrap=True, overflow="ellipsis", min_width=12)
    table.add_column("tier target")
    table.add_column("tier yield (0.1/1/9%)")
    table.add_column("elapsed")
    table.add_column("rate")
    for key in sorted(state, key=lambda k: str(state[k].get("motif"))):
        row = state[key]
        generated = int(row.get("generated", 0))
        target = max(1, int(row.get("target", 1)))
        accepted = row.get("accepted")
        accepted_target = row.get("accepted_target")
        target_fraction = row.get("target_fraction")
        elapsed = float(row.get("elapsed", 0.0))
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
        tier_label = "-"
        if target_fraction is not None:
            tier_label = f"{float(target_fraction) * 100:.3f}%"
        tier_yield = _format_tier_yield(int(accepted)) if accepted is not None else "-"
        elapsed_label = f"{max(0.0, float(elapsed)):.1f}s"
        rate = generated / elapsed if elapsed > 0 else 0.0
        rate_label = _format_rate(rate)
        table.add_row(
            str(row.get("motif", "-")),
            gen_label,
            progress_label,
            eligible_label,
            tier_label,
            tier_yield,
            elapsed_label,
            rate_label,
        )
    return make_panel(table, title="Stage-A mining")


def _stage_a_live_start(stream: TextIO):
    from rich.console import Console
    from rich.live import Live

    pixi_in_shell = bool(os.environ.get("PIXI_IN_SHELL"))
    tty = bool(getattr(stream, "isatty", lambda: False)())
    if not tty or pixi_in_shell:
        return None, None
    console = Console(file=stream)
    if not console.is_terminal:
        return None, None
    live = Live(console=console, refresh_per_second=4, transient=False)
    live.start()
    return live, console


def _stage_a_live_register(
    *,
    key: str,
    motif_id: str,
    target: int,
    accepted_target: Optional[int],
    target_fraction: Optional[float],
    stream: TextIO,
) -> bool:
    global _STAGE_A_LIVE
    global _STAGE_A_LIVE_MODE
    with _STAGE_A_LIVE_LOCK:
        if _STAGE_A_LIVE_MODE is None:
            _STAGE_A_LIVE, _STAGE_A_LIVE_CONSOLE = _stage_a_live_start(stream)
            if _STAGE_A_LIVE is None:
                _STAGE_A_LIVE_CONSOLE = None
                return False
            _STAGE_A_LIVE_MODE = "live"
        _STAGE_A_LIVE_STATE[key] = {
            "motif": motif_id,
            "generated": 0,
            "target": int(target),
            "accepted": None,
            "accepted_target": int(accepted_target) if accepted_target is not None else None,
            "target_fraction": target_fraction,
            "elapsed": 0.0,
        }
        return True


def _stage_a_live_update(
    *,
    key: str,
    generated: int,
    accepted: Optional[int],
    elapsed: float,
    target: Optional[int] = None,
    accepted_target: Optional[int] = None,
) -> None:
    with _STAGE_A_LIVE_LOCK:
        row = _STAGE_A_LIVE_STATE.get(key)
        if row is None:
            return
        row["generated"] = int(generated)
        row["accepted"] = int(accepted) if accepted is not None else None
        row["elapsed"] = float(elapsed)
        if target is not None:
            row["target"] = int(target)
        if accepted_target is not None:
            row["accepted_target"] = int(accepted_target)
        if _STAGE_A_LIVE is None:
            return
        renderable = _stage_a_live_render(_STAGE_A_LIVE_STATE)
        _STAGE_A_LIVE.update(renderable, refresh=True)


def _stage_a_live_finish(*, key: str) -> None:
    global _STAGE_A_LIVE
    global _STAGE_A_LIVE_MODE
    with _STAGE_A_LIVE_LOCK:
        _STAGE_A_LIVE_STATE.pop(key, None)
        if not _STAGE_A_LIVE_STATE:
            if _STAGE_A_LIVE is not None:
                _STAGE_A_LIVE.stop()
            _STAGE_A_LIVE = None
            _STAGE_A_LIVE_CONSOLE = None
            _STAGE_A_LIVE_MODE = None


class _PwmSamplingProgress:
    def __init__(
        self,
        *,
        motif_id: str,
        backend: str,
        target: int,
        accepted_target: Optional[int],
        stream: TextIO,
        target_fraction: Optional[float] = None,
        min_interval: float = 0.2,
    ) -> None:
        self.motif_id = motif_id
        self.backend = backend
        self.target = int(target)
        self.accepted_target = accepted_target
        self.stream = stream
        self.target_fraction = target_fraction
        self.min_interval = float(min_interval)
        self._mode = str(logging_utils.get_progress_style())
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        pixi_in_shell = bool(os.environ.get("PIXI_IN_SHELL"))
        self._enabled = bool(logging_utils.is_progress_enabled())
        self._use_live = self._enabled and self._mode == "screen" and tty and not pixi_in_shell
        self._use_table = False
        self._allow_carriage = self._mode == "screen" and tty and not pixi_in_shell
        self._live_key = f"{self.motif_id}:{id(self)}"
        self._start = time.monotonic()
        self._last_update = self._start
        self._last_len = 0
        self._last_state: tuple[int, Optional[int], Optional[int], Optional[int], int, Optional[int]] | None = None
        self._shown = False
        if self._enabled:
            logging_utils.set_progress_active(True)
        if self._use_live:
            self._use_live = _stage_a_live_register(
                key=self._live_key,
                motif_id=self.motif_id,
                target=int(self.target),
                accepted_target=self.accepted_target,
                target_fraction=self.target_fraction,
                stream=self.stream,
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
        )
        if self._shown and state == self._last_state:
            if self._mode != "screen" or logging_utils.is_progress_line_visible():
                self._last_update = now
                return
        if self._use_live:
            _stage_a_live_update(
                key=self._live_key,
                generated=int(generated),
                accepted=accepted,
                elapsed=now - self._start,
                target=int(self.target),
                accepted_target=self.accepted_target,
            )
        else:
            line = _format_pwm_progress_line(
                motif_id=self.motif_id,
                backend=self.backend,
                generated=int(generated),
                target=int(self.target),
                accepted=accepted,
                accepted_target=self.accepted_target,
                batch_index=batch_index,
                batch_total=batch_total,
                elapsed=now - self._start,
                target_fraction=self.target_fraction,
                tier_yield=_format_tier_yield(int(accepted)) if accepted is not None else None,
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

    def finish(self) -> None:
        if self._enabled:
            logging_utils.set_progress_active(False)
        if self._use_live:
            _stage_a_live_finish(key=self._live_key)
        if not self._shown:
            return
        if not self._use_live and self._allow_carriage:
            self.stream.write("\n")
            self.stream.flush()
