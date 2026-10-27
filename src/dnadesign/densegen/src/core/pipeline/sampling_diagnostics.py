"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/sampling_diagnostics.py

Sampling diagnostics helpers for Stage-B leaderboards and failure tracking.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .progress import (
    _leaderboard_snapshot,
    _summarize_diversity,
    _summarize_failure_leaderboard,
    _summarize_failure_totals,
    _summarize_leaderboard,
    _summarize_tfbs_usage_stats,
)


@dataclass
class SamplingDiagnostics:
    usage_counts: dict[tuple[str, str], int]
    tf_usage_counts: dict[str, int]
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]]
    source_label: str
    plan_name: str
    display_tf_label: Callable[[str], str]
    progress_style: str
    show_tfbs: bool
    track_failures: bool
    logger: logging.Logger
    library_tfs: list[str]
    library_tfbs: list[str]
    library_site_ids: list[str | None]

    def update_library(
        self,
        *,
        library_tfs: list[str],
        library_tfbs: list[str],
        library_site_ids: list[str | None],
    ) -> None:
        self.library_tfs = list(library_tfs)
        self.library_tfbs = list(library_tfbs)
        self.library_site_ids = list(library_site_ids)

    def snapshot(self) -> dict[str, object]:
        return _leaderboard_snapshot(
            self.usage_counts,
            self.tf_usage_counts,
            self.failure_counts,
            input_name=self.source_label,
            plan_name=self.plan_name,
            library_tfs=self.library_tfs,
            library_tfbs=self.library_tfbs,
        )

    def log_snapshot(self) -> None:
        if self.progress_style != "stream":
            return
        tf_usage_display = self.map_tf_usage(self.tf_usage_counts)
        tfbs_usage_display = self.map_tfbs_usage(self.usage_counts) if self.show_tfbs else self.usage_counts
        self.logger.info(
            "[%s/%s] Leaderboard (TF): %s",
            self.source_label,
            self.plan_name,
            _summarize_leaderboard(tf_usage_display, top=5),
        )
        if self.show_tfbs:
            self.logger.info(
                "[%s/%s] Leaderboard (TFBS): %s",
                self.source_label,
                self.plan_name,
                _summarize_leaderboard(tfbs_usage_display, top=5),
            )
            self.logger.info(
                "[%s/%s] Failed TFBS: %s",
                self.source_label,
                self.plan_name,
                _summarize_failure_leaderboard(
                    self.failure_counts,
                    input_name=self.source_label,
                    plan_name=self.plan_name,
                    top=5,
                ),
            )
        else:
            self.logger.info(
                "[%s/%s] TFBS usage: %s",
                self.source_label,
                self.plan_name,
                _summarize_tfbs_usage_stats(self.usage_counts),
            )
            failure_totals = _summarize_failure_totals(
                self.failure_counts,
                input_name=self.source_label,
                plan_name=self.plan_name,
            )
            if failure_totals:
                self.logger.info("[%s/%s] Failures: %s", self.source_label, self.plan_name, failure_totals)
        self.logger.info(
            "[%s/%s] Diversity: %s",
            self.source_label,
            self.plan_name,
            _summarize_diversity(
                self.usage_counts,
                self.tf_usage_counts,
                library_tfs=self.library_tfs,
                library_tfbs=self.library_tfbs,
            ),
        )

    def record_site_failures(self, reason: str) -> None:
        if not self.track_failures:
            return
        if not self.library_tfbs:
            return
        for idx, tfbs in enumerate(self.library_tfbs):
            tf = self.library_tfs[idx] if idx < len(self.library_tfs) else ""
            site_id = None
            if self.library_site_ids and idx < len(self.library_site_ids):
                raw = self.library_site_ids[idx]
                if raw not in (None, "", "None"):
                    site_id = str(raw)
            key = (self.source_label, self.plan_name, tf, tfbs, site_id)
            reasons = self.failure_counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1

    def map_tf_usage(self, counts: dict[str, int]) -> dict[str, int]:
        mapped: dict[str, int] = {}
        for tf, count in counts.items():
            label = self.display_tf_label(str(tf))
            mapped[label] = mapped.get(label, 0) + int(count)
        return mapped

    def map_tfbs_usage(self, counts: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
        mapped: dict[tuple[str, str], int] = {}
        for (tf, tfbs), count in counts.items():
            label = self.display_tf_label(str(tf))
            mapped[(label, str(tfbs))] = mapped.get((label, str(tfbs)), 0) + int(count)
        return mapped
