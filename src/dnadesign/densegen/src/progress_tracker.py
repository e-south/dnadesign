"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/progress_tracker.py

Lightweight YAML progress tracker for DenseGen batches.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class ProgressTracker:
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        if self.progress_file.exists():
            self.status = self._load_status()
        else:
            self.status = {
                "total_entries": 0,
                "target_quota": None,
                "last_checkpoint": None,
                "error_flags": [],
                "config": {},
                "source": "",
                "plan_name": "",
            }

    def _load_status(self) -> dict:
        with self.progress_file.open("r") as f:
            data = yaml.safe_load(f)
        return data or {}

    def update(self, used_gap_fill: bool, target_quota: int):
        self.status["total_entries"] += 1
        self.status["target_quota"] = target_quota
        self.status["last_checkpoint"] = datetime.datetime.now().isoformat()
        if used_gap_fill and "gap_fill_used_count" in self.status:
            self.status["gap_fill_used_count"] += 1
        elif used_gap_fill:
            self.status["gap_fill_used_count"] = 1
        self._save_status()

    def update_batch_config(self, config: Dict[str, Any], source_label: str, plan_name: str):
        self.status["config"] = config
        self.status["source"] = source_label
        self.status["plan_name"] = plan_name
        self._save_status()

    def _save_status(self):
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_file.open("w") as f:
            yaml.safe_dump(self.status, f, sort_keys=False)
