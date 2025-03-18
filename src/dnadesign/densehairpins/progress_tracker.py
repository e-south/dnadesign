"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/progress_tracker.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import yaml
import datetime

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
                "system_resources": {},
                "config": {},
                "meta_gap_fill_used": False,
                "source": ""
            }

    def _load_status(self) -> dict:
        with self.progress_file.open("r") as f:
            data = yaml.safe_load(f)
        return data if data else {}

    def update(self, new_entry: dict, target_quota: int):
        self.status["total_entries"] += 1
        self.status["target_quota"] = target_quota
        self.status["last_checkpoint"] = datetime.datetime.now().isoformat()
        if new_entry.get("meta_gap_fill", False):
            self.status["meta_gap_fill_used"] = True
        self._save_status()

    def update_batch_config(self, config: dict, source_label: str):
        self.status["config"] = {
            "sequence_length": config.get("sequence_length"),
            "quota": config.get("quota"),
            "subsample_size": config.get("subsample_size"),
            "arrays_generated_before_resample": config.get("arrays_generated_before_resample"),
            "solver": config.get("solver"),
            "solver_options": config.get("solver_options"),
            "fixed_elements": config.get("fixed_elements")
        }
        self.status["source"] = source_label
        self._save_status()

    def _save_status(self):
        with self.progress_file.open("w") as f:
            yaml.dump(self.status, f)
