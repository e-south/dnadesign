"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/progress_tracker.py

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
                "system_resources": {}
            }

    def _load_status(self) -> dict:
        with self.progress_file.open("r") as f:
            data = yaml.safe_load(f)
        return data if data else {}

    def update(self, new_entry: dict, target_quota: int):
        self.status["total_entries"] += 1
        self.status["target_quota"] = target_quota
        self.status["last_checkpoint"] = datetime.datetime.now().isoformat()
        # (Optional) update system resources or error flags here.
        self._save_status()

    def _save_status(self):
        with self.progress_file.open("w") as f:
            yaml.dump(self.status, f)
