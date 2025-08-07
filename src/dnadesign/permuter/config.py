"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml

_VALID_METRICS = {"log_likelihood", "log_likelihood_ratio", "embedding_distance"}
_VALID_STRATEGIES = {"top_k", "threshold"}


class ConfigError(RuntimeError):
    """Raised when a configuration file is invalid."""


class _Validator:
    """Light-weight, dependency-free config validator."""

    def __init__(self, cfg_path: str | Path) -> None:
        self.cfg_path = Path(cfg_path)
        self.raw = self._load_yaml()

    # ───────────────────────── public API ────────────────────────── #

    def experiment(self) -> Dict[str, Any]:
        exp = self.raw["permuter"]["experiment"]
        if "name" not in exp or not exp["name"]:
            exp["name"] = f"permuter_{uuid.uuid4().hex[:8]}"
        return exp

    def jobs(self) -> List[Dict[str, Any]]:
        jobs = self.raw["permuter"]["jobs"]
        if not isinstance(jobs, list) or not jobs:
            raise ConfigError("No jobs defined under `permuter.jobs`")

        for job in jobs:
            self._validate_job(job)
        return jobs

    # ───────────────────────── helpers ───────────────────────────── #

    def _load_yaml(self) -> Dict[str, Any]:
        try:
            with open(self.cfg_path, "rt", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except FileNotFoundError:  # pragma: no cover
            raise ConfigError(f"Config file not found: {self.cfg_path}")

    # ——————————— per-job validation ——————————— #
    def _validate_job(self, job: Dict[str, Any]) -> None:
        name = job.get("name", "<unnamed>")
        required = {"input_file", "references", "protocol", "evaluate", "select"}
        missing = required.difference(job)
        if missing:
            raise ConfigError(f"[{name}] Missing keys: {missing}")

        # protocol sanity
        if job["protocol"] == "scan_codon" and not job["permute"]["lookup_tables"]:
            raise ConfigError(f"[{name}] scan_codon requires a codon table")

        # metric sanity
        metric = job["evaluate"].get("metric")
        if metric not in _VALID_METRICS:
            raise ConfigError(f"[{name}] unsupported metric: {metric}")

        # selector sanity
        strategy = job["select"]["strategy"]
        if strategy not in _VALID_STRATEGIES:
            raise ConfigError(f"[{name}] Unknown selection strategy: {strategy}")
        if strategy == "top_k" and (
            "k" not in job["select"] or job["select"]["k"] <= 0
        ):
            raise ConfigError(f"[{name}] top_k strategy requires positive `k`")
        if strategy == "threshold" and "threshold" not in job["select"]:
            raise ConfigError(f"[{name}] threshold strategy requires `threshold`")

        # optional embedding ref
        if metric == "embedding_distance" and not job["evaluate"].get(
            "embedding_reference_sequence"
        ):
            raise ConfigError(
                f"[{name}] embedding_distance requires exactly one embedding_reference_sequence"
            )

        # references list must be non-empty
        if not isinstance(job["references"], list) or not job["references"]:
            raise ConfigError(f"[{name}] `references` must be a non-empty list")

        # region bounds sanity
        regions = job.get("permute", {}).get("regions", [])
        for region in regions:
            if not (isinstance(region, (list, tuple)) and len(region) == 2):
                raise ConfigError(f"[{name}] region must be [start,end) pair: {region}")
            if not 0 <= region[0] < region[1]:
                raise ConfigError(f"[{name}] invalid region bounds: {region}")

        # symbol/column sanity (quick regex)
        if not re.match(r"^[A-Za-z0-9_\-]+$", name):
            raise ConfigError(f"[{name}] job name contains illegal characters")
