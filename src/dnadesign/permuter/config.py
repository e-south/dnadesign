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

_VALID_GOALS = {"max", "min"}
_VALID_NORM_METHODS = {"rank", "z2cdf", "minmax", "identity"}
_VALID_NORM_SCOPES = {"round", "job", "fixed"}
_VALID_OBJECTIVES = {"weighted_sum"}
_VALID_STRATEGIES = {"top_k", "threshold"}
_VALID_RUN_MODES = {"full", "analysis", "auto"}


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
        required = {"input_file", "references", "evaluate", "select", "permute"}
        missing = required.difference(job)
        if missing:
            raise ConfigError(f"[{name}] Missing keys: {missing}")

        # run.mode sanity (optional, defaults to full)
        run_mode = job.get("run", {}).get("mode", "full")
        if run_mode not in _VALID_RUN_MODES:
            raise ConfigError(
                f"[{name}] run.mode must be one of {sorted(_VALID_RUN_MODES)}"
            )

        # permute.protocol sanity (protocol-specific details are validated inside the protocol)
        permute = job.get("permute", {}) or {}
        protocol = permute.get("protocol")
        if not protocol:
            raise ConfigError(f"[{name}] permute.protocol is required")

        # references list must be non-empty
        if not isinstance(job["references"], list) or not job["references"]:
            raise ConfigError(f"[{name}] `references` must be a non-empty list")

        # symbol/column sanity (quick regex)
        if not re.match(r"^[A-Za-z0-9_\-]+$", name):
            raise ConfigError(f"[{name}] job name contains illegal characters")

        # evaluate.metrics (flat list with ids)
        eval_block = job["evaluate"]
        metrics = eval_block.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ConfigError(f"[{name}] `evaluate.metrics` must be a non-empty list")

        metric_ids: set[str] = set()
        for m in metrics:
            if not isinstance(m, dict):
                raise ConfigError(
                    f"[{name}] each metric must be a mapping, got {type(m).__name__}"
                )
            # required per-metric keys
            for k in ("id", "name", "evaluator", "goal"):
                if k not in m:
                    raise ConfigError(
                        f"[{name}] metric missing required key '{k}': {m}"
                    )
            mid = m["id"]
            if not re.fullmatch(r"^[A-Za-z0-9_\-]+$", mid):
                raise ConfigError(
                    f"[{name}] metric id '{mid}' contains illegal characters"
                )
            if mid in metric_ids:
                raise ConfigError(f"[{name}] duplicate metric id '{mid}'")
            metric_ids.add(mid)
            goal = str(m["goal"]).lower()
            if goal not in _VALID_GOALS:
                raise ConfigError(
                    f"[{name}] metric '{mid}' has invalid goal '{m['goal']}', expected one of {_VALID_GOALS}"
                )

            # optional norm section
            norm = m.get("norm", {})
            if norm:
                method = str(norm.get("method", "rank")).lower()
                scope = str(norm.get("scope", "round")).lower()
                if method not in _VALID_NORM_METHODS:
                    raise ConfigError(
                        f"[{name}] metric '{mid}' norm.method '{method}' not in {_VALID_NORM_METHODS}"
                    )
                if scope not in _VALID_NORM_SCOPES:
                    raise ConfigError(
                        f"[{name}] metric '{mid}' norm.scope '{scope}' not in {_VALID_NORM_SCOPES}"
                    )
                if scope == "fixed" and not norm.get("fixed_stats"):
                    raise ConfigError(
                        f"[{name}] metric '{mid}' scope=fixed requires norm.fixed_stats"
                    )

        # selector sanity (nested objective + strategy)
        sel = job["select"]
        if "objective" not in sel or "strategy" not in sel:
            raise ConfigError(
                f"[{name}] `select` must have `objective` and `strategy` blocks"
            )
        obj = sel["objective"]
        strat = sel["strategy"]

        # objective validation
        if obj.get("type") not in _VALID_OBJECTIVES:
            raise ConfigError(f"[{name}] unknown objective type: {obj.get('type')!r}")
        weights = obj.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ConfigError(f"[{name}] objective.weights must be a non-empty mapping")
        # strict: keys must match metric_ids exactly
        weight_keys = set(weights.keys())
        if weight_keys != metric_ids:
            missing = metric_ids - weight_keys
            extra = weight_keys - metric_ids
            raise ConfigError(
                f"[{name}] objective.weights keys must exactly match metric ids. "
                f"Missing: {sorted(missing)} Extra: {sorted(extra)}"
            )
        # numeric, non-negative
        for k, v in weights.items():
            try:
                val = float(v)
            except Exception:  # pragma: no cover
                raise ConfigError(f"[{name}] objective.weights['{k}'] must be numeric")
            if not (val >= 0.0):
                raise ConfigError(
                    f"[{name}] objective.weights['{k}'] must be non-negative"
                )

        # strategy validation
        stype = strat.get("type")
        if stype not in _VALID_STRATEGIES:
            raise ConfigError(f"[{name}] Unknown selection strategy: {stype!r}")

        if stype == "top_k":
            if "k" not in strat or int(strat["k"]) <= 0:
                raise ConfigError(f"[{name}] top_k strategy requires positive `k`")

        if stype == "threshold":
            target = strat.get("target", "objective")
            if target not in {"objective", "metric"}:
                raise ConfigError(
                    f"[{name}] threshold.target must be 'objective' or 'metric'"
                )

            # exactly one of {threshold, percentile}
            has_thr = "threshold" in strat
            has_pct = "percentile" in strat
            if has_thr == has_pct:
                raise ConfigError(
                    f"[{name}] threshold strategy requires exactly one of 'threshold' or 'percentile'"
                )
            if has_thr:
                try:
                    float(strat["threshold"])
                except Exception:
                    raise ConfigError(f"[{name}] threshold must be numeric")
            if has_pct:
                try:
                    pct = float(strat["percentile"])
                except Exception:
                    raise ConfigError(f"[{name}] percentile must be numeric")
                if not (0 < pct <= 100):
                    raise ConfigError(
                        f"[{name}] percentile must be in (0, 100], got {pct}"
                    )

            if target == "metric":
                mid = strat.get("metric_id")
                if not mid:
                    raise ConfigError(
                        f"[{name}] threshold.target=metric requires 'metric_id'"
                    )
                if mid not in metric_ids:
                    raise ConfigError(
                        f"[{name}] threshold.metric_id '{mid}' is not a declared metric id"
                    )
                # percentile on raw metrics is disallowed to avoid goal ambiguity
                use_norm = bool(strat.get("use_normalized", True))
                if has_pct and not use_norm:
                    raise ConfigError(
                        f"[{name}] percentile with target=metric requires use_normalized: true"
                    )
