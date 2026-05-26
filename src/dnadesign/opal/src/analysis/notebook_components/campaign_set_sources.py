from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

from ._support import sequence
from .plot_scopes import sort_plot_scope_manifests


def campaign_plot_manifest(
    campaign_model: Mapping[str, Any],
    *,
    name: str,
    kind: str,
) -> Mapping[str, Any] | None:
    candidates = [
        manifest
        for manifest in sequence(campaign_model.get("plot_manifests"))
        if isinstance(manifest, Mapping)
        and manifest.get("status") == "written"
        and str(manifest.get("name") or "") == str(name)
        and str(manifest.get("kind") or "") == str(kind)
    ]
    if not candidates:
        return None
    return sort_plot_scope_manifests(candidates)[0]


def manifest_tidy_csv_path(manifest: Mapping[str, Any]) -> Path | None:
    tidy_csv = manifest.get("tidy_csv")
    if tidy_csv not in (None, ""):
        return Path(str(tidy_csv))
    for output in sequence(manifest.get("outputs")):
        if isinstance(output, Mapping) and output.get("role") == "tidy_csv" and output.get("path"):
            return Path(str(output["path"]))
    return None


def manifest_media_path(manifest: Mapping[str, Any]) -> Path | None:
    for output in sequence(manifest.get("outputs")):
        if isinstance(output, Mapping) and output.get("role") == "media" and output.get("path"):
            return Path(str(output["path"]))
    path = manifest.get("path")
    return Path(str(path)) if path not in (None, "") else None


def read_csv_dict_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
