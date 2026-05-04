"""Shared view matrix shape reads for status and notebook control surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..io.json_io import read_json

ViewShape = tuple[int | None, int | None]


def _manifest_shape(output_root: Path, view_id: str) -> ViewShape | None:
    manifest_path = output_root / "views" / view_id / "manifest.json"
    matrix_path = output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return None
    if str(manifest.get("artifact_id") or "") != view_id:
        return None
    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        return None
    rows = stats.get("rows")
    dims = stats.get("dims")
    if rows is None or dims is None:
        return None
    try:
        return int(rows), int(dims)
    except (TypeError, ValueError):
        return None


def read_view_shape(output_root: Path, view_id: str) -> ViewShape:
    """Return manifest-declared row and dimension counts for a view matrix."""

    return _manifest_shape(output_root, view_id) or (None, None)


@dataclass
class ViewShapeCache:
    """Process-local cache for matrix header shape reads."""

    output_root: Path
    _shapes: dict[str, ViewShape] = field(default_factory=dict)

    def get(self, view_id: str) -> ViewShape:
        if view_id not in self._shapes:
            self._shapes[view_id] = read_view_shape(self.output_root, view_id)
        return self._shapes[view_id]

    def set(self, view_id: str, shape: ViewShape) -> None:
        self._shapes[view_id] = shape


def view_shape_cache_from_inventory(output_root: Path, rows: list[dict[str, object]]) -> ViewShapeCache:
    """Seed a shape cache from a candidate inventory payload."""

    cache = ViewShapeCache(output_root=output_root)
    for row in rows:
        view_id = str(row.get("view_id") or "").strip()
        n_rows = row.get("n_rows")
        n_dims = row.get("n_dims")
        if not view_id or n_rows is None or n_dims is None:
            continue
        cache.set(view_id, (int(n_rows), int(n_dims)))
    return cache


__all__ = ["ViewShape", "ViewShapeCache", "read_view_shape", "view_shape_cache_from_inventory"]
