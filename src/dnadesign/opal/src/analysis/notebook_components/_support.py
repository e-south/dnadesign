from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def join_list(value: Any, *, sep: str) -> str:
    items = [str(item) for item in sequence(value) if str(item)]
    return sep.join(items) if items else "not recorded"


def display_name(value: Any) -> str:
    """Return a compact human label for stable identifiers."""

    text = str(value or "").strip()
    if not text:
        return "Untitled"
    words = text.replace("__", " ").replace("_", " ").replace("-", " ").split()
    return " ".join(word[:1].upper() + word[1:] for word in words) if words else text


def compact_path(value: Any, *, base: Any | None = None, max_parts: int = 3) -> str:
    """Return a notebook-friendly path label without discarding evidence fields."""

    text = str(value or "").strip()
    if not text:
        return "not recorded"
    path = Path(text)
    if base not in (None, ""):
        try:
            relative = path.expanduser().resolve().relative_to(Path(str(base)).expanduser().resolve())
            return relative.as_posix() or "."
        except Exception:
            pass
    if not path.is_absolute():
        return path.as_posix()
    parts = [part for part in path.parts if part not in {path.anchor, ""}]
    if not parts:
        return path.as_posix()
    return Path(*parts[-max(1, max_parts) :]).as_posix()


def compact_notebook_path(value: Any, *, base: Any | None = None, max_parts: int = 3) -> str:
    """Public notebook helper for compact path display labels."""

    return compact_path(value, base=base, max_parts=max_parts)


def selection_count(view_model: Mapping[str, Any]) -> int | None:
    review_manifest = mapping(view_model.get("review_manifest"))
    selection = mapping(review_manifest.get("selection"))
    for key in ("selected_count", "selection_count", "count"):
        value = selection.get(key)
        if isinstance(value, int):
            return value
    for key in ("selected_records", "preview", "rows"):
        value = selection.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def resolved_run_id(run_scope: Mapping[str, Any]) -> str | None:
    value = run_scope.get("resolved_run_id")
    if value not in (None, ""):
        return str(value)
    run_ids = sequence(run_scope.get("run_ids"))
    return str(run_ids[-1]) if run_ids else None


def predict_progress_text(predict: Mapping[str, Any]) -> str:
    batch = predict.get("batch")
    of = predict.get("of")
    rows = predict.get("rows")
    if batch is not None and of is not None:
        text = f"{batch}/{of} batches"
        if rows is not None:
            text += f", {rows} rows"
        return text
    if rows is not None:
        return f"{rows} rows"
    return "not recorded"


def plot_entries_from_manifests(manifests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for manifest in manifests:
        name = manifest.get("name")
        if not name:
            continue
        entries.append(
            {
                "name": str(name),
                "kind": manifest.get("kind") or "unknown",
                "tags": list(sequence(manifest.get("tags"))),
            }
        )
    return entries


def first_media_output(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for output in sequence(manifest.get("outputs")):
        if not isinstance(output, Mapping):
            continue
        if output.get("role") == "media" and output.get("exists") and output.get("path"):
            return output
    return None
