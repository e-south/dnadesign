"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/presets/registry.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Dict, List, Tuple

import yaml

PKG = "dnadesign.infer.src.presets"


def _scan() -> List[Tuple[str, str]]:
    """
    Returns [(id, text_yaml)] scanning package presets. The id is derived from
    subpath: e.g. 'evo2/extract_logits_ll'.
    """
    root = files(PKG)
    out: List[Tuple[str, str]] = []
    for entry in root.rglob("*"):
        if entry.is_file() and entry.name.lower().endswith((".yml", ".yaml")):
            # id: subpath without suffix, POSIX style
            rel = entry.relative_to(root)
            preset_id = str(rel.with_suffix("")).replace("\\", "/")
            out.append((preset_id, entry.read_text()))
    return out


@dataclass(frozen=True)
class _PresetRecord:
    path_id: str
    id: str
    stem: str
    kind: str
    description: str
    model: Dict[str, Any]
    outputs: List[Any]
    params: Dict[str, Any]


def _normalize_preset_record(path_id: str, text: str) -> _PresetRecord:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset '{path_id}' must be a YAML mapping at the root.")

    meta = data.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Preset '{path_id}' has invalid meta block; expected mapping.")

    preset_id_raw = data.get("id") or meta.get("id") or path_id
    preset_id = str(preset_id_raw).strip()
    if not preset_id:
        raise ValueError(f"Preset '{path_id}' has empty id.")

    kind = str(data.get("kind") or meta.get("kind") or "extract").strip() or "extract"
    description = str(data.get("description") or meta.get("description") or "")
    model = data.get("model") or meta.get("model") or {}
    outputs = data.get("outputs") or []
    params = data.get("params") or {}

    if not isinstance(model, dict):
        raise ValueError(f"Preset '{preset_id}' has invalid model block; expected mapping.")
    if not isinstance(outputs, list):
        raise ValueError(f"Preset '{preset_id}' has invalid outputs block; expected list.")
    if not isinstance(params, dict):
        raise ValueError(f"Preset '{preset_id}' has invalid params block; expected mapping.")

    return _PresetRecord(
        path_id=path_id,
        id=preset_id,
        stem=path_id.split("/")[-1],
        kind=kind,
        description=description,
        model=model,
        outputs=outputs,
        params=params,
    )


@lru_cache(maxsize=1)
def _preset_records() -> Tuple[_PresetRecord, ...]:
    seen_ids: set[str] = set()
    records: List[_PresetRecord] = []
    for path_id, text in _scan():
        record = _normalize_preset_record(path_id, text)
        if record.id in seen_ids:
            raise ValueError(f"Duplicate preset id '{record.id}' detected in packaged presets.")
        seen_ids.add(record.id)
        records.append(record)
    records.sort(key=lambda item: item.id)
    return tuple(records)


def clear_preset_cache() -> None:
    _preset_records.cache_clear()


def _record_to_preset(record: _PresetRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind,
        "description": record.description,
        "model": record.model,
        "outputs": record.outputs,
        "params": record.params,
    }


def list_presets() -> List[Dict[str, str]]:
    return [
        {
            "id": record.id,
            "kind": record.kind,
            "description": record.description,
        }
        for record in _preset_records()
    ]


def load_preset(name: str) -> Dict:
    records = _preset_records()
    for record in records:
        if name in {record.path_id, record.id}:
            return _record_to_preset(record)

    stem = name.split("/")[-1]
    stem_matches = [record for record in records if record.stem == stem]
    if len(stem_matches) == 1:
        return _record_to_preset(stem_matches[0])
    if len(stem_matches) > 1:
        matches = ", ".join(record.path_id for record in stem_matches)
        raise KeyError(f"Ambiguous preset stem '{stem}'. Use a full preset id. Matches: {matches}")
    raise KeyError(f"Preset '{name}' not found")
