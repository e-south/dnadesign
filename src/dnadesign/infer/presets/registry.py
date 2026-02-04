"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/presets/registry.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib.resources import files
from typing import Dict, List, Tuple

import yaml

PKG = "dnadesign.infer.presets"


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


def list_presets() -> List[Dict[str, str]]:
    items = []
    for pid, text in _scan():
        data = yaml.safe_load(text) or {}
        meta = data.get("meta") or {}
        kind = data.get("kind") or meta.get("kind") or "extract"
        desc = data.get("description") or meta.get("description")
        # honor explicit id if present, else derived
        rid = data.get("id") or meta.get("id") or pid
        items.append({"id": rid, "kind": kind, "description": desc or ""})
    items.sort(key=lambda x: x["id"])
    return items


def load_preset(name: str) -> Dict:
    # allow 'foo/bar' id or filename stem; prefer explicit id matches
    all_items = _scan()
    for pid, text in all_items:
        data = yaml.safe_load(text) or {}
        explicit = data.get("id") or (data.get("meta") or {}).get("id")
        if name in {pid, explicit}:
            # normalize schema
            meta = data.get("meta") or {}
            preset = {
                "id": explicit or pid,
                "kind": data.get("kind") or meta.get("kind") or "extract",
                "description": data.get("description") or meta.get("description") or "",
                "model": data.get("model") or meta.get("model") or {},
                "outputs": data.get("outputs") or [],
                "params": data.get("params") or {},
            }
            return preset
    # fallback: support 'stem' lookup (last path segment)
    stem = name.split("/")[-1]
    for pid, text in all_items:
        if pid.split("/")[-1] == stem:
            data = yaml.safe_load(text) or {}
            meta = data.get("meta") or {}
            return {
                "id": data.get("id") or meta.get("id") or pid,
                "kind": data.get("kind") or meta.get("kind") or "extract",
                "description": data.get("description") or meta.get("description") or "",
                "model": data.get("model") or meta.get("model") or {},
                "outputs": data.get("outputs") or [],
                "params": data.get("params") or {},
            }
    raise KeyError(f"Preset '{name}' not found")
