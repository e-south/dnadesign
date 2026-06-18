"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlays/support/digest_ledger.py

Overlay digest ledgers for explicit directory-overlay provenance contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .. import overlay_metadata, overlay_parts

OVERLAY_DIGEST_LEDGER_FILENAME = "digest_ledger.json"
OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION = "usr.overlay_digest_ledger.v1"


def overlay_digest_ledger_path(path: Path) -> Path | None:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / OVERLAY_DIGEST_LEDGER_FILENAME
    return None


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _payload_for_parts(path: Path, *, parts: list[Path]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    metadata = overlay_metadata(resolved)
    return {
        "schema_version": OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION,
        "overlay_path": resolved.as_posix(),
        "namespace": metadata.get("namespace") or resolved.name,
        "key": metadata.get("key"),
        "created_at": metadata.get("created_at"),
        "registry_hash": metadata.get("registry_hash"),
        "namespace_contract_hash": metadata.get("namespace_contract_hash"),
        "parts": [
            {
                "name": part.name,
                "path": part.resolve().as_posix(),
                "digest": _sha256_file(part),
            }
            for part in sorted(parts)
        ],
    }


def build_overlay_digest_ledger(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"overlay digest ledger requires a directory overlay: {resolved}")
    parts = overlay_parts(resolved)
    if not parts:
        raise FileNotFoundError(f"overlay digest ledger requires parquet parts: {resolved}")
    return _payload_for_parts(resolved, parts=parts)


def _write_overlay_digest_ledger(path: Path, payload: dict[str, Any]) -> Path:
    ledger_path = overlay_digest_ledger_path(path)
    if ledger_path is None:
        raise FileNotFoundError(f"overlay digest ledger requires a directory overlay: {path}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ledger_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, ledger_path)
    return ledger_path


def write_overlay_digest_ledger(path: Path) -> Path:
    resolved = Path(path).resolve()
    payload = build_overlay_digest_ledger(resolved)
    return _write_overlay_digest_ledger(resolved, payload)


def _read_overlay_digest_ledger(path: Path) -> dict[str, Any] | None:
    ledger_path = overlay_digest_ledger_path(path)
    if ledger_path is None or not ledger_path.is_file():
        return None
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def update_overlay_digest_ledger(path: Path, *, new_parts: list[Path]) -> Path:
    resolved = Path(path).resolve()
    ledger_path = overlay_digest_ledger_path(resolved)
    if ledger_path is None:
        raise FileNotFoundError(f"overlay digest ledger requires a directory overlay: {resolved}")
    if not ledger_path.is_file():
        return write_overlay_digest_ledger(resolved)

    payload = _read_overlay_digest_ledger(resolved)
    if (
        payload is None
        or payload.get("schema_version") != OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION
        or payload.get("overlay_path") != resolved.as_posix()
    ):
        return write_overlay_digest_ledger(resolved)

    current_parts = sorted(overlay_parts(resolved))
    current_part_names = {part.name for part in current_parts}
    existing_part_names = {
        str(entry.get("name") or "") for entry in payload.get("parts", []) if isinstance(entry, dict)
    }
    if not existing_part_names.issubset(current_part_names):
        return write_overlay_digest_ledger(resolved)

    part_rows = {
        str(entry["name"]): dict(entry)
        for entry in payload.get("parts", [])
        if isinstance(entry, dict) and str(entry.get("name") or "")
    }
    for part in new_parts:
        resolved_part = Path(part).resolve()
        part_rows[resolved_part.name] = {
            "name": resolved_part.name,
            "path": resolved_part.as_posix(),
            "digest": _sha256_file(resolved_part),
        }
    if set(part_rows) != current_part_names:
        return write_overlay_digest_ledger(resolved)

    metadata = overlay_metadata(resolved)
    payload.update(
        {
            "namespace": metadata.get("namespace") or resolved.name,
            "key": metadata.get("key"),
            "created_at": metadata.get("created_at"),
            "registry_hash": metadata.get("registry_hash"),
            "namespace_contract_hash": metadata.get("namespace_contract_hash"),
            "parts": [part_rows[part.name] for part in current_parts],
        }
    )
    return _write_overlay_digest_ledger(resolved, payload)


__all__ = [
    "OVERLAY_DIGEST_LEDGER_FILENAME",
    "OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION",
    "build_overlay_digest_ledger",
    "overlay_digest_ledger_path",
    "update_overlay_digest_ledger",
    "write_overlay_digest_ledger",
]
