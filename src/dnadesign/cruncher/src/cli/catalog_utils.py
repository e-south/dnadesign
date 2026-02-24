"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/catalog_utils.py

Utility functions used by catalog CLI commands for target resolution and assets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Sequence

import typer

from dnadesign.cruncher.store.catalog_index import CatalogEntry
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_lines, sha256_path
from dnadesign.cruncher.utils.paths import resolve_workspace_root

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
LOGO_MANIFEST_NAME = "logo_manifest.json"
_DENSEGEN_WORKSPACES_ROOT_ENV = "DNADESIGN_DENSEGEN_WORKSPACES_ROOT"


@dataclass(frozen=True)
class ResolvedTarget:
    tf_name: str
    ref: MotifRef
    entry: CatalogEntry
    site_entries: list[CatalogEntry]


def _safe_stem(label: str) -> str:
    cleaned = _SAFE_RE.sub("_", label).strip("_")
    return cleaned or "motif"


def _resolve_export_format(out_path: Path, fmt: str | None, *, label: str) -> str:
    resolved = (fmt or "").strip().lower() if fmt is not None else ""
    if resolved:
        if resolved not in {"csv", "parquet"}:
            raise typer.BadParameter(f"{label} must be 'csv' or 'parquet', got: {fmt!r}.")
        return resolved
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise typer.BadParameter(f"{label} is required when output extension is not .csv/.parquet (got {out_path}).")


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _densegen_workspaces_root(config_path: Path) -> Path | None:
    del config_path
    env_root = str(os.environ.get(_DENSEGEN_WORKSPACES_ROOT_ENV, "")).strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        return None

    densegen_spec = importlib_util.find_spec("dnadesign.densegen")
    if densegen_spec is not None:
        locations = list(densegen_spec.submodule_search_locations or [])
        for location in locations:
            package_root = Path(location).resolve()
            candidate = package_root / "workspaces"
            if candidate.is_dir():
                return candidate
    return None


def _resolve_user_path(path: Path, *, base_dir: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _resolve_densegen_workspace(selector: str, *, config_path: Path) -> Path:
    raw = str(selector or "").strip()
    if not raw:
        raise typer.BadParameter("--densegen-workspace must be a non-empty string.")
    candidate = Path(raw).expanduser()
    looks_like_path = candidate.is_absolute() or any(sep in raw for sep in (os.sep, os.altsep) if sep)
    if looks_like_path:
        resolved = _resolve_user_path(candidate, base_dir=resolve_workspace_root(config_path))
    else:
        root = _densegen_workspaces_root(config_path)
        if root is None:
            env_root = str(os.environ.get(_DENSEGEN_WORKSPACES_ROOT_ENV, "")).strip()
            if env_root:
                raise typer.BadParameter(
                    "Unable to resolve DenseGen workspaces root from "
                    f"{_DENSEGEN_WORKSPACES_ROOT_ENV}={env_root!r}. "
                    "Set it to an existing directory or pass --densegen-workspace as an absolute path."
                )
            raise typer.BadParameter(
                "Unable to locate DenseGen workspaces root from installed package metadata. "
                f"Set {_DENSEGEN_WORKSPACES_ROOT_ENV} or "
                "Pass --densegen-workspace as an absolute path."
            )
        resolved = (root / raw).resolve()
    if not resolved.exists():
        raise typer.BadParameter(f"DenseGen workspace not found: {resolved}")
    if not resolved.is_dir():
        raise typer.BadParameter(f"DenseGen workspace is not a directory: {resolved}")
    config_candidate = resolved / "config.yaml"
    if not config_candidate.is_file():
        raise typer.BadParameter(f"DenseGen workspace missing config.yaml: {config_candidate}")
    inputs_root = resolved / "inputs"
    if not inputs_root.is_dir():
        raise typer.BadParameter(f"DenseGen workspace missing inputs/ directory: {inputs_root}")
    return resolved


def _require_densegen_inputs_path(path: Path, *, inputs_root: Path, label: str, base_dir: Path) -> Path:
    resolved = _resolve_user_path(path, base_dir=base_dir)
    try:
        resolved.relative_to(inputs_root.resolve())
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be under {inputs_root} when --densegen-workspace is set.") from exc
    return resolved


def _remove_existing_artifacts(out_dir: Path, *, tf_names: Sequence[str]) -> int:
    stems = {_safe_stem(name) for name in tf_names if name}
    if not stems:
        return 0
    prefixes = {f"{stem}__" for stem in stems}
    singles = {f"{stem}.json" for stem in stems}
    removed = 0
    for path in out_dir.glob("*.json"):
        if path.name == "artifact_manifest.json":
            continue
        if path.name in singles or any(path.name.startswith(prefix) for prefix in prefixes):
            path.unlink()
            removed += 1
    return removed


def _logo_manifest_path(out_dir: Path) -> Path:
    return out_dir / LOGO_MANIFEST_NAME


def _load_logo_manifest(out_dir: Path) -> dict | None:
    path = _logo_manifest_path(out_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_logo_manifest(out_dir: Path, payload: dict) -> None:
    path = _logo_manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _matrix_site_count_from_tags(tags: dict[str, object] | None) -> int | None:
    if not tags:
        return None
    for key in ("discovery_nsites", "meme_nsites", "site_count", "nsites"):
        raw = tags.get(key)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _entry_signature(entry: CatalogEntry) -> dict[str, object]:
    return {
        "source": entry.source,
        "motif_id": entry.motif_id,
        "matrix_source": entry.matrix_source,
        "site_kind": entry.site_kind,
        "site_count": entry.site_count,
        "site_total": entry.site_total,
        "matrix_site_count": _matrix_site_count_from_tags(entry.tags),
    }


def _matrix_checksum(catalog_root: Path, entry: CatalogEntry) -> str:
    motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
    return sha256_path(motif_path)


def _sites_checksum(catalog_root: Path, entries: list[CatalogEntry]) -> str:
    lines: list[str] = []
    for candidate in sorted(entries, key=lambda e: (e.source, e.motif_id)):
        sites_path = catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
        lines.append(f"{candidate.source}:{candidate.motif_id}:{sha256_path(sites_path)}")
    return sha256_lines(lines)


def _build_logo_signature(
    *,
    cfg,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
    bits_mode: str,
    dpi: int,
) -> tuple[str, dict]:
    pwm_config = {
        "pwm_source": cfg.catalog.pwm_source,
    }
    if cfg.catalog.pwm_source == "sites":
        pwm_config.update(
            {
                "combine_sites": cfg.catalog.combine_sites,
                "site_kinds": cfg.catalog.site_kinds,
                "site_window_lengths": cfg.catalog.site_window_lengths,
                "site_window_center": cfg.catalog.site_window_center,
                "min_sites_for_pwm": cfg.catalog.min_sites_for_pwm,
                "allow_low_sites": cfg.catalog.allow_low_sites,
                "pseudocounts": cfg.catalog.pseudocounts,
            }
        )
    target_payloads: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda t: (t.tf_name, t.entry.source, t.entry.motif_id)):
        payload: dict[str, object] = {
            "tf_name": target.tf_name,
            "ref": f"{target.entry.source}:{target.entry.motif_id}",
            "entry": _entry_signature(target.entry),
        }
        if cfg.catalog.pwm_source == "matrix":
            payload["matrix_sha256"] = _matrix_checksum(catalog_root, target.entry)
        else:
            entries = target.site_entries
            payload["site_entries"] = [
                _entry_signature(candidate) for candidate in sorted(entries, key=lambda e: (e.source, e.motif_id))
            ]
            payload["sites_sha256"] = _sites_checksum(catalog_root, entries)
        target_payloads.append(payload)
    signature_payload = {
        "render": {"bits_mode": bits_mode, "dpi": dpi},
        "pwm": pwm_config,
        "targets": target_payloads,
    }
    signature = sha256_bytes(json.dumps(signature_payload, sort_keys=True).encode("utf-8"))
    return signature, signature_payload


def _matching_logo_dir(out_dir: Path, signature: str) -> Path | None:
    manifest = _load_logo_manifest(out_dir)
    if manifest and manifest.get("signature") == signature:
        return out_dir
    return None


def _clear_logo_outputs(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for png in out_dir.glob("*_logo.png"):
        png.unlink()
    manifest_file = _logo_manifest_path(out_dir)
    if manifest_file.exists():
        manifest_file.unlink()
    legacy_logo_dir = out_dir / "logos"
    if legacy_logo_dir.exists():
        shutil.rmtree(legacy_logo_dir)
    legacy_manifest_dir = out_dir / "run"
    if legacy_manifest_dir.exists():
        shutil.rmtree(legacy_manifest_dir)


def _resolve_set_tfs(cfg, set_index: int | None) -> list[str]:
    if set_index is None:
        return []
    if set_index < 1 or set_index > len(cfg.regulator_sets):
        raise typer.BadParameter(f"--set must be between 1 and {len(cfg.regulator_sets)} (got {set_index}).")
    return list(cfg.regulator_sets[set_index - 1])


def _parse_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RDBECOLITFC00214"
        )
    source, motif_id = ref.split(":", 1)
    return source, motif_id


def _ensure_entry_matches_pwm_source(
    entry: CatalogEntry,
    pwm_source: str,
    site_kinds: list[str] | None,
    *,
    tf_name: str,
    ref: str,
) -> None:
    if pwm_source == "matrix":
        if not entry.has_matrix:
            raise ValueError(f"{ref} does not have a cached motif matrix for TF '{tf_name}'.")
        return
    if pwm_source == "sites":
        if not entry.has_sites:
            raise ValueError(f"{ref} does not have cached binding sites for TF '{tf_name}'.")
        if site_kinds is not None and entry.site_kind not in site_kinds:
            raise ValueError(
                f"{ref} site kind '{entry.site_kind}' is not in site_kinds={site_kinds} for TF '{tf_name}'."
            )
        return
    raise ValueError("pwm_source must be 'matrix' or 'sites'")
