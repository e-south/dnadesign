"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/attach_motifs_from_cruncher_lockfile.py

Transform that rewrites motif_logo matrices from lockfile-resolved Cruncher motif
artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..core import PluginError, Record
from ..core.record import Effect


def _resolve_existing_path(raw: str, *, ctx: str) -> Path:
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise PluginError(f"{ctx} does not exist: {candidate}")
    return candidate


def _load_json_mapping(path: Path, *, ctx: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PluginError(f"{ctx} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PluginError(f"{ctx} must be a JSON object: {path}")
    return payload


def _parse_matrix(matrix: object, *, ctx: str) -> list[list[float]]:
    if not isinstance(matrix, list) or not matrix:
        raise PluginError(f"{ctx}.matrix must be a non-empty list")
    parsed: list[list[float]] = []
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            raise PluginError(f"{ctx}.matrix row {row_idx} must have at least 4 values [A,C,G,T]")
        parsed.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return parsed


def _resolve_provenance_paths(
    *,
    run_manifest_path: str | None,
    lockfile_path: str | None,
    motif_store_root: str | None,
) -> tuple[Path, Path]:
    run_manifest = None
    if run_manifest_path is not None:
        run_manifest = _resolve_existing_path(run_manifest_path, ctx="run_manifest_path")
        payload = _load_json_mapping(run_manifest, ctx="run_manifest_path")
        manifest_lockfile = payload.get("lockfile_path")
        motif_store = payload.get("motif_store")
        manifest_store_root = motif_store.get("catalog_root") if isinstance(motif_store, Mapping) else None
        if lockfile_path is None:
            if not isinstance(manifest_lockfile, str) or manifest_lockfile.strip() == "":
                raise PluginError("run_manifest_path missing 'lockfile_path'")
            lockfile_path = manifest_lockfile
        if motif_store_root is None:
            if not isinstance(manifest_store_root, str) or manifest_store_root.strip() == "":
                raise PluginError("run_manifest_path missing 'motif_store.catalog_root'")
            motif_store_root = manifest_store_root

    if lockfile_path is None or motif_store_root is None:
        raise PluginError(
            "attach_motifs_from_cruncher_lockfile requires either "
            "run_manifest_path or both lockfile_path and motif_store_root"
        )

    lockfile = _resolve_existing_path(lockfile_path, ctx="lockfile_path")
    store_root = _resolve_existing_path(motif_store_root, ctx="motif_store_root")
    return lockfile, store_root


def _load_pwm_map(
    *,
    lockfile_path: Path,
    motif_store_root: Path,
    verify_checksums: bool,
) -> dict[str, list[list[float]]]:
    lock_payload = _load_json_mapping(lockfile_path, ctx="lockfile")
    resolved = lock_payload.get("resolved")
    if not isinstance(resolved, Mapping) or not resolved:
        raise PluginError(f"lockfile missing non-empty 'resolved' mapping: {lockfile_path}")

    motif_root = motif_store_root / "normalized" / "motifs"
    if not motif_root.exists():
        raise PluginError(f"motif_store_root missing normalized motifs directory: {motif_root}")

    out: dict[str, list[list[float]]] = {}
    for tf_name, entry in resolved.items():
        if not isinstance(entry, Mapping):
            raise PluginError(f"lockfile entry for TF '{tf_name}' must be an object")
        source = entry.get("source")
        motif_id = entry.get("motif_id")
        if not isinstance(source, str) or source.strip() == "":
            raise PluginError(f"lockfile entry for TF '{tf_name}' missing source")
        if not isinstance(motif_id, str) or motif_id.strip() == "":
            raise PluginError(f"lockfile entry for TF '{tf_name}' missing motif_id")

        motif_path = motif_root / source / f"{motif_id}.json"
        motif_payload = _load_json_mapping(motif_path, ctx=f"motif record for TF '{tf_name}'")
        matrix = _parse_matrix(motif_payload.get("matrix"), ctx=f"motif record '{source}:{motif_id}'")

        if verify_checksums:
            lock_sha = entry.get("sha256")
            if not isinstance(lock_sha, str) or lock_sha.strip() == "":
                raise PluginError(f"lockfile entry for TF '{tf_name}' missing sha256")
            checksums = motif_payload.get("checksums")
            if not isinstance(checksums, Mapping):
                raise PluginError(f"motif record '{source}:{motif_id}' missing checksums object")
            norm_sha = checksums.get("sha256_norm")
            if not isinstance(norm_sha, str) or norm_sha.strip() == "":
                raise PluginError(f"motif record '{source}:{motif_id}' missing checksums.sha256_norm")
            if norm_sha != lock_sha:
                raise PluginError(f"checksum mismatch for TF '{tf_name}': lockfile={lock_sha} motif_record={norm_sha}")

        tf_key = str(tf_name)
        out[tf_key] = matrix
        out[tf_key.lower()] = matrix

    if not out:
        raise PluginError("no lockfile motif entries were loaded")
    return out


class AttachMotifsFromCruncherLockfileTransform:
    def __init__(
        self,
        *,
        run_manifest_path: str | None = None,
        lockfile_path: str | None = None,
        motif_store_root: str | None = None,
        tf_tag_prefix: str = "tf:",
        require_effect: bool = True,
        verify_checksums: bool = True,
    ) -> None:
        resolved_lockfile, resolved_store_root = _resolve_provenance_paths(
            run_manifest_path=run_manifest_path,
            lockfile_path=lockfile_path,
            motif_store_root=motif_store_root,
        )
        self._tf_tag_prefix = str(tf_tag_prefix)
        if self._tf_tag_prefix.strip() == "":
            raise PluginError("attach_motifs_from_cruncher_lockfile: tf_tag_prefix must be non-empty")
        self._require_effect = bool(require_effect)
        self._pwm_by_tf = _load_pwm_map(
            lockfile_path=resolved_lockfile,
            motif_store_root=resolved_store_root,
            verify_checksums=bool(verify_checksums),
        )

    def _matrix_for_tf(self, tf_name: str) -> list[list[float]]:
        matrix = self._pwm_by_tf.get(tf_name) or self._pwm_by_tf.get(tf_name.lower())
        if matrix is None:
            raise PluginError(f"attach_motifs_from_cruncher_lockfile: no motif matrix found for tf='{tf_name}'")
        return matrix

    def apply(self, record: Record) -> Record:
        feature_tf_by_id: dict[str, str] = {}
        feature_len_by_id: dict[str, int] = {}
        for feature in record.features:
            if feature.id is None:
                continue
            tf_name: str | None = None
            for tag in feature.tags:
                if tag.startswith(self._tf_tag_prefix):
                    tf_name = tag[len(self._tf_tag_prefix) :].strip()
                    break
            if tf_name:
                feature_tf_by_id[feature.id] = tf_name
                feature_len_by_id[feature.id] = feature.span.length()

        if not feature_tf_by_id:
            return record

        replaced_feature_ids: set[str] = set()
        rewritten_effects: list[Effect] = []
        for effect in record.effects:
            if effect.kind != "motif_logo":
                rewritten_effects.append(effect)
                continue

            feature_id_raw = effect.target.get("feature_id")
            if not isinstance(feature_id_raw, str) or feature_id_raw.strip() == "":
                raise PluginError("attach_motifs_from_cruncher_lockfile: motif_logo effect missing target.feature_id")
            feature_id = feature_id_raw.strip()
            tf_name = feature_tf_by_id.get(feature_id)
            if tf_name is None:
                rewritten_effects.append(effect)
                continue

            matrix = self._matrix_for_tf(tf_name)
            expected_len = feature_len_by_id[feature_id]
            if len(matrix) != expected_len:
                raise PluginError(
                    "attach_motifs_from_cruncher_lockfile: motif length mismatch for "
                    f"tf='{tf_name}' feature_id='{feature_id}'; "
                    f"matrix_len={len(matrix)} feature_len={expected_len}"
                )
            params = dict(effect.params)
            params["matrix"] = [row[:] for row in matrix]
            rewritten_effects.append(
                Effect(
                    kind=effect.kind,
                    target=effect.target,
                    params=params,
                    render=effect.render,
                )
            )
            replaced_feature_ids.add(feature_id)

        if self._require_effect:
            missing = sorted(set(feature_tf_by_id.keys()) - replaced_feature_ids)
            if missing:
                raise PluginError(
                    "attach_motifs_from_cruncher_lockfile: missing motif_logo effects for feature ids "
                    f"{missing} in record '{record.id}'"
                )

        return Record(
            id=record.id,
            alphabet=record.alphabet,
            sequence=record.sequence,
            features=record.features,
            effects=tuple(rewritten_effects),
            display=record.display,
            meta=record.meta,
        ).validate()
