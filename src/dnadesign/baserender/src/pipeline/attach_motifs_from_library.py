"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/attach_motifs_from_library.py

Transform that rewrites motif_logo matrices from a tool-provided motif library
primitives artifact.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..core import PluginError, Record, reject_unknown_keys
from ..core.record import Effect

_TOP_LEVEL_KEYS = {"schema_version", "alphabet", "motifs"}
_MOTIF_KEYS = {"source", "motif_id", "matrix"}


def _resolve_library_path(raw: str) -> Path:
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise PluginError(f"attach_motifs_from_library: library_path does not exist: {candidate}")
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


def _load_pwm_map(library_path: Path) -> dict[str, list[list[float]]]:
    payload = _load_json_mapping(library_path, ctx="motif library")
    try:
        reject_unknown_keys(payload, _TOP_LEVEL_KEYS, "attach_motifs_from_library.library")
    except Exception as exc:
        raise PluginError(str(exc)) from exc

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != "1":
        raise PluginError("attach_motifs_from_library: schema_version must be '1'")
    alphabet = str(payload.get("alphabet", "")).strip().upper()
    if alphabet != "DNA":
        raise PluginError("attach_motifs_from_library: alphabet must be 'DNA'")

    motifs = payload.get("motifs")
    if not isinstance(motifs, Mapping):
        raise PluginError("attach_motifs_from_library.library.motifs must be a mapping")
    if not motifs:
        raise PluginError("attach_motifs_from_library: motifs must be non-empty")

    out: dict[str, list[list[float]]] = {}
    for tf_name, raw_entry in motifs.items():
        if not isinstance(raw_entry, Mapping):
            raise PluginError(f"attach_motifs_from_library.library.motifs.{tf_name} must be a mapping")
        entry = raw_entry
        try:
            reject_unknown_keys(entry, _MOTIF_KEYS, f"attach_motifs_from_library.library.motifs.{tf_name}")
        except Exception as exc:
            raise PluginError(str(exc)) from exc
        matrix = _parse_matrix(entry.get("matrix"), ctx=f"attach_motifs_from_library.library.motifs.{tf_name}")
        key = str(tf_name)
        out[key] = matrix
        out[key.lower()] = matrix
    return out


class AttachMotifsFromLibraryTransform:
    def __init__(self, *, library_path: str, tf_tag_prefix: str = "tf:", require_effect: bool = True) -> None:
        self._library_path = _resolve_library_path(library_path)
        self._tf_tag_prefix = str(tf_tag_prefix)
        if self._tf_tag_prefix.strip() == "":
            raise PluginError("attach_motifs_from_library: tf_tag_prefix must be non-empty")
        self._require_effect = bool(require_effect)
        self._pwm_by_tf = _load_pwm_map(self._library_path)

    def _matrix_for_tf(self, tf_name: str) -> list[list[float]]:
        matrix = self._pwm_by_tf.get(tf_name) or self._pwm_by_tf.get(tf_name.lower())
        if matrix is None:
            raise PluginError(
                f"attach_motifs_from_library: no motif matrix found for tf='{tf_name}' in {self._library_path}"
            )
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
                raise PluginError("attach_motifs_from_library: motif_logo effect missing target.feature_id")
            feature_id = feature_id_raw.strip()
            tf_name = feature_tf_by_id.get(feature_id)
            if tf_name is None:
                rewritten_effects.append(effect)
                continue

            matrix = self._matrix_for_tf(tf_name)
            expected_len = feature_len_by_id[feature_id]
            if len(matrix) != expected_len:
                raise PluginError(
                    "attach_motifs_from_library: motif length mismatch for "
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
                    "attach_motifs_from_library: missing motif_logo effects for feature ids "
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
