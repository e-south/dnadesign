"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/attach_motifs_from_config.py

Transform that rewrites motif_logo matrices from a Cruncher motif library config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from ..core import PluginError, Record
from ..core.record import Effect


def _load_pwm_map(config_path: Path) -> dict[str, list[list[float]]]:
    payload = yaml.safe_load(config_path.read_text()) or {}
    cruncher = payload.get("cruncher")
    if not isinstance(cruncher, Mapping):
        raise PluginError("attach_motifs_from_config: config missing top-level 'cruncher' mapping")
    pwms_info = cruncher.get("pwms_info")
    if not isinstance(pwms_info, Mapping):
        raise PluginError("attach_motifs_from_config: config missing 'cruncher.pwms_info' mapping")

    out: dict[str, list[list[float]]] = {}
    for tf, info in pwms_info.items():
        if not isinstance(info, Mapping):
            continue
        matrix = info.get("pwm_matrix")
        if not isinstance(matrix, list) or not matrix:
            continue
        parsed: list[list[float]] = []
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                raise PluginError(
                    "attach_motifs_from_config: pwm_matrix rows must be lists with at least 4 values [A,C,G,T]"
                )
            parsed.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        key = str(tf)
        out[key] = parsed
        out[key.lower()] = parsed

    if not out:
        raise PluginError("attach_motifs_from_config: no pwm_matrix entries found in config")
    return out


class AttachMotifsFromConfigTransform:
    def __init__(self, *, config_path: str, tf_tag_prefix: str = "tf:", require_effect: bool = True) -> None:
        raw = Path(str(config_path)).expanduser()
        self._config_path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
        if not self._config_path.exists():
            raise PluginError(f"attach_motifs_from_config: config_path does not exist: {self._config_path}")
        self._tf_tag_prefix = str(tf_tag_prefix)
        if self._tf_tag_prefix.strip() == "":
            raise PluginError("attach_motifs_from_config: tf_tag_prefix must be non-empty")
        self._require_effect = bool(require_effect)
        self._pwm_by_tf = _load_pwm_map(self._config_path)

    def _matrix_for_tf(self, tf_name: str) -> list[list[float]]:
        matrix = self._pwm_by_tf.get(tf_name) or self._pwm_by_tf.get(tf_name.lower())
        if matrix is None:
            raise PluginError(
                f"attach_motifs_from_config: no motif matrix found for tf='{tf_name}' in {self._config_path}"
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
                raise PluginError("attach_motifs_from_config: motif_logo effect missing target.feature_id")
            feature_id = feature_id_raw.strip()
            tf_name = feature_tf_by_id.get(feature_id)
            if tf_name is None:
                rewritten_effects.append(effect)
                continue

            matrix = self._matrix_for_tf(tf_name)
            expected_len = feature_len_by_id[feature_id]
            if len(matrix) != expected_len:
                raise PluginError(
                    "attach_motifs_from_config: motif length mismatch for "
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
                    "attach_motifs_from_config: missing motif_logo effects for feature ids "
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
