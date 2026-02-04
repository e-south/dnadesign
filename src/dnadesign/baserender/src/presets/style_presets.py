"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/presets/style_presets.py

Style preset resolution + deep-merge overrides (Option A).

Effective style layers (deep-merged):
  1) styles/presentation_default.yml         (always)
  2) optional preset (job style_preset)      (optional)
  3) inline overrides (job style:)           (optional)

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import fields as _dc_fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Union, cast

import yaml

from ..contracts import SchemaError, ensure
from ..style import GlyphStyle, Style

PresetSpec = Union[str, Path]

DEFAULT_PRESET_NAME = "presentation_default"


def _baserender_root() -> Path:
    # style_presets.py lives in: baserender/src/presets/
    return Path(__file__).resolve().parent.parent.parent


def list_style_presets() -> list[str]:
    """Return available preset names from styles/*.yml|*.yaml."""
    styles_dir = _baserender_root() / "styles"
    if not styles_dir.exists():
        return []
    names = []
    for pat in ("*.yml", "*.yaml"):
        names.extend(p.stem for p in styles_dir.glob(pat))
    return sorted(set(names))


def resolve_style_preset_path(spec: PresetSpec) -> Path:
    """
    Resolve a preset spec into a concrete YAML file path.

    Supported:
      - Name: "presentation_default" -> <root>/styles/presentation_default.yml|.yaml
      - Relative path: "styles/foo.yml" -> <root>/styles/foo.yml
      - Absolute path: "/abs/path/foo.yml"

    Notes:
      - If the user provides an extension without a directory (e.g. "foo.yml"),
        we will also try <root>/styles/foo.yml for convenience.
    """
    root = _baserender_root()
    raw = spec if isinstance(spec, Path) else Path(str(spec))

    # Absolute path: accept directly.
    if raw.is_absolute():
        ensure(
            raw.exists(),
            f"Could not resolve style_preset '{spec}'. File not found: {raw}",
            SchemaError,
        )
        return raw

    # If it looks like a path (has dirs) OR has a YAML extension, resolve relative to root.
    if len(raw.parts) > 1 or raw.suffix.lower() in {".yml", ".yaml"}:
        cand = root / raw
        if cand.exists():
            return cand
        # Convenience: if user provided just "foo.yml", also try "styles/foo.yml".
        cand2 = root / "styles" / raw.name
        if cand2.exists():
            return cand2
        raise SchemaError(
            f"Could not resolve style_preset '{spec}'. "
            f"Checked: {cand} and {cand2}. "
            f"Expected presets under: {root / 'styles'}"
        )

    # Otherwise treat as a name.
    name = raw.stem if raw.suffix else str(raw)
    cand_yml = root / "styles" / f"{name}.yml"
    cand_yaml = root / "styles" / f"{name}.yaml"
    if cand_yml.exists():
        return cand_yml
    if cand_yaml.exists():
        return cand_yaml

    raise SchemaError(f"Could not resolve style_preset '{spec}'. Expected at {cand_yml} (or {cand_yaml}).")


@lru_cache(maxsize=64)
def load_style_preset_mapping(path: Path) -> dict[str, Any]:
    """
    Load a style preset YAML file as a mapping.
    Cached to avoid repeated disk reads.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as e:
        raise SchemaError(f"Could not parse style preset YAML: {path}") from e
    if raw is None:
        raw = {}
    ensure(
        isinstance(raw, Mapping),
        f"style preset must be a mapping/dict: {path}",
        SchemaError,
    )
    return dict(cast(Mapping[str, Any], raw))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep merge semantics:
      - mapping + mapping -> recurse
      - otherwise override replaces base
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(cast(Mapping[str, Any], out[k]), cast(Mapping[str, Any], v))
        else:
            out[k] = v
    return out


def _validate_kmer_keys(kmer_cfg: Mapping[str, Any]) -> None:
    allowed = {f.name for f in _dc_fields(GlyphStyle)}
    unknown = sorted(str(k) for k in kmer_cfg.keys() if str(k) not in allowed)
    if unknown:
        raise SchemaError(f"Invalid style.kmer key(s): {unknown}")


def effective_style_mapping(
    *,
    preset: Optional[PresetSpec] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Compute the final merged style mapping (not yet validated into Style).
    """
    base_path = resolve_style_preset_path(DEFAULT_PRESET_NAME)
    base = load_style_preset_mapping(base_path)

    merged: dict[str, Any] = dict(base)
    if preset is not None:
        preset_path = resolve_style_preset_path(preset)
        pm = load_style_preset_mapping(preset_path)
        merged = deep_merge(merged, pm)

    if overrides is not None:
        ensure(
            isinstance(overrides, Mapping),
            "style overrides must be a mapping/dict",
            SchemaError,
        )
        merged = deep_merge(merged, cast(Mapping[str, Any], overrides))

    # Proactive nested validation for clearer error messages.
    kmer = merged.get("kmer", None)
    if kmer is not None:
        ensure(isinstance(kmer, Mapping), "style.kmer must be a mapping/dict", SchemaError)
        _validate_kmer_keys(cast(Mapping[str, Any], kmer))

    return merged


def resolve_style(
    *,
    preset: Optional[PresetSpec] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Style:
    """
    Resolve the effective Style instance using:
      presentation_default -> optional preset -> overrides
    and strict validation via Style.from_mapping().
    """
    merged = effective_style_mapping(preset=preset, overrides=overrides)
    try:
        return Style.from_mapping(merged)
    except SchemaError:
        raise
    except TypeError as e:
        # Catch nested dataclass constructor errors, reframe as SchemaError.
        raise SchemaError(f"Invalid style mapping: {e}") from e
