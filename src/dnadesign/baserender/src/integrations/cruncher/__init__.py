"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/cruncher/__init__.py

Adapt Cruncher records and visual defaults for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core import SchemaError, require_one_of
from ..contracts import AdapterDescriptor, IntegrationProvider, TransformDescriptor


def _normalize_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "on_missing_hit" in parsed:
        require_one_of(str(parsed["on_missing_hit"]).lower(), {"error", "skip"}, f"{ctx}.on_missing_hit")
    if "on_missing_pwm" in parsed:
        require_one_of(
            str(parsed["on_missing_pwm"]).lower(),
            {"error", "skip_effect"},
            f"{ctx}.on_missing_pwm",
        )
    return parsed


def _build_adapter(cfg, alphabet: str):
    from .best_window import CruncherBestWindowAdapter

    return CruncherBestWindowAdapter.from_config(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_config_transform(params: Mapping[str, Any]):
    from .motifs_from_config import AttachMotifsFromConfigTransform

    return AttachMotifsFromConfigTransform(**params)


def _build_lockfile_transform(params: Mapping[str, Any]):
    from .motifs_from_lockfile import AttachMotifsFromCruncherLockfileTransform

    return AttachMotifsFromCruncherLockfileTransform(**params)


def _validate_lockfile_params(params: Mapping[str, Any], ctx: str) -> None:
    has_manifest = params.get("run_manifest_path") is not None
    has_lock_bundle = params.get("lockfile_path") is not None and params.get("motif_store_root") is not None
    if not has_manifest and not has_lock_bundle:
        raise SchemaError(f"{ctx} requires run_manifest_path or both lockfile_path and motif_store_root")


PROVIDER = IntegrationProvider(
    name="cruncher",
    adapters=(
        AdapterDescriptor(
            kind="cruncher_best_window",
            owner_tool="cruncher",
            contract_kind="cruncher_best_window",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA",),
            factory=_build_adapter,
            docs_slug="cruncher-best-window",
            allowed_config_columns=(
                "sequence",
                "id",
                "hits_path",
                "hits_elite_id",
                "hits_tf",
                "hits_start",
                "hits_strand",
                "hits_window_seq",
                "hits_core_seq",
                "config_path",
            ),
            required_config_columns=("sequence", "id", "hits_path", "config_path"),
            required_source_columns=("sequence", "id"),
            allowed_policy_keys=("on_missing_hit", "on_missing_pwm"),
            resolved_path_columns=("hits_path", "config_path"),
            normalize_policies=_normalize_policies,
        ),
    ),
    transforms=(
        TransformDescriptor(
            name="attach_motifs_from_config",
            owner_tool="cruncher",
            factory=_build_config_transform,
            docs_slug="cruncher-config-motifs",
            allowed_params=("config_path", "tf_tag_prefix", "require_effect"),
            required_params=("config_path",),
            path_params=("config_path",),
        ),
        TransformDescriptor(
            name="attach_motifs_from_cruncher_lockfile",
            owner_tool="cruncher",
            factory=_build_lockfile_transform,
            docs_slug="cruncher-lockfile-motifs",
            allowed_params=(
                "run_manifest_path",
                "lockfile_path",
                "motif_store_root",
                "tf_tag_prefix",
                "require_effect",
                "verify_checksums",
            ),
            path_params=("run_manifest_path", "lockfile_path", "motif_store_root"),
            validate_params=_validate_lockfile_params,
        ),
    ),
)

__all__ = ["PROVIDER"]
