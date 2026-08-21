"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/dense_arrays/publisher.py

Publish digest-bound DenseGen playback endpoint bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from dense_arrays.playback import (
    PlaybackDocument,
    dumps_playback_plan,
    dumps_realized_array,
)
from dense_arrays.playback.matplotlib_renderer import (
    render_collection_mp4,
    render_collection_poster_png,
)
from dense_arrays.playback.theme import (
    PlaybackPresentation,
    legend_entries_for_profile,
)

from .baserender_projection import (
    AnchoredIllustrationPresentation,
    BaseRenderDuplexProjection,
    DuplexPresentation,
)
from .playback import realized_array_from_densegen_record

_ENDPOINT_SCHEMA = "densegen.solution_path_playback_endpoint.v1"
_SCENE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RECORD_COLUMNS = (
    "id",
    "sequence",
    "densegen__used_tfbs_detail",
    "densegen__schema_version",
    "densegen__run_id",
    "densegen__plan",
    "densegen__input_name",
    "densegen__sampling_library_hash",
    "densegen__sampling_library_index",
    "densegen__pad_used",
    "densegen__pad_bases",
    "densegen__pad_end",
)
_ADAPTER_FIELDS = {"kind", "display_coordinate", "solver_coordinate_provenance"}
_PLAYBACK_FIELDS = {
    "authority",
    "ordering_policy",
    "graph_relation",
    "show_authority_notice",
}
_PRESENTATION_FIELDS = {
    "layout",
    "color_profile",
    "show_legend",
    "graph_detail",
    "graph_fraction",
    "show_edge_costs",
    "seconds_per_step",
    "hold_seconds",
    "lead_seconds",
    "scene_transition_seconds",
    "show_distance_bracket",
    "collection_order",
}
_DUPLEX_FIELDS = {
    "fixed_element_annotations",
    "consensus_suffix",
    "anchored_illustration",
}
_ANCHORED_ILLUSTRATION_FIELDS = {"asset", "constraint", "reveal"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping"
        raise TypeError(msg)
    return value


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return text


def _strict_fields(
    value: Mapping[str, object],
    allowed: set[str],
    *,
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported fields: {unknown}")


def _required_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _choice(value: object, choices: set[str], *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if text not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}")
    return text


def _presentation_number(
    presentation: Mapping[str, object],
    field: str,
    default: float,
    *,
    positive: bool,
) -> float:
    value = presentation.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"presentation.{field} must be numeric")
    number = float(value)
    if (positive and number <= 0.0) or (not positive and number < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"presentation.{field} must be {qualifier}")
    return number


def _repository_root(config_path: Path, repository: str) -> Path:
    for ancestor in config_path.parents:
        if ancestor.name == repository:
            return ancestor
        candidate = ancestor / repository
        if candidate.is_dir():
            return candidate
    msg = f"could not resolve repository {repository!r} from {config_path}"
    raise FileNotFoundError(msg)


def _source_path(config_path: Path, source: Mapping[str, object]) -> Path:
    relative = Path(_required_text(source.get("table"), field_name="source.table"))
    repository = source.get("repository")
    if repository is None:
        return (config_path.parent / relative).resolve()
    return (_repository_root(config_path, str(repository)) / relative).resolve()


def _selected_rows(table_path: Path, record_ids: tuple[str, ...]) -> dict[str, Mapping[str, object]]:
    parquet = pq.ParquetFile(table_path)
    missing_columns = sorted(set(_RECORD_COLUMNS) - set(parquet.schema_arrow.names))
    if missing_columns:
        msg = f"DenseGen source is missing playback columns: {missing_columns}"
        raise ValueError(msg)
    wanted = set(record_ids)
    selected: dict[str, Mapping[str, object]] = {}
    for batch in parquet.iter_batches(columns=list(_RECORD_COLUMNS), batch_size=2048):
        for row in batch.to_pylist():
            record_id = str(row.get("id") or "")
            if record_id in wanted:
                if record_id in selected:
                    msg = f"record id {record_id!r} occurs more than once"
                    raise ValueError(msg)
                selected[record_id] = row
        if len(selected) == len(wanted):
            break
    missing = sorted(wanted - set(selected))
    if missing:
        msg = f"configured playback records are absent from the source: {missing}"
        raise ValueError(msg)
    return selected


def _load_endpoint(config_path: Path) -> Mapping[str, object]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    endpoint = _required_mapping(payload, field_name="endpoint")
    if endpoint.get("schema") != _ENDPOINT_SCHEMA:
        msg = f"unsupported endpoint schema: {endpoint.get('schema')!r}"
        raise ValueError(msg)
    return endpoint


def publish_densegen_playback_endpoint(
    config_path: Path,
    *,
    replace: bool = False,
) -> Path:
    """Publish one endpoint atomically from existing DenseGen records."""
    config_path = Path(config_path).resolve()
    endpoint = _load_endpoint(config_path)
    source = _required_mapping(endpoint.get("source"), field_name="source")
    source_path = _source_path(config_path, source)
    expected_sha256 = _required_text(source.get("sha256"), field_name="source.sha256").lower()
    observed_sha256 = _sha256_file(source_path)
    if observed_sha256 != expected_sha256:
        msg = f"source digest mismatch for {source_path}: expected {expected_sha256}, observed {observed_sha256}"
        raise ValueError(msg)
    raw_records = source.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        msg = "source.records must be a non-empty list"
        raise ValueError(msg)
    record_specs = tuple(
        _required_mapping(item, field_name=f"source.records[{index}]") for index, item in enumerate(raw_records)
    )
    record_ids = tuple(_required_text(item.get("id"), field_name="source.records[].id") for item in record_specs)
    if len(record_ids) != len(set(record_ids)):
        msg = "source.records ids must be unique"
        raise ValueError(msg)
    scenes = tuple(_required_text(item.get("scene"), field_name="source.records[].scene") for item in record_specs)
    if len(scenes) != len(set(scenes)):
        raise ValueError("source.records scenes must be unique")
    for scene in scenes:
        if not _SCENE_ID.fullmatch(scene):
            raise ValueError(f"scene id must match {_SCENE_ID.pattern!r}: {scene!r}")
    selected = _selected_rows(source_path, record_ids)
    adapter = _required_mapping(endpoint.get("adapter"), field_name="adapter")
    _strict_fields(adapter, _ADAPTER_FIELDS, field_name="adapter")
    expected_adapter = {
        "kind": "densegen_realized_array_v1",
        "display_coordinate": "offset",
        "solver_coordinate_provenance": "offset_raw",
    }
    if dict(adapter) != expected_adapter:
        raise ValueError(f"adapter must equal the supported contract: {expected_adapter}")
    playback = _required_mapping(endpoint.get("playback"), field_name="playback")
    _strict_fields(playback, _PLAYBACK_FIELDS, field_name="playback")
    if playback.get("authority") != "placement_reconstructed":
        raise ValueError("playback.authority must be 'placement_reconstructed'")
    if playback.get("ordering_policy") != ["start", "shorter_first", "placement_id"]:
        raise ValueError("playback.ordering_policy must be [start, shorter_first, placement_id]")
    if playback.get("graph_relation") != "coordinate_precedence":
        raise ValueError("playback.graph_relation must be 'coordinate_precedence'")
    show_authority_notice = _required_bool(
        playback.get("show_authority_notice"),
        field_name="playback.show_authority_notice",
    )
    labels = _required_mapping(endpoint.get("labels") or {}, field_name="labels")
    overrides_raw = _required_mapping(labels.get("overrides") or {}, field_name="labels.overrides")
    label_overrides = {str(key): str(value) for key, value in overrides_raw.items()}
    presentation_spec = _required_mapping(
        endpoint.get("presentation") or {},
        field_name="presentation",
    )
    _strict_fields(presentation_spec, _PRESENTATION_FIELDS, field_name="presentation")
    if presentation_spec.get("layout") != "graph_left_duplex_right":
        raise ValueError("presentation.layout must be 'graph_left_duplex_right'")
    color_profile = _required_text(
        presentation_spec.get("color_profile") or "categorical",
        field_name="presentation.color_profile",
    )
    show_legend = _required_bool(
        presentation_spec.get("show_legend", False),
        field_name="presentation.show_legend",
    )
    graph_detail = _choice(
        presentation_spec.get("graph_detail", "full"),
        {"full", "reduced", "inset", "none"},
        field_name="presentation.graph_detail",
    )
    default_graph_fraction = 0.0 if graph_detail == "none" else 0.35
    graph_fraction = _presentation_number(
        presentation_spec,
        "graph_fraction",
        default_graph_fraction,
        positive=graph_detail != "none",
    )
    show_edge_costs = _required_bool(
        presentation_spec.get("show_edge_costs", graph_detail == "full"),
        field_name="presentation.show_edge_costs",
    )
    show_distance_bracket = _choice(
        presentation_spec.get("show_distance_bracket", "when_declared"),
        {"never", "when_declared", "always"},
        field_name="presentation.show_distance_bracket",
    )
    playback_presentation = PlaybackPresentation(
        color_profile=color_profile,
        legend_entries=(legend_entries_for_profile(color_profile) if show_legend else ()),
        graph_detail=graph_detail,
        graph_fraction=graph_fraction,
        show_edge_costs=show_edge_costs,
        show_authority_notice=show_authority_notice,
        show_distance_bracket=show_distance_bracket,
    )
    duplex_spec = _required_mapping(endpoint.get("duplex") or {}, field_name="duplex")
    _strict_fields(duplex_spec, _DUPLEX_FIELDS, field_name="duplex")
    fixed_element_annotations = _choice(
        duplex_spec.get("fixed_element_annotations", "none"),
        {"none", "variant"},
        field_name="duplex.fixed_element_annotations",
    )
    consensus_suffix = _choice(
        duplex_spec.get("consensus_suffix", "omit"),
        {"omit", "include"},
        field_name="duplex.consensus_suffix",
    )
    anchored_spec_raw = duplex_spec.get("anchored_illustration")
    anchored_presentation = None
    if anchored_spec_raw is not None:
        anchored_spec = _required_mapping(
            anchored_spec_raw,
            field_name="duplex.anchored_illustration",
        )
        _strict_fields(
            anchored_spec,
            _ANCHORED_ILLUSTRATION_FIELDS,
            field_name="duplex.anchored_illustration",
        )
        anchored_presentation = AnchoredIllustrationPresentation(
            asset_id=_choice(
                anchored_spec.get("asset"),
                {"rnap_sigma70"},
                field_name="duplex.anchored_illustration.asset",
            ),
            constraint_name=_required_text(
                anchored_spec.get("constraint"),
                field_name="duplex.anchored_illustration.constraint",
            ),
            reveal=_choice(
                anchored_spec.get("reveal", "bindings_as_placed"),
                {"bindings_as_placed"},
                field_name="duplex.anchored_illustration.reveal",
            ),
        )
    duplex_presentation = DuplexPresentation(
        fixed_element_annotations=fixed_element_annotations,
        consensus_suffix=consensus_suffix,
        anchored_illustration=anchored_presentation,
    )
    seconds_per_step = _presentation_number(
        presentation_spec,
        "seconds_per_step",
        0.46,
        positive=True,
    )
    hold_seconds = _presentation_number(
        presentation_spec,
        "hold_seconds",
        0.80,
        positive=False,
    )
    lead_seconds = _presentation_number(
        presentation_spec,
        "lead_seconds",
        0.18,
        positive=False,
    )
    scene_transition_seconds = _presentation_number(
        presentation_spec,
        "scene_transition_seconds",
        0.0,
        positive=False,
    )
    endpoint_title = _required_text(endpoint.get("title"), field_name="title")
    source_ref = _required_text(source.get("table"), field_name="source.table")
    raw_collection_order = presentation_spec.get("collection_order")
    if not isinstance(raw_collection_order, list) or not raw_collection_order:
        raise ValueError("presentation.collection_order must be a non-empty list")
    collection_order = tuple(str(value) for value in raw_collection_order)
    if len(collection_order) != len(set(collection_order)):
        raise ValueError("presentation.collection_order must not contain duplicates")
    if set(collection_order) != set(scenes):
        raise ValueError("presentation.collection_order must contain every configured scene exactly once")
    spec_by_scene = {str(spec["scene"]): spec for spec in record_specs}
    record_specs = tuple(spec_by_scene[scene] for scene in collection_order)
    record_ids = tuple(str(spec["id"]) for spec in record_specs)
    forbidden_raw = labels.get("forbidden_terms") or []
    if not isinstance(forbidden_raw, list):
        raise TypeError("labels.forbidden_terms must be a list")
    forbidden_terms = tuple(str(value).strip().casefold() for value in forbidden_raw)
    text_surfaces = [endpoint_title, *label_overrides.values()]
    for spec in record_specs:
        text_surfaces.extend((str(spec.get("title") or ""), str(spec.get("subtitle") or "")))
    for term in forbidden_terms:
        if term and any(term in surface.casefold() for surface in text_surfaces):
            raise ValueError(f"configured presentation text contains forbidden term: {term!r}")

    documents: list[PlaybackDocument] = []
    realized_payloads: dict[str, str] = {}
    plan_payloads: dict[str, str] = {}
    scene_manifest: list[dict[str, object]] = []
    realized_by_digest: dict[str, object] = {}
    for spec, record_id in zip(record_specs, record_ids, strict=True):
        scene = _required_text(spec.get("scene"), field_name="source.records[].scene")
        if not _SCENE_ID.fullmatch(scene):
            msg = f"scene id must match {_SCENE_ID.pattern!r}: {scene!r}"
            raise ValueError(msg)
        row = selected[record_id]
        realized = realized_array_from_densegen_record(
            row,
            source_ref=source_ref,
            source_sha256=observed_sha256,
        )
        from dense_arrays.playback import reconstruct_playback

        plan = reconstruct_playback(realized)
        realized_by_digest[plan.realization_digest] = realized
        scene_title = str(spec.get("title") or scene.replace("_", " ").title())
        subtitle = str(spec.get("subtitle") or row.get("densegen__plan") or "")
        documents.append(
            PlaybackDocument(
                plan=plan,
                title=scene_title,
                subtitle=subtitle,
                label_overrides=label_overrides,
                presentation=playback_presentation,
            )
        )
        realized_payloads[scene] = dumps_realized_array(realized)
        plan_payloads[scene] = dumps_playback_plan(plan)
        scene_manifest.append(
            {
                "scene": scene,
                "record_id": record_id,
                "plan": row.get("densegen__plan"),
                "authority": plan.authority.value,
                "ordering_status": plan.ordering_status.value,
                "realization_sha256": plan.realization_digest,
            }
        )

    outputs = _required_mapping(endpoint.get("outputs"), field_name="outputs")
    output_path = (
        config_path.parent / _required_text(outputs.get("directory"), field_name="outputs.directory")
    ).resolve()
    if output_path.exists() and not replace:
        msg = f"output already exists: {output_path}; pass replace=True explicitly"
        raise FileExistsError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        realized_dir = temp_path / "realized"
        plans_dir = temp_path / "plans"
        realized_dir.mkdir()
        plans_dir.mkdir()
        for scene, payload in realized_payloads.items():
            (realized_dir / f"{scene}.json").write_text(payload + "\n", encoding="utf-8")
        for scene, payload in plan_payloads.items():
            (plans_dir / f"{scene}.json").write_text(payload + "\n", encoding="utf-8")
        source_documents = tuple(documents)
        projection = BaseRenderDuplexProjection(
            source_documents,
            realized_arrays=realized_by_digest,
            presentation=duplex_presentation,
        )
        render_collection_poster_png(
            source_documents,
            temp_path / "poster.png",
            duplex_frame_renderer=projection.render_rgba,
        )
        render_collection_mp4(
            source_documents,
            temp_path / "playback.mp4",
            duplex_frame_renderer=projection.render_rgba,
            seconds_per_step=seconds_per_step,
            hold_seconds=hold_seconds,
            lead_seconds=lead_seconds,
            scene_transition_seconds=scene_transition_seconds,
        )
        artifact_paths = sorted(path for path in temp_path.rglob("*") if path.is_file())
        manifest = {
            "schema": "densegen.solution_path_playback_bundle.v1",
            "endpoint_id": endpoint.get("endpoint_id"),
            "title": endpoint_title,
            "presentation": {
                "color_profile": color_profile,
                "show_legend": show_legend,
                "graph_detail": graph_detail,
                "graph_fraction": graph_fraction,
                "show_edge_costs": show_edge_costs,
                "show_authority_notice": show_authority_notice,
                "show_distance_bracket": show_distance_bracket,
                "seconds_per_step": seconds_per_step,
                "hold_seconds": hold_seconds,
                "lead_seconds": lead_seconds,
                "scene_transition_seconds": scene_transition_seconds,
                "collection_order": list(collection_order),
            },
            "duplex": {
                "fixed_element_annotations": fixed_element_annotations,
                "consensus_suffix": consensus_suffix,
                "anchored_illustration": (
                    None
                    if anchored_presentation is None
                    else {
                        "asset": anchored_presentation.asset_id,
                        "constraint": anchored_presentation.constraint_name,
                        "reveal": anchored_presentation.reveal,
                    }
                ),
            },
            "endpoint_spec_sha256": _sha256_file(config_path),
            "source": {
                "kind": source.get("kind"),
                "repository": source.get("repository"),
                "table": source_ref,
                "sha256": observed_sha256,
            },
            "scenes": scene_manifest,
            "artifacts": [
                {
                    "path": str(path.relative_to(temp_path)),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifact_paths
            ],
        }
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        temp_path.replace(output_path)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return output_path
