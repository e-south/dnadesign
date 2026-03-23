"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/seed.py

Curated bootstrap helpers for construct demo datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import yaml

from dnadesign.usr import Dataset, compute_id, default_usr_root, normalize_sequence, normalize_usr_root
from dnadesign.usr_roots import resolve_usr_root_from_env

from .errors import ConfigError
from .output_store import _ensure_construct_registry
from .workspace import project_root_or_none

_SEED_ASSET = "anchor_template_demo.yaml"


@dataclass(frozen=True)
class SeedDatasetEntry:
    label: str
    manifest_id: str
    role: str | None
    source_ref: str
    topology: str
    sequence: str
    sha256: str
    record_id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SeedSlot:
    slot: str
    template_label: str
    incumbent_label: str
    start: int
    end: int
    expected_template_sequence: str


@dataclass(frozen=True)
class SeedResult:
    root: Path
    anchor_dataset: str
    template_dataset: str
    anchor_entries: List[SeedDatasetEntry]
    template_entries: List[SeedDatasetEntry]
    slots: List[SeedSlot]
    manifest_path: Path | None


@dataclass(frozen=True)
class ManifestDatasetResult:
    dataset: str
    notes: str
    entries: List[SeedDatasetEntry]


@dataclass(frozen=True)
class ManifestImportResult:
    root: Path
    manifest_id: str
    datasets: List[ManifestDatasetResult]


def _normalize_seed_sequence(sequence: str, *, label: str) -> str:
    text = "".join(str(sequence or "").split())
    if not text:
        raise ConfigError(f"Seed sequence for '{label}' cannot be empty.")
    return normalize_sequence(text, "dna", "dna_4")


def _seed_asset_payload() -> dict:
    asset = resources.files("dnadesign.construct").joinpath("src", "seeds", _SEED_ASSET)
    try:
        text = asset.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Construct seed asset could not be read: {_SEED_ASSET}") from exc
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Seed asset '{_SEED_ASSET}' must be a YAML mapping.")
    return payload


def _seed_entries(items: Iterable[dict], *, manifest_id: str) -> List[SeedDatasetEntry]:
    entries: list[SeedDatasetEntry] = []
    for item in items:
        if not isinstance(item, dict):
            raise ConfigError("Seed entries must be YAML mappings.")
        label = str(item.get("label") or "").strip()
        role = str(item.get("intended_role") or item.get("role") or "").strip() or None
        source_ref = str(item.get("source_ref") or "").strip()
        topology = str(item.get("topology") or "").strip()
        if not label or not topology:
            raise ConfigError("Seed entries require non-empty label and topology values.")
        seq = _normalize_seed_sequence(str(item.get("sequence") or ""), label=label)
        raw_aliases = item.get("aliases") or []
        if raw_aliases and not isinstance(raw_aliases, list):
            raise ConfigError(f"Seed entry '{label}' aliases must be a YAML list of strings.")
        aliases_out: list[str] = []
        for alias in raw_aliases:
            if not isinstance(alias, str):
                raise ConfigError(f"Seed entry '{label}' aliases must contain only strings.")
            alias_text = alias.strip()
            if alias_text:
                aliases_out.append(alias_text)
        aliases = tuple(sorted(set(aliases_out)))
        digest = hashlib.sha256(seq.encode("utf-8")).hexdigest()
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if expected_sha and digest != expected_sha:
            raise ConfigError(f"Seed entry '{label}' sha256 mismatch. Expected {expected_sha}, observed {digest}.")
        entries.append(
            SeedDatasetEntry(
                label=label,
                manifest_id=manifest_id,
                role=role,
                source_ref=source_ref,
                topology=topology,
                sequence=seq,
                sha256=digest,
                record_id=compute_id("dna", seq),
                aliases=aliases,
            )
        )
    if not entries:
        raise ConfigError("Seed asset must include at least one entry.")
    return entries


def _seed_slots(items: Iterable[dict]) -> List[SeedSlot]:
    slots: list[SeedSlot] = []
    for item in items:
        if not isinstance(item, dict):
            raise ConfigError("Seed slot entries must be YAML mappings.")
        slot = str(item.get("slot") or "").strip()
        template_label = str(item.get("template_label") or "").strip()
        incumbent_label = str(item.get("incumbent_label") or "").strip()
        if not slot or not template_label or not incumbent_label:
            raise ConfigError("Seed slot entries require non-empty slot, template_label, and incumbent_label.")
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Seed slot '{slot}' start/end must be integers.") from exc
        if start < 0 or end < 0:
            raise ConfigError(f"Seed slot '{slot}' start/end must be >= 0.")
        if end <= start:
            raise ConfigError(f"Seed slot '{slot}' end must be greater than start.")
        slots.append(
            SeedSlot(
                slot=slot,
                template_label=template_label,
                incumbent_label=incumbent_label,
                start=start,
                end=end,
                expected_template_sequence=_normalize_seed_sequence(
                    str(item.get("expected_template_sequence") or ""),
                    label=f"slot {slot}",
                ),
            )
        )
    if not slots:
        raise ConfigError("Seed asset must include at least one replacement slot.")
    return slots


def _seed_overlay_table(entries: List[SeedDatasetEntry]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("construct_seed__label", pa.string()),
            pa.field("construct_seed__manifest_id", pa.string()),
            pa.field("construct_seed__role", pa.string()),
            pa.field("construct_seed__source_ref", pa.string()),
            pa.field("construct_seed__topology", pa.string()),
            pa.field("construct_seed__sha256", pa.string()),
        ]
    )
    return pa.table(
        {
            "id": pa.array([entry.record_id for entry in entries], type=pa.string()),
            "construct_seed__label": pa.array([entry.label for entry in entries], type=pa.string()),
            "construct_seed__manifest_id": pa.array([entry.manifest_id for entry in entries], type=pa.string()),
            "construct_seed__role": pa.array([entry.role or "" for entry in entries], type=pa.string()),
            "construct_seed__source_ref": pa.array([entry.source_ref for entry in entries], type=pa.string()),
            "construct_seed__topology": pa.array([entry.topology for entry in entries], type=pa.string()),
            "construct_seed__sha256": pa.array([entry.sha256 for entry in entries], type=pa.string()),
        },
        schema=schema,
    )


def _usr_label_overlay_table(entries: List[SeedDatasetEntry]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("usr_label__primary", pa.string()),
            pa.field("usr_label__aliases", pa.list_(pa.string())),
        ]
    )
    return pa.table(
        {
            "id": pa.array([entry.record_id for entry in entries], type=pa.string()),
            "usr_label__primary": pa.array([entry.label for entry in entries], type=pa.string()),
            "usr_label__aliases": pa.array([list(entry.aliases) for entry in entries], type=pa.list_(pa.string())),
        },
        schema=schema,
    )


def _materialize_usr_labels(dataset: Dataset) -> None:
    with dataset.maintenance(reason="materialize"):
        dataset.materialize(namespaces=["usr_label"])


def _seed_dataset(
    dataset: Dataset,
    *,
    entries: List[SeedDatasetEntry],
    notes: str,
    source: str,
) -> None:
    with dataset.write_session() as session:
        session.init_if_missing(source=source, notes=notes)
        session.add_sequences(
            [entry.sequence for entry in entries],
            bio_type="dna",
            alphabet="dna_4",
            source=source,
            on_conflict="ignore",
        )
        session.write_overlay(
            "construct_seed",
            _seed_overlay_table(entries),
            overwrite=True,
            note="dnadesign.construct curated seed metadata",
        )
        session.write_overlay(
            "usr_label",
            _usr_label_overlay_table(entries),
            overwrite=True,
            note="dnadesign.usr standardized human-readable sequence labels",
        )
    _materialize_usr_labels(dataset)


def _write_manifest(
    *,
    path: Path,
    anchor_dataset: str,
    template_dataset: str,
    anchor_entries: List[SeedDatasetEntry],
    template_entries: List[SeedDatasetEntry],
    slots: List[SeedSlot],
) -> None:
    payload = {
        "demo_id": "anchor_template_demo",
        "datasets": {
            "anchors": anchor_dataset,
            "templates": template_dataset,
        },
        "anchors": {
            entry.label: {
                "record_id": entry.record_id,
                "length_bp": len(entry.sequence),
                "sha256": entry.sha256,
            }
            for entry in anchor_entries
        },
        "templates": {
            entry.label: {
                "record_id": entry.record_id,
                "length_bp": len(entry.sequence),
                "topology": entry.topology,
                "sha256": entry.sha256,
            }
            for entry in template_entries
        },
        "slots": {
            slot.slot: {
                "template_label": slot.template_label,
                "incumbent_label": slot.incumbent_label,
                "start": slot.start,
                "end": slot.end,
                "expected_template_sequence": slot.expected_template_sequence,
            }
            for slot in slots
        },
        "notes": [
            (
                "The full template_backbone_dual_slot record contains two exact "
                "anchor_part_short_ref matches; choose slot_a or slot_b explicitly."
            ),
            "This packaged demo uses the full template record, not an older scaffold-only slice.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_manifest_payload(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Seed manifest not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"Seed manifest could not be read: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in seed manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Seed manifest must be a YAML mapping: {path}")
    return payload


def _resolve_seed_usr_root(root: str | Path | None) -> Path:
    if root is not None:
        return normalize_usr_root(root)

    env_root = resolve_usr_root_from_env()
    if env_root is not None:
        return env_root

    repo_root = project_root_or_none()
    if repo_root is not None:
        usr_pkg_root = (repo_root / "src" / "dnadesign" / "usr").resolve()
        if (usr_pkg_root / "__init__.py").exists():
            return default_usr_root(pkg_root=usr_pkg_root)

    raise ConfigError("construct seed requires --root outside a dnadesign checkout unless DNADESIGN_USR_ROOT is set.")


def _dataset_results_from_manifest(payload: dict) -> tuple[str, List[ManifestDatasetResult]]:
    manifest_id = str(payload.get("manifest_id") or "").strip()
    if not manifest_id:
        raise ConfigError("Seed manifest requires a non-empty manifest_id.")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ConfigError("Seed manifest requires datasets as a non-empty YAML list.")

    results: list[ManifestDatasetResult] = []
    seen: set[str] = set()
    for item in raw_datasets:
        if not isinstance(item, dict):
            raise ConfigError("Seed manifest datasets entries must be YAML mappings.")
        dataset_id = str(item.get("id") or "").strip()
        notes = str(item.get("notes") or "").strip()
        if not dataset_id:
            raise ConfigError("Seed manifest dataset entries require a non-empty id.")
        if dataset_id in seen:
            raise ConfigError(f"Seed manifest duplicates dataset id '{dataset_id}'.")
        records = item.get("records")
        if not isinstance(records, list) or not records:
            raise ConfigError(f"Seed manifest dataset '{dataset_id}' requires a non-empty records list.")
        results.append(
            ManifestDatasetResult(
                dataset=dataset_id,
                notes=notes or f"Seeded by construct manifest '{manifest_id}'.",
                entries=_seed_entries(records, manifest_id=manifest_id),
            )
        )
        seen.add(dataset_id)
    return manifest_id, results


def import_seed_manifest(*, root: str | Path | None, manifest: str | Path) -> ManifestImportResult:
    root_path = _resolve_seed_usr_root(root)
    manifest_path = Path(manifest).expanduser().resolve()
    payload = _load_manifest_payload(manifest_path)
    manifest_id, datasets = _dataset_results_from_manifest(payload)

    _ensure_construct_registry(root_path)
    for dataset_result in datasets:
        _seed_dataset(
            Dataset(root_path, dataset_result.dataset),
            entries=dataset_result.entries,
            notes=dataset_result.notes,
            source=f"construct seed import-manifest {manifest_id}",
        )

    return ManifestImportResult(root=root_path, manifest_id=manifest_id, datasets=datasets)


def bootstrap_anchor_template_demo(*, root: str | Path | None, manifest: str | Path | None = None) -> SeedResult:
    root_path = _resolve_seed_usr_root(root)
    payload = _seed_asset_payload()
    datasets = payload.get("datasets") or {}
    manifest_id = str(payload.get("demo_id") or "anchor_template_demo").strip()
    anchor_dataset = str(datasets.get("anchors") or "").strip()
    template_dataset = str(datasets.get("templates") or "").strip()
    if not anchor_dataset or not template_dataset:
        raise ConfigError("Seed asset datasets.anchors and datasets.templates are required.")

    anchor_entries = _seed_entries(payload.get("anchors") or [], manifest_id=manifest_id)
    template_entries = _seed_entries(payload.get("templates") or [], manifest_id=manifest_id)
    slots = _seed_slots(payload.get("slots") or [])

    _ensure_construct_registry(root_path)
    anchor_ds = Dataset(root_path, anchor_dataset)
    template_ds = Dataset(root_path, template_dataset)

    _seed_dataset(
        anchor_ds,
        entries=anchor_entries,
        notes="Curated anchor parts for the packaged construct demo.",
        source="construct seed anchor-template-demo",
    )
    _seed_dataset(
        template_ds,
        entries=template_entries,
        notes="Curated template backbones for the packaged construct demo.",
        source="construct seed anchor-template-demo",
    )

    manifest_path = Path(manifest).expanduser().resolve() if manifest is not None else None
    if manifest_path is not None:
        _write_manifest(
            path=manifest_path,
            anchor_dataset=anchor_dataset,
            template_dataset=template_dataset,
            anchor_entries=anchor_entries,
            template_entries=template_entries,
            slots=slots,
        )

    return SeedResult(
        root=root_path,
        anchor_dataset=anchor_dataset,
        template_dataset=template_dataset,
        anchor_entries=anchor_entries,
        template_entries=template_entries,
        slots=slots,
        manifest_path=manifest_path,
    )
