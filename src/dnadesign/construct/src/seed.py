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
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.usr import Dataset, compute_id, default_usr_root, normalize_sequence, normalize_usr_root

from .errors import ConfigError
from .runtime import _ensure_construct_registry

_SEED_ASSET = "promoter_swap_demo.yaml"


@dataclass(frozen=True)
class SeedDatasetEntry:
    label: str
    manifest_id: str
    role: str
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
        role = str(item.get("role") or "").strip()
        source_ref = str(item.get("source_ref") or "").strip()
        topology = str(item.get("topology") or "").strip()
        if not label or not role or not topology:
            raise ConfigError("Seed entries require non-empty label, role, and topology values.")
        seq = _normalize_seed_sequence(str(item.get("sequence") or ""), label=label)
        raw_aliases = item.get("aliases") or []
        if raw_aliases and not isinstance(raw_aliases, list):
            raise ConfigError(f"Seed entry '{label}' aliases must be a YAML list of strings.")
        aliases = tuple(sorted({str(alias).strip() for alias in raw_aliases if str(alias).strip()}))
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


def _ensure_dataset(root: Path, name: str, *, notes: str) -> Dataset:
    dataset = Dataset(root, name)
    if not dataset.records_path.exists():
        dataset.init(source="construct seed promoter-swap-demo", notes=notes)
    return dataset


def _attach_seed_overlay(dataset: Dataset, entries: List[SeedDatasetEntry]) -> None:
    frame = pd.DataFrame(
        [
            {
                "id": entry.record_id,
                "construct_seed__label": entry.label,
                "construct_seed__manifest_id": entry.manifest_id,
                "construct_seed__role": entry.role,
                "construct_seed__source_ref": entry.source_ref,
                "construct_seed__topology": entry.topology,
                "construct_seed__sha256": entry.sha256,
            }
            for entry in entries
        ]
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "construct_seed.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        dataset.attach(
            path,
            namespace="construct_seed",
            key="id",
            key_col="id",
            columns=[col for col in frame.columns if col != "id"],
            allow_overwrite=True,
            note="dnadesign.construct curated seed metadata",
        )


def _attach_usr_label_overlay(dataset: Dataset, entries: List[SeedDatasetEntry]) -> None:
    frame = pd.DataFrame(
        [
            {
                "id": entry.record_id,
                "usr_label__primary": entry.label,
                "usr_label__aliases": list(entry.aliases),
            }
            for entry in entries
        ]
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "usr_label.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        dataset.attach(
            path,
            namespace="usr_label",
            key="id",
            key_col="id",
            columns=[col for col in frame.columns if col != "id"],
            allow_overwrite=True,
            note="dnadesign.usr standardized human-readable sequence labels",
        )


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
        "demo_id": "promoter_swap_pdual10",
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
            "The full pDual-10 record contains two exact J23105 matches; choose slot_a or slot_b explicitly.",
            "The earlier scaffold-only interval [405, 440) does not apply to the full pDual-10 record.",
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


def import_seed_manifest(*, root: str | Path, manifest: str | Path) -> ManifestImportResult:
    root_path = normalize_usr_root(root or default_usr_root())
    manifest_path = Path(manifest).expanduser().resolve()
    payload = _load_manifest_payload(manifest_path)
    manifest_id, datasets = _dataset_results_from_manifest(payload)

    _ensure_construct_registry(root_path)
    for dataset_result in datasets:
        dataset = _ensure_dataset(root_path, dataset_result.dataset, notes=dataset_result.notes)
        dataset.add_sequences(
            [entry.sequence for entry in dataset_result.entries],
            bio_type="dna",
            alphabet="dna_4",
            source=f"construct seed import-manifest {manifest_id}",
            on_conflict="ignore",
        )
        _attach_seed_overlay(dataset, dataset_result.entries)
        _attach_usr_label_overlay(dataset, dataset_result.entries)
        with dataset.maintenance(reason="materialize"):
            dataset.materialize(namespaces=["usr_label"])

    return ManifestImportResult(root=root_path, manifest_id=manifest_id, datasets=datasets)


def bootstrap_promoter_swap_demo(*, root: str | Path, manifest: str | Path | None = None) -> SeedResult:
    root_path = normalize_usr_root(root or default_usr_root())
    payload = _seed_asset_payload()
    datasets = payload.get("datasets") or {}
    manifest_id = str(payload.get("demo_id") or "promoter_swap_demo").strip()
    anchor_dataset = str(datasets.get("anchors") or "").strip()
    template_dataset = str(datasets.get("templates") or "").strip()
    if not anchor_dataset or not template_dataset:
        raise ConfigError("Seed asset datasets.anchors and datasets.templates are required.")

    anchor_entries = _seed_entries(payload.get("anchors") or [], manifest_id=manifest_id)
    template_entries = _seed_entries(payload.get("templates") or [], manifest_id=manifest_id)
    slots = _seed_slots(payload.get("slots") or [])

    _ensure_construct_registry(root_path)
    anchor_ds = _ensure_dataset(
        root_path,
        anchor_dataset,
        notes="Curated control anchors for construct tracer bullet.",
    )
    template_ds = _ensure_dataset(
        root_path,
        template_dataset,
        notes="Curated template records for construct tracer bullet.",
    )

    anchor_ds.add_sequences(
        [entry.sequence for entry in anchor_entries],
        bio_type="dna",
        alphabet="dna_4",
        source="construct seed promoter-swap-demo",
        on_conflict="ignore",
    )
    template_ds.add_sequences(
        [entry.sequence for entry in template_entries],
        bio_type="dna",
        alphabet="dna_4",
        source="construct seed promoter-swap-demo",
        on_conflict="ignore",
    )
    _attach_seed_overlay(anchor_ds, anchor_entries)
    _attach_seed_overlay(template_ds, template_entries)
    _attach_usr_label_overlay(anchor_ds, anchor_entries)
    _attach_usr_label_overlay(template_ds, template_entries)
    with anchor_ds.maintenance(reason="materialize"):
        anchor_ds.materialize(namespaces=["usr_label"])
    with template_ds.maintenance(reason="materialize"):
        template_ds.materialize(namespaces=["usr_label"])

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
