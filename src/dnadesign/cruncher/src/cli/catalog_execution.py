"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/catalog_execution.py

Execution helpers for catalog CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from dnadesign.cruncher.app.motif_artifacts import (
    artifact_filename,
    build_densegen_artifact,
    load_motif_payload,
)
from dnadesign.cruncher.cli.catalog_utils import ResolvedTarget, _matrix_site_count_from_tags, _safe_stem
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.viz.logos import logo_subtitle, logo_title


def collect_pwm_payloads(
    *,
    cfg: CruncherConfig,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
    output_format: str,
    log_odds: bool,
) -> tuple[list[dict[str, object]], list[tuple[ResolvedTarget, object]], list[dict[str, str]]]:
    store = CatalogMotifStore(
        catalog_root,
        pwm_source=cfg.catalog.pwm_source,
        site_kinds=cfg.catalog.site_kinds,
        combine_sites=cfg.catalog.combine_sites,
        site_window_lengths=cfg.catalog.site_window_lengths,
        site_window_center=cfg.catalog.site_window_center,
        apply_pwm_window=False,
        min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
        allow_low_sites=cfg.catalog.allow_low_sites,
        pseudocounts=cfg.catalog.pseudocounts,
    )
    payloads: list[dict[str, object]] = []
    resolved: list[tuple[ResolvedTarget, object]] = []
    rows: list[dict[str, str]] = []

    for target in targets:
        pwm = store.get_pwm(target.ref)
        resolved.append((target, pwm))
        info_bits = pwm.information_bits()
        cached_sites = f"{target.entry.site_count}/{target.entry.site_total}" if target.entry.has_sites else "-"
        site_sets = "-" if cfg.catalog.pwm_source == "matrix" else str(len(target.site_entries))
        window = "-"
        if pwm.source_length is not None and pwm.window_start is not None:
            window = f"{pwm.window_start}:{pwm.window_start + pwm.length}/{pwm.source_length}"
        rows.append(
            {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "pwm_source": cfg.catalog.pwm_source,
                "length": str(pwm.length),
                "window": window,
                "info_bits": f"{info_bits:.2f}",
                "sites_cached": cached_sites,
                "matrix_sites": str(pwm.nsites or "-"),
                "site_sets": site_sets,
            }
        )
        record = {
            "tf_name": target.tf_name,
            "source": target.entry.source,
            "motif_id": target.entry.motif_id,
            "pwm_source": cfg.catalog.pwm_source,
            "length": pwm.length,
            "window_start": pwm.window_start,
            "source_length": pwm.source_length,
            "window_strategy": pwm.window_strategy,
            "window_score": pwm.window_score,
            "info_bits": info_bits,
            "nsites": pwm.nsites,
            "sites_cached": target.entry.site_count if target.entry.has_sites else None,
            "sites_cached_total": target.entry.site_total if target.entry.has_sites else None,
            "site_sets": len(target.site_entries) if cfg.catalog.pwm_source == "sites" else None,
        }
        if output_format == "json":
            record["matrix"] = pwm.matrix.tolist()
            if log_odds:
                record["log_odds"] = pwm.log_odds().tolist()
        payloads.append(record)
    return payloads, resolved, rows


def write_densegen_artifacts(
    *,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
    out_dir: Path,
    background: str,
    pseudocount: float | None,
    producer: str,
    overwrite: bool,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    manifest_entries: list[dict[str, object]] = []

    for target in sorted(targets, key=lambda t: (t.entry.source, t.entry.motif_id)):
        motif_path = catalog_root / "normalized" / "motifs" / target.entry.source / f"{target.entry.motif_id}.json"
        if not motif_path.exists():
            raise FileNotFoundError(str(motif_path))

        payload = load_motif_payload(motif_path)
        artifact = build_densegen_artifact(
            payload,
            producer=producer,
            background_policy=background,
            pseudocount=pseudocount,
        )
        filename = artifact_filename(
            tf_name=target.tf_name,
            source=target.entry.source,
            motif_id=target.entry.motif_id,
        )
        dest = out_dir / filename
        if dest.exists() and not overwrite:
            raise FileExistsError(str(dest))
        dest.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        rows.append(
            {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "artifact": str(dest),
            }
        )
        manifest_entries.append(
            {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "path": dest.name,
            }
        )
    return rows, manifest_entries


def collect_site_export_rows(
    *,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    table_rows: list[dict[str, str]] = []

    for target in sorted(targets, key=lambda t: (t.tf_name, t.entry.source, t.entry.motif_id)):
        site_entries = target.site_entries or [target.entry]
        for entry in site_entries:
            sites_path = catalog_root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
            if not sites_path.exists():
                raise FileNotFoundError(str(sites_path))
            count = 0
            with sites_path.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    seq = payload.get("sequence")
                    if not seq:
                        site_id = payload.get("site_id") or "unknown"
                        motif_ref = payload.get("motif_ref") or f"{entry.source}:{entry.motif_id}"
                        raise ValueError(f"Missing sequence for site {site_id} in {motif_ref}.")
                    rows.append(
                        {
                            "tf": target.tf_name,
                            "tfbs": seq,
                            "site_id": payload.get("site_id"),
                            "source": payload.get("motif_ref") or f"{entry.source}:{entry.motif_id}",
                        }
                    )
                    count += 1
            table_rows.append(
                {
                    "tf_name": target.tf_name,
                    "source": entry.source,
                    "motif_id": entry.motif_id,
                    "count": str(count),
                }
            )

    return rows, table_rows


def write_site_export(rows: list[dict[str, object]], *, out_path: Path, fmt: str) -> int:
    import pandas as pd

    df = pd.DataFrame(rows)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    return int(len(df))


def render_logo_outputs(
    *,
    cfg: CruncherConfig,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
    catalog: CatalogIndex,
    bits_mode: str,
    dpi: int,
    out_base: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    store = CatalogMotifStore(
        catalog_root,
        pwm_source=cfg.catalog.pwm_source,
        site_kinds=cfg.catalog.site_kinds,
        combine_sites=cfg.catalog.combine_sites,
        site_window_lengths=cfg.catalog.site_window_lengths,
        site_window_center=cfg.catalog.site_window_center,
        apply_pwm_window=False,
        min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
        allow_low_sites=cfg.catalog.allow_low_sites,
        pseudocounts=cfg.catalog.pseudocounts,
    )
    from dnadesign.cruncher.viz.pwm import plot_pwm

    rows: list[dict[str, str]] = []
    outputs: list[str] = []
    for target in targets:
        pwm = store.get_pwm(target.ref)
        info_bits = pwm.information_bits()
        subtitle = logo_subtitle(
            pwm_source=cfg.catalog.pwm_source,
            entry=target.entry,
            catalog=catalog,
            combine_sites=cfg.catalog.combine_sites,
            site_kinds=cfg.catalog.site_kinds,
        )
        title = logo_title(
            tf_name=target.tf_name,
            motif_id=target.entry.motif_id,
            nsites=pwm.nsites,
        )
        stem = _safe_stem(f"{target.tf_name}_{target.entry.source}_{target.entry.motif_id}")
        out_path = out_base / f"{stem}_logo.png"
        plot_pwm(
            pwm,
            mode=bits_mode,
            out=out_path,
            title=title,
            dpi=dpi,
            subtitle=f"sites: {subtitle}" if cfg.catalog.pwm_source == "sites" else subtitle,
        )
        outputs.append(str(out_path))
        rows.append(
            {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "length": str(pwm.length),
                "bits": f"{info_bits:.2f}",
                "output": str(out_path),
            }
        )
    return rows, outputs


def entry_table_rows(entries: Sequence[object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "-"
        matrix_sites = _matrix_site_count_from_tags(entry.tags)
        matrix_sites_flag = str(matrix_sites) if matrix_sites is not None else "-"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        rows.append(
            {
                "tf_name": entry.tf_name,
                "source": entry.source,
                "motif_id": entry.motif_id,
                "organism": organism_label,
                "matrix": matrix_flag,
                "sites": sites_flag,
                "matrix_sites": matrix_sites_flag,
                "site_kind": entry.site_kind or "-",
                "dataset_id": entry.dataset_id or "-",
                "dataset_method": entry.dataset_method or "-",
                "mean_len": mean_len,
                "updated": entry.updated_at.split("T")[0],
            }
        )
    return rows
