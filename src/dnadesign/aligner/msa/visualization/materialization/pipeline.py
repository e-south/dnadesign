"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/materialization/pipeline.py

Generic MSA QC and visualization sidecar materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.aligner.msa.fasta import load_fasta_records
from dnadesign.aligner.msa.validation import validate_aligned_fasta_records
from dnadesign.aligner.msa.visualization.contracts.annotation_tracks import (
    load_annotation_tracks,
    validate_annotation_track_ranges,
)
from dnadesign.aligner.msa.visualization.contracts.exemplar_rows import (
    load_exemplar_rows,
    validate_exemplar_rows,
)
from dnadesign.aligner.msa.visualization.contracts.models import (
    MsaVisualizationRequest,
    MsaVisualizationResult,
    PositionQc,
    ProfileQc,
)
from dnadesign.aligner.msa.visualization.contracts.panel_spec import load_panel_spec
from dnadesign.aligner.msa.visualization.materialization.qc import (
    build_position_qc,
    build_profile_qc,
    sha256_text,
)
from dnadesign.aligner.msa.visualization.materialization.writers import (
    write_html_report,
    write_index_manifest,
    write_position_qc_csv,
    write_profile_qc,
)
from dnadesign.aligner.msa.visualization.renderers.exemplar_windows import (
    write_exemplar_windows_svg,
)
from dnadesign.aligner.msa.visualization.renderers.panels import (
    write_alignment_overview_svg,
    write_consensus_histogram_svg,
)
from dnadesign.aligner.msa.visualization.renderers.profile_qc import write_profile_qc_svg


def materialize_msa_visualizations(request: MsaVisualizationRequest) -> MsaVisualizationResult:
    """Materialize generic QC, CSV, SVG, and HTML sidecars for aligned FASTAs."""

    request.output_root.mkdir(parents=True, exist_ok=True)
    profile_qcs: list[ProfileQc] = []
    position_rows: list[PositionQc] = []
    missing_profile_ids: list[str] = []
    annotation_tracks = load_annotation_tracks(request.annotation_tracks_yaml)
    exemplar_spec = load_exemplar_rows(request.exemplar_rows_yaml)
    panel_spec = load_panel_spec(request.panel_spec_yaml)

    for profile_id in request.profile_ids:
        aligned_fasta = request.alignment_root / f"{profile_id}.aligned.fasta"
        if not aligned_fasta.exists():
            if request.allow_missing_profiles:
                missing_profile_ids.append(profile_id)
                continue
            raise FileNotFoundError(aligned_fasta)

        records = load_fasta_records(aligned_fasta, alphabet="protein", allow_gaps=True)
        validate_aligned_fasta_records(records, target_row_id=request.target_row_id)
        target_aligned = records[request.target_row_id]
        target_ungapped = target_aligned.replace("-", "")
        target_hash = sha256_text(target_ungapped)
        if request.target_sequence_hash and target_hash != request.target_sequence_hash:
            raise ValueError(
                f"{profile_id} target row hash {target_hash} does not match "
                f"declared target row hash {request.target_sequence_hash}"
            )

        positions = build_position_qc(profile_id=profile_id, records=records, target_row_id=request.target_row_id)
        qc = build_profile_qc(
            profile_id=profile_id,
            aligned_fasta=aligned_fasta,
            records=records,
            target_aligned=target_aligned,
            target_hash=target_hash,
            output_root=request.output_root,
        )
        validate_annotation_track_ranges(
            profile_id=profile_id,
            tracks=annotation_tracks,
            canonical_position_count=qc.canonical_position_count,
        )
        exemplar_rows = exemplar_spec.rows_for_profile(profile_id)
        validate_exemplar_rows(profile_id=profile_id, records=records, rows=exemplar_rows)
        if panel_spec.overview_enabled and not exemplar_rows:
            raise ValueError(f"{profile_id} alignment overview requires exemplar rows")
        write_profile_qc(qc, request)
        write_profile_qc_svg(qc, positions, annotation_tracks)
        if annotation_tracks and exemplar_rows:
            write_exemplar_windows_svg(
                qc=qc,
                records=records,
                target_row_id=request.target_row_id,
                tracks=annotation_tracks,
                exemplar_rows=exemplar_rows,
            )
        if panel_spec.overview_enabled:
            write_alignment_overview_svg(
                qc.profile_alignment_overview_svg_path,
                qc=qc,
                records=records,
                target_row_id=request.target_row_id,
                tracks=annotation_tracks,
                exemplar_rows=exemplar_rows,
                panel_spec=panel_spec,
            )
        if panel_spec.consensus_histogram_enabled:
            write_consensus_histogram_svg(
                qc.profile_consensus_histogram_svg_path,
                qc=qc,
                positions=positions,
                tracks=annotation_tracks,
                panel_spec=panel_spec,
            )
        profile_qcs.append(qc)
        position_rows.extend(positions)

    if not profile_qcs:
        raise ValueError("No aligned FASTA profiles were available for visualization")

    position_qc_csv_path = request.output_root / "msa_position_qc.csv"
    write_position_qc_csv(position_qc_csv_path, position_rows)

    html_report_path = request.output_root / "msa_qc.html"
    write_html_report(
        html_report_path,
        profile_qcs,
        missing_profile_ids,
        has_exemplar_rows=exemplar_spec.has_rows,
        has_alignment_overview=panel_spec.overview_enabled,
        has_consensus_histogram=panel_spec.consensus_histogram_enabled,
    )

    index_manifest_path = request.output_root / "msa_visualization_index.yaml"
    write_index_manifest(
        index_manifest_path,
        request=request,
        profile_qcs=profile_qcs,
        missing_profile_ids=tuple(missing_profile_ids),
        position_qc_csv_path=position_qc_csv_path,
        html_report_path=html_report_path,
        has_alignment_overview=panel_spec.overview_enabled,
        has_consensus_histogram=panel_spec.consensus_histogram_enabled,
    )

    return MsaVisualizationResult(
        profile_ids=tuple(qc.profile_id for qc in profile_qcs),
        missing_profile_ids=tuple(missing_profile_ids),
        index_manifest_path=index_manifest_path,
        position_qc_csv_path=position_qc_csv_path,
        html_report_path=html_report_path,
        profile_qc_paths={qc.profile_id: qc.profile_qc_path for qc in profile_qcs},
        profile_svg_paths={qc.profile_id: qc.profile_svg_path for qc in profile_qcs},
        profile_exemplar_svg_paths={
            qc.profile_id: qc.profile_exemplar_svg_path
            for qc in profile_qcs
            if request.exemplar_rows_yaml and request.annotation_tracks_yaml
        },
        profile_alignment_overview_svg_paths={
            qc.profile_id: qc.profile_alignment_overview_svg_path for qc in profile_qcs if panel_spec.overview_enabled
        },
        profile_consensus_histogram_svg_paths={
            qc.profile_id: qc.profile_consensus_histogram_svg_path
            for qc in profile_qcs
            if panel_spec.consensus_histogram_enabled
        },
    )
