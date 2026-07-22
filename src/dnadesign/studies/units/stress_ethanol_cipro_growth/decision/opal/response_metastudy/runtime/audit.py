"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/audit.py

Orchestrate the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_bundle import (
    build_all_primary_measurements,
)

from ...source_evidence import response_window_round0_source_evidence_root
from ..core.contracts import (
    DEFAULT_RECOMMENDATION_THRESHOLDS,
    SFXI_SOURCE_PROVENANCE,
    MetastudyPaths,
)
from ..core.policies import CANONICAL_SFXI_POLICY_ID, audit_policy_specs
from ..evaluation.candidates import build_top_candidate_table
from ..evaluation.comparison_panel import build_policy_comparison_panel
from ..evaluation.correlations import build_pairwise_correlations
from ..evaluation.metric_behavior import build_denominator_sensitivity
from ..evaluation.metric_contract import build_metric_contract_tests, build_rmf_cardinality_pressure
from ..evaluation.model_validation import (
    cross_validate_random_forest,
    cross_validate_random_forest_by_group,
    summarize_model_validation,
)
from ..evaluation.overlap import build_overlap_by_k
from ..evaluation.pressure_tests import build_pressure_tests
from ..evaluation.recommendation import choose_recommendation
from ..evaluation.recompute import assert_canonical_sfxi_recompute, validate_canonical_sfxi_recompute
from ..evaluation.response_examples import build_response_example_rows
from ..evaluation.scoring import score_sfxi_evidence
from ..evaluation.sfxi_greedy_replay import build_historical_sfxi_greedy_replay
from ..evaluation.summaries import summarize_policies
from ..evaluation.support import build_setpoint_support
from .campaign_calibration import compare_campaign_to_screen_calibration
from .candidate_identity import load_response_candidate_identity_bindings
from .label_truth import resolve_configured_label_truth
from .loading import (
    assert_candidate_alignment,
    assert_shared_observed_labels,
    load_campaign_reader_bundle,
    load_candidate_matrix,
    load_label_source_frame,
    load_observed_label_frame,
    load_sfxi_evidence_frame,
    load_stress_campaign_contract,
    load_training_matrix,
)
from .manifest import write_metastudy_manifest
from .measurement_selection import load_response_measurement_selection
from .observed_sfxi import build_historical_observed_sfxi_evidence
from .publication import create_staging_dir, publish_staging_dir, remove_staging_dir, sha256_arrays
from .response_screen import build_response_metric_screen
from .response_screen_publication import response_screen_manifest
from .review_bundle import ReviewBundleEvidence, materialize_review_bundle
from .run_contracts import assert_shared_label_sources, predictor_parity
from .selected_reader_rows import build_selected_bootstrap_draws, build_selected_response_labels


def run_metastudy(
    *,
    repo_root: Path,
    reader_bundle_root: Path,
    candidate_binding_bundle_root: Path,
    out_dir: Path,
    overwrite: bool,
    top_k: int = 6,
) -> dict[str, object]:
    final_dir = out_dir.resolve()
    stage = create_staging_dir(final_dir, overwrite=overwrite)
    try:
        manifest = _materialize_metastudy(
            repo_root=repo_root,
            reader_bundle_root=reader_bundle_root,
            candidate_binding_bundle_root=candidate_binding_bundle_root,
            out_dir=stage,
            top_k=top_k,
        )
        publish_staging_dir(stage, final_dir, overwrite=overwrite)
    except BaseException:
        remove_staging_dir(stage)
        raise
    manifest["output_dir"] = str(final_dir)
    return manifest


def _materialize_metastudy(
    *,
    repo_root: Path,
    reader_bundle_root: Path,
    candidate_binding_bundle_root: Path,
    out_dir: Path,
    top_k: int,
) -> dict[str, object]:
    paths = MetastudyPaths(
        repo_root=repo_root.resolve(),
        reader_bundle_root=reader_bundle_root.resolve(),
        out_dir=out_dir.resolve(),
        campaign_root=response_window_round0_source_evidence_root(repo_root).resolve(),
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    stress_campaign = load_stress_campaign_contract(paths)
    label_truth_state = resolve_configured_label_truth(stress_campaign.config_path)
    target_views = stress_campaign.target_views
    target_views_by_id = {view.id: view for view in target_views}
    sfxi_evidence = tuple(
        load_sfxi_evidence_frame(
            paths,
            source,
            target_view=target_views_by_id[source.target_view_id],
            stress_campaign=stress_campaign,
        )
        for source in SFXI_SOURCE_PROVENANCE
    )
    assert_candidate_alignment(sfxi_evidence)
    predictor_parity_record = predictor_parity(sfxi_evidence)
    label_frames = tuple(load_observed_label_frame(paths, source) for source in SFXI_SOURCE_PROVENANCE)
    assert_shared_observed_labels(label_frames)
    observed_labels = label_frames[0]
    label_source_frames = tuple(
        load_label_source_frame(paths, source, labels=labels)
        for source, labels in zip(SFXI_SOURCE_PROVENANCE, label_frames, strict=True)
    )
    assert_shared_label_sources(label_source_frames)
    label_sources = label_source_frames[0]
    reader_bundle = load_campaign_reader_bundle(paths, stress_campaign)
    measurement_selection = load_response_measurement_selection(
        Path(__file__).resolve().parents[1] / "config/response_model_screen_selection.yaml",
        reader_designs=reader_bundle.designs,
        primary_reduction_id=reader_bundle.primary_reduction_id,
    )
    candidate_identity_bindings = load_response_candidate_identity_bindings(
        measurement_selection=measurement_selection.rows,
        excluded_designs=measurement_selection.excluded_designs,
        bundle_root=candidate_binding_bundle_root,
    )
    observed_sfxi = build_historical_observed_sfxi_evidence(
        label_sources,
        observed_labels,
        sfxi_evidence=sfxi_evidence,
        label_truth_state=label_truth_state,
        candidate_bindings=candidate_identity_bindings,
    )
    response_labels = build_selected_response_labels(
        reader_bundle,
        candidate_identity_bindings=candidate_identity_bindings.rows,
    )
    response_draws = build_selected_bootstrap_draws(
        reader_bundle,
        candidate_identity_bindings=candidate_identity_bindings.rows,
    )
    all_primary_measurements = build_all_primary_measurements(reader_bundle)
    for evidence in sfxi_evidence:
        if evidence.stats_n_train != len(observed_labels):
            raise ValueError(
                f"{evidence.source.source_id}: run metadata reports {evidence.stats_n_train} training rows; "
                f"the shared label ledger has {len(observed_labels)}."
            )
        if evidence.stats_n_scored != len(evidence.predictions):
            raise ValueError(
                f"{evidence.source.source_id}: run metadata reports {evidence.stats_n_scored} scored rows; "
                f"the prediction ledger has {len(evidence.predictions)}."
            )
    reference_evidence = sfxi_evidence[0]
    if reference_evidence.records_path is None:
        raise RuntimeError("response metric metastudy did not resolve candidate records for model validation.")
    sfxi_x_train, sfxi_y_train = load_training_matrix(
        reference_evidence.records_path,
        x_column=reference_evidence.x_column_name,
        labels=observed_labels,
    )
    response_ids = candidate_identity_bindings.rows["id"].astype(str).tolist()
    response_x_train = load_candidate_matrix(
        stress_campaign.candidate_records_path,
        x_column=stress_campaign.x_column_name,
        candidate_ids=response_ids,
    )
    response_screen = build_response_metric_screen(
        response_labels,
        response_draws,
        all_primary_measurements,
        reader_bundle.events,
        reader_designs=reader_bundle.designs,
        reader_wells=reader_bundle.wells,
        reader_traces=reader_bundle.traces,
        reference_design_id=reader_bundle.reference_design_id,
        primary_reduction_id=reader_bundle.primary_reduction_id,
        label_ids=response_ids,
        x_train=response_x_train,
        groups=candidate_identity_bindings.rows["reader_experiment_id"].astype(str).to_numpy(),
        random_forest_params=stress_campaign.model_params,
        target_views=target_views,
    )
    campaign_to_screen_calibration = compare_campaign_to_screen_calibration(
        response_screen.calibration, configured_by_view=stress_campaign.rmf_calibration_by_view
    )
    response_examples = build_response_example_rows(
        response_screen.uncertainty,
        examples=reader_bundle.response_examples,
        selection_view_ids=tuple(target_view.id for target_view in target_views),
    )
    shuffled_validation = cross_validate_random_forest(
        sfxi_x_train,
        sfxi_y_train,
        target_views=target_views,
        target_view_denoms={evidence.target_view.id: evidence.denom for evidence in sfxi_evidence},
        model_params=reference_evidence.model_params,
        seeds=(3, 7, 19, 29, 43),
        n_splits=5,
        yops_eps=reference_evidence.yops_eps,
        scaling_percentile=reference_evidence.scaling_percentile,
        scaling_min_n=reference_evidence.scaling_min_n,
        scaling_eps=reference_evidence.scaling_eps,
        intensity_log2_offset_delta=reference_evidence.intensity_log2_offset_delta,
    )
    grouped_validation = cross_validate_random_forest_by_group(
        sfxi_x_train,
        sfxi_y_train,
        groups=label_sources["reader_experiment_id"].astype(str).to_numpy(),
        target_views=target_views,
        target_view_denoms={evidence.target_view.id: evidence.denom for evidence in sfxi_evidence},
        model_params=reference_evidence.model_params,
        seeds=(3, 7, 19, 29, 43),
        yops_eps=reference_evidence.yops_eps,
        scaling_percentile=reference_evidence.scaling_percentile,
        scaling_min_n=reference_evidence.scaling_min_n,
        scaling_eps=reference_evidence.scaling_eps,
        intensity_log2_offset_delta=reference_evidence.intensity_log2_offset_delta,
    )
    model_validation = pd.concat([shuffled_validation, grouped_validation], ignore_index=True)
    grouped_model_validation_summary = summarize_model_validation(
        model_validation,
        split_strategy="leave_one_experiment_out",
    )
    shuffled_model_validation_summary = summarize_model_validation(
        model_validation,
        split_strategy="shuffled_kfold",
    )
    sfxi_training_matrix_sha256 = sha256_arrays(
        np.ascontiguousarray(sfxi_x_train),
        np.ascontiguousarray(sfxi_y_train),
    )
    response_x_matrix_sha256 = sha256_arrays(np.ascontiguousarray(response_x_train))
    policy_specs = audit_policy_specs()
    scored = {
        policy.id: {evidence.target_view.id: score_sfxi_evidence(evidence, policy) for evidence in sfxi_evidence}
        for policy in policy_specs
    }
    summary = summarize_policies(policy_specs, sfxi_evidence, scored, top_k=top_k)
    pairwise = build_pairwise_correlations(policy_specs, sfxi_evidence, scored)
    candidates = build_top_candidate_table(policy_specs, sfxi_evidence, scored, top_k=top_k)
    overlap_by_k = build_overlap_by_k(
        policy_specs,
        sfxi_evidence,
        scored,
        k_values=(6, 10, 20, 50, 100, 500, 1000),
    )
    canonical_scored = scored[CANONICAL_SFXI_POLICY_ID]
    canonical_sfxi_validation = validate_canonical_sfxi_recompute(sfxi_evidence, canonical_scored)
    assert_canonical_sfxi_recompute(canonical_sfxi_validation)
    sfxi_greedy_replay = build_historical_sfxi_greedy_replay(sfxi_evidence, canonical_scored, top_k=top_k)
    thresholds = DEFAULT_RECOMMENDATION_THRESHOLDS
    support_thresholds = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    setpoint_support = build_setpoint_support(sfxi_evidence, canonical_scored, thresholds=support_thresholds)
    intrinsic_tests = build_metric_contract_tests(target_views)
    rmf_cardinality_pressure = build_rmf_cardinality_pressure()
    recommendation = choose_recommendation(
        summary,
        thresholds=thresholds,
        model_validation_summary=grouped_model_validation_summary,
    )
    policy_by_id = {policy.id: policy for policy in policy_specs}
    denominator_policy_ids = (CANONICAL_SFXI_POLICY_ID, str(recommendation["comparison_policy_id"]))
    denominator_policies = tuple(
        policy_by_id[policy_id] for policy_id in denominator_policy_ids if policy_id in policy_by_id
    )
    denominator_sensitivity = build_denominator_sensitivity(
        denominator_policies,
        sfxi_evidence,
        scored,
        factors=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        top_k=top_k,
    )
    observed_label_ids = set(observed_labels["id"].astype(str))
    comparison_panel = build_policy_comparison_panel(
        candidates,
        recommendation=recommendation,
        observed_label_ids=observed_label_ids,
        per_target_view=3,
    )
    pressure_tests = build_pressure_tests(
        summary=summary,
        pairwise=pairwise,
        canonical_sfxi_validation=canonical_sfxi_validation,
        recommendation=recommendation,
        thresholds=thresholds,
        model_validation_summary=grouped_model_validation_summary,
        setpoint_support=setpoint_support,
        intrinsic_tests=intrinsic_tests,
    )
    artifact_records = materialize_review_bundle(
        paths,
        ReviewBundleEvidence(
            summary=summary,
            pairwise=pairwise,
            candidates=candidates,
            comparison_panel=comparison_panel,
            overlap_by_k=overlap_by_k,
            denominator_sensitivity=denominator_sensitivity,
            pressure_tests=pressure_tests,
            model_validation=model_validation,
            setpoint_support=setpoint_support,
            response_screen=response_screen,
            response_examples=response_examples,
            rmf_cardinality_pressure=rmf_cardinality_pressure,
            observed_sfxi_components=observed_sfxi.components,
            observed_sfxi_robustness=observed_sfxi.robustness,
            sfxi_greedy_replay=sfxi_greedy_replay,
            scored=scored,
            sfxi_evidence=sfxi_evidence,
            thresholds=thresholds,
            recommendation=recommendation,
            canonical_sfxi_validation=canonical_sfxi_validation,
            primary_reduction_id=reader_bundle.primary_reduction_id,
            label_truth_state=label_truth_state,
        ),
    )
    return write_metastudy_manifest(
        paths=paths,
        sfxi_evidence=sfxi_evidence,
        stress_campaign=stress_campaign,
        reader_bundle=reader_bundle,
        measurement_selection=measurement_selection,
        label_truth_state=label_truth_state,
        candidate_identity_bindings=candidate_identity_bindings,
        policy_specs=policy_specs,
        top_k=top_k,
        sfxi_training_matrix_sha256=sfxi_training_matrix_sha256,
        response_x_matrix_sha256=response_x_matrix_sha256,
        recommendation=recommendation,
        canonical_sfxi_validation=canonical_sfxi_validation,
        artifact_records=artifact_records,
        predictor_parity=predictor_parity_record,
        grouped_model_validation_summary=grouped_model_validation_summary,
        shuffled_model_validation_summary=shuffled_model_validation_summary,
        response_metric_screen=response_screen_manifest(
            response_screen,
            primary_reduction_id=reader_bundle.primary_reduction_id,
            campaign_to_screen_calibration=campaign_to_screen_calibration,
            campaign_model_params=stress_campaign.model_params,
        ),
    )
