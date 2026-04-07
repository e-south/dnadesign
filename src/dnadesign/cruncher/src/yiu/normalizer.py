"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/normalizer.py

Normalize YIU v4 specs into deterministic payload bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.bio import reverse_complement_iupac
from dnadesign.cruncher.yiu.candidate_generation import enumerate_candidates, resolve_window_starts
from dnadesign.cruncher.yiu.domain_models import (
    JunctionSelection,
    MismatchSelection,
    NormalizedMotifContext,
    NormalizedPayload,
    OptimizationDecision,
    OptimizationObjective,
    OptimizationWinner,
    build_ligation_search_state,
)
from dnadesign.cruncher.yiu.errors import YIU_PWM_CONTEXT_INVALID, YIU_PWM_CONTEXT_REQUIRED, raise_yiu_error
from dnadesign.cruncher.yiu.optimizer import select_best_candidate
from dnadesign.cruncher.yiu.payload_resolution import resolve_input_payload
from dnadesign.cruncher.yiu.pwm_context import resolve_motif_context
from dnadesign.cruncher.yiu.scoring import apply_candidate_sequences, build_scorable_motifs
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec


def aligned_complement_3to5(sequence_5to3: str) -> str:
    return "".join(reverse_complement_iupac(base) for base in sequence_5to3)


def _bounded_motif_context(
    *,
    motif_context: NormalizedMotifContext,
    payload_length: int,
) -> NormalizedMotifContext:
    if not motif_context.effective:
        return motif_context
    invalid = [
        motif.motif_instance_id
        for motif in motif_context.motifs
        if motif.start < 0 or motif.end > payload_length or motif.end <= motif.start
    ]
    if not invalid:
        return motif_context
    message = "resolved PWM context includes motif coordinates outside the payload bounds: " + ", ".join(invalid)
    if motif_context.requested_mode == "require":
        raise_yiu_error(YIU_PWM_CONTEXT_REQUIRED, message)
    return NormalizedMotifContext(
        requested_mode=motif_context.requested_mode,
        effective=False,
        source_kind=motif_context.source_kind,
        fallback_reason=message,
        motifs=[],
    )


def normalize_payload(spec: YiuPayloadRenderingSpec, *, workspace_root: Path) -> NormalizedPayload:
    resolved_input = resolve_input_payload(spec.input, workspace_root=workspace_root, spec_name=spec.yiu.name)
    reference_payload_sequence = resolved_input.payload_sequence
    reference_complement_sequence = aligned_complement_3to5(reference_payload_sequence)
    ligation_state = build_ligation_search_state(
        ligation_profile=spec.optimization.mismatches.ligation_profile,
        ligation_awareness_mode=spec.optimization.mismatches.ligation_awareness_mode,
        candidate_positions=spec.optimization.mismatches.candidate_positions,
    )
    motif_context = _bounded_motif_context(
        motif_context=resolve_motif_context(
            pwm_spec=spec.optimization.pwm,
            resolved_input=resolved_input,
            workspace_root=workspace_root,
            spec_name=spec.yiu.name,
        ),
        payload_length=len(reference_payload_sequence),
    )
    junction_starts = resolve_window_starts(
        payload_length=len(reference_payload_sequence),
        junction_spec=spec.optimization.junction,
    )
    candidates = enumerate_candidates(
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
        junction_starts=junction_starts,
        mismatches_spec=spec.optimization.mismatches,
    )
    if not candidates:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            "candidate enumeration produced no valid YIU v4 plans after hard-constraint filtering",
        )
    scorable_motifs = (
        build_scorable_motifs(
            reference_payload_sequence=reference_payload_sequence,
            reference_complement_sequence=reference_complement_sequence,
            motifs=motif_context.motifs,
        )
        if motif_context.effective
        else ()
    )
    optimizer_result = select_best_candidate(
        candidates=candidates,
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
        scorable_motifs=scorable_motifs,
        pwm_effective=motif_context.effective,
        ligation_state=ligation_state,
        bad_pattern_heuristics=spec.optimization.mismatches.bad_pattern_heuristics,
    )
    selected_payload_sequence, selected_complement_sequence = apply_candidate_sequences(
        candidate=optimizer_result.winner,
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
    )
    winner = optimizer_result.winner
    mismatches = [
        MismatchSelection(
            payload_index=mutation.payload_index,
            junction_offset=mutation.junction_offset,
            mutated_strand=mutation.mutated_strand,
            native_base=mutation.native_base,
            mutated_base=mutation.mutated_base,
            opposing_base=mutation.opposing_base,
        )
        for mutation in sorted(winner.mutations, key=lambda item: item.junction_offset)
    ]
    return NormalizedPayload(
        name=spec.yiu.name,
        input_kind=resolved_input.input_kind,  # type: ignore[arg-type]
        payload_label=resolved_input.payload_label,
        site_label=resolved_input.site_label,
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
        selected_payload_sequence=selected_payload_sequence,
        selected_complement_sequence=selected_complement_sequence,
        source_provenance=resolved_input.provenance,
        ligation_profile=spec.optimization.mismatches.ligation_profile,
        ligation_awareness_mode=spec.optimization.mismatches.ligation_awareness_mode,
        bad_pattern_heuristics=spec.optimization.mismatches.bad_pattern_heuristics,
        ligation_state=optimizer_result.ligation_state,
        chosen_ligation_key=optimizer_result.chosen_ligation_key,
        ligation_rationale=list(optimizer_result.ligation_rationale),
        junction=JunctionSelection(
            start=winner.junction_start,
            end=winner.junction_end,
            offsets=[0, 1, 2, 3],
            mode=spec.optimization.junction.mode,
            left_body_length=winner.junction_start,
            right_body_length=len(reference_payload_sequence) - winner.junction_end,
        ),
        mismatches=mismatches,
        motif_context=motif_context,
        optimization_decision=OptimizationDecision(
            candidate_count=optimizer_result.candidate_count,
            objective=OptimizationObjective(
                primary=spec.optimization.pwm.objective.primary,
                secondary=list(spec.optimization.pwm.objective.secondary),
            ),
            winner=OptimizationWinner(
                junction_start=winner.junction_start,
                junction_end=winner.junction_end,
                selected_positions=list(winner.mismatch_positions),
                mutated_strands=[
                    item.mutated_strand for item in sorted(winner.mutations, key=lambda entry: entry.junction_offset)
                ],
                mutated_bases=[
                    item.mutated_base for item in sorted(winner.mutations, key=lambda entry: entry.junction_offset)
                ],
                worst_loss=optimizer_result.score.worst_loss,
                total_loss=optimizer_result.score.total_loss,
                midpoint_distance=winner.midpoint_distance,
                middle_mismatch_count=winner.middle_mismatch_count,
                double_middle_flag=winner.double_middle_flag,
                default_strand_preference_count=winner.default_strand_preference_count,
                lexical_key=winner.lexical_key,
            ),
            trace_sample=optimizer_result.trace_sample,
            trace=list(optimizer_result.trace),
        ),
    )
