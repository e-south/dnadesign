"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/planner.py

Deterministic validation/report planning for dual-context cassette specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.cassette.errors import CassettePlanningError
from dnadesign.cruncher.cassette.models import (
    BoundedSegment,
    CassetteCandidateDesign,
    CassetteEvaluationReport,
    HairpinCassetteSpec,
    NickaseCatalog,
    NickaseCatalogEntry,
    PairContract,
    PlannedNick,
    SpanContract,
    UnsatReason,
    reverse_complement,
)


@dataclass(frozen=True)
class _NickaseMatch:
    nickase: str
    recognition_sequence: str
    site_start_context: int
    site_end_context: int
    site_orientation: str
    nicked_strand: str
    nick_coordinate_context: int

    def to_planned_nick(self, *, context_offset: int) -> PlannedNick:
        return PlannedNick(
            nickase=self.nickase,
            recognition_sequence=self.recognition_sequence,
            site_start=self.site_start_context - context_offset,
            site_end=self.site_end_context - context_offset,
            site_orientation="forward" if self.site_orientation == "+" else "reverse",
            nicked_strand=self.nicked_strand,  # type: ignore[arg-type]
            nick_coordinate=self.nick_coordinate_context - context_offset,
            nick_coordinate_context=self.nick_coordinate_context,
        )

    def key(self) -> tuple[str, int, int, str, str, int]:
        return (
            self.nickase,
            self.site_start_context,
            self.site_end_context,
            self.site_orientation,
            self.nicked_strand,
            self.nick_coordinate_context,
        )


def _pair_map(cassette_length: int, stem_length: int) -> list[PairContract]:
    return [PairContract(left=index, right=cassette_length - 1 - index) for index in range(stem_length)]


def _scan_entry(sequence: str, entry: NickaseCatalogEntry) -> list[_NickaseMatch]:
    recognition = entry.recognition_sequence
    recognition_rc = reverse_complement(recognition)
    width = len(recognition)
    matches: list[_NickaseMatch] = []
    seen: set[tuple[int, str]] = set()
    for start in range(0, len(sequence) - width + 1):
        window = sequence[start : start + width]
        if window == recognition:
            orientation = "+"
        elif window == recognition_rc:
            orientation = "-"
        else:
            continue
        key = (start, orientation)
        if key in seen:
            continue
        seen.add(key)
        if orientation == "+":
            nicked_strand = "primary_strand" if entry.nicked_site_strand == "forward" else "complement_strand"
        else:
            nicked_strand = "complement_strand" if entry.nicked_site_strand == "forward" else "primary_strand"
        if nicked_strand == "primary_strand":
            coordinate = start + entry.cut_offset
        else:
            coordinate = start + (width - entry.cut_offset)
        matches.append(
            _NickaseMatch(
                nickase=entry.id,
                recognition_sequence=entry.recognition_sequence,
                site_start_context=start,
                site_end_context=start + width,
                site_orientation=orientation,
                nicked_strand=nicked_strand,
                nick_coordinate_context=coordinate,
            )
        )
    return matches


def _matches_for_requested_window(
    matches: list[_NickaseMatch],
    *,
    designated_strand: str,
    context_offset: int,
    window_start: int,
    window_end: int,
) -> list[_NickaseMatch]:
    selected: list[_NickaseMatch] = []
    for match in matches:
        if match.nicked_strand != designated_strand:
            continue
        cassette_coordinate = match.nick_coordinate_context - context_offset
        if window_start <= cassette_coordinate <= window_end:
            selected.append(match)
    return selected


def _markdown_report(report: CassetteEvaluationReport) -> str:
    lines = [
        f"# Cassette Report: {report.spec_name}",
        "",
        f"- status: {report.status}",
        f"- designated_strand: {report.designated_strand}",
        f"- spec_path: {report.spec_path}",
        f"- catalog_path: {report.catalog_path}",
    ]
    if report.run_dir:
        lines.append(f"- run_dir: {report.run_dir}")
    if report.candidate is not None:
        candidate = report.candidate
        lines.extend(
            [
                "",
                "## Candidate",
                f"- cassette_sequence: `{candidate.cassette_sequence}`",
                f"- context_sequence: `{candidate.context_sequence}`",
                f"- left_nick: {candidate.left_nick.nickase} at {candidate.left_nick.nick_coordinate}",
                f"- right_nick: {candidate.right_nick.nickase} at {candidate.right_nick.nick_coordinate}",
                (
                    "- bounded_segment: "
                    f"{candidate.bounded_segment.start}..{candidate.bounded_segment.end} "
                    f"(length={candidate.bounded_segment.length})"
                ),
            ]
        )
        if candidate.additional_designated_strand_nicks:
            lines.append("- additional_designated_strand_nicks:")
            for extra in candidate.additional_designated_strand_nicks:
                lines.append(
                    f"  - {extra.nickase} {extra.site_orientation} "
                    f"site=[{extra.site_start},{extra.site_end}) nick={extra.nick_coordinate}"
                )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


def build_cassette_report(
    spec: HairpinCassetteSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog_path: Path,
    catalog: NickaseCatalog,
) -> CassetteEvaluationReport:
    stem5p = spec.topology.stem5p_arm
    loop = spec.topology.loop
    stem3p = reverse_complement(stem5p)
    cassette_sequence = f"{stem5p}{loop}{stem3p}"
    cassette_length = len(cassette_sequence)
    if cassette_length < 2:
        raise CassettePlanningError("cassette sequence must have length >= 2")

    context_offset = len(spec.duplex_context.upstream)
    context_sequence = f"{spec.duplex_context.upstream}{cassette_sequence}{spec.duplex_context.downstream}"
    complement_sequence = reverse_complement(context_sequence)

    for label, request in (("left", spec.nicking.left), ("right", spec.nicking.right)):
        if request.nick_window.end >= cassette_length:
            raise CassettePlanningError(
                f"{label}.nick_window.end={request.nick_window.end} exceeds cassette length {cassette_length}."
            )

    catalog_by_id = catalog.by_id()
    missing = [
        nickase_id
        for nickase_id in {spec.nicking.left.nickase, spec.nicking.right.nickase}
        if nickase_id not in catalog_by_id
    ]
    if missing:
        raise CassettePlanningError(f"Nickase ids not found in catalog: {', '.join(sorted(missing))}")

    all_matches: list[_NickaseMatch] = []
    for nickase_id in {spec.nicking.left.nickase, spec.nicking.right.nickase}:
        all_matches.extend(_scan_entry(context_sequence, catalog_by_id[nickase_id]))

    left_matches = _matches_for_requested_window(
        [match for match in all_matches if match.nickase == spec.nicking.left.nickase],
        designated_strand=spec.nicking.designated_strand,
        context_offset=context_offset,
        window_start=spec.nicking.left.nick_window.start,
        window_end=spec.nicking.left.nick_window.end,
    )
    right_matches = _matches_for_requested_window(
        [match for match in all_matches if match.nickase == spec.nicking.right.nickase],
        designated_strand=spec.nicking.designated_strand,
        context_offset=context_offset,
        window_start=spec.nicking.right.nick_window.start,
        window_end=spec.nicking.right.nick_window.end,
    )

    issues: list[UnsatReason] = []
    if len(left_matches) == 0:
        issues.append(
            UnsatReason(
                code="missing_left_nick",
                message="No designated-strand nick matched the left window.",
                details={"nickase": spec.nicking.left.nickase},
            )
        )
    elif len(left_matches) > 1:
        issues.append(
            UnsatReason(
                code="ambiguous_left_nick",
                message="Multiple designated-strand nicks matched the left window.",
                details={"nickase": spec.nicking.left.nickase, "count": len(left_matches)},
            )
        )
    if len(right_matches) == 0:
        issues.append(
            UnsatReason(
                code="missing_right_nick",
                message="No designated-strand nick matched the right window.",
                details={"nickase": spec.nicking.right.nickase},
            )
        )
    elif len(right_matches) > 1:
        issues.append(
            UnsatReason(
                code="ambiguous_right_nick",
                message="Multiple designated-strand nicks matched the right window.",
                details={"nickase": spec.nicking.right.nickase, "count": len(right_matches)},
            )
        )

    candidate: CassetteCandidateDesign | None = None
    render_contract: dict[str, object] | None = None

    if not issues:
        left_match = left_matches[0]
        right_match = right_matches[0]
        left_nick = left_match.to_planned_nick(context_offset=context_offset)
        right_nick = right_match.to_planned_nick(context_offset=context_offset)
        if left_nick.nick_coordinate >= right_nick.nick_coordinate:
            issues.append(
                UnsatReason(
                    code="invalid_nick_order",
                    message="Left nick coordinate must be strictly less than right nick coordinate.",
                    details={
                        "left": left_nick.nick_coordinate,
                        "right": right_nick.nick_coordinate,
                    },
                )
            )
        else:
            selected_keys = {left_match.key(), right_match.key()}
            extras = [
                match.to_planned_nick(context_offset=context_offset)
                for match in all_matches
                if match.nicked_strand == spec.nicking.designated_strand and match.key() not in selected_keys
            ]
            if spec.nicking.forbid_additional_designated_strand_nicks and extras:
                issues.append(
                    UnsatReason(
                        code="extra_designated_strand_nicks",
                        message="Additional designated-strand nicking sites were detected.",
                        details={"count": len(extras)},
                    )
                )
            else:
                candidate = CassetteCandidateDesign(
                    cassette_sequence=cassette_sequence,
                    context_sequence=context_sequence,
                    complement_sequence=complement_sequence,
                    cassette_length=cassette_length,
                    context_offset=context_offset,
                    stem5p_span=SpanContract(start=0, end=len(stem5p)),
                    loop_span=SpanContract(start=len(stem5p), end=len(stem5p) + len(loop)),
                    stem3p_span=SpanContract(start=len(stem5p) + len(loop), end=cassette_length),
                    pair_map=_pair_map(cassette_length=cassette_length, stem_length=len(stem5p)),
                    left_nick=left_nick,
                    right_nick=right_nick,
                    bounded_segment=BoundedSegment(
                        start=left_nick.nick_coordinate,
                        end=right_nick.nick_coordinate,
                        length=right_nick.nick_coordinate - left_nick.nick_coordinate,
                    ),
                    additional_designated_strand_nicks=extras,
                )
                render_contract = {
                    "schema_version": 1,
                    "workflow": "cassette",
                    "views": {
                        "ssdna_hairpin": {
                            "sequence": cassette_sequence,
                            "stem5p_span": candidate.stem5p_span.model_dump(mode="json"),
                            "loop_span": candidate.loop_span.model_dump(mode="json"),
                            "stem3p_span": candidate.stem3p_span.model_dump(mode="json"),
                            "pair_map": [pair.model_dump(mode="json") for pair in candidate.pair_map],
                        },
                        "linear_duplex": {
                            "primary_sequence": context_sequence,
                            "complement_sequence": complement_sequence,
                            "context_offset": context_offset,
                            "designated_strand": spec.nicking.designated_strand,
                            "left_nick": candidate.left_nick.model_dump(mode="json"),
                            "right_nick": candidate.right_nick.model_dump(mode="json"),
                            "bounded_segment": candidate.bounded_segment.model_dump(mode="json"),
                        },
                    },
                }

    status = "satisfied" if not issues and candidate is not None else "unsatisfied"
    return CassetteEvaluationReport(
        status=status,
        spec_name=spec.name,
        designated_strand=spec.nicking.designated_strand,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        catalog_path=str(catalog_path),
        issues=issues,
        candidate=candidate,
        render_contract=render_contract,
    )


def render_markdown_report(report: CassetteEvaluationReport) -> str:
    return _markdown_report(report)
