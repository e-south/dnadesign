"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/sequence_constraints/engine.py

Compilation and validation engine for final-sequence motif constraints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from .kmers import find_kmer_matches, reverse_complement


def _as_dict(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=False)
    if isinstance(value, dict):
        return dict(value)
    return value


def _normalize_motif(value: str, *, label: str) -> str:
    seq = str(value).strip().upper()
    if not seq:
        raise ValueError(f"{label} must be a non-empty DNA motif.")
    if any(ch not in {"A", "C", "G", "T"} for ch in seq):
        raise ValueError(f"{label} must contain only A/C/G/T characters.")
    return seq


def _find_motif_positions(seq: str, motif: str, bounds: tuple[int, int] | list[int] | None) -> list[int]:
    text = str(seq)
    pattern = _normalize_motif(motif, label="motif")
    lo = None
    hi = None
    if bounds is not None:
        try:
            lo = int(bounds[0])
            hi = int(bounds[1])
        except Exception:
            lo = None
            hi = None
    positions: list[int] = []
    start = 0
    while start < len(text):
        idx = text.find(pattern, start)
        if idx == -1:
            break
        if lo is not None and idx < lo:
            start = idx + 1
            continue
        if hi is not None and idx > hi:
            start = idx + 1
            continue
        positions.append(int(idx))
        start = idx + 1
    return positions


def _spacer_range(raw) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo = int(min(raw))
        hi = int(max(raw))
        return lo, hi
    if raw is None:
        return 0, 0
    value = int(raw)
    return value, value


@dataclass(frozen=True)
class ForbiddenRule:
    name: str
    patterns: tuple[str, ...]
    strands: str


@dataclass(frozen=True)
class CompiledSequenceConstraints:
    forbid_rules: tuple[ForbiddenRule, ...]
    allow_components: tuple[str, ...]
    generation_forbidden_patterns: tuple[str, ...]

    def has_rules(self) -> bool:
        return bool(self.forbid_rules)


@dataclass(frozen=True)
class SequenceValidationResult:
    validation_passed: bool
    violations: list[dict]
    promoter_detail: dict


def _extract_allow_components(sequence_constraints: dict) -> tuple[str, ...]:
    allowlist = list(sequence_constraints.get("allowlist") or [])
    components: list[str] = []
    for item in allowlist:
        block = _as_dict(item) or {}
        if str(block.get("kind") or "") != "fixed_element_instance":
            continue
        selector = _as_dict(block.get("selector")) or {}
        if str(selector.get("fixed_element") or "") != "promoter":
            continue
        for raw in list(selector.get("component") or []):
            comp = str(raw).strip().lower()
            if comp not in {"upstream", "downstream"}:
                raise ValueError("allowlist.selector.component must contain only upstream/downstream.")
            if comp not in components:
                components.append(comp)
    if not components:
        return ("upstream", "downstream")
    return tuple(components)


def _patterns_from_motif_sets(motif_sets: dict[str, dict[str, str]], names: list[str]) -> list[str]:
    patterns: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        set_name = str(raw_name).strip()
        if not set_name:
            raise ValueError("patterns_from_motif_sets entries must be non-empty strings.")
        if set_name not in motif_sets:
            raise ValueError(f"Unknown motif set in sequence constraints: {set_name}")
        values = motif_sets.get(set_name) or {}
        if not values:
            raise ValueError(f"Motif set '{set_name}' is empty.")
        for motif in values.values():
            seq = _normalize_motif(motif, label=f"motif_sets.{set_name}")
            if seq in seen:
                continue
            seen.add(seq)
            patterns.append(seq)
    return patterns


def compile_sequence_constraints(
    *,
    sequence_constraints,
    motif_sets: dict[str, dict[str, str]] | None,
    fixed_elements_dump: dict | None = None,
) -> CompiledSequenceConstraints:
    constraints = _as_dict(sequence_constraints) or {}
    motif_sets_map = _as_dict(motif_sets) or {}
    allow_components = _extract_allow_components(constraints)

    forbid_rules: list[ForbiddenRule] = []
    generation_patterns: list[str] = []
    generation_seen: set[str] = set()
    for raw_rule in list(constraints.get("forbid_kmers") or []):
        rule = _as_dict(raw_rule) or {}
        name = str(rule.get("name") or "").strip()
        if not name:
            raise ValueError("sequence_constraints.forbid_kmers[].name must be set.")
        strands = str(rule.get("strands") or "").strip().lower()
        if strands not in {"forward", "both"}:
            raise ValueError("sequence_constraints.forbid_kmers[].strands must be forward or both.")
        set_names = list(rule.get("patterns_from_motif_sets") or [])
        if not set_names:
            raise ValueError("sequence_constraints.forbid_kmers[].patterns_from_motif_sets must be non-empty.")
        include_rc = bool(rule.get("include_reverse_complements", False))
        patterns = _patterns_from_motif_sets(motif_sets_map, set_names)
        if include_rc:
            for pattern in list(patterns):
                rc = reverse_complement(pattern)
                if rc not in patterns:
                    patterns.append(rc)
        unique_patterns = tuple(sorted(set(patterns)))
        forbid_rules.append(
            ForbiddenRule(
                name=name,
                patterns=unique_patterns,
                strands=strands,
            )
        )
        for pattern in unique_patterns:
            if pattern not in generation_seen:
                generation_seen.add(pattern)
                generation_patterns.append(pattern)
            if strands == "both":
                rc = reverse_complement(pattern)
                if rc not in generation_seen:
                    generation_seen.add(rc)
                    generation_patterns.append(rc)

    return CompiledSequenceConstraints(
        forbid_rules=tuple(forbid_rules),
        allow_components=allow_components,
        generation_forbidden_patterns=tuple(sorted(generation_patterns)),
    )


def _resolve_promoter_placements(*, sequence: str, fixed_elements_dump: dict) -> list[dict]:
    fixed = _as_dict(fixed_elements_dump) or {}
    constraints = list(fixed.get("promoter_constraints") or [])
    placements: list[dict] = []
    for raw in constraints:
        pc = _as_dict(raw) or {}
        upstream = _normalize_motif(pc.get("upstream"), label="promoter upstream")
        downstream = _normalize_motif(pc.get("downstream"), label="promoter downstream")
        spacer_min, spacer_max = _spacer_range(pc.get("spacer_length"))
        upstream_positions = _find_motif_positions(sequence, upstream, pc.get("upstream_pos"))
        downstream_positions = _find_motif_positions(sequence, downstream, pc.get("downstream_pos"))
        candidates: list[tuple[int, int, int]] = []
        for up_start in upstream_positions:
            up_end = up_start + len(upstream)
            for down_start in downstream_positions:
                if down_start < up_end:
                    continue
                spacer = down_start - up_end
                if spacer < spacer_min or spacer > spacer_max:
                    continue
                candidates.append((int(up_start), int(down_start), int(spacer)))
        if not candidates:
            raise ValueError(
                "sequence constraints could not resolve promoter placement in final sequence "
                f"for constraint={pc.get('name')!r}."
            )
        candidates = sorted(candidates, key=lambda item: (int(item[0]), int(item[1])))
        up_start, down_start, spacer = candidates[0]
        placements.append(
            {
                "name": str(pc.get("name") or ""),
                "upstream_seq": upstream,
                "downstream_seq": downstream,
                "upstream_start": int(up_start),
                "downstream_start": int(down_start),
                "spacer_length": int(spacer),
                "candidate_count": int(len(candidates)),
                "variant_ids": {
                    "up_id": pc.get("upstream_variant_id"),
                    "down_id": pc.get("downstream_variant_id"),
                },
            }
        )
    return placements


def _allowed_windows(*, placements: list[dict], components: tuple[str, ...]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for placement in placements:
        if "upstream" in components:
            start = int(placement["upstream_start"])
            end = start + len(str(placement["upstream_seq"]))
            windows.append((start, end))
        if "downstream" in components:
            start = int(placement["downstream_start"])
            end = start + len(str(placement["downstream_seq"]))
            windows.append((start, end))
    return windows


def validate_sequence_constraints(
    *,
    sequence: str,
    compiled: CompiledSequenceConstraints,
    fixed_elements_dump: dict | None,
) -> SequenceValidationResult:
    seq = _normalize_motif(sequence, label="final sequence")
    if not compiled.has_rules():
        return SequenceValidationResult(
            validation_passed=True,
            violations=[],
            promoter_detail={"placements": []},
        )

    placements = _resolve_promoter_placements(sequence=seq, fixed_elements_dump=fixed_elements_dump or {})
    windows = _allowed_windows(placements=placements, components=compiled.allow_components)
    if not windows:
        raise ValueError("sequence constraints require promoter allowlist windows but none were resolved.")

    violations: list[dict] = []
    for rule in compiled.forbid_rules:
        for pattern in rule.patterns:
            for match in find_kmer_matches(sequence=seq, pattern=pattern, strands=rule.strands):
                start = int(match["position"])
                end = start + len(str(match["matched_seq"]))
                allowed = any(start >= win_start and end <= win_end for win_start, win_end in windows)
                if allowed:
                    continue
                context_start = max(0, start - 4)
                context_end = min(len(seq), end + 4)
                violations.append(
                    {
                        "constraint": rule.name,
                        "pattern": str(match["pattern"]),
                        "strand": str(match["strand"]),
                        "position": int(start),
                        "matched_seq": str(match["matched_seq"]),
                        "context_window": seq[context_start:context_end],
                    }
                )
    violations = sorted(
        violations,
        key=lambda item: (
            int(item["position"]),
            str(item["pattern"]),
            str(item["strand"]),
        ),
    )
    return SequenceValidationResult(
        validation_passed=not bool(violations),
        violations=violations,
        promoter_detail={"placements": placements},
    )
