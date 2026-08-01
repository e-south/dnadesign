"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/query.py

Read-only exact query surface for compositional RT-lnRNA subjects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .contracts import SubjectBinding, SubjectBindingContractError
from .loader import load_registered_subject_bindings


def query_registered_subjects(
    *,
    repo_root: Path | None = None,
    subject_id: str | None = None,
    rt_part_id: str | None = None,
    lnrna_part_id: str | None = None,
    reader_design_id: str | None = None,
    reader_assay_subject_id: str | None = None,
) -> dict[str, object]:
    """Return subjects matching exactly one explicit selector."""

    selectors = {
        "subject_id": subject_id,
        "rt_part_id": rt_part_id,
        "lnrna_part_id": lnrna_part_id,
        "reader_design_id": reader_design_id,
        "reader_assay_subject_id": reader_assay_subject_id,
    }
    selected = [(name, value) for name, value in selectors.items() if value is not None]
    if len(selected) != 1:
        raise SubjectBindingContractError("subject query requires exactly one selector")
    selector, raw_value = selected[0]
    value = _exact_text(raw_value, label=selector)
    registry = load_registered_subject_bindings(repo_root=repo_root)
    matches = _matches(registry.subjects, selector=selector, value=value)
    if not matches:
        raise SubjectBindingContractError(f"no subject matches exact {selector} {value!r}")
    return {
        "schema_id": "rt_lnrna_subject_query_result_v1",
        "binding_set_id": registry.binding_set_id,
        "selector": {"field": selector, "value": value, "match": "exact"},
        "match_count": len(matches),
        "subjects": [asdict(subject) for subject in matches],
    }


def _matches(subjects: Sequence[SubjectBinding], *, selector: str, value: str) -> tuple[SubjectBinding, ...]:
    if selector == "subject_id":
        return tuple(subject for subject in subjects if subject.subject_id == value)
    if selector == "rt_part_id":
        return tuple(subject for subject in subjects if subject.rt_part.part_id == value)
    if selector == "lnrna_part_id":
        return tuple(subject for subject in subjects if subject.lnrna_part.part_id == value)
    namespace = {
        "reader_design_id": "reader.design_id",
        "reader_assay_subject_id": "reader.assay_subject_id",
    }[selector]
    return tuple(
        subject
        for subject in subjects
        if any(alias.namespace == namespace and alias.value == value for alias in subject.aliases)
    )


def _exact_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SubjectBindingContractError(f"{label} must be a non-empty exact string without outer whitespace")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--subject-id")
    selectors.add_argument("--rt-part-id")
    selectors.add_argument("--lnrna-part-id")
    selectors.add_argument("--reader-design-id")
    selectors.add_argument("--reader-assay-subject-id")
    args = parser.parse_args(argv)
    payload = query_registered_subjects(
        repo_root=args.repo_root,
        subject_id=args.subject_id,
        rt_part_id=args.rt_part_id,
        lnrna_part_id=args.lnrna_part_id,
        reader_design_id=args.reader_design_id,
        reader_assay_subject_id=args.reader_assay_subject_id,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _entrypoint() -> None:
    try:
        raise SystemExit(main())
    except SubjectBindingContractError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    _entrypoint()


__all__ = ["query_registered_subjects"]
