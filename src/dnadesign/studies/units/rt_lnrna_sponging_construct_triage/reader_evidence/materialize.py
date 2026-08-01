"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_evidence/materialize.py

CLI for materializing study-owned Reader evidence bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dnadesign.studies.core.reader_records import (
    ReaderDataframeRecordError,
    resolve_digest_verified_dataframe_record,
)

from ..subject_bindings import load_registered_subject_bindings
from .bindings import (
    ReaderEvidenceBindingError,
    build_reader_evidence_bindings,
    materialize_reader_evidence_bindings_json,
)
from .experiment_routes import (
    ReaderExperimentRouteError,
    require_route_readiness,
    selected_experiments_for_route,
)

_RECORD_ID = "sample_measurements/df"
_RECORD_CONTRACT_ID = "plate_reader.annotated.v1"
_PROTOCOL_ID = "plate_reader/single_reporter_screen"
_EXPERIMENT_ROUTE_ID = "rt_competence_subject_binding"


class ReaderEvidenceMaterializationError(ValueError):
    """Raised when an exact Reader experiment cannot be materialized."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize exact RT-lnRNA subject bindings from one verified public Reader record."
    )
    parser.add_argument("--reader-root", type=Path, required=True, help="Reader repository root")
    parser.add_argument(
        "--experiment-route-registry",
        type=Path,
        required=True,
        help="PhD retron bridge experiment-route registry",
    )
    parser.add_argument("--experiment-id", required=True, help="Exact Reader experiment ID selected by the route")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reader_root = Path(args.reader_root).expanduser().resolve()
    selected = selected_experiments_for_route(
        args.experiment_route_registry,
        route_id=_EXPERIMENT_ROUTE_ID,
    )
    matches = [member for member in selected if member.experiment_id == args.experiment_id]
    if len(matches) != 1:
        raise ReaderEvidenceMaterializationError(
            f"experiment {args.experiment_id!r} is not selected exactly once by Reader route {_EXPERIMENT_ROUTE_ID!r}"
        )
    require_route_readiness(
        args.experiment_route_registry,
        route_id=_EXPERIMENT_ROUTE_ID,
        reader_root=reader_root,
    )
    member = matches[0]
    config_path = (reader_root.parent / member.reader_config).resolve()
    record = resolve_digest_verified_dataframe_record(
        config_path,
        reader_root=reader_root,
        experiment_id=member.experiment_id,
        protocol_id=_PROTOCOL_ID,
        record_id=_RECORD_ID,
        contract_id=_RECORD_CONTRACT_ID,
    )
    binding_set = build_reader_evidence_bindings(
        record=record,
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )
    output_path = materialize_reader_evidence_bindings_json(binding_set, args.output)
    print(f"wrote {output_path} bindings={len(binding_set.rows)} unbound={binding_set.unbound_count}")
    return 0


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ReaderEvidenceMaterializationError("cannot locate DNA Design repository root")


def _entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (
        ReaderDataframeRecordError,
        ReaderEvidenceBindingError,
        ReaderEvidenceMaterializationError,
        ReaderExperimentRouteError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    _entrypoint()


__all__ = ["ReaderEvidenceMaterializationError", "build_parser", "main"]
