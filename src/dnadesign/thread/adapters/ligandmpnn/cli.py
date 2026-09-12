"""File-in, JSON-out commands; computation and upstream parsing stay producer-owned."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ._regular_files import open_regular_file
from .context_probe import materialize_ligandmpnn_context_inventory
from .handoff_requests import canonical_json, context_request_from_document, upstream_from_document
from .handoffs import normalize_score_plan, plan_designs, plan_scores, producer_identity
from .preflight import preflight_ligandmpnn


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _read(path: Path) -> dict:
    with open_regular_file(path) as source:
        value = json.loads(source.read(), object_pairs_hook=_unique_object)
    canonical_json(value)  # Reject NaN and infinities before request admission.
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("identity")
    for name in ("context", "score-plan", "score-normalize", "design-plan", "preflight"):
        command = commands.add_parser(name)
        command.add_argument(
            "--request", type=Path, required=True, help="request JSON; score-normalize accepts its published plan"
        )
        if name != "preflight":
            command.add_argument("--execution-root", type=Path, required=True)
        if name != "score-normalize":
            command.add_argument("--checkout-root", type=Path, required=True)
        if name in {"score-plan", "design-plan"}:
            command.add_argument("--python-executable", default="python")
        if name == "score-plan":
            command.add_argument(
                "--input-root", type=Path, help="staged inputs; commands bind the distinct final execution root"
            )
        if name == "score-normalize":
            command.add_argument("--trust", choices=["pinned_local_execution"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            result = producer_identity()
        elif args.command == "score-normalize":
            result = normalize_score_plan(_read(args.request), execution_root=args.execution_root, trust=args.trust)
        elif args.command == "context":
            reference = materialize_ligandmpnn_context_inventory(
                context_request_from_document(_read(args.request)),
                execution_root=args.execution_root,
                checkout_root=args.checkout_root,
            )
            result = reference.to_dict()
        elif args.command == "preflight":
            report = preflight_ligandmpnn(args.checkout_root, upstream_from_document(_read(args.request)))
            result = {
                "ok": report.ok,
                "issues": [asdict(issue) for issue in report.issues],
                "provenance": report.provenance.to_dict(),
            }
            if not report.ok:
                print(canonical_json(result).decode(), file=sys.stderr)
                return 1
        else:
            kwargs = {
                "checkout_root": args.checkout_root,
                "execution_root": args.execution_root,
                "python_executable": args.python_executable,
            }
            if args.command == "score-plan":
                result = plan_scores(_read(args.request), input_root=args.input_root, **kwargs)
            else:
                result = plan_designs(_read(args.request), **kwargs)
        print(canonical_json(result).decode())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"LigandMPNN handoff failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
