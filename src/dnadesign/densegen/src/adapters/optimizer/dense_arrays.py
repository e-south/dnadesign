"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/optimizer/dense_arrays.py

Dense-arrays optimizer adapter and helpers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

import dense_arrays as da

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerRun:
    optimizer: da.Optimizer
    generator: Iterable
    forbid_each: bool = False


class OptimizerAdapter(Protocol):
    def probe_solver(self, backend: str, *, test_length: int = 10) -> None: ...

    def build(
        self,
        *,
        library: list[str],
        sequence_length: int,
        solver: str | None,
        strategy: str,
        solver_options: list[str],
        fixed_elements: dict | None,
        strands: str = "double",
        regulator_by_index: list[str] | None = None,
        required_regulators: list[str] | None = None,
        min_count_by_regulator: dict[str, int] | None = None,
        min_required_regulators: int | None = None,
    ) -> OptimizerRun: ...


_EXTRA_REGULATOR_LABEL = "__densegen__extra__"


def _normalize_regulator_labels(labels: list[str]) -> list[str]:
    cleaned = []
    for label in labels:
        s = str(label).strip()
        if not s:
            raise ValueError("regulator_by_index labels must be non-empty strings")
        cleaned.append(s)
    return cleaned


def _apply_regulator_constraints(
    opt: da.Optimizer,
    *,
    regulator_by_index: list[str] | None,
    required_regulators: list[str] | None,
    min_count_by_regulator: dict[str, int] | None,
    min_required_regulators: int | None,
) -> None:
    if not regulator_by_index:
        return
    if not (required_regulators or min_count_by_regulator or min_required_regulators):
        return

    labels = list(regulator_by_index)
    if len(labels) > len(opt.library):
        raise ValueError(
            f"regulator_by_index length exceeds optimizer library length ({len(labels)} > {len(opt.library)})"
        )
    if len(labels) < len(opt.library):
        labels.extend([_EXTRA_REGULATOR_LABEL] * (len(opt.library) - len(labels)))

    labels = _normalize_regulator_labels(labels)
    label_set = set(labels)

    required = [str(r).strip() for r in (required_regulators or []) if str(r).strip()]
    missing = sorted(set(required) - label_set)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Required regulators missing from optimizer mapping: {preview}")

    if min_count_by_regulator:
        extras = sorted(set(min_count_by_regulator.keys()) - label_set)
        if extras:
            preview = ", ".join(extras[:10])
            raise ValueError(f"min_count_by_regulator includes unknown regulators: {preview}")

    opt.add_regulator_constraints(
        labels,
        required=set(required) if required else None,
        min_count_by_regulator=min_count_by_regulator,
        min_required_regulators=min_required_regulators,
    )


class DenseArraysAdapter:
    def probe_solver(self, backend: str, *, test_length: int = 10) -> None:
        try:
            dummy = da.Optimizer(library=["AT"], sequence_length=test_length)
            _ = dummy.optimal(solver=backend)
            log.info("Solver selected: %s", backend)
        except Exception as e:
            raise RuntimeError(
                f"Requested solver '{backend}' failed during probe: {e}\n"
                "Please install/configure this solver or choose another in solver.backend."
            ) from e

    def build(
        self,
        *,
        library: list[str],
        sequence_length: int,
        solver: str | None,
        strategy: str,
        solver_options: list[str],
        fixed_elements: dict | None,
        strands: str = "double",
        regulator_by_index: list[str] | None = None,
        required_regulators: list[str] | None = None,
        min_count_by_regulator: dict[str, int] | None = None,
        min_required_regulators: int | None = None,
    ) -> OptimizerRun:
        if strategy != "approximate" and not solver:
            raise ValueError("solver.backend is required unless strategy=approximate")
        solver_name = solver or ""
        wrapper = DenseArrayOptimizer(
            library=library,
            sequence_length=sequence_length,
            solver=solver_name,
            solver_options=solver_options,
            fixed_elements=fixed_elements,
            strands=strands,
        )
        opt = wrapper.get_optimizer_instance()
        _apply_regulator_constraints(
            opt,
            regulator_by_index=regulator_by_index,
            required_regulators=required_regulators,
            min_count_by_regulator=min_count_by_regulator,
            min_required_regulators=min_required_regulators,
        )
        if strategy == "diverse":
            if not hasattr(opt, "solutions_diverse"):
                raise RuntimeError("dense-arrays does not support solutions_diverse on this install.")
            gen = opt.solutions_diverse(solver=solver_name, solver_options=solver_options)
        elif strategy == "iterate":
            gen = opt.solutions(solver=solver_name, solver_options=solver_options)
        elif strategy == "optimal":

            def _gen():
                yield opt.optimal(solver=solver_name, solver_options=solver_options)

            gen = _gen()
        elif strategy == "approximate":

            def _gen():
                yield opt.approximate()

            gen = _gen()
        else:
            raise ValueError(f"Unknown solver strategy: {strategy}")
        return OptimizerRun(optimizer=opt, generator=gen, forbid_each=False)


class DenseArrayOptimizer:
    def __init__(
        self,
        library: list[str],
        sequence_length: int,
        solver: str | None = None,
        solver_options: list | None = None,
        fixed_elements: dict | None = None,
        strands: str = "double",
    ):
        valid = {"A", "T", "G", "C"}
        filtered = []
        invalid = []
        for motif in library:
            if not isinstance(motif, str):
                invalid.append(motif)
                continue
            s = motif.strip().upper()
            if not s:
                invalid.append(motif)
                continue
            if not set(s).issubset(valid):
                invalid.append(motif)
                continue
            filtered.append(s)
        if invalid:
            preview = ", ".join(repr(x) for x in invalid[:5])
            raise ValueError(f"Invalid motifs in library (showing up to 5): {preview}")
        if not filtered:
            raise ValueError("Motif library is empty after validation.")

        self.library = filtered
        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options or []
        self.fixed_elements = (fixed_elements or {}).copy()
        if strands not in {"single", "double"}:
            raise ValueError("strands must be 'single' or 'double'")
        self.strands = strands

    def get_optimizer_instance(self) -> da.Optimizer:
        lib = self.library.copy()
        converted = []
        cons = (self.fixed_elements or {}).get("promoter_constraints")
        if cons:
            if not isinstance(cons, list):
                raise ValueError("promoter_constraints must be a list of dicts")
            for pc in cons:
                if not isinstance(pc, dict):
                    raise ValueError("Each promoter constraint must be a dict")
                unknown = set(pc.keys()) - {
                    "name",
                    "upstream",
                    "downstream",
                    "spacer_length",
                    "upstream_pos",
                    "downstream_pos",
                }
                if unknown:
                    raise ValueError(f"Unknown promoter constraint keys: {sorted(unknown)}")
                # Ensure motifs in library
                for mkey in ("upstream", "downstream"):
                    mv = pc.get(mkey)
                    if mv and mv not in lib:
                        lib.append(mv)
                clean = {
                    k: v
                    for k, v in pc.items()
                    if k in {"upstream", "downstream", "spacer_length", "upstream_pos", "downstream_pos"}
                    and v is not None
                }
                if clean:
                    converted.append(clean)

        opt = da.Optimizer(
            library=lib,
            sequence_length=self.sequence_length,
            strands=self.strands,
        )
        for c in converted:
            opt.add_promoter_constraints(**c)

        sb = (self.fixed_elements or {}).get("side_biases") or {}
        left, right = sb.get("left"), sb.get("right")
        if (left and any(left)) or (right and any(right)):
            missing = [m for m in (left or []) + (right or []) if m not in lib]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"side_biases motifs must exist in the library: {preview}")
            opt.add_side_biases(left=left, right=right)
        return opt
