"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/merge_overlay_carry.py

Explicit overlay-carry planning and application for USR dataset merges.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pyarrow as pa

from .dataset_overlay_catalog import load_overlay_catalog
from .dataset_overlay_ops import write_overlay_dataset
from .errors import SchemaError

if TYPE_CHECKING:
    from .dataset import Dataset


@dataclass(frozen=True)
class OverlayCarryPlan:
    namespace: str
    rows_from_src: int
    table: pa.Table | None


@contextmanager
def _held_write_lock(_dataset_dir):
    yield


def _namespace_policy():
    # Imported lazily to avoid extra dataset import churn at module load time.
    from .dataset import _NS_RE, MUTATION_RESERVED_NAMESPACES

    return _NS_RE, MUTATION_RESERVED_NAMESPACES


def _normalized_namespaces(namespaces: Iterable[str] | None) -> tuple[str, ...]:
    ordered: list[str] = []
    for namespace in namespaces or ():
        cleaned = str(namespace or "").strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return tuple(ordered)


def _sql_str(path: Path) -> str:
    return str(path).replace("'", "''")


def _raise_duplicate_key_error(*, con, relation_sql: str, namespace: str, label: str) -> None:
    dup_keys = int(
        con.execute(f"SELECT COUNT(*) FROM (SELECT id FROM {relation_sql} GROUP BY id HAVING COUNT(*) > 1)").fetchone()[
            0
        ]
    )
    if dup_keys <= 0:
        return
    preview = con.execute(
        f"SELECT id, COUNT(*) AS cnt FROM {relation_sql} GROUP BY id HAVING cnt > 1 ORDER BY cnt DESC, id LIMIT 3"
    ).fetchall()
    sample = "; ".join(f"{row[0]} (count {row[1]})" for row in preview)
    raise SchemaError(
        f"Carry namespace '{namespace}' found duplicate keys in {label}: {dup_keys} key(s) repeated. Sample: {sample}."
    )


def plan_overlay_carry(
    *,
    con,
    dest_dataset: Dataset,
    src_dataset: Dataset,
    src_keep_relation: str,
    namespaces: Iterable[str] | None,
    dry_run: bool,
) -> tuple[OverlayCarryPlan, ...]:
    normalized = _normalized_namespaces(namespaces)
    if not normalized:
        return ()
    _namespace_pattern, reserved_namespaces = _namespace_policy()

    src_catalog = {
        str(overlay["namespace"]): overlay
        for overlay in load_overlay_catalog(src_dataset, reserved_namespaces=frozenset())
    }
    dest_catalog = {
        str(overlay["namespace"]): overlay
        for overlay in load_overlay_catalog(dest_dataset, reserved_namespaces=frozenset())
    }

    plans: list[OverlayCarryPlan] = []
    for namespace in normalized:
        if namespace in reserved_namespaces:
            raise SchemaError(f"Carry namespace '{namespace}' is reserved and cannot be transferred with merge.")
        src_overlay = src_catalog.get(namespace)
        if src_overlay is None:
            raise SchemaError(
                f"Requested carry namespace '{namespace}' was not found in source dataset '{src_dataset.name}'."
            )
        src_path = Path(src_overlay["path"])
        if src_path.is_dir():
            raise SchemaError(
                f"Carry namespace '{namespace}' requires a compact source overlay file; compact '{namespace}' in "
                f"dataset '{src_dataset.name}' first."
            )
        if str(src_overlay["key"]) != "id":
            raise SchemaError(
                f"Carry namespace '{namespace}' only supports id-keyed overlays; source key is '{src_overlay['key']}'."
            )

        dest_overlay = dest_catalog.get(namespace)
        if dest_overlay is not None:
            dest_path = Path(dest_overlay["path"])
            if dest_path.is_dir():
                raise SchemaError(
                    f"Carry namespace '{namespace}' requires a compact destination overlay file; compact "
                    f"'{namespace}' in dataset '{dest_dataset.name}' first."
                )
            if str(dest_overlay["key"]) != "id":
                raise SchemaError(
                    f"Carry namespace '{namespace}' only supports id-keyed overlays; destination key is "
                    f"'{dest_overlay['key']}'."
                )
            _raise_duplicate_key_error(
                con=con,
                relation_sql=f"read_parquet('{_sql_str(dest_path)}')",
                namespace=namespace,
                label=f"destination overlay '{dest_dataset.name}/{namespace}'",
            )

        src_sql = _sql_str(src_path)
        query = (
            f"SELECT s.* FROM read_parquet('{src_sql}') AS s "
            f"JOIN (SELECT DISTINCT id FROM {src_keep_relation}) AS keep USING (id)"
        )
        rows_from_src = int(con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0])
        if rows_from_src > 0:
            _raise_duplicate_key_error(
                con=con,
                relation_sql=f"({query})",
                namespace=namespace,
                label=f"source overlay '{src_dataset.name}/{namespace}' after merge filtering",
            )
        table = None
        if rows_from_src > 0 and not dry_run:
            table = con.execute(query).fetch_arrow_table()
            dest_dataset._validate_registry_schema(namespace=namespace, schema=table.schema, key="id")  # noqa: SLF001

        plans.append(OverlayCarryPlan(namespace=namespace, rows_from_src=rows_from_src, table=table))
    return tuple(plans)


def apply_overlay_carry(
    *,
    dataset: Dataset,
    plans: Iterable[OverlayCarryPlan],
    src_name: str,
) -> None:
    namespace_pattern, reserved_namespaces = _namespace_policy()
    for plan in plans:
        if plan.rows_from_src == 0 or plan.table is None:
            continue
        write_overlay_dataset(
            dataset=dataset,
            namespace=plan.namespace,
            table_or_batches=plan.table,
            key="id",
            overwrite=True,
            allow_missing=False,
            note=f"carried from merge source '{src_name}'",
            namespace_pattern=namespace_pattern,
            reserved_namespaces=reserved_namespaces,
            write_lock=_held_write_lock,
        )
