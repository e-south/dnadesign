"""
Scalar derivation helpers for latentdna.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


def _replace_or_append_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if values.ndim == 0:
        values = np.repeat(values.astype(np.float32), table.num_rows)
    array = pa.array(values.tolist())
    if name in table.column_names:
        index = table.column_names.index(name)
        return table.set_column(index, name, array)
    return table.append_column(name, array)


class _ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, env: dict[str, np.ndarray]) -> None:
        self._env = env

    def visit_Expression(self, node: ast.Expression) -> np.ndarray:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> np.ndarray:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ContractViolationError(f"unsupported scalar expression operator: {ast.dump(node.op)}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> np.ndarray:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ContractViolationError(f"unsupported scalar expression unary operator: {ast.dump(node.op)}")

    def visit_Name(self, node: ast.Name) -> np.ndarray:
        if node.id not in self._env:
            raise ContractViolationError(f"unknown column in scalar expression: {node.id}")
        return self._env[node.id]

    def visit_Constant(self, node: ast.Constant) -> np.ndarray:
        if not isinstance(node.value, int | float):
            raise ContractViolationError("scalar expressions only allow numeric constants")
        return np.asarray(node.value, dtype=np.float32)

    def generic_visit(self, node: ast.AST) -> np.ndarray:
        raise ContractViolationError(f"unsupported scalar expression syntax: {ast.dump(node)}")


def _evaluate_expression(expression: str, table: pa.Table) -> np.ndarray:
    env: dict[str, np.ndarray] = {}
    for name in table.column_names:
        column = table[name]
        if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
            env[name] = np.asarray(column.to_pylist(), dtype=np.float32)
    tree = ast.parse(expression, mode="eval")
    return np.asarray(_ExpressionEvaluator(env).visit(tree), dtype=np.float32)


def _resolve_table_source(context: WorkspaceContext, source_id: str) -> tuple[Path, pa.Table]:
    scalar_path = context.output_root / "scalars" / source_id / "table.parquet"
    if scalar_path.exists():
        return scalar_path, read_table(scalar_path)
    distance_path = context.output_root / "distances" / source_id / "table.parquet"
    if distance_path.exists():
        return distance_path, read_table(distance_path)
    raise MissingArtifactError(f"scalar source table not found for {source_id!r}")


def _join_tables(*, source_ids: list[str], tables: list[pa.Table], key_columns: list[str]) -> pa.Table:
    mappings: list[dict[tuple[object, ...], dict[str, object]]] = []
    first_order: list[tuple[object, ...]] = []
    output_columns: list[str] = []
    seen_columns: set[str] = set()
    shared_keys: set[tuple[object, ...]] | None = None

    for index, (source_id, table) in enumerate(zip(source_ids, tables, strict=True)):
        missing_keys = [column for column in key_columns if column not in table.column_names]
        if missing_keys:
            raise ContractViolationError(
                f"scalar join_tables source {source_id!r} is missing key columns: {missing_keys}"
            )
        rows = table.to_pylist()
        mapping: dict[tuple[object, ...], dict[str, object]] = {}
        order: list[tuple[object, ...]] = []
        for row in rows:
            key = tuple(row[column] for column in key_columns)
            if key in mapping:
                raise ContractViolationError(
                    f"scalar join_tables requires unique keys in source {source_id!r} for {key_columns}"
                )
            mapping[key] = row
            order.append(key)
        mappings.append(mapping)
        if index == 0:
            first_order = order
            output_columns.extend(table.column_names)
            seen_columns.update(table.column_names)
        else:
            duplicate_columns = [
                column for column in table.column_names if column not in key_columns and column in seen_columns
            ]
            if duplicate_columns:
                raise ContractViolationError(
                    f"scalar join_tables source {source_id!r} reuses non-key columns: {duplicate_columns}"
                )
            new_columns = [column for column in table.column_names if column not in key_columns]
            output_columns.extend(new_columns)
            seen_columns.update(new_columns)
        key_set = set(mapping)
        shared_keys = key_set if shared_keys is None else shared_keys.intersection(key_set)

    if not shared_keys:
        raise ContractViolationError("scalar join_tables produced an empty key intersection")

    output_rows: list[dict[str, object]] = []
    for key in first_order:
        if key not in shared_keys:
            continue
        merged: dict[str, object] = {}
        for mapping in mappings:
            row = mapping[key]
            for column in output_columns:
                if column in merged:
                    continue
                if column in row:
                    merged[column] = row[column]
        output_rows.append(merged)
    return pa.Table.from_pylist(output_rows)


def derive_scalar_artifact(context: WorkspaceContext, *, scalar_id: str) -> tuple[Path, int, list[str]]:
    scalar = context.require_scalar(scalar_id)
    derive = scalar.derive
    artifact_dir = context.output_root / "scalars" / scalar_id

    if derive.kind == "vector_norm":
        matrix_path = context.output_root / "views" / derive.view / "matrix.npy"
        rows_path = context.output_root / "views" / derive.view / "rows.parquet"
        if not matrix_path.exists() or not rows_path.exists():
            raise MissingArtifactError(f"view artifact is missing for scalar {scalar_id}: {derive.view}")
        matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
        if derive.norm == "l1":
            values = np.abs(matrix).sum(axis=1)
        else:
            values = np.linalg.norm(matrix, axis=1)
        output_column = derive.output_column or scalar_id
        table = _replace_or_append_column(read_table(rows_path), output_column, values)
    elif derive.kind in {"column_expression", "select_columns", "rename_columns"}:
        source_path, input_table = _resolve_table_source(context, derive.source)
        del source_path  # only used by the service manifest path selection
        if derive.kind == "column_expression":
            values = _evaluate_expression(derive.expression, input_table)
            table = _replace_or_append_column(input_table, derive.output_column, values)
        elif derive.kind == "select_columns":
            missing = [column for column in derive.columns if column not in input_table.column_names]
            if missing:
                raise ContractViolationError(f"scalar select_columns is missing required columns: {missing}")
            table = input_table.select(derive.columns)
        elif derive.kind == "rename_columns":
            missing = [column for column in derive.renames if column not in input_table.column_names]
            if missing:
                raise ContractViolationError(f"scalar rename_columns is missing required columns: {missing}")
            table = input_table
            for source_name, target_name in derive.renames.items():
                column_index = table.column_names.index(source_name)
                table = table.rename_columns(
                    [target_name if index == column_index else name for index, name in enumerate(table.column_names)]
                )
        else:
            raise AssertionError("unreachable scalar derive branch")
    else:
        source_tables = [_resolve_table_source(context, source_id) for source_id in derive.sources]
        table = _join_tables(
            source_ids=derive.sources,
            tables=[source_table for _, source_table in source_tables],
            key_columns=derive.on,
        )

    write_table(table, artifact_dir / "table.parquet")
    return artifact_dir, table.num_rows, table.column_names
