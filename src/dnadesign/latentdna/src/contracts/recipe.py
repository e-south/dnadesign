"""
Recipe contracts and shared graph helpers for latentdna.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

SUPPORTED_RECIPE_OPS: frozenset[str] = frozenset(
    {
        "agreement.compare",
        "alignment.build",
        "cluster.fit",
        "distance.score",
        "enrich.score",
        "export.matrix",
        "export.table",
        "neighbors.fit",
        "notebook.generate",
        "plot.render",
        "projection.fit",
        "sample.build",
        "scalar.derive",
        "snapshot.build",
        "view.derive",
        "view.materialize",
        "view.reduce",
    }
)


class RecipeValidationResult(BaseModel):
    schema_version: Literal["latentdna.recipe_validation_result.v1"] = "latentdna.recipe_validation_result.v1"
    workspace_id: str
    recipe_id: str
    status: Literal["ok", "attention", "missing", "error"]
    step_order: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def expected_step_artifacts(op: str, params: dict[str, Any]) -> list[tuple[str, str]]:
    def require_param(*keys: str) -> Any:
        for key in keys:
            if key in params:
                return params[key]
        expected = ", ".join(keys)
        raise ValueError(f"recipe step is missing required params: {expected}")

    def optional_param(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in params:
                return params[key]
        return default

    if op == "agreement.compare":
        return [("agreement_set", str(require_param("agreement_id", "agreement")))]
    if op == "alignment.build":
        return [("alignment_set", str(require_param("alignment_id", "alignment")))]
    if op == "cluster.fit":
        return [("cluster_set", str(require_param("cluster_id", "cluster")))]
    if op == "distance.score":
        return [("distance_set", str(require_param("distance_id", "distance")))]
    if op == "enrich.score":
        return [("enrichment_set", str(require_param("enrichment_id", "enrichment")))]
    if op == "export.matrix":
        return [("export_bundle", str(require_param("export_id", "export")))]
    if op == "export.table":
        return [("export_bundle", str(require_param("export_id", "export")))]
    if op == "neighbors.fit":
        return [("neighbor_set", str(require_param("neighbor_id", "neighbors_id", "neighbors")))]
    if op == "notebook.generate":
        return [("notebook", str(require_param("notebook_id", "notebook")))]
    if op == "plot.render":
        return [("plot", str(require_param("plot_id", "plot")))]
    if op == "projection.fit":
        return [("projection", str(require_param("projection_id", "run_id")))]
    if op == "sample.build":
        return [("sample_set", str(require_param("sample_id", "sample")))]
    if op == "scalar.derive":
        return [("scalar_table", str(require_param("scalar_id", "scalar")))]
    if op == "snapshot.build":
        return [("snapshot", str(require_param("snapshot_id", "snapshot")))]
    if op in {"view.derive", "view.materialize"}:
        return [("view", str(require_param("view_id", "view")))]
    if op == "view.reduce":
        refs = [("reducer", str(require_param("reducer_id", "run_id")))]
        reduced_view_id = optional_param("reduced_view_id", default=None)
        if reduced_view_id is not None:
            refs.append(("reduced_view", str(reduced_view_id)))
        return refs
    raise ValueError(f"unsupported recipe op: {op}")


def topological_step_order(steps: Iterable[Any]) -> list[str]:
    step_list = list(steps)
    ids = [str(step.id) for step in step_list]
    dependencies = {str(step.id): {str(dep) for dep in getattr(step, "depends_on", [])} for step in step_list}
    dependents: dict[str, list[str]] = {step_id: [] for step_id in ids}
    indegree = {step_id: len(dependencies[step_id]) for step_id in ids}
    for step in step_list:
        step_id = str(step.id)
        for dependency in getattr(step, "depends_on", []):
            dependency_id = str(dependency)
            if dependency_id not in dependents:
                continue
            dependents[dependency_id].append(step_id)

    ready = deque(step_id for step_id in ids if indegree[step_id] == 0)
    order: list[str] = []
    while ready:
        step_id = ready.popleft()
        order.append(step_id)
        for dependent in dependents[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(ids):
        raise ValueError("recipe graph contains a cycle")
    return order
