"""
Stage-B library artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...utils.logging_utils import install_native_stderr_filters

LIBRARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LibraryArtifact:
    manifest_path: Path
    builds_path: Path
    members_path: Path
    schema_version: str
    run_id: str
    run_root: str
    config_path: str

    @classmethod
    def load(cls, manifest_path: Path) -> "LibraryArtifact":
        payload = json.loads(manifest_path.read_text())
        return cls(
            manifest_path=manifest_path,
            builds_path=Path(payload.get("library_builds_path", "")),
            members_path=Path(payload.get("library_members_path", "")),
            schema_version=str(payload.get("schema_version")),
            run_id=str(payload.get("run_id")),
            run_root=str(payload.get("run_root")),
            config_path=str(payload.get("config_path")),
        )


def _library_manifest_path(out_dir: Path) -> Path:
    return out_dir / "library_manifest.json"


def write_library_artifact(
    *,
    out_dir: Path,
    builds: list[dict],
    members: list[dict],
    cfg_path: Path,
    run_id: str,
    run_root: Path,
    overwrite: bool = False,
) -> LibraryArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    install_native_stderr_filters(suppress_solver_messages=False)
    builds_path = out_dir / "library_builds.parquet"
    members_path = out_dir / "library_members.parquet"

    if not overwrite:
        if builds_path.exists():
            raise FileExistsError(f"Library builds already exist: {builds_path}")
        if members_path.exists():
            raise FileExistsError(f"Library members already exist: {members_path}")

    pd.DataFrame(builds).to_parquet(builds_path, index=False)
    pd.DataFrame(members).to_parquet(members_path, index=False)

    manifest = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "library_builds_path": str(builds_path),
        "library_members_path": str(members_path),
    }
    manifest_path = _library_manifest_path(out_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return LibraryArtifact(
        manifest_path=manifest_path,
        builds_path=builds_path,
        members_path=members_path,
        schema_version=LIBRARY_SCHEMA_VERSION,
        run_id=str(run_id),
        run_root=str(run_root),
        config_path=str(cfg_path),
    )


def load_library_artifact(out_dir: Path) -> LibraryArtifact:
    manifest_path = _library_manifest_path(out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Library manifest not found: {manifest_path}")
    return LibraryArtifact.load(manifest_path)


@dataclass(frozen=True)
class LibraryRecord:
    input_name: str
    plan_name: str
    library_index: int
    library_hash: str
    library_id: str
    library_tfbs: list[str]
    library_tfs: list[str]
    library_site_ids: list[str | None]
    library_sources: list[str | None]
    library_tfbs_ids: list[str | None]
    library_motif_ids: list[str | None]
    pool_strategy: str | None
    library_sampling_strategy: str | None
    library_size: int
    target_length: int | None
    achieved_length: int | None
    relaxed_cap: bool | None
    final_cap: int | None
    iterative_max_libraries: int | None
    iterative_min_new_solutions: int | None
    required_regulators_selected: list[str] | None

    def sampling_info(self) -> dict:
        return {
            "library_index": int(self.library_index),
            "library_hash": str(self.library_hash),
            "pool_strategy": self.pool_strategy,
            "library_sampling_strategy": self.library_sampling_strategy,
            "library_size": int(self.library_size),
            "target_length": self.target_length,
            "achieved_length": self.achieved_length,
            "relaxed_cap": self.relaxed_cap,
            "final_cap": self.final_cap,
            "iterative_max_libraries": self.iterative_max_libraries,
            "iterative_min_new_solutions": self.iterative_min_new_solutions,
            "required_regulators_selected": self.required_regulators_selected,
            "site_id_by_index": list(self.library_site_ids),
            "source_by_index": list(self.library_sources),
            "tfbs_id_by_index": list(self.library_tfbs_ids),
            "motif_id_by_index": list(self.library_motif_ids),
        }


def _ensure_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return list(val)
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, str):
        text = val.strip()
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return list(parsed)
            except Exception:
                return []
        return []
    return []


def _required_columns(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_library_records(artifact: LibraryArtifact) -> dict[tuple[str, str], list[LibraryRecord]]:
    install_native_stderr_filters(suppress_solver_messages=False)
    builds_df = pd.read_parquet(artifact.builds_path)
    members_df = pd.read_parquet(artifact.members_path)
    _required_columns(
        builds_df,
        [
            "input_name",
            "plan_name",
            "library_index",
            "library_hash",
            "library_id",
        ],
        label="library_builds.parquet",
    )
    _required_columns(
        members_df,
        [
            "input_name",
            "plan_name",
            "library_index",
            "position",
            "tfbs",
        ],
        label="library_members.parquet",
    )
    records: dict[tuple[str, str], list[LibraryRecord]] = {}
    for _, row in builds_df.iterrows():
        input_name = str(row.get("input_name") or "")
        plan_name = str(row.get("plan_name") or "")
        key = (input_name, plan_name)
        library_index = int(row.get("library_index") or 0)
        library_hash = str(row.get("library_hash") or "")
        library_id = str(row.get("library_id") or library_hash)
        if not input_name or not plan_name:
            raise ValueError("library_builds.parquet contains empty input_name/plan_name")
        if not library_hash:
            raise ValueError(f"library_builds.parquet missing library_hash for {input_name}/{plan_name}")
        sub = members_df[
            (members_df["input_name"] == input_name)
            & (members_df["plan_name"] == plan_name)
            & (members_df["library_index"] == library_index)
        ].sort_values("position")
        if sub.empty:
            raise ValueError(
                f"library_members.parquet missing members for {input_name}/{plan_name} index={library_index}"
            )
        library_tfbs = [str(x) for x in sub["tfbs"].tolist()]
        library_tfs = [str(x) for x in _ensure_list(sub.get("tf"))] if "tf" in sub.columns else []
        if not library_tfs and "tf" in sub.columns:
            library_tfs = [str(x) for x in sub["tf"].tolist()]
        library_site_ids = [x if x not in ("", None, "None") else None for x in _ensure_list(sub.get("site_id"))]
        if not library_site_ids and "site_id" in sub.columns:
            library_site_ids = [x if x not in ("", None, "None") else None for x in sub["site_id"].tolist()]
        library_sources = [x if x not in ("", None, "None") else None for x in _ensure_list(sub.get("source"))]
        if not library_sources and "source" in sub.columns:
            library_sources = [x if x not in ("", None, "None") else None for x in sub["source"].tolist()]
        library_tfbs_ids = [x if x not in ("", None, "None") else None for x in _ensure_list(sub.get("tfbs_id"))]
        if not library_tfbs_ids and "tfbs_id" in sub.columns:
            library_tfbs_ids = [x if x not in ("", None, "None") else None for x in sub["tfbs_id"].tolist()]
        library_motif_ids = [x if x not in ("", None, "None") else None for x in _ensure_list(sub.get("motif_id"))]
        if not library_motif_ids and "motif_id" in sub.columns:
            library_motif_ids = [x if x not in ("", None, "None") else None for x in sub["motif_id"].tolist()]
        pool_strategy = row.get("pool_strategy")
        library_sampling_strategy = row.get("library_sampling_strategy")
        library_size = int(row.get("library_size") or len(library_tfbs))
        target_length = row.get("target_length")
        achieved_length = row.get("achieved_length")
        relaxed_cap = row.get("relaxed_cap")
        final_cap = row.get("final_cap")
        iterative_max_libraries = row.get("iterative_max_libraries")
        iterative_min_new_solutions = row.get("iterative_min_new_solutions")
        required_regulators_selected = row.get("required_regulators_selected")
        if isinstance(required_regulators_selected, str):
            try:
                parsed = json.loads(required_regulators_selected)
                if isinstance(parsed, list):
                    required_regulators_selected = parsed
            except Exception:
                required_regulators_selected = None
        record = LibraryRecord(
            input_name=input_name,
            plan_name=plan_name,
            library_index=library_index,
            library_hash=library_hash,
            library_id=library_id,
            library_tfbs=library_tfbs,
            library_tfs=library_tfs,
            library_site_ids=library_site_ids,
            library_sources=library_sources,
            library_tfbs_ids=library_tfbs_ids,
            library_motif_ids=library_motif_ids,
            pool_strategy=str(pool_strategy) if pool_strategy is not None else None,
            library_sampling_strategy=str(library_sampling_strategy) if library_sampling_strategy is not None else None,
            library_size=library_size,
            target_length=int(target_length) if target_length is not None else None,
            achieved_length=int(achieved_length) if achieved_length is not None else None,
            relaxed_cap=bool(relaxed_cap) if relaxed_cap is not None else None,
            final_cap=int(final_cap) if final_cap is not None else None,
            iterative_max_libraries=int(iterative_max_libraries) if iterative_max_libraries is not None else None,
            iterative_min_new_solutions=int(iterative_min_new_solutions)
            if iterative_min_new_solutions is not None
            else None,
            required_regulators_selected=list(required_regulators_selected)
            if isinstance(required_regulators_selected, list)
            else None,
        )
        records.setdefault(key, []).append(record)

    for key, items in records.items():
        items.sort(key=lambda item: int(item.library_index))
        seen = set()
        for item in items:
            if item.library_index in seen:
                raise ValueError(f"Duplicate library_index={item.library_index} for {key[0]}/{key[1]}")
            seen.add(item.library_index)
    return records
