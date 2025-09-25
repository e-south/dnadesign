"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/convert_legacy.py

Single-purpose converter: legacy .pt (list[dict]) -> fresh USR dataset.

- Fail-fast validation of .pt structure and required keys
- Computes canonical USR id from (bio_type, normalized sequence)
- Fills essentials then adds a small set of namespaced columns
- Honors typed columns for the 60bp_dual_promoter_cpxR_LexA profile
- Writes snapshots + logs via Dataset & write_parquet_atomic
- De-duplicates by canonical id across all provided inputs (first occurrence wins)

Usage (via CLI):
    usr convert-legacy 60bp_dual_promoter_cpxR_LexA \
        --paths archived/densebatch_*/densegenbatch_*.pt \
        --expected-length 60 \
        --plan sigma70_mid

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import torch

from .dataset import Dataset
from .errors import SchemaError, ValidationError
from .io import append_event, read_parquet, write_parquet_atomic
from .normalize import compute_id, normalize_sequence

# ---------------- profile typing (60bp_dual_promoter_cpxR_LexA) ----------------


@dataclass(frozen=True)
class Profile:
    name: str
    expected_length: Optional[int]
    logits_key: str
    logits_dst: str
    logits_expected_dim: int
    densegen_plan: str


def profile_60bp_dual_promoter() -> Profile:
    return Profile(
        name="60bp_dual_promoter_cpxR_LexA",
        expected_length=60,
        logits_key="evo2_logits_mean_pooled",
        logits_dst="infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean",
        logits_expected_dim=512,
        densegen_plan="sigma70_mid",
    )


# Arrow types for derived columns in this profile
PA = pa
PA_STRUCT_USED_TF_COUNTS = PA.struct(
    [PA.field("cpxr", PA.int64()), PA.field("lexa", PA.int64())]
)
PA_LIST_STR = PA.list_(PA.string())
PA_LIST_F64 = PA.list_(PA.float64())
PA_LIST_STRUCT_TFBS_DETAIL = PA.list_(
    PA.struct(
        [
            PA.field("offset", PA.int64()),
            PA.field("orientation", PA.string()),
            PA.field("tf", PA.string()),
            PA.field("tfbs", PA.string()),
        ]
    )
)


# ---------------- small helpers ----------------


def _coerce_logits(v: Any, *, want_dim: int) -> Optional[List[float]]:
    """Return a flat list[float] of length want_dim, or None if unavailable."""
    if v is None:
        return None
    # torch.Tensor path
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu()
        # Allow [1, D] or [D]
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.reshape(t.shape[1])
        if t.ndim != 1 or int(t.shape[0]) != want_dim:
            raise ValidationError(
                f"logits shape mismatch (got {tuple(t.shape)}, expected [{want_dim}])"
            )
        # default to float32 then Python floats
        return t.to(dtype=torch.float32).tolist()
    # list-like path
    if isinstance(v, (list, tuple)):
        arr = list(v)
        # flatten [1, D] to [D]
        if len(arr) == 1 and isinstance(arr[0], (list, tuple)):
            arr = list(arr[0])
        if len(arr) != want_dim:
            raise ValidationError(
                f"logits length mismatch (got {len(arr)}, expected {want_dim})"
            )
        return [float(x) for x in arr]
    return None


def _tf_from_parts(parts: Sequence[str]) -> List[str]:
    """Extract unique TF names from strings like 'lexa:...' or 'cpxr:...'"""
    out = set()
    for s in parts:
        if not isinstance(s, str):
            continue
        if ":" in s:
            tf = s.split(":", 1)[0].strip().lower()
            if tf:
                out.add(tf)
    return sorted(out)


def _count_tf(parts: Sequence[str]) -> Dict[str, int]:
    counts = {"cpxr": 0, "lexa": 0}
    for s in parts:
        if not isinstance(s, str):
            continue
        if s.lower().startswith("lexa:"):
            counts["lexa"] += 1
        elif s.lower().startswith("cpxr:"):
            counts["cpxr"] += 1
    return counts


def _ensure_pt_list_of_dicts(p: Path) -> List[dict]:
    checkpoint = torch.load(str(p), map_location=torch.device("cpu"))
    if not isinstance(checkpoint, list) or not checkpoint:
        raise SchemaError(f"{p} must be a non-empty list.")
    for i, e in enumerate(checkpoint):
        if not isinstance(e, dict):
            raise SchemaError(f"{p} entry {i} is not a dict.")
        if "sequence" not in e:
            raise SchemaError(f"{p} entry {i} missing 'sequence'.")
    return checkpoint


def _gather_pt_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.pt")))
        elif p.suffix == ".pt":
            files.append(p)
        else:
            raise SchemaError(f"Not a .pt file or directory: {p}")
    if not files:
        raise SchemaError("No .pt files found.")
    return files


# ---------------- core conversion ----------------


@dataclass
class ConvertStats:
    files: int
    rows: int
    skipped_bad_len: int
    skipped_duplicates: int  # NEW: number of duplicate ids skipped pre-import


def convert_legacy(
    *,
    dataset_root: Path,
    dataset_name: str,
    pt_paths: Sequence[Path],
    profile: Profile | None = None,
    expected_length: Optional[int] = None,
    plan_override: Optional[str] = None,
    force: bool = False,
) -> ConvertStats:
    """
    Convert .pt → new USR dataset (records.parquet).

    - Creates dataset folder (fails if exists unless force=True)
    - Imports essentials
    - Adds profile-specific derived columns (typed)
    - De-duplicates by canonical id across all provided inputs (first occurrence wins)
    """
    prof = profile or profile_60bp_dual_promoter()
    want_len = expected_length if expected_length is not None else prof.expected_length
    plan_value = (plan_override or prof.densegen_plan) or ""

    # Resolve & validate sources
    files = _gather_pt_files([Path(p) for p in pt_paths])
    all_entries: List[Tuple[str, dict]] = []  # (source, entry)
    for f in files:
        for e in _ensure_pt_list_of_dicts(f):
            all_entries.append((str(f), e))
    total_entries = len(all_entries)

    # Essentials (sequence-normalized)
    rows_ess: List[Dict[str, object]] = []
    # cache derived per-id so we can align by current table ids later
    per_id: Dict[str, Dict[str, Any]] = {}
    # NEW: track duplicates (by canonical id) seen across all inputs
    seen_ids: set[str] = set()
    skipped_bad_len = 0
    skipped_dups = 0

    for src, e in all_entries:
        seq_raw = str(e["sequence"])
        bt = "dna"
        alph = "dna_4"
        seq = normalize_sequence(seq_raw, bt, alph)
        if want_len is not None and len(seq) != int(want_len):
            # fail-fast for systemic errors, but skip rare one-offs
            skipped_bad_len += 1
            continue

        rid = compute_id(bt, seq)

        # Skip duplicates (first seen wins)
        if rid in seen_ids:
            skipped_dups += 1
            continue
        seen_ids.add(rid)

        rows_ess.append(
            {
                "sequence": seq,  # Dataset.import_rows will compute id/created_at
                "bio_type": bt,
                "alphabet": alph,
                "source": src,  # provenance per-row: archived filename
            }
        )

        # Derived (minimal & typed)
        derived: Dict[str, Any] = {}

        # densegen__compression_ratio
        if "meta_compression_ratio" in e:
            derived["densegen__compression_ratio"] = float(e["meta_compression_ratio"])

        # densegen__diverse
        if "diverse_solution" in e:
            derived["densegen__diverse"] = bool(e["diverse_solution"])

        # densegen__gap_fill_used / details
        if "meta_gap_fill" in e:
            derived["densegen__gap_fill_used"] = bool(e["meta_gap_fill"])
        d = e.get("meta_gap_fill_details") or {}
        if isinstance(d, dict):
            if "fill_gap" in d:
                derived["densegen__gap_fill_bases"] = float(d["fill_gap"])
            if "fill_end" in d:
                derived["densegen__gap_fill_end"] = str(d["fill_end"])
            if (
                "fill_gc_range" in d
                and isinstance(d["fill_gc_range"], (list, tuple))
                and len(d["fill_gc_range"]) == 2
            ):
                gmin, gmax = d["fill_gc_range"]
                derived["densegen__gap_fill_gc_min"] = float(gmin)
                derived["densegen__gap_fill_gc_max"] = float(gmax)

        # densegen__plan (constant)
        if plan_value:
            derived["densegen__plan"] = plan_value

        # densegen__sequence_length (int64)
        if "sequence_length" in e:
            derived["densegen__sequence_length"] = int(e["sequence_length"])
        else:
            derived["densegen__sequence_length"] = int(len(seq))

        # densegen__solver
        if "solver" in e:
            derived["densegen__solver"] = str(e["solver"])

        # densegen__tfbs_parts (list<string>)
        tfbs_parts = None
        if "meta_tfbs_parts" in e and isinstance(e["meta_tfbs_parts"], list):
            tfbs_parts = [str(x) for x in e["meta_tfbs_parts"]]
            derived["densegen__tfbs_parts"] = tfbs_parts
        elif "meta_tfbs_parts_in_array" in e and isinstance(
            e["meta_tfbs_parts_in_array"], list
        ):
            tfbs_parts = [str(x) for x in e["meta_tfbs_parts_in_array"]]
            derived["densegen__tfbs_parts"] = tfbs_parts

        # densegen__visual
        if "meta_sequence_visual" in e:
            derived["densegen__visual"] = str(e["meta_sequence_visual"])

        # densegen__tf_list / used_tf_list / used_tf_counts / used_tfbs
        if tfbs_parts:
            tf_list = _tf_from_parts(tfbs_parts)
            if tf_list:
                derived["densegen__tf_list"] = tf_list
                derived["densegen__used_tf_list"] = tf_list
            counts = _count_tf(tfbs_parts)
            derived["densegen__used_tf_counts"] = {
                "cpxr": int(counts["cpxr"]),
                "lexa": int(counts["lexa"]),
            }
        if isinstance(e.get("meta_tfbs_parts_in_array"), list):
            derived["densegen__used_tfbs"] = [
                str(x) for x in e["meta_tfbs_parts_in_array"]
            ]

        # densegen__promoter_constraint (best-effort from fixed_elements.promoter_constraints[].name)
        try:
            fe = e.get("fixed_elements") or {}
            pcs = fe.get("promoter_constraints") or []
            if (
                pcs
                and isinstance(pcs, list)
                and isinstance(pcs[0], dict)
                and "name" in pcs[0]
            ):
                derived["densegen__promoter_constraint"] = str(pcs[0]["name"])
        except Exception:
            pass

        # logits → infer__...__logits_mean (list<double>[512])
        if prof.logits_key in e:
            try:
                vec = _coerce_logits(
                    e[prof.logits_key], want_dim=int(prof.logits_expected_dim)
                )
                if vec is not None:
                    derived[prof.logits_dst] = vec
            except ValidationError as ve:
                raise ValidationError(f"{src} → id={rid[:8]}…: {ve}") from None

        per_id[rid] = derived

    if not rows_ess:
        raise ValidationError("No valid rows to import (all skipped?).")

    # ---------------- write dataset essentials ----------------
    ds = Dataset(dataset_root, dataset_name)
    if ds.dir.exists():
        if not force:
            raise ValidationError(
                f"Dataset '{dataset_name}' already exists at {ds.dir}. Use --force to overwrite."
            )
        # destructive reset (only if requested)
        for p in sorted(ds.dir.glob("*")):
            if p.is_dir():
                for q in sorted(p.rglob("*")):
                    try:
                        q.unlink()
                    except Exception:
                        pass
                try:
                    p.rmdir()
                except Exception:
                    pass
            else:
                try:
                    p.unlink()
                except Exception:
                    pass

    ds.init(source="convert-legacy")
    # import rows (Dataset will compute id/created_at; we pass per-row source)
    ess_df = pd.DataFrame(rows_ess)
    ds.import_rows(
        ess_df,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source=None,
        strict_id_check=False,
    )

    # ---------------- add typed derived columns ----------------
    tbl = read_parquet(ds.records_path)
    ids: List[str] = tbl.column("id").to_pylist()
    N = len(ids)

    def build_array(col: str, pa_type: pa.DataType) -> pa.Array:
        vals = []
        for rid in ids:
            d = per_id.get(rid, {})
            v = d.get(col, None)
            vals.append(v)
        return pa.array(vals, type=pa_type)

    # Minimal set (make present with correct types; missing stay NULL)
    add_specs: List[Tuple[str, pa.DataType]] = [
        ("densegen__compression_ratio", PA.float64()),
        ("densegen__covers_all_tfs_in_solution", PA.bool_()),
        ("densegen__diverse", PA.bool_()),
        ("densegen__gap_fill_attempts", PA.float64()),
        ("densegen__gap_fill_bases", PA.float64()),
        ("densegen__gap_fill_end", PA.string()),
        ("densegen__gap_fill_gc_actual", PA.float64()),
        ("densegen__gap_fill_gc_max", PA.float64()),
        ("densegen__gap_fill_gc_min", PA.float64()),
        ("densegen__gap_fill_relaxed", PA.bool_()),
        ("densegen__gap_fill_used", PA.bool_()),
        ("densegen__library_size", PA.int64()),
        ("densegen__min_count_per_tf_required", PA.int64()),
        ("densegen__plan", PA.string()),
        ("densegen__promoter_constraint", PA.string()),
        ("densegen__sequence_length", PA.int64()),
        ("densegen__solver", PA.string()),
        ("densegen__tf_list", PA_LIST_STR),
        ("densegen__tfbs_parts", PA_LIST_STR),
        ("densegen__used_tf_counts", PA_STRUCT_USED_TF_COUNTS),
        ("densegen__used_tf_list", PA_LIST_STR),
        ("densegen__used_tfbs", PA_LIST_STR),
        ("densegen__used_tfbs_detail", PA_LIST_STRUCT_TFBS_DETAIL),
        ("densegen__visual", PA.string()),
        (profile_60bp_dual_promoter().logits_dst, PA_LIST_F64),
    ]

    out = tbl
    existing = set(out.schema.names)
    for name, t in add_specs:
        arr = build_array(name, t)
        if name in existing:
            # replace type-safely
            idx = out.schema.get_field_index(name)
            out = out.set_column(idx, pa.field(name, t, nullable=True), arr)
        else:
            out = out.add_column(out.num_columns, pa.field(name, t, nullable=True), arr)

    write_parquet_atomic(out, ds.records_path, ds.snapshot_dir)
    append_event(
        ds.events_path, {"action": "convert_legacy", "dataset": ds.name, "rows": N}
    )
    # Append a helpful note to the scratch pad
    try:
        skipped_str = ""
        if skipped_bad_len:
            skipped_str += f"; skipped_bad_length={skipped_bad_len}"
        if skipped_dups:
            skipped_str += f"; duplicates={skipped_dups}"
        note = f"Converted legacy .pt files into new dataset (rows={N}, files={len(files)}{skipped_str})."
        ds.append_meta_note(note, f"# example\nusr convert-legacy {dataset_name} --paths ...")
    except Exception:
        pass

    # ---------------- informative console summary ----------------
    valid_len = total_entries - skipped_bad_len
    denom = valid_len if valid_len > 0 else total_entries if total_entries > 0 else 1
    dup_pct = (skipped_dups / denom) * 100.0
    # Always print a compact one-liner summary so the CLI surfaces it
    print(
        f"[convert-legacy] scanned {total_entries} entry/entries from {len(files)} file(s); "
        f"kept {N}; skipped bad-length={skipped_bad_len}"
        + (f", duplicates={skipped_dups} ({dup_pct:.1f}%)" if skipped_dups else "")
    )

    return ConvertStats(
        files=len(files),
        rows=N,
        skipped_bad_len=skipped_bad_len,
        skipped_duplicates=skipped_dups,
    )
