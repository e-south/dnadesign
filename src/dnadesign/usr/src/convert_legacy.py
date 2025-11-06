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

import re
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
    seen_casefold: set[tuple[str, str]] = set()

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
        cf_key = (bt.lower(), seq.upper())

        # Skip duplicates (first seen wins)
        if rid in seen_ids or cf_key in seen_casefold:
            skipped_dups += 1
            continue
        seen_ids.add(rid)
        seen_casefold.add(cf_key)

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
        ds.append_meta_note(
            note, f"# example\nusr convert-legacy {dataset_name} --paths ..."
        )
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
        + (
            f", duplicates(incl. case-insensitive)={skipped_dups} ({dup_pct:.1f}%)"
            if skipped_dups
            else ""
        )
    )

    return ConvertStats(
        files=len(files),
        rows=N,
        skipped_bad_len=skipped_bad_len,
        skipped_duplicates=skipped_dups,
    )


_PROMOTERS = {
    "sigma70_high": {"upstream": "TTGACA", "downstream": "TATAAT"},
    "sigma70_mid": {"upstream": "ACCGCG", "downstream": "TATAAT"},
    "sigma70_low": {"upstream": "GCAGGT", "downstream": "TATAAT"},
}

_DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_DNA_COMP)[::-1]


def _parse_tfbs_parts(parts: Sequence[str], *, min_len: int) -> list[tuple[str, str]]:
    """
    Parse ['tf:motif', ...] → [('tf', 'MOTIF'), ...], filtering out short motifs.
    TF is lower-cased; motif upper-cased.
    """
    out: list[tuple[str, str]] = []
    for raw in parts or []:
        if not isinstance(raw, str) or ":" not in raw:
            continue
        tf, motif = raw.split(":", 1)
        tf = (tf or "").strip().lower()
        motif = (motif or "").strip().upper()
        if not tf or not motif or len(motif) < int(min_len):
            continue
        out.append((tf, motif))
    return out


def _scan_used_tfbs(
    seq: str, tfbs_parts: list[tuple[str, str]]
) -> tuple[list[str], list[dict], dict]:
    """
    From a sequence and cleaned parts [('tf','MOTIF')...], compute:
      - used_tfbs: ['tf:motif', ...] for motifs found in seq (fwd or revcmp)
      - used_tfbs_detail: [{'offset':int,'orientation':'fwd|rev','tf':str,'tfbs':str}, ...]
      - used_tf_counts: {'cpxr':int, 'lexa':int}
    Rule: take the *earliest* (lowest offset) match among fwd/rev per motif, one entry per part.
    """
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts = {"cpxr": 0, "lexa": 0}

    sU = (seq or "").upper()

    for tf, motif in tfbs_parts:
        if not motif:
            continue
        fwd = sU.find(motif)
        rev_m = _revcomp(motif)
        rev = sU.find(rev_m)

        if fwd < 0 and rev < 0:
            continue

        if fwd >= 0 and (rev < 0 or fwd <= rev):
            used_simple.append(f"{tf}:{motif}")
            used_detail.append(
                {"offset": int(fwd), "orientation": "fwd", "tf": tf, "tfbs": motif}
            )
        else:
            used_simple.append(f"{tf}:{motif}")
            used_detail.append(
                {"offset": int(rev), "orientation": "rev", "tf": tf, "tfbs": motif}
            )

        if tf in counts:
            counts[tf] += 1

    used_detail.sort(key=lambda d: (d["offset"], d["tf"]))
    return used_simple, used_detail, counts


def _detect_promoter_forward(seq: str, plan_name: str) -> list[dict]:
    """
    Find forward-strand promoter hexamers for the row's plan.
    Adds entries in the same detail schema with 'tf' like 'sigma70_low_upstream'.
    """
    plan = (plan_name or "").strip() or profile_60bp_dual_promoter().densegen_plan
    pc = _PROMOTERS.get(plan, _PROMOTERS[profile_60bp_dual_promoter().densegen_plan])
    sU = (seq or "").upper()
    extras: list[dict] = []

    for key in ("upstream", "downstream"):
        motif = pc.get(key, "")
        if not motif:
            continue
        start = 0
        while True:
            idx = sU.find(motif, start)
            if idx < 0:
                break
            extras.append(
                {
                    "offset": int(idx),
                    "orientation": "fwd",
                    "tf": f"{plan}_{key}",
                    "tfbs": motif,
                }
            )
            start = idx + 1  # allow multiple occurrences

    extras.sort(key=lambda d: (d["offset"], d["tf"]))
    return extras


@dataclass
class RepairStats:
    rows_total: int
    rows_with_parts: int
    rows_with_cpxr_g_bug: int
    rows_missing_used_tfbs: int
    rows_missing_detail: int
    rows_single_tf: int
    rows_id_seq_only: int
    rows_changed_tfbs_parts: int
    rows_changed_used_tfbs: int
    rows_changed_used_detail: int
    rows_changed_used_counts: int
    rows_changed_used_list: int
    rows_touched: int


def repair_densegen_used_tfbs(
    *,
    dataset_root: Path,
    dataset_name: str,
    min_tfbs_len: int = 6,
    dry_run: bool = False,
    assume_yes: bool = False,
    dedupe_policy: str | None = None,  # None/off | "keep-first" | "keep-last" | "ask"
    drop_missing_used_tfbs: bool = False,
    drop_single_tf: bool = False,
    drop_id_seq_only: bool = False,
    filter_single_tf: bool = False,
) -> RepairStats:
    """
    Clean/repair a USR dataset in-place:
      - Drop 1-nt TFBS (e.g., 'cpxr:G') from densegen__tfbs_parts
      - Recompute densegen__used_tfbs / __used_tfbs_detail (search seq, fwd+rev)
      - Recompute densegen__used_tf_counts / __used_tf_list
      - Append forward-strand sigma70 upstream/downstream details based on densegen__plan
      - Assert post-conditions (no single-base TFBS anywhere, detail shape for all rows)
      - (optional): Case-insensitive sequence de-duplication (policy-driven)
      - (optional): Drop rows whose recomputed 'used_tfbs' is empty
    """
    ds = Dataset(dataset_root, dataset_name)
    tbl = read_parquet(ds.records_path)

    # ----------- (optional) Case-insensitive de-duplication (pre-clean) -----------
    if dedupe_policy and dedupe_policy.lower() != "off":
        import pyarrow as _pa
        import pyarrow.compute as _pc

        df = tbl.select(["id", "bio_type", "sequence", "created_at"]).to_pandas()
        df["_key"] = df["bio_type"].str.lower() + "|" + df["sequence"].str.upper()
        groups = df.groupby("_key").agg({"id": "count"})
        dup_keys = groups[groups["id"] > 1].index.tolist()
        if not dup_keys:
            print(
                "[repair-densegen] dedupe: OK — no case-insensitive duplicates found."
            )
        else:
            # Decide which to keep/drop per group
            keep_ids: list[str] = []
            drop_ids: list[str] = []
            for k, g in df[df["_key"].isin(dup_keys)].groupby("_key"):
                # deterministic order: created_at then id
                g_sorted = g.sort_values(
                    ["created_at", "id"], ascending=True, kind="stable"
                )
                if dedupe_policy == "keep-last":
                    g_sorted = g.sort_values(
                        ["created_at", "id"], ascending=False, kind="stable"
                    )
                if dedupe_policy == "ask" and not dry_run:
                    print(f"\nduplicate sequence (casefold): {k.split('|',1)[1]}")
                    for i, r in enumerate(
                        g_sorted.reset_index(drop=True).itertuples(index=False), start=1
                    ):
                        print(f"  {i}: id={r.id}  created_at={r.created_at}")
                    ans = (
                        input("Keep which row? [1..n], 0=drop all, s=skip group: ")
                        .strip()
                        .lower()
                    )
                    if ans in {"s", "skip"}:
                        keep_ids.extend(g_sorted["id"].tolist())
                        continue
                    if ans in {"0", "drop-all"}:
                        drop_ids.extend(g_sorted["id"].tolist())
                        continue
                    try:
                        kidx = int(ans)
                        if 1 <= kidx <= len(g_sorted):
                            keep_ids.append(g_sorted.iloc[kidx - 1]["id"])
                            drop_ids.extend(
                                g_sorted.drop(g_sorted.index[kidx - 1])["id"].tolist()
                            )
                            continue
                    except Exception:
                        pass
                    # default: keep-first semantics on invalid input
                    keep_ids.append(g_sorted.iloc[0]["id"])
                    drop_ids.extend(g_sorted.iloc[1:]["id"].tolist())
                else:
                    keep_ids.append(g_sorted.iloc[0]["id"])
                    drop_ids.extend(g_sorted.iloc[1:]["id"].tolist())

            print(
                f"[repair-densegen] dedupe plan: groups={len(dup_keys)}  would_drop={len(drop_ids)}"
            )
            if not dry_run and drop_ids:
                if not assume_yes:
                    ans2 = input("Proceed with de-duplication? [y/N]: ").strip().lower()
                    if ans2 not in {"y", "yes"}:
                        print("Skipping de-duplication.")
                    else:
                        drop_set = set(drop_ids)
                        mask = _pc.is_in(
                            tbl.column("id"), value_set=_pa.array(list(drop_set))
                        )
                        tbl = tbl.filter(_pc.invert(mask))
                        print(
                            f"[repair-densegen] dedupe: dropped {len(drop_set)} row(s); rows now {tbl.num_rows}."
                        )

    names = set(tbl.schema.names)
    need_cols = {"id", "sequence", "densegen__plan", "densegen__tfbs_parts"}
    missing = need_cols - names
    if missing:
        raise ValidationError(
            f"Missing required columns for repair: {', '.join(sorted(missing))}"
        )

    # Pull required columns into Python lists
    ids = tbl.column("id").to_pylist()
    seqs = tbl.column("sequence").to_pylist()
    plans = (
        tbl.column("densegen__plan").to_pylist()
        if "densegen__plan" in names
        else [None] * len(ids)
    )
    parts_col = tbl.column("densegen__tfbs_parts").to_pylist()

    used_tfbs_col = (
        tbl.column("densegen__used_tfbs").to_pylist()
        if "densegen__used_tfbs" in names
        else [None] * len(ids)
    )
    detail_col = (
        tbl.column("densegen__used_tfbs_detail").to_pylist()
        if "densegen__used_tfbs_detail" in names
        else [None] * len(ids)
    )
    counts_col = (
        tbl.column("densegen__used_tf_counts").to_pylist()
        if "densegen__used_tf_counts" in names
        else [None] * len(ids)
    )
    used_list_col = (
        tbl.column("densegen__used_tf_list").to_pylist()
        if "densegen__used_tf_list" in names
        else [None] * len(ids)
    )

    # Scan & plan changes
    rows_total = len(ids)
    rows_with_parts = 0
    rows_with_cpxr_g_bug = 0
    rows_missing_used_tfbs = 0
    rows_missing_detail = 0

    changed_parts = 0
    changed_used = 0
    changed_detail = 0
    changed_counts = 0
    changed_used_list = 0
    touched = 0

    new_parts_all: list[list[str] | None] = []
    new_used_all: list[list[str] | None] = []
    new_detail_all: list[list[dict] | None] = []
    new_counts_all: list[dict | None] = []
    new_used_list_all: list[list[str] | None] = []

    def _json_like(x):
        import json

        try:
            return json.dumps(x, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(x)

    for i in range(rows_total):
        seq = str(seqs[i] or "")
        plan = str(plans[i] or "")
        parts_raw = parts_col[i] or []
        parts_clean = _parse_tfbs_parts(parts_raw, min_len=min_tfbs_len)
        rows_with_parts += 1 if parts_raw else 0
        # Bug signal: any single-base item present originally?
        if any(
            isinstance(p, str) and re.search(r"^[a-z]+:[ACGTacgt]$", p)
            for p in parts_raw
        ):
            rows_with_cpxr_g_bug += 1

        if _json_like(parts_clean) != _json_like(parts_raw):
            changed_parts += 1

        used_simple, used_detail, used_counts = _scan_used_tfbs(seq, parts_clean)
        if not (used_simple and isinstance(used_simple, list)):
            rows_missing_used_tfbs += 1

        # promoter extras
        promo = _detect_promoter_forward(seq, plan)
        full_detail = sorted(
            (used_detail + promo), key=lambda d: (d["offset"], d["tf"])
        )

        # used_tf_list
        used_tf_list = (
            sorted({(s.split(":", 1)[0]) for s in used_simple}) if used_simple else []
        )

        # Compare with existing
        old_used = used_tfbs_col[i] or []
        old_detail = detail_col[i] or []
        old_counts = counts_col[i] or None
        old_used_list = used_list_col[i] or []

        if not old_detail:
            rows_missing_detail += 1

        if _json_like(old_used) != _json_like(used_simple):
            changed_used += 1
        if _json_like(old_detail) != _json_like(full_detail):
            changed_detail += 1
        if _json_like(old_counts) != _json_like(
            {"cpxr": used_counts.get("cpxr", 0), "lexa": used_counts.get("lexa", 0)}
        ):
            changed_counts += 1
        if _json_like(old_used_list) != _json_like(used_tf_list):
            changed_used_list += 1

        if any(
            [
                _json_like(parts_clean) != _json_like(parts_raw),
                _json_like(old_used) != _json_like(used_simple),
                _json_like(old_detail) != _json_like(full_detail),
                _json_like(old_counts)
                != _json_like(
                    {
                        "cpxr": used_counts.get("cpxr", 0),
                        "lexa": used_counts.get("lexa", 0),
                    }
                ),
                _json_like(old_used_list) != _json_like(used_tf_list),
            ]
        ):
            touched += 1

        # Store
        new_parts_all.append(
            [f"{tf}:{motif}" for tf, motif in parts_clean] if parts_clean else []
        )
        new_used_all.append(used_simple if used_simple else [])
        new_detail_all.append(full_detail if full_detail else [])
        new_counts_all.append(
            {
                "cpxr": int(used_counts.get("cpxr", 0)),
                "lexa": int(used_counts.get("lexa", 0)),
            }
            if used_simple
            else None
        )
        new_used_list_all.append(used_tf_list if used_tf_list else [])

    # ----------- Optional: summarize & (optionally) drop single-TF rows -----------
    def _is_single_tf(idx: int) -> bool:
        lst = new_used_list_all[idx] or []
        if len(lst) == 1:
            return True
        cnt = new_counts_all[idx] or {}
        # treat missing/None as not single-tf; else exactly one strictly >0
        vals = [int(cnt.get("cpxr", 0)), int(cnt.get("lexa", 0))]
        return sum(1 for v in vals if v > 0) == 1

    rows_single_tf_count = sum(1 for i in range(rows_total) if _is_single_tf(i))

    if filter_single_tf:
        single_mask = [_is_single_tf(i) for i in range(rows_total)]
        n_single = rows_single_tf_count
        pct = 100.0 * n_single / max(1, rows_total)
        print(f"[repair-densegen] rows with only one TF: {n_single} ({pct:.1f}%)")
        do_drop = False
        if n_single > 0:
            if assume_yes:
                do_drop = True
            else:
                ans = (
                    input("Drop rows that include only one TF? [y/N]: ").strip().lower()
                )
                do_drop = ans in {"y", "yes"}
        if do_drop and n_single > 0:
            # Filter the table and all planned arrays in lockstep
            keep_bools = [not b for b in single_mask]
            import pyarrow as _pa

            tbl = tbl.filter(_pa.array(keep_bools))

            def _flt(xs):
                return [x for x, k in zip(xs, keep_bools) if k]

            new_parts_all = _flt(new_parts_all)
            new_used_all = _flt(new_used_all)
            new_detail_all = _flt(new_detail_all)
            new_counts_all = _flt(new_counts_all)
            new_used_list_all = _flt(new_used_list_all)
            rows_total = tbl.num_rows
            print(
                f"[repair-densegen] dropped {n_single} single-TF row(s); rows now {rows_total}."
            )

    # ----------- (optional) Drop rows missing used_tfbs after recompute -----------
    drop_miss_count = sum(1 for u in new_used_all if not u)
    if drop_missing_used_tfbs:
        print(
            f"[repair-densegen] rows with empty used_tfbs after recompute: {drop_miss_count}"
        )
        if not dry_run and drop_miss_count > 0:
            if not assume_yes:
                ans3 = input("Drop rows missing used_tfbs? [y/N]: ").strip().lower()
                if ans3 not in {"y", "yes"}:
                    print("Skipping drop of rows missing used_tfbs.")
                else:
                    import pyarrow as _pa

                    # Convert to simple boolean list for filtering arrays
                    keep_bools = [bool(u) for u in new_used_all]
                    import pyarrow.compute as _pc

                    tbl = tbl.filter(_pa.array(keep_bools))

                    # Filter the “planned arrays” to match filtered rows
                    def _flt(xs):
                        return [x for x, k in zip(xs, keep_bools) if k]

                    new_parts_all = _flt(new_parts_all)
                    new_used_all = _flt(new_used_all)
                    new_detail_all = _flt(new_detail_all)
                    new_counts_all = _flt(new_counts_all)
                    new_used_list_all = _flt(new_used_list_all)
                    # Update counters for accurate write summary
                    rows_total = tbl.num_rows
                    print(
                        f"[repair-densegen] dropped {drop_miss_count} row(s) with empty used_tfbs; rows now {rows_total}."  # noqa
                    )

    # ----------- Summarize & (optionally) drop rows that are "id+sequence only" -----------
    # Definition: a row is "id/sequence-only" if *all* non-essential columns are empty/null.
    # Essentials (always allowed to be present): id, sequence, bio_type, alphabet, length, source, created_at
    import pyarrow as _pa

    essentials = {
        "id",
        "sequence",
        "bio_type",
        "alphabet",
        "length",
        "source",
        "created_at",
    }
    payload_cols = [c for c in tbl.column_names if c not in essentials]
    rows_id_seq_only_count = 0
    id_seq_only_mask: list[bool] | None = None
    if payload_cols:
        n = rows_total
        has_any_payload = [False] * n

        # Robust per-cell "meaningful" check that works across Arrow types *and* JSON-encoded strings.
        def _cell_has_payload(v) -> bool:
            if v is None:
                return False
            # Lists/dicts may already be typed or JSON-dumped strings.
            if isinstance(v, (list, dict)):
                return len(v) > 0
            s = str(v).strip()
            if not s:
                return False
            if s in {"[]", "{}", "null", "None", "nan", "NaN"}:
                return False
            return True

        for col in payload_cols:
            vals = tbl.column(col).to_pylist()
            for i, v in enumerate(vals):
                if not has_any_payload[i] and _cell_has_payload(v):
                    has_any_payload[i] = True
        id_seq_only_mask = [not b for b in has_any_payload]
        rows_id_seq_only_count = sum(1 for b in id_seq_only_mask if b)
    else:
        # No non-essential columns exist → every row is effectively "id+sequence-only"
        id_seq_only_mask = [True] * rows_total
        rows_id_seq_only_count = rows_total

    pct_iso = 100.0 * rows_id_seq_only_count / max(1, rows_total)
    print(
        f"[repair-densegen] rows with only essentials (id/sequence & USR core): {rows_id_seq_only_count} ({pct_iso:.1f}%)"  # noqa
    )
    if drop_id_seq_only and rows_id_seq_only_count > 0 and id_seq_only_mask:
        do_drop_iso = assume_yes
        if not assume_yes:
            ans_iso = (
                input(
                    "Drop rows that lack any non-essential metadata (id/sequence-only)? [y/N]: "
                )
                .strip()
                .lower()
            )  # noqa
            do_drop_iso = ans_iso in {"y", "yes"}
        if do_drop_iso and not dry_run:
            keep_bools = [not b for b in id_seq_only_mask]
            tbl = tbl.filter(_pa.array(keep_bools))

            def _flt(xs):
                return [x for x, k in zip(xs, keep_bools) if k]

            new_parts_all = _flt(new_parts_all)
            new_used_all = _flt(new_used_all)
            new_detail_all = _flt(new_detail_all)
            new_counts_all = _flt(new_counts_all)
            new_used_list_all = _flt(new_used_list_all)
            rows_total = tbl.num_rows
            print(
                f"[repair-densegen] dropped {rows_id_seq_only_count} id/sequence-only row(s); rows now {rows_total}."
            )

    # ----------- Logging (dry-run preview) -----------
    def _pct(n: int, d: int) -> str:
        d = max(1, d)
        return f"{(100.0*n)/d:.1f}%"

    print("\n[repair-densegen] Preflight")
    print(f"  rows total                : {rows_total}")
    print(
        f"  rows with tfbs_parts      : {rows_with_parts} ({_pct(rows_with_parts, rows_total)})"
    )
    print(
        f"  rows w/ single-base bug   : {rows_with_cpxr_g_bug} ({_pct(rows_with_cpxr_g_bug, rows_total)})"
    )
    print(
        f"  rows missing used_tfbs    : {rows_missing_used_tfbs} ({_pct(rows_missing_used_tfbs, rows_total)})"
    )
    print(
        f"  rows missing detail       : {rows_missing_detail} ({_pct(rows_missing_detail, rows_total)})"
    )
    rows_single_tf_after = sum(
        1 for ulist in new_used_list_all if isinstance(ulist, list) and len(ulist) == 1
    )
    print(
        f"  rows with single TF       : {rows_single_tf_after} ({_pct(rows_single_tf_after, rows_total)})"
    )
    print(
        f"  rows id/sequence-only     : {rows_id_seq_only_count} ({_pct(rows_id_seq_only_count, rows_total)})"
    )
    print("  planned changes:")
    print(f"    densegen__tfbs_parts      : {changed_parts}")
    print(f"    densegen__used_tfbs       : {changed_used}")
    print(f"    densegen__used_tfbs_detail: {changed_detail}")
    print(f"    densegen__used_tf_counts  : {changed_counts}")
    print(f"    densegen__used_tf_list    : {changed_used_list}")
    print(f"  rows touched overall      : {touched}\n")

    # Sample diff (first 3)
    if rows_total and dry_run:
        shown = 0
        print("  Examples (first 3 touched rows):")
        for i in range(rows_total):
            if shown >= 3:
                break
            if not (
                new_used_all[i]
                or new_detail_all[i]
                or new_parts_all[i]
                or new_counts_all[i]
                or new_used_list_all[i]
            ):
                continue
            if _json_like(new_used_all[i]) != _json_like(
                used_tfbs_col[i]
            ) or _json_like(new_detail_all[i]) != _json_like(detail_col[i]):
                print(f"   • id={ids[i][:8]}…")
                print(f"     sequence: {seqs[i]}")
                if _json_like(new_parts_all[i]) != _json_like(parts_col[i]):
                    print(
                        f"     tfbs_parts: OLD={parts_col[i]}  →  NEW={new_parts_all[i]}"
                    )
                if _json_like(new_used_all[i]) != _json_like(used_tfbs_col[i]):
                    print(
                        f"     used_tfbs:  OLD={used_tfbs_col[i]}  →  NEW={new_used_all[i]}"
                    )
                if _json_like(new_detail_all[i]) != _json_like(detail_col[i]):
                    print(
                        f"     detail:     OLD={detail_col[i]}  →  NEW={new_detail_all[i]}"
                    )
                shown += 1
        if shown == 0:
            print("   (no touched rows to preview)")

    if dry_run:
        return RepairStats(
            rows_total=rows_total,
            rows_with_parts=rows_with_parts,
            rows_with_cpxr_g_bug=rows_with_cpxr_g_bug,
            rows_missing_used_tfbs=rows_missing_used_tfbs,
            rows_missing_detail=rows_missing_detail,
            rows_single_tf=rows_single_tf_count,
            rows_id_seq_only=rows_id_seq_only_count,
            rows_changed_tfbs_parts=changed_parts,
            rows_changed_used_tfbs=changed_used,
            rows_changed_used_detail=changed_detail,
            rows_changed_used_counts=changed_counts,
            rows_changed_used_list=changed_used_list,
            rows_touched=touched,
        )

    if not assume_yes:
        ans = input("Apply these changes? [Enter = yes / n = abort]: ").strip().lower()
        if ans in {"n", "no"}:
            print("Aborted.")
            return RepairStats(
                rows_total=rows_total,
                rows_with_parts=rows_with_parts,
                rows_with_cpxr_g_bug=rows_with_cpxr_g_bug,
                rows_missing_used_tfbs=rows_missing_used_tfbs,
                rows_missing_detail=rows_missing_detail,
                rows_single_tf=rows_single_tf_count,
                rows_id_seq_only=rows_id_seq_only_count,
                rows_changed_tfbs_parts=0,
                rows_changed_used_tfbs=0,
                rows_changed_used_detail=0,
                rows_changed_used_counts=0,
                rows_changed_used_list=0,
                rows_touched=0,
            )

    # ----------- Build Arrow arrays with types and write -----------
    # Ensure column presence with correct Arrow types, then set.
    out = tbl
    existing = set(out.schema.names)

    def _ensure_col(name: str, t: pa.DataType):
        nonlocal out, existing
        if name not in existing:
            out = out.add_column(
                out.num_columns,
                pa.field(name, t, nullable=True),
                pa.nulls(rows_total, type=t),
            )
            existing.add(name)

    _ensure_col("densegen__tfbs_parts", PA_LIST_STR)
    _ensure_col("densegen__used_tfbs", PA_LIST_STR)
    _ensure_col("densegen__used_tfbs_detail", PA_LIST_STRUCT_TFBS_DETAIL)
    _ensure_col("densegen__used_tf_counts", PA_STRUCT_USED_TF_COUNTS)
    _ensure_col("densegen__used_tf_list", PA_LIST_STR)

    arr_parts = pa.array(new_parts_all, type=PA_LIST_STR)
    arr_used = pa.array(new_used_all, type=PA_LIST_STR)
    arr_detail = pa.array(new_detail_all, type=PA_LIST_STRUCT_TFBS_DETAIL)
    # Struct may contain None → pa.array([...], type=struct)
    arr_counts = pa.array(new_counts_all, type=PA_STRUCT_USED_TF_COUNTS)
    arr_used_list = pa.array(new_used_list_all, type=PA_LIST_STR)

    def _set(name: str, arr: pa.Array, typ: pa.DataType):
        nonlocal out
        idx = out.schema.get_field_index(name)
        out = out.set_column(idx, pa.field(name, typ, nullable=True), arr)

    _set("densegen__tfbs_parts", arr_parts, PA_LIST_STR)
    _set("densegen__used_tfbs", arr_used, PA_LIST_STR)
    _set("densegen__used_tfbs_detail", arr_detail, PA_LIST_STRUCT_TFBS_DETAIL)
    _set("densegen__used_tf_counts", arr_counts, PA_STRUCT_USED_TF_COUNTS)
    _set("densegen__used_tf_list", arr_used_list, PA_LIST_STR)

    write_parquet_atomic(
        out, ds.records_path, ds.snapshot_dir, preserve_metadata_from=tbl
    )
    append_event(
        ds.events_path,
        {
            "action": "repair_densegen",
            "dataset": ds.name,
            "rows": rows_total,
            "touched": touched,
            "min_tfbs_len": min_tfbs_len,
        },
    )

    # ----------- Assertions (no fallbacks) -----------
    out2 = read_parquet(
        ds.records_path,
        columns=[
            "densegen__tfbs_parts",
            "densegen__used_tfbs",
            "densegen__used_tfbs_detail",
        ],
    )
    tp = out2.column("densegen__tfbs_parts").to_pylist()
    tu = out2.column("densegen__used_tfbs").to_pylist()
    td = out2.column("densegen__used_tfbs_detail").to_pylist()

    # 1) no single-base entries remain in parts or used
    if any(
        any(isinstance(x, str) and re.search(r"^[a-z]+:[ACGT]$", x) for x in (p or []))
        for p in tp
    ):
        raise ValidationError(
            "Post-condition failed: single-base TFBS still present in densegen__tfbs_parts."
        )
    if any(
        any(isinstance(x, str) and re.search(r"^[a-z]+:[ACGT]$", x) for x in (u or []))
        for u in tu
    ):
        raise ValidationError(
            "Post-condition failed: single-base TFBS still present in densegen__used_tfbs."
        )

    # 2) every row has well-formed detail (non-null list of dicts)
    for det in td:
        if det is None:
            raise ValidationError(
                "Post-condition failed: densegen__used_tfbs_detail has NULL entries."
            )
        if not isinstance(det, list):
            raise ValidationError(
                "Post-condition failed: densegen__used_tfbs_detail contains non-list."
            )
        for d in det:
            if not (
                isinstance(d, dict)
                and {"offset", "orientation", "tf", "tfbs"} <= set(d.keys())
            ):
                raise ValidationError(
                    "Post-condition failed: malformed dict in densegen__used_tfbs_detail."
                )

    print("[repair-densegen] Applied successfully.")
    return RepairStats(
        rows_total=rows_total,
        rows_with_parts=rows_with_parts,
        rows_with_cpxr_g_bug=rows_with_cpxr_g_bug,
        rows_missing_used_tfbs=rows_missing_used_tfbs,
        rows_missing_detail=rows_missing_detail,
        rows_single_tf=rows_single_tf_count,
        rows_id_seq_only=rows_id_seq_only_count,
        rows_changed_tfbs_parts=changed_parts,
        rows_changed_used_tfbs=changed_used,
        rows_changed_used_detail=changed_detail,
        rows_changed_used_counts=changed_counts,
        rows_changed_used_list=changed_used_list,
        rows_touched=touched,
    )
