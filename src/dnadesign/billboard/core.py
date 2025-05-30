"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/core.py

Core processing and metric‐computation routines for the billboard pipeline.

This module handles:
  - Loading and validating sequence data.
  - Parsing TFBS annotations (meta_tfbs_parts).
  - Building motif‐order strings.
  - Computing positional occupancy matrices.
  - Calculating core diversity metrics:
      • tf_richness
      • 1_minus_gini (inverted Gini)
      • min_jaccard_dissimilarity
      • min_tf_entropy
      • min_motif_string_levenshtein
      
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging

import Levenshtein
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy

from dnadesign.aligner.metrics import compute_alignment_scores

# set up module‐level logger
logger = logging.getLogger(__name__)


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    """
    comp_map = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp_map)[::-1]


def load_pt_files(paths):
    """
    Load .pt files containing lists of sequence dictionaries.
    Raises if the loaded object is not a list.
    """
    logger.info(f"Loading .pt files: {paths}")
    seqs = []
    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {p}, got {type(data)}")
        seqs.extend(data)
    return seqs


def validate_sequence(seq):
    """
    Ensure each sequence dict has the required keys:
      - sequence (the DNA string)
      - meta_tfbs_parts (TF→motif annotations)
      - meta_tfbs_parts_in_array (motif instances)
    """
    for key in ("sequence", "meta_tfbs_parts", "meta_tfbs_parts_in_array"):
        if key not in seq:
            raise KeyError(f"Sequence missing '{key}'")


def robust_parse_tfbs(part, seq_id="unknown"):
    """
    Parse a TFBS annotation string into (tf_name, motif).
    Supports two formats:
      - "TF:ACGT..." (colon‐delimited)
      - "idx_#_TF_motif" (underscore‐delimited legacy)
    """
    if ":" in part:
        tf, motif = part.split(":", 1)
        tf = tf.lower()
        if not motif.isalpha():
            raise ValueError(f"Invalid motif '{motif}' in {seq_id}")
        return tf, motif

    # legacy form: strip leading "idx_"
    s = part[4:] if part.startswith("idx_") else part
    tokens = s.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Bad legacy format '{part}' in {seq_id}")
    motif = tokens[-1]
    tf = "_".join(tokens[1:-1]).lower() if len(tokens) > 2 else tokens[0].lower()
    return tf, motif


def build_motif_string(seq_dict, cfg):
    """
    Construct a comma‐delimited motif string for one sequence.
    Steps:
      1. Identify any fixed 'upstream'/'downstream' motifs and skip them.
      2. Build a mapping motif→tf from meta_tfbs_parts.
      3. For each motif in meta_tfbs_parts_in_array:
           - Skip fixed elements.
           - Lookup tf; warn if missing.
           - Find position on forward or reverse strand.
      4. Sort all hits by genomic position and join with commas.
    """
    seq = seq_dict["sequence"]

    # 1) gather fixed elements (we suppress warnings for these)
    fixed = set()
    for pc in seq_dict.get("fixed_elements", {}).get("promoter_constraints", []):
        if pc.get("upstream"):
            fixed.add(pc["upstream"])
        if pc.get("downstream"):
            fixed.add(pc["downstream"])

    # 2) build motif→tf mapping
    mapping = {}
    for part in seq_dict["meta_tfbs_parts"]:
        try:
            tf, motif = robust_parse_tfbs(part, seq_dict.get("id"))
            mapping[motif] = tf
        except ValueError as e:
            logger.warning(str(e))

    hits = []
    # 3) scan for each motif in the sequence
    for motif in seq_dict["meta_tfbs_parts_in_array"]:
        if motif in fixed:
            continue
        tf = mapping.get(motif)
        if tf is None:
            logger.warning(f"Skipped unknown motif {motif}")
            continue

        pos = seq.find(motif)
        strand = "+"
        if pos < 0:
            rc = reverse_complement(motif)
            pos = seq.find(rc)
            if pos >= 0:
                strand = "-"
            else:
                logger.warning(f"Motif '{motif}' not found in sequence {seq_dict.get('id')}")
                continue

        hits.append((pos, f"{tf}{strand}"))

    # 4) sort and join
    hits.sort(key=lambda x: x[0])
    return ",".join(tok for _, tok in hits)


def token_edit_distance(t1, t2, tf_penalty, strand_penalty, partial_penalty):
    """
    Compute a token‐based Levenshtein distance between two motif‐token lists.
    Costs:
      - insertion/deletion: 1
      - substitution:
          • 0 if tokens identical
          • strand_penalty if same TF but opposite strand
          • partial_penalty if TF appears elsewhere in the other list
          • tf_penalty otherwise
    Returns the raw edit distance.
    """
    m, n = len(t1), len(t2)
    dp = np.zeros((m + 1, n + 1))
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a, b = t1[i - 1], t2[j - 1]
            if a == b:
                cost = 0
            else:
                tf1, st1 = a[:-1], a[-1]
                tf2, st2 = b[:-1], b[-1]
                if tf1 == tf2:
                    cost = strand_penalty
                elif any(x.startswith(tf1) for x in t2):
                    cost = partial_penalty
                else:
                    cost = tf_penalty

            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

    return dp[m, n]


def process_sequences(pt_paths, cfg):
    """
    Main data‐loading and processing pipeline.
    If `pt_paths` is a list of dicts, treat it as raw sequences (for cluster‐level reuse);
    otherwise load .pt files from disk.
    Returns a results dict containing:
      - raw sequences
      - sequence_length
      - motif_strings
      - motif_info list of dicts for plotting
      - occupancy matrices (forward/reverse)
      - tf_frequency counts
      - per_tf_entropy list
      - total_tfbs_instances
    """
    # 1) load and validate. support raw sequence dicts directly
    if pt_paths and isinstance(pt_paths[0], dict):
        seqs = pt_paths
    else:
        seqs = load_pt_files(pt_paths)

    assert seqs, "No sequences loaded"
    logger.info(f"Processing {len(seqs)} sequences")

    # 2) build motif strings & collect lengths
    lengths, motif_strings = [], []
    for s in seqs:
        validate_sequence(s)
        lengths.append(len(s["sequence"]))
        motif_strings.append(build_motif_string(s, cfg))

    # 3) enforce consistent length
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1 and not cfg.get("allow_variable_sequence_length", False):
        raise ValueError(f"Inconsistent lengths: {unique_lengths}")
    sequence_length = max(unique_lengths)

    # 4) initialize accumulators
    tf_frequency = {}
    occupancy_fwd = {}
    occupancy_rev = {}
    motif_info = []
    total_instances = 0

    # 5) count only placed motifs (meta_tfbs_parts_in_array)
    for s in seqs:
        seq = s["sequence"]
        # build motif->tf map
        mapping = {}
        for part in s["meta_tfbs_parts"]:
            try:
                tf, motif = robust_parse_tfbs(part, s.get("id"))
                mapping[motif] = tf
            except ValueError:
                continue

        for motif in s["meta_tfbs_parts_in_array"]:
            tf = mapping.get(motif)
            if tf is None:
                continue
            # TF frequency
            tf_frequency[tf] = tf_frequency.get(tf, 0) + 1
            total_instances += 1
            motif_info.append({"tf": tf, "motif": motif})

            # occupancy
            idx = seq.find(motif)
            if idx >= 0:
                arr, pos = occupancy_fwd, idx
            else:
                arr, pos = occupancy_rev, seq.find(reverse_complement(motif))

            if pos >= 0:
                arr.setdefault(tf, np.zeros(sequence_length, dtype=int))[pos : pos + len(motif)] += 1

    # 6) build occupancy matrices
    tf_list = sorted(set(occupancy_fwd) | set(occupancy_rev))
    F = np.vstack([occupancy_fwd.get(tf, np.zeros(sequence_length, dtype=int)) for tf in tf_list])
    R = np.vstack([occupancy_rev.get(tf, np.zeros(sequence_length, dtype=int)) for tf in tf_list])

    # 7) compute per‐TF positional entropy (base‐2, normalized by log₂(L))
    per_tf_entropy = []
    log2L = np.log2(sequence_length)
    for i, tf in enumerate(tf_list):
        pf, pr = F[i], R[i]

        def ent(arr):
            s = arr.sum()
            return 0.0 if s == 0 else float(scipy_entropy(arr / s, base=2) / log2L)

        per_tf_entropy.append({"tf": tf, "avg_entropy": (ent(pf) + ent(pr)) / 2})

    logger.info(
        f"Done processing: {len(seqs)} seqs | " f"{total_instances} TFBS instances | " f"{len(tf_list)} unique TFs"
    )

    return {
        "sequences": seqs,
        "num_sequences": len(seqs),
        "sequence_length": sequence_length,
        "tf_frequency": tf_frequency,
        "occupancy_tf_list": tf_list,
        "occupancy_forward_matrix": F,
        "occupancy_reverse_matrix": R,
        "motif_strings": motif_strings,
        "motif_info": motif_info,
        "per_tf_entropy": per_tf_entropy,
        "total_tfbs_instances": total_instances,
    }


# -------------------------------------------------------------------------
# Core diversity metrics
# -------------------------------------------------------------------------


def compute_tf_richness(results, cfg):
    """
    TF Richness = |unique TFs placed|
    """
    return len(results["tf_frequency"])


def compute_inverted_gini(results, cfg):
    """
    Inverted Gini: 1 - (∑₁ⁿ∑₁ⁿ|f_i - f_j|)/(2 n ∑ f_i)
    """
    freqs = np.array(list(results["tf_frequency"].values()), dtype=float)
    n = freqs.size
    total = freqs.sum()
    if n == 0 or total == 0:
        return 0.0
    diffs = np.abs(freqs.reshape(-1, 1) - freqs.reshape(1, -1)).sum()
    gini = diffs / (2 * n * total)
    return 1.0 - gini


def compute_min_jaccard_dissimilarity(results, cfg):
    """
    Min pairwise Jaccard dissimilarity over TF sets from meta_tfbs_parts_in_array.
    """
    sets = []
    for s in results["sequences"]:
        mp = {}
        for part in s["meta_tfbs_parts"]:
            try:
                tf, motif = robust_parse_tfbs(part, s.get("id"))
                mp[motif] = tf
            except ValueError:
                continue
        tset = {mp[m] for m in s["meta_tfbs_parts_in_array"] if m in mp}
        sets.append(tset)

    best = 1.0
    N = len(sets)
    if N < 2:
        return 0.0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = sets[i], sets[j]
            U = len(a | b)
            if U:
                d = 1 - len(a & b) / U
                best = min(best, d)
                if best == 0.0:
                    return 0.0
    return best


def compute_min_tf_entropy(results, cfg):
    """
    Min per‑TF positional entropy (base‑2, normalized).
    """
    entropies = [e["avg_entropy"] for e in results["per_tf_entropy"]]
    return float(min(entropies)) if entropies else 0.0


def compute_min_motif_string_levenshtein(results, cfg):
    """
    Min normalized motif-string edit distance (python-Levenshtein).
    """
    p = cfg["motif_string_levenshtein"]
    toks = [ms.split(",") for ms in results["motif_strings"] if ms]
    best = float("inf")
    N = len(toks)
    for i in range(N):
        si = "|".join(toks[i])
        li = len(toks[i])
        for j in range(i + 1, N):
            sj = "|".join(toks[j])
            lj = len(toks[j])
            if li == 0 or lj == 0:
                raw = max(li, lj)
            else:
                raw = Levenshtein.distance(si, sj)
            norm = raw / max(li, lj) if max(li, lj) else 0.0
            best = min(best, norm)
            if best == 0.0:
                return 0.0
    return 0.0 if best == float("inf") else best


def compute_min_nw_dissimilarity(results: dict, cfg: dict) -> float | None:
    """
    Compute the *minimum* pairwise Needleman–Wunsch dissimilarity
    (1 - normalized similarity) across all sequences.

    If skip_aligner_call is True in cfg, returns None.
    """
    if cfg.get("skip_aligner_call", False):
        return None

    seqs = [s["sequence"] for s in results["sequences"]]
    # get the condensed vector of *normalized* similarities
    out = compute_alignment_scores(
        sequences=seqs,
        sequence_key="sequence",
        return_formats=("condensed",),
        normalize=True,
        return_dissimilarity=False,
        verbose=False,
    )
    # compute_alignment_scores returns a dict if return_formats != ("mean",)
    sims = out["condensed"] if isinstance(out, dict) else np.asarray(out)
    if sims.size == 0:
        return 0.0
    diss = 1.0 - sims
    return float(np.min(diss))


def compute_core_metrics(results: dict, cfg: dict) -> dict:
    """
    Dispatch to each metric implementation based on cfg["diversity_metrics"].
    """
    mapping = {
        "tf_richness": compute_tf_richness,
        "1_minus_gini": compute_inverted_gini,
        "min_jaccard_dissimilarity": compute_min_jaccard_dissimilarity,
        "min_tf_entropy": compute_min_tf_entropy,
        "min_motif_string_levenshtein": compute_min_motif_string_levenshtein,
        "min_nw_dissimilarity": compute_min_nw_dissimilarity,
    }

    out = {}
    for m in cfg["diversity_metrics"]:
        if m not in mapping:
            raise KeyError(f"Unknown metric '{m}' in diversity_metrics")
        out[m] = mapping[m](results, cfg)
        logger.info(f"Metric {m}: {out[m]}")
    return out
