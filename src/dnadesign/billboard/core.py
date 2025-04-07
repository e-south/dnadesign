"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/core.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
import numpy as np
import warnings
from scipy.stats import entropy as scipy_entropy
# Removed dependency on rapidfuzz; using our own CPython implementation

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def load_pt_files(pt_paths):
    """Load sequence entries from the provided .pt file paths."""
    sequences = []
    for path in pt_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, list):
            raise ValueError(f"Data in '{path}' is not a list of sequences.")
        sequences.extend(data)
    return sequences

def validate_sequence(seq):
    """Ensure a sequence dictionary contains required keys."""
    required_keys = ["sequence", "meta_tfbs_parts", "meta_tfbs_parts_in_array"]
    for key in required_keys:
        if key not in seq:
            raise ValueError(f"Sequence entry (id: {seq.get('id', 'unknown')}) is missing required key: {key}")

def robust_parse_tfbs(part, seq_id="unknown"):
    """
    Parse a meta_tfbs_parts string robustly.
    Supports two formats:
      - Colon-delimited: "tf:motif"
      - Underscore-delimited (legacy): "idx_{index}_{TF_name}_{motif}"
    Returns a tuple (tf_name, motif).
    """
    if ":" in part:
        tf_name, motif = part.split(":", 1)
        tf_name = tf_name.lower()
        if not motif.isalpha() or not all(c in "ATCGatcg" for c in motif):
            raise ValueError(f"Motif candidate '{motif}' is not a valid nucleotide sequence in sequence (id: {seq_id})")
        return tf_name, motif
    if part.startswith("idx_"):
        s = part[4:]
    else:
        s = part
    parts = s.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid meta_tfbs_parts format in sequence (id: {seq_id}): {part}")
    motif_candidate = parts[-1]
    if not motif_candidate.isalpha() or not all(c in "ATCGatcg" for c in motif_candidate):
        raise ValueError(f"Motif candidate '{motif_candidate}' is not a valid nucleotide sequence in sequence (id: {seq_id})")
    tf_name = "_".join(parts[1:-1]).lower() if len(parts) > 2 else parts[0].lower()
    return tf_name, motif_candidate

# --- NEW FUNCTION: Build motif string from a sequence dictionary ---
def build_motif_string(seq_dict, config):
    """
    Generate a motif string representation for a sequence.
    The motif string is constructed as follows:
      1. Parse each entry in meta_tfbs_parts to build a mapping: {motif: tf_name}.
      2. For each motif in meta_tfbs_parts_in_array, scan the sequence (seq_dict["sequence"]) to
         find the first occurrence.
         - If found, mark strand as '+'.
         - If not found, check for its reverse complement; if found, mark strand as '-'.
         - If not found at all, log a warning unless the motif is a fixed element.
      3. If fixed elements are present in seq_dict["fixed_elements"]["promoter_constraints"]:
         - If config["include_fixed_elements_in_combos"] is True, include tokens ("upstream" or "downstream")
           for each fixed element.
         - Otherwise, ignore fixed element motifs without warning.
      4. Sort the found motifs by their 5′ position.
      5. Return a comma-delimited string in the format: "crp+,gadX-,fliZ+"
    """
    seq = seq_dict["sequence"]
    mapping = {}
    for part in seq_dict.get("meta_tfbs_parts", []):
        try:
            tf_name, motif = robust_parse_tfbs(part, seq_id=seq_dict.get("id", "unknown"))
            mapping[motif] = tf_name
        except ValueError as e:
            warnings.warn(str(e))
    
    # Get fixed motifs from fixed_elements if present.
    fixed_motifs = set()
    fixed_tokens_info = []  # List of tuples (position, token) for fixed elements, if included.
    if "fixed_elements" in seq_dict and "promoter_constraints" in seq_dict["fixed_elements"]:
        for pc in seq_dict["fixed_elements"]["promoter_constraints"]:
            upstream = pc.get("upstream")
            downstream = pc.get("downstream")
            # For fixed elements, we don't want to issue warnings.
            if upstream:
                fixed_motifs.add(upstream)
                # Optionally, if including, add with a token label.
                if config.get("include_fixed_elements_in_combos", False):
                    # Use the provided upstream position if available; else, default to 0.
                    pos = pc.get("upstream_pos", [0])[0]
                    fixed_tokens_info.append((pos, "upstream"))
            if downstream:
                fixed_motifs.add(downstream)
                if config.get("include_fixed_elements_in_combos", False):
                    # Without a defined position, append downstream at the end.
                    fixed_tokens_info.append((len(seq), "downstream"))
    
    motif_positions = []
    for motif in seq_dict.get("meta_tfbs_parts_in_array", []):
        # If the motif is a fixed element, skip warning and (optionally) let it be handled later.
        if motif not in mapping:
            if motif in fixed_motifs:
                # If including fixed elements, we will add them later from fixed_tokens_info.
                # Otherwise, silently skip.
                continue
            else:
                warnings.warn(f"Motif '{motif}' not found in meta_tfbs_parts mapping for sequence {seq_dict.get('id', 'unknown')}. Skipping.")
                continue
        tf_name = mapping[motif]
        pos = seq.find(motif)
        strand = "+"
        if pos == -1:
            rc_motif = reverse_complement(motif)
            pos = seq.find(rc_motif)
            if pos != -1:
                strand = "-"
            else:
                warnings.warn(f"Motif '{motif}' (tf: {tf_name}) not found in sequence {seq_dict.get('id', 'unknown')}.")
                continue
        motif_positions.append((pos, f"{tf_name}{strand}"))
    
    # If fixed elements are to be included, add their tokens.
    if config.get("include_fixed_elements_in_combos", False):
        for pos, token in fixed_tokens_info:
            motif_positions.append((pos, token))
    
    # Sort by position (ascending)
    motif_positions.sort(key=lambda x: x[0])
    motif_tokens = [token for pos, token in motif_positions]
    return ",".join(motif_tokens)


# --- NEW FUNCTION: Custom token-based Levenshtein distance with partial match support ---
def token_edit_distance(tokens1, tokens2, tf_penalty, strand_penalty, partial_penalty=0.8):
    """
    Compute the edit distance between two lists of tokens.
    Costs:
      - Insertion and deletion: 1.
      - Substitution:
           * 0 if tokens match exactly.
           * If tokens differ but share the same TF (ignoring strand):
               - If strands differ, cost = strand_penalty.
           * Else if token1's TF is found anywhere in tokens2, cost = partial_penalty.
           * Otherwise, cost = tf_penalty.
    """
    m, n = len(tokens1), len(tokens2)
    dp = np.zeros((m+1, n+1))
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            token1 = tokens1[i-1]
            token2 = tokens2[j-1]
            if token1 == token2:
                cost = 0
            else:
                tf1, strand1 = token1[:-1], token1[-1]
                tf2, strand2 = token2[:-1], token2[-1]
                if tf1 == tf2:
                    # Same TF but different strand
                    cost = strand_penalty
                else:
                    # Check if token1's TF is present anywhere in tokens2
                    if any(t.startswith(tf1) for t in tokens2):
                        cost = partial_penalty
                    else:
                        cost = tf_penalty
            dp[i][j] = min(
                dp[i-1][j] + 1,         # deletion
                dp[i][j-1] + 1,         # insertion
                dp[i-1][j-1] + cost     # substitution
            )
    return dp[m][n]

# --- UPDATED FUNCTION: Compute motif_string_levenshtein metric ---
def compute_motif_string_levenshtein(motif_strings, config):
    """
    Given a list of motif strings (one per sequence), compute all pairwise
    normalized Levenshtein distances using the custom token-based algorithm.
    Normalized distance = raw_distance / max(len(tokens in s1, len(tokens in s2)).
    Returns a dictionary with summary statistics.
    
    If 'motif_string_lev_debug' is True in config, prints debug information:
      - The pair with the minimum distance.
      - The pair with the maximum distance.
      - The pair closest to the median.
    """
    msl_config = config.get("motif_string_levenshtein", {})
    tf_penalty = msl_config.get("tf_penalty", 1.0)
    strand_penalty = msl_config.get("strand_penalty", 0.5)
    partial_penalty = msl_config.get("partial_penalty", 0.8)
    
    distances = []  # List of tuples: (normalized_distance, i, j)
    n = len(motif_strings)
    for i in range(n):
        tokens_i = motif_strings[i].split(",") if motif_strings[i] else []
        for j in range(i+1, n):
            tokens_j = motif_strings[j].split(",") if motif_strings[j] else []
            if not tokens_i or not tokens_j:
                raw_distance = max(len(tokens_i), len(tokens_j))
            else:
                raw_distance = token_edit_distance(tokens_i, tokens_j, tf_penalty, strand_penalty, partial_penalty)
            norm_factor = max(len(tokens_i), len(tokens_j))
            normalized_distance = raw_distance / norm_factor if norm_factor > 0 else 0
            distances.append((normalized_distance, i, j))
    
    if distances:
        norm_vals = [d[0] for d in distances]
        mean_val = np.mean(norm_vals)
        median_val = np.median(norm_vals)
        std_val = np.std(norm_vals)
    else:
        mean_val = median_val = std_val = 0.0

    if config.get("motif_string_lev_debug", False) and distances:
        sorted_distances = sorted(distances, key=lambda x: x[0])
        min_distance, i_min, j_min = sorted_distances[0]
        max_distance, i_max, j_max = sorted_distances[-1]
        median_pair = min(distances, key=lambda x: abs(x[0] - median_val))
        med_distance, i_med, j_med = median_pair
        
        print("==== Motif String Levenshtein Debug Info ====")
        print(f"Minimum normalized distance: {min_distance:.4f}")
        print(f"  Motif String {i_min}: {motif_strings[i_min]}")
        print(f"  Motif String {j_min}: {motif_strings[j_min]}")
        print(f"Median normalized distance: {med_distance:.4f} (target median: {median_val:.4f})")
        print(f"  Motif String {i_med}: {motif_strings[i_med]}")
        print(f"  Motif String {j_med}: {motif_strings[j_med]}")
        print(f"Maximum normalized distance: {max_distance:.4f}")
        print(f"  Motif String {i_max}: {motif_strings[i_max]}")
        print(f"  Motif String {j_max}: {motif_strings[j_max]}")
        print("===============================================")
    
    return {
        "motif_string_levenshtein_mean": mean_val,
        "motif_string_levenshtein_median": median_val,
        "motif_string_levenshtein_std": std_val
    }

# --- Modification in process_sequences: Build motif strings for each sequence ---
def process_sequences(pt_paths, config):
    """
    Load, validate, and process sequences from one or more .pt files.
    Computes occupancy matrices, TF frequency counts, per-sequence TF rosters,
    motif info, and additional statistics.
    
    Additionally, for each sequence, builds an ordered motif string using build_motif_string().
    """
    sequences = load_pt_files(pt_paths)
    if not sequences:
        raise ValueError("No sequences loaded from provided pt files.")
    
    seq_lengths = []
    motif_strings = []
    for seq in sequences:
        validate_sequence(seq)
        seq_lengths.append(len(seq["sequence"]))
        motif_str = build_motif_string(seq, config)
        motif_strings.append(motif_str)
    
    unique_lengths = set(seq_lengths)
    if len(unique_lengths) > 1:
        msg = f"Inconsistent sequence lengths found: {unique_lengths}"
        if not config.get("allow_variable_sequence_length", False):
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    sequence_length = max(unique_lengths)
    
    tf_frequency = {}
    occupancy_forward_dict = {}
    occupancy_reverse_dict = {}
    per_sequence_combos = []
    motif_info = []
    motif_stats = []
    mapping_failures = []
    total_tfbs_instances = 0
    
    for seq in sequences:
        seq_str = seq["sequence"]
        tf_names_in_seq = []
        tfbs_parts = seq["meta_tfbs_parts"]
        motif_to_tf = {}
        for part in tfbs_parts:
            try:
                tf_name, motif = robust_parse_tfbs(part, seq_id=seq.get("id", "unknown"))
                motif_to_tf[motif] = tf_name
            except ValueError as e:
                mapping_failures.append({
                    "sequence_id": seq.get("id", "unknown"),
                    "tf": "",
                    "motif": "",
                    "reason": str(e)
                })
                continue
        for motif in seq.get("meta_tfbs_parts_in_array", []):
            if motif not in motif_to_tf:
                mapping_failures.append({
                    "sequence_id": seq.get("id", "unknown"),
                    "tf": "",
                    "motif": motif,
                    "reason": "Motif not found in meta_tfbs_parts mapping"
                })
                continue
            tf_name = motif_to_tf[motif]
            tf_names_in_seq.append(tf_name)
            tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
            total_tfbs_instances += 1
            motif_info.append({
                "tf": tf_name,
                "motif": motif,
                "status": "pass"
            })
            motif_length = len(motif)
            index_fwd = seq_str.find(motif)
            if index_fwd != -1:
                if tf_name not in occupancy_forward_dict:
                    occupancy_forward_dict[tf_name] = np.zeros(sequence_length, dtype=int)
                occupancy_forward_dict[tf_name][index_fwd:index_fwd+motif_length] += 1
            else:
                rc_motif = reverse_complement(motif)
                index_rev = seq_str.find(rc_motif)
                if index_rev != -1:
                    if tf_name not in occupancy_reverse_dict:
                        occupancy_reverse_dict[tf_name] = np.zeros(sequence_length, dtype=int)
                    occupancy_reverse_dict[tf_name][index_rev:index_rev+motif_length] += 1
        unique_roster = set(tf_names_in_seq)
        per_sequence_combos.append(sorted(unique_roster))
    
    sorted_tf_list = sorted(list(set(occupancy_forward_dict.keys()) | set(occupancy_reverse_dict.keys())))
    full_forward_matrix = []
    full_reverse_matrix = []
    for tf in sorted_tf_list:
        forward = occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int))
        reverse = occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int))
        full_forward_matrix.append(forward)
        full_reverse_matrix.append(reverse)
    full_forward_matrix = np.array(full_forward_matrix)
    full_reverse_matrix = np.array(full_reverse_matrix)
    
    occupancy_tf_list = [tf for tf in sorted_tf_list if not (tf.endswith("_upstream") or tf.endswith("_downstream"))]
    occupancy_forward_matrix = np.array([occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int)) for tf in occupancy_tf_list])
    occupancy_reverse_matrix = np.array([occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int)) for tf in occupancy_tf_list])
    
    def compute_entropy(matrix):
        pos_totals = matrix.sum(axis=0)
        if pos_totals.sum() == 0:
            return 0.0
        prob_dist = pos_totals / pos_totals.sum()
        return float(scipy_entropy(prob_dist))
    
    positional_entropy_forward = compute_entropy(full_forward_matrix)
    positional_entropy_reverse = compute_entropy(full_reverse_matrix)
    global_positional_entropy = (positional_entropy_forward + positional_entropy_reverse) / 2.0
    pos_norm_factor = np.log2(sequence_length) if sequence_length > 1 else 1
    global_positional_entropy_norm = global_positional_entropy / pos_norm_factor if pos_norm_factor > 0 else 0.0
    
    per_tf_entropy = []
    for tf in sorted_tf_list:
        forward_array = occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int))
        reverse_array = occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int))
        def safe_entropy(arr):
            return 0.0 if arr.sum() == 0 else float(scipy_entropy(arr / arr.sum()))
        avg_entropy = (safe_entropy(forward_array) + safe_entropy(reverse_array)) / 2.0
        per_tf_entropy.append({
            "tf": tf,
            "entropy_forward": safe_entropy(forward_array),
            "entropy_reverse": safe_entropy(reverse_array),
            "avg_entropy": avg_entropy
        })
    
    if "positional_entropy_method" in config:
        method = config["positional_entropy_method"]
    else:
        method = "median"
    tf_entropies = [entry["avg_entropy"] for entry in per_tf_entropy if entry["avg_entropy"] is not None]
    if tf_entropies:
        if method == "median":
            aggregated_pos_entropy = float(np.median(tf_entropies))
        else:
            aggregated_pos_entropy = float(np.mean(tf_entropies))
    else:
        aggregated_pos_entropy = 0.0

    num_sequences = len(sequences)
    average_tfbs_per_sequence = total_tfbs_instances / num_sequences if num_sequences > 0 else 0
    
    results = {
        "sequences": sequences,
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "unique_tfs": sorted_tf_list,
        "occupancy_tf_list": occupancy_tf_list,
        "total_tfbs_instances": total_tfbs_instances,
        "average_tfbs_per_sequence": average_tfbs_per_sequence,
        "failed_mappings": mapping_failures,
        "full_coverage_forward_matrix": full_forward_matrix,
        "full_coverage_reverse_matrix": full_reverse_matrix,
        "occupancy_forward_matrix": occupancy_forward_matrix,
        "occupancy_reverse_matrix": occupancy_reverse_matrix,
        "tf_frequency": tf_frequency,
        "positional_entropy_forward": positional_entropy_forward,
        "positional_entropy_reverse": positional_entropy_reverse,
        "per_tf_entropy": per_tf_entropy,
        "aggregated_positional_entropy": aggregated_pos_entropy,
        "per_sequence_combos": per_sequence_combos,
        "combo_counts": { "|".join(combo): count for combo, count in zip(per_sequence_combos, [1]*len(per_sequence_combos)) },
        "global_positional_entropy": global_positional_entropy,
        "global_positional_entropy_norm": global_positional_entropy_norm,
        "motif_info": motif_info,
        "motif_stats": motif_stats,
        "motif_strings": motif_strings  # New key for motif string representations
    }
    return results

def compute_tf_richness(sequences, config):
    """Compute the number of unique TFs across all sequences."""
    unique_tfs = set()
    for seq in sequences:
        roster = set()
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                roster.add(tf_name.lower().strip())
        unique_tfs.update(roster)
    return len(unique_tfs)

def compute_gini(tf_frequency):
    """Compute the Gini coefficient for TF frequency counts and return 1 - Gini."""
    values = np.array(list(tf_frequency.values()), dtype=float)
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    if cumulative[-1] == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return 1 - gini

def compute_jaccard_index(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union

def compute_mean_jaccard(sequences, config):
    """Compute the mean pairwise Jaccard dissimilarity (1 - Jaccard index) of TF rosters."""
    tf_rosters = []
    for seq in sequences:
        roster = set()
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                roster.add(tf_name.lower().strip())
        tf_rosters.append(roster)
    n = len(tf_rosters)
    if n < 2:
        return 0.0
    dissimilarities = []
    for i in range(n):
        for j in range(i+1, n):
            jaccard = compute_jaccard_index(tf_rosters[i], tf_rosters[j])
            dissimilarities.append(1 - jaccard)
    return np.mean(dissimilarities) if dissimilarities else 0.0

def compute_core_metrics(results, config):
    """
    Compute core diversity metrics:
      - tf_richness
      - 1_minus_gini
      - mean_jaccard
      - median_tf_entropy (or mean_tf_entropy if configured)
      - motif_string_levenshtein (if enabled)
    """
    sequences = results["sequences"]
    tf_richness = compute_tf_richness(sequences, config)
    
    tf_frequency = {}
    for seq in sequences:
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                tf_name = tf_name.lower().strip()
                tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
    
    one_minus_gini = compute_gini(tf_frequency)
    mean_jaccard = compute_mean_jaccard(sequences, config)
    
    if "per_tf_entropy" in results:
        tf_entropies = [(entry["entropy_forward"] + entry["entropy_reverse"]) / 2.0 for entry in results["per_tf_entropy"]]
    else:
        tf_entropies = []
    method = config.get("positional_entropy_method", "median")
    if tf_entropies:
        if method == "median":
            aggregated_pos_entropy = float(np.median(tf_entropies))
        else:
            aggregated_pos_entropy = float(np.mean(tf_entropies))
    else:
        aggregated_pos_entropy = 0.0

    core_metrics = {
        "tf_richness": tf_richness,
        "1_minus_gini": one_minus_gini,
        "mean_jaccard": mean_jaccard,
        "median_tf_entropy": aggregated_pos_entropy
    }
    if "motif_string_levenshtein" in config.get("diversity_metrics", []):
        motif_strings = results.get("motif_strings", [])
        msl_metrics = compute_motif_string_levenshtein(motif_strings, config)
        core_metrics.update(msl_metrics)
    
    return core_metrics

def compute_composite_metrics(core_metrics, config):
    """
    Compute composite metrics from the core metrics using a weighted sum.
    Expected keys in core_metrics: "tf_richness", "1_minus_gini", "mean_jaccard",
    "median_tf_entropy", and optionally "motif_string_levenshtein_mean".
    """
    weights = config.get("composite_weights", {})
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, value in core_metrics.items():
        w = weights.get(metric, 0)
        weighted_sum += w * value
        total_weight += w
    if total_weight > 0:
        weighted_sum /= total_weight
    else:
        weighted_sum = 0.0

    return {
        "billboard_weighted_sum": weighted_sum
    }