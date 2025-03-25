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
from scipy.stats import entropy as scipy_entropy
import warnings

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def load_pt_files(pt_paths):
    """Load and pool sequence entries from a list of .pt files."""
    sequences = []
    for path in pt_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, list):
            raise ValueError(f"Data in '{path}' is not a list of sequences.")
        sequences.extend(data)
    return sequences

def validate_sequence(seq):
    """Ensure a sequence dictionary contains required keys."""
    required_keys = ["sequence", "meta_tfbs_parts", "meta_offsets"]
    for key in required_keys:
        if key not in seq:
            raise ValueError(f"Sequence entry (id: {seq.get('id', 'unknown')}) is missing required key: {key}")

def normalize_visual(visual_str):
    """
    Process the meta_sequence_visual string.
    Lines starting with '-->' are taken as-is (after stripping the marker),
    and lines starting with '<--' are reverse-complemented.
    The normalized lines are concatenated.
    """
    normalized_lines = []
    for line in visual_str.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-->"):
            normalized_lines.append(line[3:].strip())
        elif line.startswith("<--"):
            normalized_lines.append(reverse_complement(line[3:].strip()))
        else:
            normalized_lines.append(line)
    return "".join(normalized_lines)

def robust_parse_tfbs(part, seq_id="unknown"):
    """
    Parse a meta_tfbs_parts string robustly.
    Expected format: "idx_{index}_{TF_name}_{motif}"
    where the TF_name may contain underscores and the motif is a nucleotide string.
    Returns (tf_name, motif).
    """
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
    # Use all parts except the first (index) and the last (motif) as TF name.
    tf_name = "_".join(parts[1:-1]).lower()
    if not tf_name:
        tf_name = parts[1].lower()
    motif = motif_candidate
    return tf_name, motif

def process_sequences(pt_paths, config):
    """
    Load, validate, and process sequences from one or more .pt files.
    Computes coverage matrices, positional entropy metrics, per-TF entropy statistics,
    and TF combination (roster) entropy.
    
    Key Entropy Calculations:
      - Global Positional Entropy: Shannon entropy computed over all positions (from forward and 
        reverse matrices), averaged and normalized by log2(sequence_length). It reflects whether 
        motifs are clustered or evenly distributed across the library.
      - Per-TF Positional Entropy: For each TF, the entropy of its positional binding is computed 
        (averaged over forward and reverse strands and normalized), capturing binding flexibility. 
        Summary statistics (mean, weighted mean, top-K average) are derived from these values.
      - TF Combination (Roster) Entropy: Shannon entropy computed over the unique sets of TFs 
        present in each sequence (normalized by log2(number of unique combinations)).
      - **TF Frequency Entropy**: Newly added metric. It computes the Shannon entropy of the TF 
        frequency distribution (as used in the TF frequency bar plot) to assess how evenly the TFs 
        are represented across the library. This value is normalized by log2(n_unique_TFs).
    """
    sequences_list = load_pt_files(pt_paths)
    if not sequences_list:
        raise ValueError("No sequences loaded from provided pt files.")
    
    allowed_variable_length = config.get("allow_variable_sequence_length", False)
    seq_lengths = []
    for seq in sequences_list:
        validate_sequence(seq)
        seq_lengths.append(len(seq["sequence"]))
    unique_lengths = set(seq_lengths)
    if len(unique_lengths) > 1:
        msg = f"Inconsistent sequence lengths found: {unique_lengths}"
        if not allowed_variable_length:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    sequence_length = max(unique_lengths)
    
    mapping_failures = []
    total_tfbs_instances = 0
    tf_frequency = {}
    
    # Coverage dictionaries.
    coverage_forward_dict = {}
    coverage_reverse_dict = {}
    per_sequence_combos = []
    
    # Lists for motif information.
    motif_passed_info = []
    motif_failed_info = []
    
    track_unique_roster = config.get("track_unique_roster", True)
    include_fixed_in_combos = config.get("include_fixed_elements_in_combos", False)
    
    # Set flag for fixed element warnings so that they print only once.
    fixed_element_warning_printed = False

    for seq in sequences_list:
        seq_str = seq["sequence"]
        tf_names_in_seq = []
        tfbs_parts = seq["meta_tfbs_parts"]
        visual_str = seq.get("meta_sequence_visual", None)
        if visual_str:
            normalized_visual = normalize_visual(visual_str)
        else:
            normalized_visual = None
        
        for part in tfbs_parts:
            try:
                tf_name, motif = robust_parse_tfbs(part, seq_id=seq.get("id", "unknown"))
            except ValueError as e:
                mapping_failures.append({"sequence_id": seq.get("id", "unknown"), "tf": "", "motif": "", "reason": str(e)})
                continue
            motif_length = len(motif)
            # Verify using normalized visual if available.
            if normalized_visual is not None:
                idx_v = normalized_visual.find(motif)
                if idx_v == -1:
                    rc_motif = reverse_complement(motif)
                    idx_v = normalized_visual.find(rc_motif)
                    if idx_v == -1:
                        mapping_failures.append({
                            "sequence_id": seq.get("id", "unknown"),
                            "tf": tf_name,
                            "motif": motif,
                            "reason": "Motif not found in normalized meta_sequence_visual"
                        })
                        motif_failed_info.append({"tf": tf_name, "length": motif_length, "status": "fail"})
                        continue  # Skip this motif
            if tf_name not in coverage_forward_dict:
                coverage_forward_dict[tf_name] = np.zeros(sequence_length, dtype=int)
                coverage_reverse_dict[tf_name] = np.zeros(sequence_length, dtype=int)
            index_fwd = seq_str.find(motif)
            if index_fwd != -1:
                coverage_forward_dict[tf_name][index_fwd:index_fwd+motif_length] += 1
                tf_names_in_seq.append(tf_name)
                tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
                total_tfbs_instances += 1
                motif_passed_info.append({"tf": tf_name, "length": motif_length, "status": "pass"})
            else:
                rc_motif = reverse_complement(motif)
                index_rev = seq_str.find(rc_motif)
                if index_rev != -1:
                    coverage_reverse_dict[tf_name][index_rev:index_rev+motif_length] += 1
                    tf_names_in_seq.append(tf_name)
                    tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
                    total_tfbs_instances += 1
                    motif_passed_info.append({"tf": tf_name, "length": motif_length, "status": "pass"})
                else:
                    mapping_failures.append({
                        "sequence_id": seq.get("id", "unknown"),
                        "tf": tf_name,
                        "motif": motif,
                        "reason": "No match found in full sequence"
                    })
                    motif_failed_info.append({"tf": tf_name, "length": motif_length, "status": "fail"})
        
        # Process fixed promoter elements.
        # Instead of warning for each sequence, we print one combined warning if any sequence is missing fixed elements.
        if "config" in seq and seq["config"]:
            if "fixed_elements" in seq["config"] and seq["config"]["fixed_elements"]:
                fixed_elements = seq["config"]["fixed_elements"]
                if "promoter_constraints" in fixed_elements and fixed_elements["promoter_constraints"]:
                    for constraint in fixed_elements["promoter_constraints"]:
                        name = constraint.get("name", "fixed")
                        for label, element in [("upstream", constraint.get("upstream")), ("downstream", constraint.get("downstream"))]:
                            if element:
                                pseudo_tf = f"{name}_{label}".lower()
                                if pseudo_tf not in coverage_forward_dict:
                                    coverage_forward_dict[pseudo_tf] = np.zeros(sequence_length, dtype=int)
                                    coverage_reverse_dict[pseudo_tf] = np.zeros(sequence_length, dtype=int)
                                if normalized_visual:
                                    idx_v = normalized_visual.find(element)
                                    if idx_v == -1:
                                        rc_elem = reverse_complement(element)
                                        idx_v = normalized_visual.find(rc_elem)
                                        if idx_v == -1:
                                            mapping_failures.append({
                                                "sequence_id": seq.get("id", "unknown"),
                                                "tf": pseudo_tf,
                                                "motif": element,
                                                "reason": f"Fixed element {label} not found in normalized meta_sequence_visual"
                                            })
                                            continue
                                pos = seq_str.find(element)
                                if pos != -1:
                                    coverage_forward_dict[pseudo_tf][pos:pos+len(element)] += 1
                                    if include_fixed_in_combos:
                                        tf_names_in_seq.append(pseudo_tf)
                                    tf_frequency[pseudo_tf] = tf_frequency.get(pseudo_tf, 0) + 1
                                    total_tfbs_instances += 1
                                else:
                                    rc_element = reverse_complement(element)
                                    pos = seq_str.find(rc_element)
                                    if pos != -1:
                                        coverage_reverse_dict[pseudo_tf][pos:pos+len(element)] += 1
                                        if include_fixed_in_combos:
                                            tf_names_in_seq.append(pseudo_tf)
                                        tf_frequency[pseudo_tf] = tf_frequency.get(pseudo_tf, 0) + 1
                                        total_tfbs_instances += 1
                                    else:
                                        mapping_failures.append({
                                            "sequence_id": seq.get("id", "unknown"),
                                            "tf": pseudo_tf,
                                            "motif": element,
                                            "reason": f"Fixed element {label} not found in full sequence"
                                        })
                else:
                    if not fixed_element_warning_printed:
                        warnings.warn("Warning: Some sequences were found without promoter_constraints in fixed_elements; these entries will be skipped.")
                        fixed_element_warning_printed = True
            else:
                if not fixed_element_warning_printed:
                    warnings.warn("Warning: Some sequences were found without fixed_elements; skipping fixed element processing.")
                    fixed_element_warning_printed = True
        else:
            if not fixed_element_warning_printed:
                warnings.warn("Warning: Some sequences were found without config; skipping fixed element processing.")
                fixed_element_warning_printed = True
        
        if track_unique_roster:
            unique_roster = sorted(set(tf_names_in_seq))
        else:
            unique_roster = sorted(tf_names_in_seq)
        per_sequence_combos.append(unique_roster)
    
    sorted_tf_list = sorted(coverage_forward_dict.keys())
    # For occupancy plots, exclude fixed elements.
    occupancy_tf_list = [tf for tf in sorted_tf_list if not (tf.endswith("_upstream") or tf.endswith("_downstream"))]
    
    full_forward_matrix = np.array([coverage_forward_dict[tf] for tf in sorted_tf_list])
    full_reverse_matrix = np.array([coverage_reverse_dict[tf] for tf in sorted_tf_list])
    occupancy_forward_matrix = np.array([coverage_forward_dict[tf] for tf in occupancy_tf_list])
    occupancy_reverse_matrix = np.array([coverage_reverse_dict[tf] for tf in occupancy_tf_list])
    
    def compute_entropy(matrix):
        pos_totals = matrix.sum(axis=0)
        if pos_totals.sum() == 0:
            return 0.0
        prob_dist = pos_totals / pos_totals.sum()
        return float(scipy_entropy(prob_dist))
    
    # Global positional entropy for the entire library (forward and reverse combined)
    positional_entropy_forward = compute_entropy(full_forward_matrix)
    positional_entropy_reverse = compute_entropy(full_reverse_matrix)
    global_positional_entropy = (positional_entropy_forward + positional_entropy_reverse) / 2.0
    # Normalization factor: log2(sequence_length)
    pos_norm_factor = np.log2(sequence_length) if sequence_length > 1 else 1
    global_positional_entropy_norm = global_positional_entropy / pos_norm_factor if pos_norm_factor > 0 else 0.0
    
    # Compute per-TF positional entropy.
    per_tf_entropy = []
    for tf in sorted_tf_list:
        forward_array = coverage_forward_dict[tf]
        reverse_array = coverage_reverse_dict[tf]
        def safe_entropy(arr):
            return 0.0 if arr.sum() == 0 else float(scipy_entropy(arr / arr.sum()))
        per_tf_entropy.append({
            "tf": tf,
            "entropy_forward": safe_entropy(forward_array),
            "entropy_reverse": safe_entropy(reverse_array)
        })
    
    # Prepare per-TF entropy details: raw average and normalized value for each TF.
    per_tf_entropy_details = []
    for entry in per_tf_entropy:
        raw = (entry["entropy_forward"] + entry["entropy_reverse"]) / 2.0
        norm = raw / pos_norm_factor if pos_norm_factor > 0 else 0.0
        per_tf_entropy_details.append({"tf": entry["tf"], "raw": raw, "norm": norm})
    
    # Compute summary statistics for per-TF entropy:
    if per_tf_entropy_details:
        raw_values = np.array([d["raw"] for d in per_tf_entropy_details])
        norm_values = np.array([d["norm"] for d in per_tf_entropy_details])
        per_tf_mean_raw = float(np.mean(raw_values))
        per_tf_mean_norm = float(np.mean(norm_values))
        # Weighted mean: weight each TF's entropy by its frequency.
        tf_freq = []
        weighted_raws = []
        weighted_norms = []
        for d in per_tf_entropy_details:
            freq = tf_frequency.get(d["tf"], 0)
            tf_freq.append(freq)
            weighted_raws.append(d["raw"] * freq)
            weighted_norms.append(d["norm"] * freq)
        total_freq = sum(tf_freq) if sum(tf_freq) > 0 else 1
        per_tf_weighted_raw = float(sum(weighted_raws) / total_freq)
        per_tf_weighted_norm = float(sum(weighted_norms) / total_freq)
        # Top-K: average of the top-K most frequent TFs.
        top_k = config.get("top_k", 5)
        per_tf_details_sorted = sorted(per_tf_entropy_details, key=lambda d: tf_frequency.get(d["tf"], 0), reverse=True)
        topk_entries = per_tf_details_sorted[:top_k] if len(per_tf_details_sorted) >= top_k else per_tf_details_sorted
        if topk_entries:
            per_tf_topk_raw = float(np.mean([d["raw"] for d in topk_entries]))
            per_tf_topk_norm = float(np.mean([d["norm"] for d in topk_entries]))
        else:
            per_tf_topk_raw = 0.0
            per_tf_topk_norm = 0.0
    else:
        per_tf_mean_raw = per_tf_mean_norm = per_tf_weighted_raw = per_tf_weighted_norm = per_tf_topk_raw = per_tf_topk_norm = 0.0

    # Calculate TF Combination (Roster) Entropy (based on unique TF rosters in each sequence)
    combo_counts = {}
    for combo in per_sequence_combos:
        combo_key = "|".join(combo)
        combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
    combo_values = np.array(list(combo_counts.values()))
    if combo_values.sum() > 0:
        prob_combo = combo_values / combo_values.sum()
        combo_entropy_value = float(scipy_entropy(prob_combo))
    else:
        combo_entropy_value = 0.0
    n_combos = len(combo_counts)
    combo_norm_factor = np.log2(n_combos) if n_combos > 1 else 1
    tf_combo_entropy_norm = combo_entropy_value / combo_norm_factor if combo_norm_factor > 0 else 0.0

    # Compute TF Frequency Entropy: Shannon entropy of the TF frequency distribution.
    if tf_frequency:
        total_tf_freq = sum(tf_frequency.values())
        tf_probs = np.array([count / total_tf_freq for count in tf_frequency.values()])
        tf_frequency_entropy = float(scipy_entropy(tf_probs))
        num_tfs = len(tf_frequency)
        tf_freq_entropy_norm = tf_frequency_entropy / (np.log2(num_tfs) if num_tfs > 1 else 1)
    else:
        tf_frequency_entropy = 0.0
        tf_freq_entropy_norm = 0.0

    num_sequences = len(sequences_list)
    average_tfbs_per_sequence = total_tfbs_instances / num_sequences if num_sequences > 0 else 0
    
    # Combine motif info.
    motif_info = motif_passed_info + motif_failed_info
    
    # Compute per-TF motif statistics.
    from collections import defaultdict
    stats = defaultdict(lambda: {"pass": 0, "fail": 0, "lengths": []})
    for d in motif_info:
        stats[d["tf"]][d["status"]] += 1
        if d["status"] == "pass":
            stats[d["tf"]]["lengths"].append(d["length"])
    motif_stats = []
    for tf, s in stats.items():
        total = s["pass"] + s["fail"]
        ratio = s["pass"] / total if total > 0 else 0
        avg_length = sum(s["lengths"]) / len(s["lengths"]) if s["lengths"] else 0
        motif_stats.append({"tf": tf, "pass_count": s["pass"], "fail_count": s["fail"], "ratio": ratio, "avg_length": avg_length})
    
    results = {
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
        "per_sequence_combos": per_sequence_combos,
        "combo_counts": combo_counts,
        "combo_entropy": combo_entropy_value,
        "motif_info": motif_info,
        "motif_stats": motif_stats,
        # New entropy summary metrics:
        "global_positional_entropy": global_positional_entropy,
        "global_positional_entropy_norm": global_positional_entropy_norm,
        "per_tf_entropy_mean_raw": per_tf_mean_raw,
        "per_tf_entropy_mean_norm": per_tf_mean_norm,
        "per_tf_entropy_weighted_raw": per_tf_weighted_raw,
        "per_tf_entropy_weighted_norm": per_tf_weighted_norm,
        "per_tf_entropy_topk_raw": per_tf_topk_raw,
        "per_tf_entropy_topk_norm": per_tf_topk_norm,
        "tf_combo_entropy_norm": tf_combo_entropy_norm,
        "tf_frequency_entropy": tf_frequency_entropy,
        "tf_frequency_entropy_norm": tf_freq_entropy_norm
    }
    return results
