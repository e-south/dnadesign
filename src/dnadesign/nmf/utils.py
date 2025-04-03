"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    """
    complement = str.maketrans("ATCG", "TAGC")
    return seq.translate(complement)[::-1]

def build_motif2tf_map(sequences):
    """
    Returns a dictionary mapping motif.upper() -> tf_name
    as extracted from meta_tfbs_parts. If multiple TFs claim the same motif,
    the last one in the data is used.
    """
    motif2tf = {}
    for seq in sequences:
        tfbs_parts = seq.get("meta_tfbs_parts", [])
        for part in tfbs_parts:
            # part looks like "tf:motif" or "something_something"
            # we parse it
            # robust_parse_tfbs can do it, but let's do a quick approach:
            if ":" in part:
                left, right = part.split(":", 1)
                tf_name = left.strip().lower()
                motif = right.strip().upper()
                motif2tf[motif] = tf_name
    return motif2tf