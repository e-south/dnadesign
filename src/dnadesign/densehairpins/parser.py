"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/parser.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import re

def parse_meme_file(file_path, consensus_only=True, min_length=5):
    """
    Parses a MEME format file to extract binding site sequences.

    When consensus_only is True:
      - Returns the consensus sequence found on the "Multilevel" line.

    When consensus_only is False:
      - Returns a dictionary with:
          "consensus": consensus sequence (if found)
          "others": a list of dictionaries, each with keys:
                     "sequence": the binding site,
                     "type": the identifier from the line (e.g. TF name)
    """
    consensus = None
    others = []  # For additional binding sites.
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Look for consensus on the "Multilevel" line.
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Multilevel"):
            match = re.search(r"^Multilevel\s+([ACGTacgt]+)", stripped)
            if match:
                candidate = match.group(1).upper()
                if len(candidate) >= min_length:
                    # Validate candidate contains only valid nucleotides.
                    assert set(candidate).issubset({"A", "C", "G", "T"}), f"Invalid nucleotides found in consensus: {candidate}"
                    consensus = candidate
                    break

    if consensus_only:
        assert consensus is not None, f"Consensus sequence not found in {file_path} when consensus_only=True"
        return consensus

    # If consensus_only is False, look for additional binding sites in the BLOCKS section.
    in_blocks = False
    for line in lines:
        stripped = line.strip()
        if not in_blocks and "in blocks format" in stripped.lower():
            in_blocks = True
            continue
        if in_blocks:
            if stripped.startswith("//"):
                break  # End of BLOCKS section.
            if not stripped or set(stripped) == {"-"}:
                continue
            fields = re.split(r"\s+", stripped)
            if len(fields) >= 4:
                tf_name = fields[0]
                candidate = fields[3].upper()
                if len(candidate) >= min_length and set(candidate).issubset({"A", "T", "G", "C"}):
                    others.append({"sequence": candidate, "type": tf_name})
    return {"consensus": consensus, "others": others}
