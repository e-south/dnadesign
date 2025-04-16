"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/align.py

Alignment module using Bio.Align.PairwiseAligner for global alignment
with affine gap penalties.

This module replaces the deprecated Bio.pairwise2 interface with the recommended
PairwiseAligner class from Biopython. For further details, see:
    https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html#pairwise-alignments

The aligner is configured to perform **global alignment** (Needleman–Wunsch) with the following:
  - match_score: Awarded for a match (default: 2)
  - mismatch_score: Penalty for a mismatch (default: -1)
  - gap_open: Gap-opening penalty (provided as a positive value by the API but internally converted to negative).
              Default is 10 (affine gap open = -10).
  - gap_extend: Gap-extension penalty (provided as a positive value by the API but internally converted to negative).
                Default is 1 (affine gap extension = -1).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from Bio.Align import PairwiseAligner

def global_alignment(seqA: str, seqB: str, 
                     match: int = 2, 
                     mismatch: int = -1, 
                     gap_open: int = 10,   # Updated default for affine gap: 10 (=> -10 internally)
                     gap_extend: int = 1,  # Updated default for affine gap: 1 (=> -1 internally)
                     return_alignment_str: bool = False):
    """
    Perform a global alignment between two sequences using PairwiseAligner
    with affine gap penalties.
    
    Parameters:
        seqA (str): First nucleotide sequence.
        seqB (str): Second nucleotide sequence.
        match (int): Score for a match (default: 2).
        mismatch (int): Penalty for a mismatch (default: -1).
        gap_open (int): Gap opening penalty (positive value; default: 10), 
                        internally used as -10.
        gap_extend (int): Gap extension penalty (positive value; default: 1),
                          internally used as -1.
        return_alignment_str (bool): If True, returns a tuple (score, alignment_str)
                                     where alignment_str is the full (global) alignment.
                                     Otherwise, returns only the score.
    
    Returns:
        If return_alignment_str is False:
            float: The best alignment score.
        Else:
            tuple: (score (float), alignment_str (str)) where the alignment_str is the
                   full, end-to-end global alignment.
    
    Raises:
        RuntimeError: If an error occurs during alignment.
    """
    try:
        aligner = PairwiseAligner()
        aligner.mode = "global"  # Global alignment (Needleman–Wunsch)
        aligner.match_score = match
        aligner.mismatch_score = mismatch
        # Convert the positive gap penalties to negatives for proper affine gap scoring.
        aligner.open_gap_score = -abs(gap_open)
        aligner.extend_gap_score = -abs(gap_extend)
        
        # Compute alignments.
        alignments = aligner.align(seqA, seqB)
        if not alignments:
            return (0.0, "") if return_alignment_str else 0.0
        best_alignment = alignments[0]
        score = best_alignment.score
        if return_alignment_str:
            alignment_str = str(best_alignment)
            return score, alignment_str
        else:
            return score
    except Exception as e:
        raise RuntimeError(f"Alignment error: {e}")