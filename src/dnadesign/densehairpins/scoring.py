"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/scoring.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

def compute_cumulative_score(solution_dict, pancardo_df, score_weights):
    """
    Computes a cumulative score for a solution dictionary.
    
    For each TF in the solution's tf_roster, it computes:
      score_TF = (max_rank + 1) - rank + α*(silenced_genes) - β*(induced_genes)
    
    Returns the sum of these scores.
    """
    alpha = score_weights.get("silenced_genes", 1)
    beta = score_weights.get("induced_genes", 1)
    
    max_rank = pancardo_df['Rank'].max()
    cumulative_score = 0
    roster = solution_dict.get("tf_roster", [])
    
    for entry in roster:
        # Expect entry in format "TF:binding_site"
        tf_name = entry.split(":", 1)[0].strip()
        row = pancardo_df[pancardo_df['TF'].str.lower() == tf_name.lower()]
        if row.empty:
            continue
        rank = int(row.iloc[0]['Rank'])
        silenced = int(row.iloc[0]['Silenced Genes'])
        induced = int(row.iloc[0]['Induced Genes'])
        score_tf = (max_rank + 1) - rank + alpha * silenced - beta * induced
        cumulative_score += score_tf
    return cumulative_score

def add_scores_to_solutions(solution_dicts, pancardo_df, score_weights):
    """
    For each solution dictionary, compute and attach cumulative score,
    sequence length, and TF count.
    """
    for sol in solution_dicts:
        cum_score = compute_cumulative_score(sol, pancardo_df, score_weights)
        seq = sol.get("sequence", "")
        sol["Cumulative_Score"] = cum_score
        sol["Sequence_Length"] = len(seq)
        sol["TF_Count"] = len(sol.get("tf_roster", []))
    return solution_dicts
