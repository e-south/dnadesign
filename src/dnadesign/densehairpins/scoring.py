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
    
    Additionally:
      - Rewards diversity with γ*(unique_tf_count) factor
      - Rewards shorter sequences with δ*(sequence_efficiency) factor
    
    Returns the sum of these scores.
    """
    alpha = score_weights.get("silenced_genes", 1)
    beta = score_weights.get("induced_genes", 1)
    gamma = score_weights.get("tf_diversity", 0)  # Diversity weight
    delta = score_weights.get("sequence_length", 0)  # Sequence length weight
    
    max_rank = pancardo_df['Rank'].max()
    cumulative_score = 0
    roster = solution_dict.get("tf_roster", [])
    sequence = solution_dict.get("sequence", "")
    
    # Calculate base score from TF ranks and gene effects
    for tf in roster:
        row = pancardo_df[pancardo_df['TF'].str.lower() == tf.lower()]
        if row.empty:
            continue
        rank = int(row.iloc[0]['Rank'])
        silenced = int(row.iloc[0]['Silenced Genes'])
        induced = int(row.iloc[0]['Induced Genes'])
        score_tf = (max_rank + 1) - rank + alpha * silenced - beta * induced
        cumulative_score += score_tf
    
    # Add diversity bonus based on unique TF count
    unique_tfs = len(set(tf.lower() for tf in roster))
    diversity_bonus = gamma * unique_tfs
    cumulative_score += diversity_bonus
    
    # Add sequence efficiency bonus (shorter is better)
    if len(sequence) > 0 and len(roster) > 0:
        # Higher score for more TFs per nucleotide
        efficiency = len(roster) / len(sequence) * 100  # Scale to make it more noticeable
        sequence_bonus = delta * efficiency
        cumulative_score += sequence_bonus
    
    return cumulative_score

def add_scores_to_solutions(solution_dicts, pancardo_df, score_weights):
    alpha = score_weights.get("silenced_genes", 1)
    beta = score_weights.get("induced_genes", 1)
    gamma = score_weights.get("tf_diversity", 0)
    delta = score_weights.get("sequence_length", 0)
    
    for sol in solution_dicts:
        # Ensure tf_roster is properly handled regardless of format
        roster = sol.get("tf_roster", [])
        original_roster = roster  # Keep for debugging
        
        # Parse roster if it's a string representation
        if isinstance(roster, str):
            try:
                import ast
                # Use ast.literal_eval for safe parsing
                roster = ast.literal_eval(roster)
                sol["tf_roster"] = roster  # Update the roster in the dict
            except Exception as e:
                print(f"Error parsing roster for entry {sol.get('entry_id', 'unknown')}: {e}")
                roster = []
        
        # Ensure roster is a list
        if not isinstance(roster, list):
            print(f"Warning: tf_roster is not a list for entry {sol.get('entry_id', 'unknown')}")
            roster = []
        
        seq = sol.get("sequence", "")
        
        # Correctly calculate metrics with proper normalization
        normalized_roster = [tf.strip().lower() for tf in roster if isinstance(tf, str)]
        unique_tfs = set(normalized_roster)  # This creates a set of unique TF names
        unique_tf_count = len(unique_tfs)
        
        # Store the counts
        sol["Unique_TF_Count"] = unique_tf_count
        sol["TF_Count"] = len(roster)
        sol["Sequence_Length"] = len(seq)
        
        # Calculate TF density (TFs per nucleotide)
        if len(seq) > 0:
            sol["TF_Density"] = len(roster) / len(seq)
        else:
            sol["TF_Density"] = 0
        
        # Calculate score components
        tf_score_components = []
        cumulative_score = 0
        
        for tf in roster:
            if not isinstance(tf, str):
                continue
                
            row = pancardo_df[pancardo_df['TF'].str.lower() == tf.strip().lower()]
            if row.empty:
                continue
                
            rank = int(row.iloc[0]['Rank'])
            silenced = int(row.iloc[0]['Silenced Genes'])
            induced = int(row.iloc[0]['Induced Genes'])
            
            score_tf = (pancardo_df['Rank'].max() + 1) - rank + alpha * silenced - beta * induced
            cumulative_score += score_tf
            
            tf_score_components.append({
                "TF": tf,
                "Rank": rank,
                "Silenced_Genes": silenced,
                "Induced_Genes": induced,
                "Score_Contribution": score_tf
            })
        
        # Add diversity bonus using the correct unique TF count
        diversity_bonus = gamma * unique_tf_count
        cumulative_score += diversity_bonus
        
        # Add sequence efficiency bonus
        if len(seq) > 0 and len(roster) > 0:
            efficiency = len(roster) / len(seq) * 100
            sequence_bonus = delta * efficiency
            cumulative_score += sequence_bonus
            sol["Sequence_Bonus"] = sequence_bonus
        else:
            sol["Sequence_Bonus"] = 0
        
        # Store all score components
        sol["Cumulative_Score"] = cumulative_score
        sol["Diversity_Bonus"] = diversity_bonus
        sol["TF_Score_Details"] = tf_score_components
    
    return solution_dicts
