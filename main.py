"""
Dense Arrays allows to create densely packed string arrays from a library of motifs.

It models the String Packing Problem as an Integer Linear Programming problem.
"""

import dense_arrays as da

# motifs = [
#     "GAAATAACATAATTGA",
#     "TGTTAATAATAAGTAAT",
#     "TTATATTTTACCCATTT",
#     "AGGTTAATCCTAAAA",
#     "ATTGAAACGATTCAGC",
#     "CTCTGTCATAAAACTGTCATAT",
#     "TTACGCATTTTTAC",
#     "ATTTGTACACAA",
#     "AAGGCATAACCTATCACTGT",
#     "ACGCAAACGTTTTCTT",
#     "TACATTTAGTTACA",
#     "TTAATAAAACCTTAAGGTT",
#     "CCTTTTAGGTGCTT",
#     "TACTGTATATAAAAACAGTA",
#     "TAAAATTCATGGTAATTAT",
#     "AATGAGAATGATTATTAT",
#     "TGTTTATATTTTGTTTA",
#     "CATAAGAAAAA",
#     "CATTCATTTG",
#     "TTGACA",
#     "TATAAT",
#     "TATACT",    
#     "TGGCAGG",
#     "TTGCA"
# ]

# left = [
#     "GAAATAACATAATTGA",
#     "TGTTAATAATAAGTAAT",
#     "TTATATTTTACCCATTT",
#     "AGGTTAATCCTAAAA",
#     "ATTGAAACGATTCAGC",
#     "CTCTGTCATAAAACTGTCATAT",
#     "TTACGCATTTTTAC",
#     "ATTTGTACACAA",
#     "AAGGCATAACCTATCACTGT",
#     "ACGCAAACGTTTTCTT",
# ]

# right = [
#     "TTAATAAAACCTTAAGGTT",
#     "CCTTTTAGGTGCTT",
#     "TACTGTATATAAAAACAGTA",
#     "TAAAATTCATGGTAATTAT",
#     "AATGAGAATGATTATTAT",
#     "TGTTTATATTTTGTTTA",
#     "CATAAGAAAAA",
#     "CATTCATTTG",
# ]

motifs = [
    "ATAATATTCTGAATT",
    "TCCCTATAAGAAAATTA",
    "TAATTGATTGATT",
    "GCTTAAAAAATGAAC",
    "TGCACTAAAATGGTGCAA",
    "AATATGTAACCAAAAGTAA",
    "ACTGAATTTTTATGCAAA",
    "CGGGGATGAG",
    "TTGACA",
    "TATAAT",
]

opt = da.Optimizer(
    library=motifs,
    sequence_length=100,
    strands="double",
)

# # sigma D -35 and -10 motifs
# opt.add_promoter_constraints(
#     upstream="TTGACA",
#     downstream="TATAAT",
#     upstream_pos=(40, 80),
#     spacer_length=(16, 18),
# )

# # sigma S -35 and -10 motifs
# opt.add_promoter_constraints(
#     upstream="TTGACA",
#     downstream="TATACT",
#     upstream_pos=(40, 80),
#     spacer_length=(16, 18),
# )

# # # sigma E -24 and -12 motifs
# opt.add_promoter_constraints(
#     upstream="TGGCAGG",
#     downstream="TTGCA",
#     upstream_pos=(40, 80),
#     spacer_length=(3, 5),
# )

# opt.add_side_biases(
#     left=left, 
#     right=right
# )

best = opt.optimal()
print(f"Optimal solution, score {best.nb_motifs}")
print(best)


# print("List of all solutions")
# for solution in opt.solutions():
#     print(solution)
    
# print("Return solutions in a diversity-driven order.")
# for solution in opt.solutions_diverse():
#     print(solution)