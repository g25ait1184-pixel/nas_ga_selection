# nas-ga-basic
Basic demo of Neural Architecture Search (NAS) using Genetic Algorithm (GA)
summarizes the implementation and analysis for Neural Architecture Search (NAS) using a Genetic Algorithm (GA), covering Q1A (Roulette-Wheel Selection) and Q2B (Weighted Fitness Function).
# Implementing Roulette-Wheel Selection
🔹 Original Method: Tournament Selection

The original GA used tournament selection, where:

A small subset of chromosomes is sampled.

The fittest in that subset is selected.

This promotes exploitation but limits exploration, and may cause premature convergence.
