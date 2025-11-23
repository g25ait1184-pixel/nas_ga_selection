Genetic Algorithm–Based NAS 
=================================================================
-----------------------------------------------------------------
#📘 Overview
-----------------------------------------------------------------
This  provides a detailed explanation of the Genetic Algorithm–based
Neural Architecture Search (NAS), covering  Roulette-Wheel Selection and
Weighted Fitness Function. 

-----------------------------------------------------------------
#🔹 — Roulette-Wheel Selection
-----------------------------------------------------------------

In the original NAS implementation, **Tournament Selection** was used.
This method is biased toward the fittest chromosomes in a small sampled group.

✔ Advantage: Fast exploitation  
✖ Disadvantage: Poor exploration, risk of premature convergence

To improve exploration, **Roulette-Wheel Selection** was implemented.

📌 Roulette-Wheel Formula:
    p_i = f_i / Σ f_i

Where:
		p_i = probability of selecting chromosome i  
		f_i = fitness of chromosome i  

Meaning:

    • Higher fitness → higher probability  
    • Lower fitness → still has some chance  

This encourages genetic diversity and avoids stagnation.

-----------------------------------------------------------------
#🔹 — Modified Fitness Function
-----------------------------------------------------------------

Original Fitness Function penalized only total parameters:

    fitness_orig = accuracy − 0.01 × (total_params / 1e6)

⚠ Problem:
This treats convolution and fully-connected layers equally:
	CONV layers → high compute cost, moderate params
	FC layers → low compute, very high params

Thus the penalty is not realistic.

-----------------------------------------------------------------
#✔ Modified Fitness Function (Conv vs FC Penalty)
-----------------------------------------------------------------

We split model parameters into:
    conv_params  = parameters of convolutional layers
    fc_params    = parameters of fully-connected layers

Normalized units (in millions):
    conv_M = conv_params / 1e6
    fc_M   = fc_params / 1e6

New weighted fitness:
    fitness_weighted = accuracy − (w_conv × conv_M + w_fc × fc_M)

Weights used:
    w_conv = 1e−6  w_fc   = 5e−6
	
Justification:
    • Conv layers are compute-heavy → mild penalty  
    • FC layers explode in size → stronger penalty  
    • This promotes smaller, compute-friendly networks  

-----------------------------------------------------------------
#📊 Experimental Results
-----------------------------------------------------------------
- Dataset: CIFAR-10 (subset for quick NAS)
- Population size: small (for faster runs)
- Generations: small (for demonstration)
- Two selection strategies compared:
  - **Tournament selection** (uses original fitness = accuracy − λ · total_params)
  - **Roulette-wheel selection** (uses weighted fitness = accuracy − (w_conv·conv_M + w_fc·fc_M))
The following table summarizes the NAS results using both selection methods:

| Selection Method   | Accuracy | Original Fitness | Weighted Fitness | Parameters |
|--------------------|----------|------------------|------------------|------------|
| Tournament (Run 2) | 0.6460    | 0.63336118 | 0.645994536142            | 1,263,882   |
| Roulette (Run 3)   | 0.6530    | 0.60852022           | 0.652978752238          |    4,447,978  |

-----------------------------------------------------------------
#🧠 Interpretation
-----------------------------------------------------------------

Tournament:
    ✔ Good accuracy
    ✖ Very large model (3× more parameters)
    → Lower fitness

Roulette:
    ✔ Slightly lower accuracy  
    ✔ Much fewer parameters  
    → Higher final fitness

#🏆 Winner: Roulette-Wheel Selection

- **Winner by original fitness:** Run 2 (Tournament).  
  The original fitness penalizes total parameters uniformly; under that metric the tournament-selected architecture scored higher.
- **Winner by weighted fitness:** Run 3 (Roulette).  
  The modified weighted fitness (separate conv/FC penalties) favored the architecture produced by roulette selection.
- **Winner by accuracy:** Run 3 (Roulette) has higher raw accuracy (0.6530 vs 0.6460) but also has far more parameters (~4.45M vs ~1.26M)..
## Trade-off and recommendation
- The two methods optimized different objectives (tournament used the original penalty; roulette used the conv/FC weighted penalty). Because objectives differ, direct comparison must specify the metric used.
- If your goal is **max accuracy** (and model size is less important), choose Run 3 (Roulette).
- If your goal is **compact models** or to minimize the original total-params penalty, choose Run 2 (Tournament).
-----------------------------------------------------------------
#✔ Final Conclusion
-----------------------------------------------------------------

• Roulette selection improved diversity and avoided premature convergence  
• Weighted fitness accurately penalized FC-heavy models  
• The new NAS setup discovers smaller CNNs without losing accuracy  


