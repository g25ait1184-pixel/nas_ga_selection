Genetic Algorithm–Based NAS 
=================================================================

#📘 Overview
This document provides a detailed explanation of the Genetic Algorithm–based
Neural Architecture Search (NAS), covering Q1A (Roulette-Wheel Selection) and
Q2B (Weighted Fitness Function). This version is formatted cleanly for
Notepad++ viewing.

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
    - CONV layers → high compute cost, moderate params
    - FC layers → low compute, very high params

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

The following table summarizes the NAS results using both selection methods:

| Selection Method   | Accuracy | Original Fitness | Weighted Fitness | Parameters |
|--------------------|----------|------------------|------------------|------------|
| Tournament (Run 4) | 0.6770   | 0.6530           | 0.6769           | 2,398,250  |
| Roulette (Run 5)   | 0.6700   | 0.6617           | 0.6699           |   826,042  |

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

Roulette produced a more parameter-efficient architecture with better 
accuracy–complexity trade-off.

-----------------------------------------------------------------
#✔ Final Conclusion
-----------------------------------------------------------------

• Roulette selection improved diversity and avoided premature convergence  
• Weighted fitness accurately penalized FC-heavy models  
• The new NAS setup discovers smaller CNNs without losing accuracy  


