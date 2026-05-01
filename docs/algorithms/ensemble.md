# Deep Ensemble Sampling

## How It Works

An ensemble of M independent MLP approximators is trained on the same dataset (with different random initialisations). At sampling time, the predictive variance across the ensemble is computed for each candidate point. Points with high variance — where the ensemble members disagree most — are selected for labelling and added to the training set. The ensemble is then retrained, and the process repeats.

## Pros

- Simple to implement — only requires training multiple standard networks.
- Predictive variance is a natural, interpretable uncertainty metric.
- Captures both epistemic uncertainty and approximation difficulty.
- Ensemble members can be trained in parallel.

## Cons

- Training M networks multiplies computational cost by M.
- Diversity comes only from random initialisation; no explicit diversity regularisation.
- Variance estimates may be poorly calibrated.
- Variance can be high in smooth, well-approximated regions due to network noise.

## Reference

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
