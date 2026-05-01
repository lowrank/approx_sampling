# Importance Sampling

## How It Works

A proposal distribution generates candidate points, which are weighted by the ratio of a target density to the proposal density. The approximator is trained to minimise an importance-weighted MSE loss. The proposal itself is trained via REINFORCE to reduce the variance of the weighted loss. Self-normalised importance weights are used to handle unnormalised target densities. The proposal and approximator are optimised jointly.

## Pros

- Unbiased estimator of the target loss, correcting for mismatch between proposal and target.
- Self-normalised weights are stable and avoid extreme weight magnitudes.
- Joint optimisation of proposal and approximator improves sample efficiency.
- Applicable when the optimal sampling distribution is known up to a normalising constant.

## Cons

- REINFORCE training of the proposal suffers from high variance.
- Importance weights can degenerate (effective sample size collapse) if proposal is poor.
- Self-normalisation introduces bias in finite samples.
- Requires careful learning rate scheduling for stable joint training.

## Reference

Katharopoulos, A., & Fleuret, F. (2018). Not all samples are created equal: Deep learning with importance sampling. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
