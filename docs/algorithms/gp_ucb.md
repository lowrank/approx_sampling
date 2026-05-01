# Gaussian Process UCB Sampling

## How It Works

A Gaussian Process (GP) is fitted to the residual error surface of the approximator using observed (x, error(x)) pairs. The GP provides a posterior mean and variance at every point in the domain. An Upper Confidence Bound (UCB) acquisition function, combining predicted error and uncertainty, is maximised to select the next sampling point. The selected point is evaluated, added to the training set, and the GP is updated.

## Pros

- Principled exploration–exploitation trade-off via the UCB acquisition function.
- GP provides well-calibrated uncertainty estimates with closed-form posteriors.
- Naturally sequential — each new point optimally improves the error surface model.
- Works with small sample budgets due to GP sample efficiency.

## Cons

- GP inference scales cubically with the number of observed points.
- Kernel choice and UCB hyperparameter strongly affect behaviour.
- GP assumes smooth error surface; may fail for discontinuous errors.
- Sequential point selection limits parallelisation.

## Reference

Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. *Proceedings of the 27th International Conference on Machine Learning (ICML)*.
