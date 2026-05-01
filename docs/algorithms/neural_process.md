# Neural Process Sampling

## How It Works

An encoder–decoder architecture processes a context set of {(x, f(x))} pairs. The encoder aggregates context points into a latent representation via a permutation-invariant function (e.g. mean pooling). The decoder conditions on this representation to output a sampling distribution over the domain — typically bin probabilities or mixture parameters. Because the model is amortised, a single forward pass produces a sampling strategy for any context set without further optimisation.

## Pros

- Amortised inference — no per-function training or optimisation needed at deployment.
- Permutation-invariant encoder handles context sets of arbitrary size.
- Learns to generalise sampling strategies across a family of functions.
- Fast inference: one forward pass produces the full sampling distribution.

## Cons

- Requires a distribution of training functions similar to the test distribution.
- May underfit if the function family is too diverse relative to model capacity.
- Latent bottleneck can lose fine-grained spatial information.
- Evaluating the decoupled training pipeline (NP + approximator) adds implementation complexity.

## Reference

Garnelo, M., Rosenbaum, D., Maddison, C. J., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D. J., & Eslami, S. M. A. (2018). Conditional Neural Processes. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
