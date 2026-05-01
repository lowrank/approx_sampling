# Adversarial Sampling

## How It Works

A min-max game between a generator and an approximator. The generator outputs a piecewise-constant sampling density over the domain, parameterised by bin probabilities. It is trained via REINFORCE to maximise the approximator's mean squared error. The approximator is simultaneously trained to minimise MSE on points drawn from the generator's distribution. The two networks alternate updates, driving the generator to focus mass on the hardest-to-approximate regions.

## Pros

- Directly optimises for worst-case error, providing a principled min-max objective.
- Piecewise-constant parameterisation is simple and interpretable.
- Generator and approximator can be trained jointly end-to-end.
- Naturally handles multi-scale error landscapes.

## Cons

- REINFORCE gradient estimates have high variance, leading to unstable training.
- Min-max games are prone to non-convergence and mode collapse.
- Piecewise-constant density cannot represent smooth sampling distributions.
- Requires careful balancing of generator and approximator learning rates.

## Reference

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256.
