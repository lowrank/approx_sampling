# Policy Network Sampling

## How It Works

A policy network takes as input a histogram of the current residual error over the domain bins and outputs a categorical distribution over bins. Points are sampled from the bin distribution and the approximator is trained on them. The policy is updated via REINFORCE, using the improvement in approximation error as the reward signal. Over successive iterations, the policy learns to allocate samples to bins where they most reduce the overall error.

## Pros

- State-aware — the policy conditions on the current error landscape.
- Learns a sampling strategy rather than a fixed distribution.
- Bin-level output is compact and interpretable.
- Reward signal (error improvement) is directly aligned with the objective.

## Cons

- REINFORCE gradient has high variance; actor-critic variants add complexity.
- Histogram representation loses spatial resolution within bins.
- Credit assignment is difficult when many bins contribute to error reduction.
- Training the policy requires many episodes, which is expensive with full retraining.

## Reference

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*.
