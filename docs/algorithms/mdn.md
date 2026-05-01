# Mixture Density Network Sampling

## How It Works

A mixture density network (MDN) outputs the parameters of a Gaussian mixture model over the domain. Component selection is performed via the Gumbel-Softmax reparameterisation, which provides a continuous relaxation of categorical sampling. The temperature is annealed from high to low during training, gradually sharpening the selection toward hard assignments. The MDN is trained to maximise the expected error of the approximator under the sampled points.

## Pros

- Gaussian mixtures can represent complex, multi-modal sampling densities.
- Gumbel-Softmax enables differentiable component selection and low-variance gradients.
- Temperature annealing balances exploration (high temperature) with exploitation (low temperature).
- Well-studied architecture with stable training dynamics.

## Cons

- Fixed number of mixture components must be specified a priori.
- Gumbel-Softmax introduces bias that vanishes only as temperature approaches zero.
- Mixture model likelihood can have pathological local optima.
- Training is sensitive to the annealing schedule.

## Reference

Bishop, C. M. (1994). Mixture Density Networks. *Technical Report NCRG/94/004*, Aston University.
