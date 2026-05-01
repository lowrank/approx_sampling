# QMC Halton

Scrambled Halton low-discrepancy sequence in 1D. Unlike Sobol', no power-of-two constraint.

## How It Works

The Halton sequence uses co-prime bases (here base 2 for 1D). Points fill \([0,1]\)
with low discrepancy, avoiding the gaps that random sampling produces.

## Pros / Cons

Same as Sobol' but without the power-of-two constraint, making it more flexible for
arbitrary \(N\).

## Reference

Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM, 1992.
