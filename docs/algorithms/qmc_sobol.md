# QMC Sobol'

Scrambled Sobol' low-discrepancy sequence in 1D. **Only evaluated at power-of-two sample sizes**
for optimal balance properties.

## How It Works

Sobol' sequences are \((t,s)\)-sequences in base 2 that minimise the star discrepancy
\(D_N^* = O(N^{-1}\log N)\). The Koksma–Hlawka inequality bounds quadrature error by
\(D_N^* \cdot V(f)\) where \(V(f)\) is the Hardy–Krause variation.

## Pros

* Better space-filling than i.i.d. uniform (avoids gaps and clusters).
* Deterministic, reproducible.
* \(O(N^{-1}\log N)\) convergence for functions of bounded variation,
  vs \(O(N^{-1/2})\) for Monte Carlo.

## Cons

* Still non-adaptive — does not use information about \(f\).
* Requires \(N\) to be a power of 2 for full balance properties.
* Limited benefit for smooth functions where Chebyshev already excels.

## Reference

Dick & Pillichshammer, *Digital Nets and Sequences*, Cambridge, 2010.
