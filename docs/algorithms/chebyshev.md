# Chebyshev Nodes

\(x_i = \frac12\bigl(1 + \cos\frac{(2i-1)\pi}{2N}\bigr)\), mapped from \([-1,1]\) to \([0,1]\).

## How It Works

Chebyshev nodes cluster near the boundaries \(x=0\) and \(x=1\). This minimises the
Runge phenomenon for polynomial interpolation and often yields superior \(L^2\)
approximation for functions that are analytic in a neighbourhood of the Bernstein
ellipse.

## Pros

* Near-optimal for smooth, analytic functions.
* Better boundary resolution than uniform grids.
* Zero overhead — precomputed once.

## Cons

* Offers no benefit for interior-localised features.
* Clustering near boundaries is wasted if the function is flat there.
* No adaptivity to the specific function.

## Reference

Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2019.
