# Downstream Tasks

The sampling algorithms are decoupled from the problem being solved via the
`DownstreamTask` abstract interface.

## Interface

```python
class DownstreamTask(ABC):
    def pointwise_loss(self, model, x) -> Tensor    # loss at x for training
    def reference(self, x) -> np.ndarray             # ground truth for L² error
    def predict(self, model, x) -> Tensor            # model output at x
    def compute_l2_error(self, model, grid) -> float # ||predict - reference||_L²
    def boundary_loss(self, model) -> Tensor         # optional BC penalty
```

## Built-in Tasks

### Function Approximation
Fit \(u_\theta(x) \approx f(x)\).
Pointwise loss: \(|u_\theta(x) - f(x)|^2\).

### Poisson PINN
Solve \(-u'' = f\) on \([0,1]\) with \(u(0)=u(1)=0\).
Hard boundary encoding: \(u_\theta(x) = x(1-x) \cdot \mathrm{NN}(x)\).
Pointwise loss: \(|-u_\theta''(x) - f(x)|^2\).
Reference solution computed by linear FEM on 2000 elements.

### Deep Ritz
Minimise Dirichlet energy \(E(u) = \int_0^1 \bigl[\frac12|u'|^2 - f u\bigr]\,dx\).
Same hard boundary encoding as PINN.
Pointwise loss: \(\frac12|u'(x)|^2 - f(x)u(x)\) (energy density, not squared).

## Adding a New Task

```python
from tasks.base import DownstreamTask, register_task

class MyTask(DownstreamTask):
    def __init__(self, ...):
        super().__init__("my_task")
        ...

    def pointwise_loss(self, model, x):
        ...

    def reference(self, x):
        ...

    def predict(self, model, x):
        ...

register_task("my_task", MyTask)
```

Then use `python main.py --task my_task`.
