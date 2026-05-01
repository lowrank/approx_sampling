"""
Neural network architectures for 1D function approximation.

Provides a model registry so that different architectures can be swapped
in via the experiment runner API.  Two architectures are included:

* ``MLP`` — standard fully-connected network with tanh activations.
* ``MMNN`` — Matrix-Matrix Neural Network with frozen random expansion
  layers and trainable low-rank bottleneck layers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Standard MLP
# ======================================================================


class MLP(nn.Module):
    """Fully-connected network with tanh activations.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden layer widths, e.g. ``[32, 32, 32, 32]`` for four layers.
    activation : callable, optional
        Activation function.  Default: ``torch.tanh``.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]
        if activation is None:
            activation = nn.Tanh()

        layers: List[nn.Module] = []
        prev_dim = 1
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(activation)
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self._net(x)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            out = self.forward(t).squeeze(-1).cpu().numpy()
        self.train()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# MMNN — Matrix-Matrix Neural Network
# ======================================================================


class MMNN(nn.Module):
    """Feedforward network with frozen random-expansion layers.

    Layers fc1 and fc3 are frozen random projections (e.g. Gaussian)
    that expand to/from a high-dimensional hidden space.  Layers fc2
    and fc4 are trainable low-rank bottleneck layers.  The activation
    is ``sin``, following random-feature / Fourier-feature conventions.

    Parameters
    ----------
    input_size : int
        Input dimension (default 1).
    rank : int
        Bottleneck dimension between the two random-expansion layers.
    hidden_size : int
        Dimension of the random-expansion space.
    seed : int
        Seed for reproducibility of the frozen random weights.
    """

    def __init__(
        self,
        input_size: int = 1,
        rank: int = 10,
        hidden_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank

        g = torch.Generator()
        g.manual_seed(seed)

        # Frozen expansion: input → hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.normal_(self.fc1.weight, std=1.0, generator=g)
        nn.init.zeros_(self.fc1.bias)
        self.fc1.weight.requires_grad_(False)
        self.fc1.bias.requires_grad_(False)

        # Trainable bottleneck: hidden → rank
        self.fc2 = nn.Linear(hidden_size, rank)
        self.fc2.weight.requires_grad_(True)
        self.fc2.bias.requires_grad_(True)

        # Frozen expansion: rank → hidden
        self.fc3 = nn.Linear(rank, hidden_size)
        nn.init.normal_(self.fc3.weight, std=1.0, generator=g)
        nn.init.zeros_(self.fc3.bias)
        self.fc3.weight.requires_grad_(False)
        self.fc3.bias.requires_grad_(False)

        # Trainable output: hidden → 1
        self.fc4 = nn.Linear(hidden_size, 1)
        self.fc4.weight.requires_grad_(True)
        self.fc4.bias.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.fc1(x)
        h = torch.sin(h)
        h = self.fc2(h)
        h = self.fc3(h)
        h = torch.sin(h)
        return self.fc4(h)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            out = self.forward(t).squeeze(-1).cpu().numpy()
        self.train()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Piecewise-constant generator (for adversarial / IS / diffusion)
# ======================================================================


class PiecewiseConstantGenerator(nn.Module):
    """Learnable piecewise-constant probability density on [0, 1].

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins partitioning [0, 1].
    """

    def __init__(self, n_bins: int = 32) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.logits = nn.Parameter(torch.zeros(n_bins))
        self._bin_width = 1.0 / n_bins

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    @property
    def density(self) -> torch.Tensor:
        return self.probs * self.n_bins

    def sample(self, n_samples: int) -> torch.Tensor:
        probs = self.probs
        bins = torch.multinomial(probs, n_samples, replacement=True)
        offsets = torch.rand(n_samples, device=bins.device)
        samples = bins.float() * self._bin_width + offsets * self._bin_width
        return samples.detach()

    def sample_with_log_prob(
        self, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.probs
        bins = torch.multinomial(probs, n_samples, replacement=True)
        offsets = torch.rand(n_samples, device=bins.device)
        samples = bins.float() * self._bin_width + offsets * self._bin_width
        log_prob = torch.log(probs[bins]) + np.log(self.n_bins)
        return samples.detach(), log_prob

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.clamp(
            (x / self._bin_width).long(), 0, self.n_bins - 1
        )
        log_p = torch.log(self.probs[bin_idx]) + np.log(self.n_bins)
        return log_p
