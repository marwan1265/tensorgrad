import numpy as np
from typing import List, Sequence

from tensorGrad.engine import Tensor, randn, zeros


class Module:
    """Base class mirroring PyTorch's nn.Module – tracks parameters & offers zero_grad()."""

    def parameters(self) -> List[Tensor]:
        """Return a list of all learnable tensors."""
        return []

    # Convenience to clear grads recursively
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)


class Linear(Module):
    """Fully-connected layer: y = x @ W + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming-like initialisation (scaled Gaussian)
        scale = 1.0 / np.sqrt(in_features)
        self.W = randn(in_features, out_features) * scale
        self.b = zeros(out_features) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.b is not None else out

    def parameters(self) -> List[Tensor]:
        return [self.W] + ([self.b] if self.b is not None else [])


class ReLU(Module):
    """Functional ReLU as a module for Sequential nets."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    """A sequence of layers applied in order."""

    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MLP(Module):
    """Convenience multi-layer perceptron (Linear-ReLU blocks)."""

    def __init__(self, in_features: int, hidden: Sequence[int], out_features: int):
        layers: List[Module] = []
        sizes = [in_features] + list(hidden) + [out_features]
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # no activation after last layer
                layers.append(ReLU())
        self.net = Sequential(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        return self.net(x)

    def parameters(self) -> List[Tensor]:
        return self.net.parameters()


# Public re-exports for * import convenience
__all__ = [
    "Module",
    "Linear",
    "ReLU",
    "Sequential",
    "MLP",
]
    