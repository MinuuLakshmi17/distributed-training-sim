"""Tiny deterministic linear regression model used by the simulator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

Vector = list[float]

@dataclass(frozen=True)
class Example:
    x: Vector
    y: float

@dataclass
class LinearModel:
    weights: Vector
    bias: float = 0.0

    def predict(self, x: Sequence[float]) -> float:
        return sum(w * v for w, v in zip(self.weights, x)) + self.bias

    def loss(self, data: Sequence[Example]) -> float:
        if not data:
            return 0.0
        return sum((self.predict(e.x) - e.y) ** 2 / 2 for e in data) / len(data)

    def gradient(self, data: Sequence[Example]) -> Vector:
        """Mean gradient of half squared error, including bias as last element."""
        if not data:
            raise ValueError("gradient requires at least one example")
        g = [0.0] * (len(self.weights) + 1)
        for e in data:
            if len(e.x) != len(self.weights):
                raise ValueError("feature dimension mismatch")
            err = self.predict(e.x) - e.y
            for i, v in enumerate(e.x):
                g[i] += err * v / len(data)
            g[-1] += err / len(data)
        return g

    def apply_gradient(self, gradient: Sequence[float], learning_rate: float) -> None:
        if len(gradient) != len(self.weights) + 1:
            raise ValueError("gradient dimension mismatch")
        self.weights = [w - learning_rate * g for w, g in zip(self.weights, gradient[:-1])]
        self.bias -= learning_rate * gradient[-1]

    def parameters(self) -> Vector:
        return [*self.weights, self.bias]

    @classmethod
    def from_parameters(cls, params: Sequence[float]) -> "LinearModel":
        if len(params) < 2:
            raise ValueError("need at least one weight and a bias")
        return cls(list(params[:-1]), float(params[-1]))
