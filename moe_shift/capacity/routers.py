"""Parameter-matched routers.

Linear and cosine routers share the SAME E x d weight matrix and a scalar temperature, so the
linear/cosine factor changes only the similarity geometry -- never the parameter count
(plan: "Linear and cosine routers must share the same E x d weight matrix and temperature.
Cosine routing normalizes the input and router rows; it does not receive an additional learned
projection.").

    linear:  r_e(x) = <w_e, x> / tau
    cosine:  r_e(x) = <w_e/||w_e||, x/||x||> / tau
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    GEOMETRIES = ("linear", "cosine")

    def __init__(self, dim: int, n_experts: int, geometry: str = "cosine", temperature: float = 0.07):
        super().__init__()
        if geometry not in self.GEOMETRIES:
            raise ValueError(f"geometry must be one of {self.GEOMETRIES}, got {geometry!r}")
        self.dim, self.n_experts, self.geometry = dim, n_experts, geometry
        # identical parameterisation for both geometries -> parameter-matched by construction
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=dim ** -0.5)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, dim] -> logits [M, n_experts]."""
        if self.geometry == "linear":
            logits = x @ self.weight.t()
        else:
            logits = F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()
        return (logits / self.temperature).clamp(-30.0, 30.0)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_experts={self.n_experts}, geometry={self.geometry}"
