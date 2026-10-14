from __future__ import annotations

import torch
from torch import nn

from .baselines import MultiBranchGRU
from .dasr import DASR
from .encoders import CausalStreamEncoder, CausalTCN


class TCNStreamEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernels: tuple[int, int],
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.projection = CausalStreamEncoder(input_dim, hidden_dim, kernels, dropout)
        self.tcn = CausalTCN(hidden_dim, layers, dropout)
        self.context_length = self.projection.context_length + self.tcn.context_length

    def forward(self, values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        return self.tcn(self.projection(values, available))


class TCNDASR(DASR):
    def __init__(self, stream_dims: dict[str, int], tcn_layers: int, **kwargs):
        kernels = tuple(kwargs.get("encoder_kernels", (5, 3)))
        hidden_dim = int(kwargs["hidden_dim"])
        dropout = float(kwargs.get("dropout", 0.1))
        kwargs["num_layers"] = 1
        super().__init__(stream_dims=stream_dims, **kwargs)
        self.encoders = nn.ModuleDict(
            {
                name: TCNStreamEncoder(
                    dimension, hidden_dim, kernels, tcn_layers, dropout
                )
                for name, dimension in stream_dims.items()
            }
        )


class TCNBase(MultiBranchGRU):
    def __init__(self, stream_dims: dict[str, int], tcn_layers: int, **kwargs):
        kernels = tuple(kwargs.get("encoder_kernels", (5, 3)))
        hidden_dim = int(kwargs["hidden_dim"])
        dropout = float(kwargs.get("dropout", 0.1))
        kwargs["num_layers"] = 1
        super().__init__(stream_dims=stream_dims, **kwargs)
        self.encoders = nn.ModuleDict(
            {
                name: TCNStreamEncoder(
                    dimension, hidden_dim, kernels, tcn_layers, dropout
                )
                for name, dimension in stream_dims.items()
            }
        )
