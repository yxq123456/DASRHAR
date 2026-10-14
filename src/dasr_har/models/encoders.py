from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        values = F.pad(values, (self.kernel_size - 1, 0))
        values = self.conv(values).transpose(1, 2)
        return self.dropout(F.gelu(self.norm(values)))


class CausalStreamEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernels: tuple[int, int] = (5, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = sum(kernel - 1 for kernel in kernels)
        self.first = CausalConvLayer(input_dim + 1, hidden_dim, kernels[0], dropout)
        self.second = CausalConvLayer(hidden_dim, hidden_dim, kernels[1], dropout)

    def forward(self, values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        mask = available.to(values.dtype).unsqueeze(-1)
        encoded = torch.cat((values * mask, mask), dim=-1)
        return self.second(self.first(encoded))


class RecurrentStack(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        states: torch.Tensor,
        value: torch.Tensor,
        top_predecessor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        layer_input = value
        for index, cell in enumerate(self.cells):
            predecessor = states[index]
            if index == self.num_layers - 1 and top_predecessor is not None:
                predecessor = top_predecessor
            updated = cell(layer_input, predecessor)
            outputs.append(updated)
            layer_input = self.dropout(updated) if index + 1 < self.num_layers else updated
        return torch.stack(outputs)


class CausalTCN(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.context_length = 2 * (2**layers - 1)
        self.blocks = nn.ModuleList()
        for index in range(layers):
            dilation = 2**index
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=3,
                            dilation=dilation,
                        ),
                        "norm": nn.LayerNorm(hidden_dim),
                        "dropout": nn.Dropout(dropout),
                    }
                )
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = values
        for index, block in enumerate(self.blocks):
            dilation = 2**index
            convolved = block["conv"](
                F.pad(result.transpose(1, 2), (2 * dilation, 0))
            ).transpose(1, 2)
            result = result + block["dropout"](F.gelu(block["norm"](convolved)))
        return result

