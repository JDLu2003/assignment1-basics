from typing import Any

import torch
import torch.nn as nn
import einx

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {'device': device, 'dtype': dtype}
        weight_tensor: torch.Tensor = torch.empty(out_features, in_features, **factory_kwargs)
        self.weight: torch.Tensor = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i, oi -> ...o", x, self.weight)

class Embedding(nn.Module):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {'device': device, 'dtype': dtype}
        weight_tensor: torch.Tensor = torch.empty(num_embedding, embedding_dim, **factory_kwargs)
        self.weight: torch.Tensor = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
    


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int, # hidden dimension
        eps: float = 1e-5, # numerical stability constant
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        factory_kwargs: dict[str, Any] = {'device': device, 'dtype': dtype}
        self.weight: torch.Tensor = nn.Parameter(torch.empty(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normalized_fp32 = torch.rsqrt(variance + self.eps) * x_fp32
        y = torch.einsum("... d, d -> ... d", x_normalized_fp32, self.weight)
        return y.to(original_dtype)

