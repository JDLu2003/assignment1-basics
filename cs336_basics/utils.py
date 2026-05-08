from typing import Any

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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

class Swiglu(nn.Module):
    """
    传统 FFN：
    x
     └─W1─ReLU─W2

    SwiGLU：
    x
     ├─W1─SiLU──┐
     └─W3───────⊙──W2
    """
    def __init__(
            self, d_model: int, 
            d_ff: int | None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super().__init__()

        if d_ff == None:
            d_ff_cal: int = int((8.0 / 3.0) * d_model)
            self.d_ff:int = 64 * ((d_ff_cal + 63) // 64)
        else:
            self.d_ff = d_ff

        factory_kwargs: dict[str, Any] = {'device': device, 'dtype': dtype}

        self.w1: Linear = Linear(self.d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(self.d_ff, d_model, **factory_kwargs)
        self.w2: Linear = Linear(d_model, self.d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x1: torch.Tensor = self.w1(x)
        # silu_out: torch.Tensor = x1 * torch.sigmoid(x1)
        # x3: torch.Tensor = self.w3(x)
        # hidden: torch.Tensor = silu_out * x3
        # output: torch.Tensor = self.w2(hidden)
        # return output
        x1: torch.Tensor = torch.einsum('...d, fd -> ...f', x, self.w1.weight)
        silu_out: torch.Tensor = x1 * torch.sigmoid(x1)
        x3: torch.Tensor = torch.einsum('...d, fd -> ...f', x, self.w3.weight)
        hidden: torch.Tensor = torch.einsum('...f, ...f -> ...f', silu_out, x3)
        output: torch.Tensor = torch.einsum('...f, df -> ...d', hidden, self.w2.weight)
        return output

