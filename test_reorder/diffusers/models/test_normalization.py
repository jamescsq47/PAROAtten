# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_torch_npu_available, is_torch_version
from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX and OmniGen for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class SD35AdaLayerNormZeroX(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )
        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_eps: float, norm_elementwise_affine: bool):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])

        return x, gate_msa, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class CogView3PlusAdaLayerNormZeroTextImage(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, dim: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)
        normed_x = self.norm_x(x)
        normed_context = self.norm_c(context)
        x = normed_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        context = normed_context * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        r"""
        LayerNorm with the bias parameter.

        Args:
            dim (`int`): Dimensionality to use for the parameters.
            eps (`float`, defaults to 1e-5): Epsilon factor.
            elementwise_affine (`bool`, defaults to `True`):
                Boolean flag to denote if affine transformation should be applied.
            bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        """

        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    r"""
    RMS Norm as introduced in https://arxiv.org/abs/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        if is_torch_npu_available():
            import torch_npu

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = hidden_states * self.weight
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias
            else:
                hidden_states = hidden_states.to(input_dtype)

        return hidden_states


# TODO: (Dhruv) This can be replaced with regular RMSNorm in Mochi once `_keep_in_fp32_modules` is supported
# for sharded checkpoints, see: https://github.com/huggingface/diffusers/issues/10013
class MochiRMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class GlobalResponseNorm(nn.Module):
    r"""
    Global response normalization as introduced in ConvNeXt-v2 (https://arxiv.org/abs/2301.00808).

    Args:
        dim (`int`): Number of dimensions to use for the `gamma` and `beta`.
    """

    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class LpNorm(nn.Module):
    def __init__(self, p: int = 2, dim: int = -1, eps: float = 1e-12):
        super().__init__()

        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden_states, p=self.p, dim=self.dim, eps=self.eps)


def get_normalization(
    norm_type: str = "batch_norm",
    num_features: Optional[int] = None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> nn.Module:
    if norm_type == "rms_norm":
        norm = RMSNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    elif norm_type == "layer_norm":
        norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    elif norm_type == "batch_norm":
        norm = nn.BatchNorm2d(num_features, eps=eps, affine=elementwise_affine)
    else:
        raise ValueError(f"{norm_type=} is not supported.")
    return norm


def reorder_tensor(x: torch.Tensor, pattern: str, H: int, W: int, F: int) -> torch.Tensor:
    """
    根据指定的模式对输入张量进行重排序
    
    Args:
        x: 输入张量，形状为 [B, T, C]，其中 T = H * W * F
        pattern: 重排序模式，如 "FWH", "FHW", "WFH", "WHF", "HFW", "HWF"
        H: 高度
        W: 宽度
        F: 帧数
    """
    B, T, C = x.shape
    assert T == H * W * F, f"T ({T}) must equal H * W * F ({H * W * F})"
    
    # 将输入重塑为 [B, F, H, W, C]（默认顺序）
    x = x.view(B, F, H, W, C)
    
    # 根据模式重排序
    if pattern == "FWH":
        # 保持默认顺序 [F, W, H]
        x = x.permute(0, 1, 3, 2, 4)
    elif pattern == "FHW":
        # 保持默认顺序 [F, H, W]
        pass
    elif pattern == "WFH":
        # [W, F, H]
        x = x.permute(0, 3, 1, 2, 4)
    elif pattern == "WHF":
        # [W, H, F]
        x = x.permute(0, 3, 2, 1, 4)
    elif pattern == "HFW":
        # [H, F, W]
        x = x.permute(0, 2, 1, 3, 4)
    elif pattern == "HWF":
        # [H, W, F]
        x = x.permute(0, 2, 3, 1, 4)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")
    
    # 重新展平为 [B, T, C]
    return x.reshape(B, T, C)

def inv_reorder_tensor(x: torch.Tensor, pattern: str, H: int, W: int, F: int) -> torch.Tensor:
    """
    根据指定的模式对重排序后的张量进行逆操作
    
    Args:
        x: 重排序后的张量，形状为 [B, T, C]，其中 T = H * W * F
        pattern: 重排序模式，如 "FWH", "FHW", "WFH", "WHF", "HFW", "HWF"
        H: 高度
        W: 宽度
        F: 帧数
    """
    B, T, C = x.shape
    assert T == H * W * F, f"T ({T}) must equal H * W * F ({H * W * F})"
    
    # 将输入重塑为 [B, F, H, W, C]（默认顺序）
    x = x.view(B, F, H, W, C)
    
    # 根据模式进行逆重排序
    if pattern == "FWH":
        # 从 [F, W, H] 回到 [F, H, W]
        x = x.permute(0, 1, 3, 2, 4)
    elif pattern == "FHW":
        # 保持默认顺序 [F, H, W]
        pass
    elif pattern == "WFH":
        # 从 [W, F, H] 回到 [F, H, W]
        x = x.permute(0, 2, 3, 1, 4)
    elif pattern == "WHF":
        # 从 [W, H, F] 回到 [F, H, W]
        x = x.permute(0, 3, 2, 1, 4)
    elif pattern == "HFW":
        # 从 [H, F, W] 回到 [F, H, W]
        x = x.permute(0, 2, 1, 3, 4)
    elif pattern == "HWF":
        # 从 [H, W, F] 回到 [F, H, W]
        x = x.permute(0, 3, 1, 2, 4)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")
    
    # 重新展平为 [B, T, C]
    return x.reshape(B, T, C)

class ReorderLayerNorm(nn.Module):
    """
    融合了重排序操作的 LayerNorm
    
    Args:
        normalized_shape: 需要归一化的维度
        pattern: 重排序模式，如 "FWH", "FHW", "WFH", "WHF", "HFW", "HWF"
        H: 高度
        W: 宽度
        F: 帧数
        eps: 数值稳定性参数
        elementwise_affine: 是否使用可学习的参数
    """
    def __init__(
        self,
        normalized_shape: int,
        pattern: str,
        H: int,
        W: int,
        F: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.pattern = pattern
        self.H = H
        self.W = W
        self.F = F
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 重排序
        x = reorder_tensor(x, self.pattern, self.H, self.W, self.F)
        
        # 2. 归一化
        x = self.norm(x)
        
        # 3. 逆重排序
        x = inv_reorder_tensor(x, self.pattern, self.H, self.W, self.F)
        
        return x

def test_reorder_layer_norm():
    """测试融合了重排序的 LayerNorm"""
    # 设置测试参数
    B = 2  # batch size
    H = 4  # height
    W = 4  # width
    F = 3  # frames
    C = 8  # channels
    T = H * W * F  # total time steps
    
    # 创建测试数据
    x = torch.randn(B, T, C)
    
    # 测试所有可能的排列模式
    patterns = ["FWH", "FHW", "WFH", "WHF", "HFW", "HWF"]
    for pattern in patterns:
        # 创建融合了重排序的 LayerNorm
        reorder_norm = ReorderLayerNorm(
            normalized_shape=C,
            pattern=pattern,
            H=H,
            W=W,
            F=F,
            eps=1e-5,
            elementwise_affine=True
        )
        
        # 使用融合后的 LayerNorm
        output = reorder_norm(x)
        
        # 验证输出形状
        assert output.shape == (B, T, C), f"Output shape mismatch for pattern {pattern}"
        
        # 验证重排序和归一化的正确性
        # 1. 手动重排序
        reordered = reorder_tensor(x, pattern, H, W, F)
        # 2. 手动归一化
        normed = nn.LayerNorm(C)(reordered)
        # 3. 手动逆重排序
        restored = inv_reorder_tensor(normed, pattern, H, W, F)
        
        # 比较结果
        assert torch.allclose(output, restored), f"Reordering LayerNorm failed for pattern {pattern}"
        
        # 打印测试通过信息
        print(f"Test passed for pattern: {pattern}")

class ReorderCogVideoXLayerNormZero(nn.Module):
    """
    融合了重排序功能的 CogVideoXLayerNormZero
    
    Args:
        conditioning_dim: 条件嵌入的维度
        embedding_dim: 嵌入维度
        pattern: 重排序模式，如 "FWH", "FHW", "WFH", "WHF", "HFW", "HWF"
        H: 高度
        W: 宽度
        F: 帧数
        elementwise_affine: 是否使用可学习的参数
        eps: 数值稳定性参数
        bias: 是否使用偏置
    """
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        pattern: str,
        Height: int,
        Width: int,
        F: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        mode: str = "reorder",
    ) -> None:
        super().__init__()
        
        self.pattern = pattern
        self.Height = Height
        self.Width = Width
        self.F = F
        self.mode = mode
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def reorder_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """重排序张量"""
        B, T, C = x.shape
        assert T == self.Height * self.Width * self.F, f"T ({T}) must equal H * W * F ({self.Height * self.Width * self.F})"
        
        # 将输入重塑为 [B, F, H, W, C]（默认顺序）
        x = x.view(B, self.F, self.Height, self.Width, C)
        
        # 根据模式重排序
        if self.pattern == "FWH":
            # 保持默认顺序 [F, W, H]
            x = x.permute(0, 1, 3, 2, 4)
        elif self.pattern == "FHW":
            # 保持默认顺序 [F, H, W]
            pass
        elif self.pattern == "WFH":
            # [W, F, H]
            x = x.permute(0, 3, 1, 2, 4)
        elif self.pattern == "WHF":
            # [W, H, F]
            x = x.permute(0, 3, 2, 1, 4)
        elif self.pattern == "HFW":
            # [H, F, W]
            x = x.permute(0, 2, 1, 3, 4)
        elif self.pattern == "HWF":
            # [H, W, F]
            x = x.permute(0, 2, 3, 1, 4)
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")
        
        # 重新展平为 [B, T, C]
        return x.reshape(B, T, C)

    def inv_reorder_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """逆重排序张量"""
        B, T, C = x.shape
        assert T == self.Height * self.Width * self.F, f"T ({T}) must equal H * W * F ({self.Height * self.Width * self.F})"
        
        # 将输入重塑为 [B, F, H, W, C]（默认顺序）
        x = x.view(B, self.F, self.Height, self.Width, C)
        
        # 根据模式进行逆重排序
        if self.pattern == "FWH":
            # 从 [F, W, H] 回到 [F, H, W]
            x = x.view(B, self.F, self.Width, self.Height, C)
            x = x.permute(0, 1, 3, 2, 4)
        elif self.pattern == "FHW":
            # 保持默认顺序 [F, H, W]
            x = x.view(B, self.F, self.Height, self.Width, C)
            pass
        elif self.pattern == "WFH":
            # 从 [W, F, H] 回到 [F, H, W]
            x = x.view(B, self.Width, self.F, self.Height, C)
            x = x.permute(0, 2, 3, 1, 4)
        elif self.pattern == "WHF":
            # 从 [W, H, F] 回到 [F, H, W]
            x = x.view(B, self.Width, self.Height, self.F, C)
            x = x.permute(0, 3, 2, 1, 4)
        elif self.pattern == "HFW":
            # 从 [H, F, W] 回到 [F, H, W]
            x = x.view(B, self.Height, self.F, self.Width, C)
            x = x.permute(0, 2, 1, 3, 4)
        elif self.pattern == "HWF":
            # 从 [H, W, F] 回到 [F, H, W]
            x = x.view(B, self.Height, self.Width, self.F, C)
            x = x.permute(0, 3, 1, 2, 4)
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")
        
        # 重新展平为 [B, T, C]
        return x.reshape(B, T, C)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "reorder":
            # 1. 重排序
            hidden_states = self.reorder_tensor(hidden_states)
            reordered_hidden_states = hidden_states
            # 2. 原有的归一化操作
            shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
            hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
            encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
            return reordered_hidden_states, hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]
        elif self.mode == "inv_reorder":
            # 1. 原有的归一化操作
            inv_reordered_hidden_states = self.inv_reorder_tensor(hidden_states)
            shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
            hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
            encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
            hidden_states = self.inv_reorder_tensor(hidden_states)
            return inv_reordered_hidden_states,hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]
            #print("I'm inv_reordering!")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

def test_reorder_cogvideo_x_layer_norm_zero():
    """测试融合了重排序的 CogVideoXLayerNormZero"""
    # 设置测试参数
    B = 2  # batch size
    H = 4  # height
    W = 4  # width
    F = 3  # frames
    C = 8  # channels
    T = H * W * F  # total time steps
    
    # 创建测试数据
    hidden_states = torch.randn(B, T, C)
    encoder_hidden_states = torch.randn(B, T, C)
    temb = torch.randn(B, 64)  # 假设 conditioning_dim = 64
    
    # 创建一个随机权重矩阵用于矩阵乘法
    weight = torch.randn(C, C)
    
    # 测试所有可能的排列模式
    patterns = ["FWH", "FHW", "WFH", "WHF", "HFW", "HWF"]
    for pattern in patterns:
        # 创建融合了重排序的模型
        model = ReorderCogVideoXLayerNormZero(
            conditioning_dim=64,
            embedding_dim=C,
            pattern=pattern,
            Height=H,
            Width=W,
            F=F,
            elementwise_affine=True,
            eps=1e-5,
            bias=True,
            mode="reorder"
        )
        
        # 1. 先进行重排序和归一化
        reordered_hidden, reordered_encoder, gate, enc_gate = model(hidden_states, encoder_hidden_states, temb)
        
        # 2. 对重排序后的结果进行矩阵乘法操作
        transformed_hidden = torch.matmul(reordered_hidden, weight)
        
        # 3. 切换到逆重排序模式
        model.mode = "inv_reorder"
        
        # 4. 进行逆重排序
        final_hidden, final_encoder, _, _ = model(transformed_hidden, reordered_encoder, temb)
        
        # 验证输出形状
        assert final_hidden.shape == (B, T, C), f"Output shape mismatch for pattern {pattern}"
        
        # 验证重排序和矩阵乘法的正确性        
        # 2. 手动归一化
        shift, scale, gate, enc_shift, enc_scale, enc_gate = model.linear(model.silu(temb)).chunk(6, dim=1)
        manual_normed = model.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        
        # 3. 手动矩阵乘法
        manual_transformed = torch.matmul(manual_normed, weight)
        
        # 4. 手动归一化
        manual_normed2 = model.norm(manual_transformed) * (1 + scale)[:, None, :] + shift[:, None, :]
        
        # 5. 手动逆重排序
        
        # 比较结果
        is_close = torch.allclose(final_hidden, manual_normed2, rtol=1e-5, atol=1e-5)
        if not is_close:
            print(f"Max difference: {torch.max(torch.abs(final_hidden - manual_normed2))}")
            print(f"Mean difference: {torch.mean(torch.abs(final_hidden - manual_normed2))}")
            print(f"Final hidden shape: {final_hidden.shape}")
            print(f"Manual normed2 shape: {manual_normed2.shape}")
        
        assert is_close, f"Reordering and matrix multiplication failed for pattern {pattern}"
        
        # 打印测试通过信息
        print(f"Test passed for pattern: {pattern}")

if __name__ == "__main__":
    test_reorder_cogvideo_x_layer_norm_zero()
