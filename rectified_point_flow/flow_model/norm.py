"""Normalization layers for DiT point cloud model.

This module provides various normalization techniques including RMS normalization
and adaptive layer normalization with timestep conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from ..utils.point_clouds import repeat_by_cu_seqlens


class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization layer. 

    Ref: 
        https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head RMS normalization."""
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with timestep conditioning."""

    def __init__(
        self, dim: int, act_fn: nn.Module = nn.SiLU, num_channels: int = 256
    ):
        """Initialize the adaptive layer normalization.
        
        Args:
            dim (int): Dimension of embeddings.
            act_fn (nn.Module): Activation function. Default: nn.SiLU.
            num_channels (int): Number of channels for timestep projection. Default: 256.
        """
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=num_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=num_channels, time_embed_dim=dim
        )
        self.activation = act_fn()
        self.linear = nn.Linear(dim, dim * 2)                       # for scale and shift
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, cu_seqlens_batch: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.
        
        Args:
            x (TP, dim): Input tensor.
            timestep (B,): Timestep tensor.
            cu_seqlens_batch (B + 1,): Cumulative sequence lengths for each batch.

        Returns:
            (TP, dim): Normalized tensor.
        """
        emb = self.timestep_embedder(self.timestep_proj(timestep))    # (B, dim)
        emb = self.linear(self.activation(emb))                       # (B, dim * 2)
        scale, shift = emb.chunk(2, dim=-1)                           # (B, dim) for both
        scale = repeat_by_cu_seqlens(scale, cu_seqlens_batch)         # (TP, dim)
        shift = repeat_by_cu_seqlens(shift, cu_seqlens_batch)         # (TP, dim)
        return self.norm(x) * (1 + scale) + shift