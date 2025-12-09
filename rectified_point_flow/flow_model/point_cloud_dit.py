"""Multi-part Point Cloud Diffusion Transformer (DiT) model."""

import torch
import torch.nn as nn

from .embedding import PointCloudEncodingManager
from .layer import DiTLayer
from ..utils.point_clouds import repeat_by_cu_seqlens

class PointCloudDiT(nn.Module):
    """A transformer-based diffusion model for multi-part point cloud data.

    Ref:
        DiT: https://github.com/facebookresearch/DiT
        mmdit: https://github.com/lucidrains/mmdit/tree/main/mmdit
        GARF: https://github.com/ai4ce/GARF
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        softcap: float = 0.0,
        qk_norm: bool = True,
        attn_dtype: str = "float16",
        final_mlp_act: nn.Module = nn.SiLU,
        max_points_per_part: int = 500,
        max_points_per_batch: int = 40000,
        scale_emb_on: bool = True,
        local_feat_concat_on: bool = True,
        local_feat_dim: int = 0,
    ):
        """
        Args:
            in_dim: Input dimension of the point features (e.g., 64).
            out_dim: Output dimension (e.g., 3 for velocity field).
            embed_dim: Hidden dimension of the transformer layers (e.g., 512).
            num_layers: Number of transformer layers (e.g., 6). 
            num_heads: Number of attention heads (e.g., 8).
            dropout_rate: Dropout rate, default 0.0.
            softcap: Soft cap for attention scores, default 0.0.
            qk_norm: Whether to use query-key normalization, default True.
            attn_dtype: Attention data type, default float16.
            final_mlp_act: Activation function for the final MLP, default SiLU.
            max_points_per_part: Maximum number of points per part, used for flash attention.
            max_points_per_batch: Maximum number of points per batch, used for flash attention.
            scale_emb_on: Whether to use scale embedding, default True.
            local_feat_concat_on: Whether to use local feature concatenation, default True.
            local_feat_dim: Dimension of the local features, default 0.
        """
        super().__init__()
        self.in_dim = in_dim # ptv3 output feature dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.final_mlp_act = final_mlp_act
        self.max_points_per_part = max_points_per_part
        self.max_points_per_batch = max_points_per_batch

        self.scale_emb_on = scale_emb_on
        self.local_feat_concat_on = local_feat_concat_on
        self.local_feat_dim = local_feat_dim


        # Parse attn_dtype
        if attn_dtype == "float16" or attn_dtype == "fp16":
            self.attn_dtype = torch.float16
        elif attn_dtype == "bfloat16" or attn_dtype == "bf16":
            self.attn_dtype = torch.bfloat16
        elif attn_dtype == "float32" or attn_dtype == "fp32":
            self.attn_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported attn_dtype: {attn_dtype}")

        # Reference part embedding for distinguishing anchor vs. moving parts # learnable
        self.anchor_part_emb = nn.Embedding(2, self.embed_dim) # 2, 512

        # Point cloud encoding manager
        self.encoding_manager = PointCloudEncodingManager(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            multires=10,
            scale_emb_on=self.scale_emb_on,
            local_feat_concat_on=self.local_feat_concat_on,
            local_feat_dim=self.local_feat_dim
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            DiTLayer(
                dim=self.embed_dim,
                num_attention_heads=self.num_heads,
                attention_head_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                softcap=softcap,
                qk_norm=qk_norm,
                attn_dtype=self.attn_dtype,
                max_points_per_part=self.max_points_per_part,
                max_points_per_batch=self.max_points_per_batch,
            )
            for _ in range(self.num_layers)
        ])

        # MLP for final predictions
        self.final_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim // 2, out_dim, bias=False)  # No bias for 3D coordinates
        )

    def _add_anchor_embedding(
        self,
        x: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Add anchor part embeddings to distinguish anchor from moving parts.
        
        Args:
            x (B, N, dim): Input point cloud features.
            anchor_indices (B, N): bool tensor, True => anchor parts.
            
        Returns:
            (B, N, dim) Point cloud features with anchor part information added.
        """
        # anchor_part_emb.weight[0] for non-anchor part
        # anchor_part_emb.weight[1] for anchor part
        TP = len(anchor_indices)
        anchor_part_emb = self.anchor_part_emb.weight[0].repeat(TP, 1)
        anchor_part_emb[anchor_indices] = self.anchor_part_emb.weight[1]
        x = x + anchor_part_emb
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond_coord: torch.Tensor,
        local_features: torch.Tensor | None,
        latent_features: torch.Tensor | None,
        scales: torch.Tensor,
        anchor_indices: torch.Tensor,
        cu_seqlens_batch: torch.Tensor,
        cu_seqlens_part: torch.Tensor,
        return_transformer_features: bool = False,
    ) -> torch.Tensor | dict:
        """Forward pass through the PointCloudDiT model.
        
        Args:
            x (TP, 3): Noise point coordinates at timestep t.
            timesteps (B, ): Timestep values.
            cond_coord (TP, 3): Point coordinates of condition point cloud.
            local_features (TP, f_dim): Local point features of condition point cloud.
            latent_features (TP, f2_dim): Latent point features of condition point cloud.
            scales (TP, ): Scale factor for the point cloud.
            anchor_indices (TP, ): bool tensor, True => anchor parts.
            cu_seqlens_batch (B + 1, ): Cumulative sequence lengths for each batch.
            cu_seqlens_part (VP + 1, ): Cumulative sequence lengths for each part.
            return_transformer_features (bool): If True, return dict with both velocity and transformer features.
            
        Returns:
            If return_transformer_features=False: Tensor of shape (TP, out_dim) representing the predicted velocity field.
            If return_transformer_features=True: Dict with 'velocity' (TP, out_dim) and 'transformer_features' (TP, embed_dim).
        """

        # Encoding
        scales = repeat_by_cu_seqlens(scales, cu_seqlens_batch)
        embed = self.encoding_manager(x, cond_coord, local_features, latent_features, scales)   # (TP, dim) 512
        embed = self._add_anchor_embedding(embed, anchor_indices)                               # (TP, dim)

        # Transformer layers
        for i, layer in enumerate(self.transformer_layers):
            embed = layer(embed, timesteps, cu_seqlens_batch, cu_seqlens_part)                  # (TP, dim)                  

        # Final MLP, use float32 for better numerical stability
        with torch.amp.autocast(x.device.type, enabled=False):
            out = self.final_mlp(embed.float())                                                 # (TP, out_dim)
        
        if return_transformer_features:
            return {
                'velocity': out,
                'transformer_features': embed.float()  # (TP, embed_dim)
            }
        return out


if __name__ == "__main__":
    model = PointCloudDiT(
        in_dim=64,
        out_dim=6,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.0,
    )
    print(f"PointCloudDiT with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")