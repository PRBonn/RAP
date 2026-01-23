"""Point embedding utilities."""

import torch
import torch.nn as nn


class PointCloudEmbedding:
    """Generate positional encodings for multi-part point clouds.
    
    Ref:
        Nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    """

    def __init__(
        self,
        include_input: bool = True,
        input_dims: int = 3,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: list = None,
    ):
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns or [torch.sin, torch.cos]
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Create the embedding function and compute output dimension."""
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

        # with default setting, the output dim is then 60+3

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed input tensor using sinusoidal encoding."""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PointCloudEncodingManager(nn.Module):
    """Generate PointCloudEmbedding from the input.

    It includes PointCloudEmbedding for:
        - Coordinate of condition PCs
        - Coordinate of noise PCs # do we need to do embedding for this? # Or we directly use the original coordinate?
        - Normal vector of condition PCs
        - Scale of condition PCs

    Args:
        in_dim: Input feature dimension.
        embed_dim: Output embedding dimension.
        multires: Multiresolution level for frequency encoding.
    """

    def __init__(self, in_dim: int, embed_dim: int, multires: int = 10, scale_emb_on: bool = False, local_feat_concat_on: bool = False, local_feat_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.multires = multires
        self.scale_emb_on = scale_emb_on
        self.local_feat_concat_on = local_feat_concat_on
        self.local_feat_dim = local_feat_dim

        # without encoder, the in_dim is 0

        # Coordinate of condition PCs (63 dim)
        self.coord_embedding = PointCloudEmbedding(
            input_dims=3,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )
        
        # Coordinate of noise PCs (63 dim)
        self.noise_embedding = PointCloudEmbedding(
            input_dims=3,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )

        # Scale factor of condition PCs (check if this is needed, or do we need to input the part embedding here), 21 dim
        self.scale_embedding = PointCloudEmbedding(
            input_dims=1,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )

        # Do not concatenate the view idx embedding to keep the pertubation equivariance and then can handle a flexible number of input views.

        # Embedding projection
        embed_input_dim = (
            self.in_dim
            + self.coord_embedding.out_dim
            + self.noise_embedding.out_dim
        )
        if self.scale_emb_on:
            embed_input_dim += self.scale_embedding.out_dim # 21
        if self.local_feat_concat_on:
            embed_input_dim += self.local_feat_dim # 32

        # print(f"in_dim: {self.in_dim}") # 0
        # print(f"noise_embedding.out_dim: {self.noise_embedding.out_dim}") # 3 or 63
        # print(f"coord_embedding.out_dim: {self.coord_embedding.out_dim}") # 63
        # print(f"scale_embedding.out_dim: {self.scale_embedding.out_dim}") # 21
        # print(f"local_feat_dim: {self.local_feat_dim}") # 32
        # print(f"embed_input_dim: {embed_input_dim}")

        self.emb_proj = nn.Linear(embed_input_dim, self.embed_dim)
        # linearly map to embed_dim (512)

    def forward(
        self,
        x: torch.Tensor,
        cond_coord: torch.Tensor,
        local_features: torch.Tensor | None,
        latent_features: torch.Tensor | None,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Generate PointCloudEmbedding from the input.
        
        Args:
            x (TP, 3): Noise point coordinates at timestep t.
            cond_coord (TP, 3): Point coordinates of condition point cloud.
            local_features (TP, f_dim): Local point features of condition point cloud.
            latent_features (TP, f2_dim): Latent point features of condition point cloud.
            scales (TP, ): Scale factor for the point cloud.
            
        Returns:
            Shape embeddings of shape (TP, dim).
        """

        TP, _ = x.shape

        # Coordinate of noise PCs
        x_pos_emb = self.noise_embedding.embed(x)                         # (TP, dim)

        # Coordinate of condition PCs
        coord = cond_coord.view(TP, 3)
        c_pos_emb = self.coord_embedding.embed(coord)                     # (TP, dim)

        embed = torch.cat([c_pos_emb, x_pos_emb], dim=-1)

        # Concatenate with point features (after ptv3)
        if latent_features is not None:
            feat = latent_features.view(TP, -1)                           # (TP, dim) 
            embed = torch.cat([embed, feat], dim=-1)
        
        # Scale of condition PCs
        if self.scale_emb_on:
            scales = scales.unsqueeze(-1)
            scale_emb = self.scale_embedding.embed(scales)                # (TP, dim)
            embed = torch.cat([embed, scale_emb], dim=-1)
        
        # Local point features
        if self.local_feat_concat_on and local_features is not None:
            local_feat = local_features.view(TP, -1)                      # (TP, dim)
            embed = torch.cat([embed, local_feat], dim=-1)
        
        return self.emb_proj(embed) # linear project to the desired dimension for DiT
