"""Utility functions for point cloud reshaping."""

import torch


def split_parts(
    pointclouds: torch.Tensor, 
    points_per_part: torch.Tensor,
    cu_seqlens_batch: torch.Tensor | None = None,
) -> list[list[torch.Tensor]]:
    """Split a packed tensor into per-part point clouds.

    Args:
        pointclouds: Tensor of shape (B, N, 3) for fixed batching or (TP, 3) for dynamic batching.
        points_per_part: Tensor of shape (B, P) giving the number of points in each part.
        cu_seqlens_batch: Tensor of shape (B + 1,) giving cumulative sequence lengths. 
                         Required for dynamic batching (when pointclouds has shape TP, 3).

    Returns: 
        parts: A list of length B, where parts[b] is itself a list of P (or fewer)
            tensors of shape (N_i, 3), where N_i is the number of points in the i-th part.
    """
    B, P = points_per_part.shape
    device = pointclouds.device
    
    # Handle dynamic batching case (TP, 3)
    if pointclouds.ndim == 2 and pointclouds.shape[1] == 3:
        if cu_seqlens_batch is None:
            raise ValueError("cu_seqlens_batch is required when pointclouds has shape (TP, 3)")
        
        # Split the flattened tensor back into batches and then into parts directly
        counts_per_batch = points_per_part.tolist()
        parts: list[list[torch.Tensor]] = []
        
        for b in range(B):
            start_idx = cu_seqlens_batch[b]
            end_idx = cu_seqlens_batch[b + 1]
            batch_pointclouds = pointclouds[start_idx:end_idx]  # (N_b, 3)
            
            counts = counts_per_batch[b]
            assert sum(counts) == batch_pointclouds.size(0), (
                f"Mismatch detected: sum(counts)={sum(counts)} does not equal "
                f"batch_pointclouds.size(0)={batch_pointclouds.size(0)} for batch {b}."
            )
            
            splits = torch.split(batch_pointclouds, counts, dim=0)
            parts.append([s for s in splits if s.size(0) > 0])
        
        return parts
    else:
        # Handle fixed batching case - ensure proper shape
        if pointclouds.ndim == 3:
            pass  # Already in (B, N, 3) format
        else:
            pointclouds = pointclouds.view(B, -1, 3)
        
        # Now proceed with the original splitting logic
        counts_per_batch = points_per_part.tolist()
        parts: list[list[torch.Tensor]] = []
        for b, counts in enumerate(counts_per_batch):
            assert sum(counts) == pointclouds[b].size(0), (
                f"Mismatch detected: sum(counts)={sum(counts)} does not equal "
                f"pointclouds[b].size(0)={pointclouds[b].size(0)} for batch {b}."
            )
            splits = torch.split(pointclouds[b], counts, dim=0)
            parts.append([s for s in splits if s.size(0) > 0])
        return parts


def ppp_to_ids(points_per_part: torch.Tensor) -> torch.Tensor:
    """Convert a points_per_part tensor into a part-IDs tensor.

    Args:
        points_per_part: Tensor of shape (B, P).

    Returns:
        Long tensor of shape (B, max_points), where for each batch b, the first
        N_b = points_per_part[b].sum() entries are the part-indices (0...P-1)
        repeated according to points_per_part[b], and any remaining positions
        (out to max_points) are zero.
    """
    B, P = points_per_part.shape
    device = points_per_part.device
    max_points = int(points_per_part.sum(dim=1).max().item())
    result = torch.zeros(B, max_points, dtype=torch.long, device=device)
    part_ids = torch.arange(P, device=device)

    for b in range(B):
        # Repeat each part index by its count in this batch
        id_repeated = torch.repeat_interleave(part_ids, points_per_part[b])  # (N_b,)
        result[b, : id_repeated.size(0)] = id_repeated
    return result


def flatten_valid_parts(x: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Flatten tensor by selecting only valid parts.
    
    Args:
        x: Batched tensor of shape (B, P, ...).
        points_per_part: Number of points per part of shape (B, P).
        
    Returns:
        Tensor of shape (valid_P, ...).
    """
    part_valids = points_per_part != 0                        # (B, P)
    return x[part_valids]


def create_batch_indices(points_per_part, num_parts):
    """Create batch indices equivalent to latent['batch'] from data_dict.
    
    This function creates batch indices that consider both part level and batch level,
    similar to batch_level1 in the encoder.
    
    Args:
        points_per_part: Tensor of shape (B, P) giving number of points per part
        num_parts: Tensor of shape (B,) giving number of valid parts per batch
    """
    
    # Calculate cumulative part count across batches to create unique offsets
    # This handles cases where different batches have different numbers of parts
    cumulative_parts = torch.cumsum(num_parts, dim=0) - num_parts  # (B,)
    
    # Create part indices for each batch: [0, 1, 2, ..., max_parts-1]
    max_parts = num_parts.max()
    part_indices = torch.arange(max_parts, device=points_per_part.device).unsqueeze(0)  # (1, max_parts)
    
    # Create batch offsets: each batch gets offset based on cumulative parts
    batch_offsets = cumulative_parts.unsqueeze(1)  # (B, 1)
    
    # Add batch offsets to part indices: (B, max_parts)
    batch_part_indices = part_indices + batch_offsets
    
    # Create mask for valid parts in each batch
    valid_parts_mask = torch.arange(max_parts, device=points_per_part.device).unsqueeze(0) < num_parts.unsqueeze(1)  # (B, max_parts)
    
    # Apply mask and repeat according to points_per_part
    # First, flatten the valid parts and their indices
    valid_batch_indices = valid_parts_mask.nonzero(as_tuple=True)  # (valid_parts, 2) - (batch_idx, part_idx)
    batch_idx, part_idx = valid_batch_indices
    
    # Get the corresponding batch_part_indices for valid parts
    valid_indices = batch_part_indices[batch_idx, part_idx]  # (valid_parts,)
    
    # Repeat each valid index according to the number of points in that part
    valid_points_per_part = points_per_part[batch_idx, part_idx]  # (valid_parts,)
    
    # Use repeat_interleave to create the final batch indices
    batch_indices = torch.repeat_interleave(valid_indices, valid_points_per_part)
    
    return batch_indices

def create_part_cu_seqlens(batch_indices):
    part_seqlen = torch.bincount(batch_indices)                    # (n_valid_parts, )
    max_seqlen = part_seqlen.max().item()                            # .item() is used to allow torch.compile
    part_cu_seqlens = torch.nn.functional.pad(torch.cumsum(part_seqlen, 0), (1, 0))
    part_cu_seqlens = part_cu_seqlens.to(torch.int32)                # (n_valid_parts + 1, ) 

    return max_seqlen, part_cu_seqlens        

def repeat_by_cu_seqlens(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Repeat a tensor by the cumulative sequence lengths (cu_seqlens).
    
    Args:
        x: Tensor of shape (B, ...).
        cu_seqlens: Tensor of shape (B + 1, ).
        
    Returns:
        Tensor of shape (TP, ...).

    Example:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        cu_seqlens = torch.tensor([0, 2, 5])
        repeat_by_cu_seqlens(x, cu_seqlens)
        # tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
    """
    cu = cu_seqlens.to(device=x.device, dtype=torch.int32)
    assert cu.ndim == 1 and cu.numel() == x.shape[0] + 1, "Bad cu_seqlens shape"

    lens = cu[1:] - cu[:-1]
    idx = torch.repeat_interleave(
        torch.arange(x.shape[0], device=x.device), lens
    )
    return x.index_select(0, idx)  