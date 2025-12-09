import torch

from .utils.point_clouds import split_parts


def solve_procrustes(source_pcd: torch.Tensor, target_pcd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Procrustes problem via SVD to find optimal rotation and translation, i.e., 

        min_{R, t} ||R * source_pcd + t - target_pcd||^2.

    Args:
        source_pcd (torch.Tensor): Source point cloud of shape (N, 3).
        target_pcd (torch.Tensor): Target point cloud of shape (N, 3).

    Returns: 
        R (torch.Tensor): Rotation matrix of shape (3, 3).
        t (torch.Tensor): Translation vector of shape (3,).
    """

    source_mean = source_pcd.mean(dim=0, keepdim=True)
    target_mean = target_pcd.mean(dim=0, keepdim=True)
    source_centered = source_pcd - source_mean
    target_centered = target_pcd - target_mean
    
    # Kabsch algorithm
    H = source_centered.t() @ target_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.t() @ U.t()
    
    # Ensure det(R) = 1
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()

    # Solve translation
    t = target_mean - source_mean @ R.t()
    return R, t.squeeze()


def fit_transformations(
    source_pcds: torch.Tensor,
    target_pcds: torch.Tensor,
    points_per_part: torch.Tensor,
    cu_seqlens_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-part rigid transformations between two multi-part point clouds.

    Args:
        source_pcds (torch.Tensor): Source point clouds of shape (TP, 3) for dynamic batching or (B, N, 3) for fixed batching.
        target_pcds (torch.Tensor): Target point clouds of shape (TP, 3) for dynamic batching or (B, N, 3) for fixed batching.
        points_per_part (torch.Tensor): Points per part of shape (B, P) where P is the maximum number of parts.
        cu_seqlens_batch (torch.Tensor, optional): Cumulative sequence lengths for each batch of shape (B + 1,). 
                                                  Required for dynamic batching (when input is TP, 3).

    Returns:
        rotations_pred (torch.Tensor): Rotation matrices of shape (B, P, 3, 3).
        translations_pred (torch.Tensor): Translation vectors of shape (B, P, 3).
    """

    device = source_pcds.device
    bs, n_parts = points_per_part.shape

    # print(f"source_pcds shape: {source_pcds.shape}")
    # print(f"target_pcds shape: {target_pcds.shape}")
    # print(f"points_per_part shape: {points_per_part.shape}")

    # Use the updated split_parts function that handles both batching formats
    parts_source = split_parts(source_pcds, points_per_part, cu_seqlens_batch)
    parts_target = split_parts(target_pcds, points_per_part, cu_seqlens_batch)

    rotations_pred = torch.zeros(bs, n_parts, 3, 3, device=device)
    translations_pred = torch.zeros(bs, n_parts, 3, device=device)
    for b in range(bs):
        for p in range(n_parts):
            if points_per_part[b, p] == 0:
                continue

            # Use float32 for SVD operations to run on CUDA
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                rot, trans = solve_procrustes(parts_source[b][p], parts_target[b][p])
                rotations_pred[b, p] = rot
                translations_pred[b, p] = trans
            
    return rotations_pred, translations_pred

def rigidify_prediction_with_procrustes(
    prediction: torch.Tensor,
    condition: torch.Tensor,
    points_per_part: torch.Tensor,
    cu_seqlens_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rigidify the prediction with the estimated procrustes transformation applied to the condition to make the prediction more rigid.
    """
    device = prediction.device
    bs, n_parts = points_per_part.shape

    # Use the updated split_parts function that handles both batching formats
    parts_source = split_parts(condition, points_per_part, cu_seqlens_batch)
    parts_target = split_parts(prediction, points_per_part, cu_seqlens_batch)

    # rotations_pred = torch.zeros(bs, n_parts, 3, 3, device=device)
    # translations_pred = torch.zeros(bs, n_parts, 3, device=device)
    rigidified_prediction = torch.zeros_like(prediction)
    offset = 0
    for b in range(bs):
        for p in range(n_parts):
            n_points = points_per_part[b, p]
            if n_points == 0:
                continue

            # Use float32 for SVD operations to run on CUDA
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                rot, trans = solve_procrustes(parts_source[b][p], parts_target[b][p])
                rigidified_prediction[offset:offset+n_points] = (parts_source[b][p] @ rot.t()) + trans
                offset += n_points
                # rmse = torch.sqrt(((rigidified_prediction[offset:offset+n_points] - parts_target[b][p]) ** 2).mean())
                # breakpoint()
    return rigidified_prediction