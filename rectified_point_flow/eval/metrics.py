"""Metrics for evaluation."""

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops import iterative_closest_point
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from ..utils.point_clouds import split_parts, ppp_to_ids


def compute_cd(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    cu_seqlens_batch: torch.Tensor,
    anchor_indices: torch.Tensor | None = None,
):
    """Compute the whole object Chamfer Distance (CD) between ground truth and predicted point clouds.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        cu_seqlens_batch (B + 1, ): Cumulative sequence lengths for each batch.

    Returns:
        Tensor of shape (B,) with Chamfer distance per batch.
    """
    
    pointclouds_gt = pointclouds_gt.view(-1, 3)
    pointclouds_pred = pointclouds_pred.view(-1, 3)
    object_cd = []
    for i in range(len(cu_seqlens_batch) - 1):
        st, ed = cu_seqlens_batch[i], cu_seqlens_batch[i + 1]
        # norm = 2 by default (chamfer l2 squared)
        cd, _ = chamfer_distance(
            x=pointclouds_gt[st:ed].unsqueeze(0),
            y=pointclouds_pred[st:ed].unsqueeze(0),
            single_directional=False,
            norm=2,
            point_reduction="mean",
        )
        cd = (0.5 * cd).sqrt() # get root mean square error (need to divide by 2 for the mean of two directions)
        
        object_cd.append(cd)
    object_cd = torch.stack(object_cd)
    return object_cd

def align_anchor(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    anchor_parts: torch.Tensor,
) -> torch.Tensor:
    """Align the predicted anchor parts to the ground truth anchor parts using ICP.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        anchor_parts (B, P): Whether the part is an anchor part; we use the first part with the flag of True as the anchor part.

    Returns:
        pointclouds_pred_aligned (B, N, 3): Aligned sampled point clouds.
    """
    B, P = anchor_parts.shape
    device = pointclouds_pred.device
    pointclouds_pred_aligned = pointclouds_pred.clone()

    with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
        for b in range(B):
            pts_count = 0
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                if anchor_parts[b, p]:
                    ed = pts_count + points_per_part[b, p]
                    anchor_align_icp = iterative_closest_point(pointclouds_pred[b, pts_count:ed].unsqueeze(0), pointclouds_gt[b, pts_count:ed].unsqueeze(0)).RTs
                    break

            pts_count = 0
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                ed = pts_count + points_per_part[b, p]
                pointclouds_pred_aligned[b, pts_count:ed] = pointclouds_pred[b, pts_count:ed] @ anchor_align_icp.R[0].T + anchor_align_icp.T[0]
                pts_count = ed

    return pointclouds_pred_aligned

def compute_part_acc(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    threshold: float = 0.01,
    return_matched_part_ids: bool = True,
):
    """Compute Part Accuracy (PA), the ratio of successfully posed parts over the total number of parts.

    The success is defined as the Chamfer Distance (CD) between a predicted part and a ground truth part is 
    less than the threshold (0.01 meter by default). Here, we use Hungarian matching to find the best matching 
    between predicted and ground truth parts, which is necessary due to the part interchangeability.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        threshold (float): Chamfer distance threshold.
        return_matched_part_ids (bool): Whether to return the matched part ids.

    Returns:
        Tensor of shape (B,) with part accuracy per batch.
        Tensor of shape (B, P): For each batch, the i-th part is matched to the j-th part if 
            matched_part_ids[b, i] == j.

    """
    device = pointclouds_gt.device
    B, P = points_per_part.shape
    part_acc = torch.zeros(B, device=device)
    matched_part_ids = torch.zeros(B, P, device=device, dtype=torch.long)
    parts_gt = split_parts(pointclouds_gt, points_per_part)
    parts_pred = split_parts(pointclouds_pred, points_per_part) # needed to be B, N, 3

    for b in range(B):
        lengths = points_per_part[b]                                # (P,)
        valid = lengths > 0
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        n_parts = idx.numel()
        lens = lengths[idx]                                         # (n_parts,)
        pts_gt = pad_sequence(parts_gt[b], batch_first=True)        # (n_parts, max_len, 3)
        pts_pred = pad_sequence(parts_pred[b], batch_first=True)    # (n_parts, max_len, 3)
        n_parts, max_len, _ = pts_gt.shape

        # Compute pairwise Chamfer distances between all parts (n_parts^2, max_len, 3)
        pts_gt = pts_gt.unsqueeze(1).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        pts_pred = pts_pred.unsqueeze(0).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        len_x = lens.unsqueeze(1).expand(n_parts, n_parts).reshape(-1)
        len_y = lens.unsqueeze(0).expand(n_parts, n_parts).reshape(-1)
        cd_all, _ = chamfer_distance(
            x=pts_gt,
            y=pts_pred,
            x_lengths=len_x,
            y_lengths=len_y,
            single_directional=False,
            point_reduction="mean",
            batch_reduction=None,
        )
        cd_mat = cd_all.view(n_parts, n_parts)

        # Find best matching using Hungarian algorithm
        cost_mat = (cd_mat >= threshold).float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matched = (cd_mat[row_ind, col_ind] < threshold).sum().item()
        part_acc[b] = matched / n_parts

        for i, j in zip(row_ind, col_ind):
            matched_part_ids[b, i] = j

    if return_matched_part_ids:
        return part_acc, matched_part_ids
    
    return part_acc

def compute_transform_errors(
    pointclouds: torch.Tensor,
    pointclouds_gt: torch.Tensor,
    rotations_gt: torch.Tensor,
    translations_gt: torch.Tensor,
    rotations_pred: torch.Tensor,
    translations_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    anchor_part: torch.Tensor,
    matched_part_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    cu_seqlens_batch: torch.Tensor | None = None,
    use_icp: bool = False,
):
    """Compute the per-part rotation and translation errors between ground truth and predicted point clouds.

    To factor out the symmetry of parts, we estimate the minimum transformation by ICP between the ground truth 
    and predicted parts.  (this is for the interchangable case, which does not apply for point cloud registration tasks)
    The rotation error (RE) is computed using the angular difference (Rodrigues formula). 
    The translation error (TE) is computed using the L2 norm of the translation vectors.

    Note that the scale of the point clouds is considered in the computation of the translation errors.
    
    Args:
        pointclouds (B, N, 3) or (TP, 3): Condition point clouds. (without scaling back yet)
        pointclouds_gt (B, N, 3) or (TP, 3): Ground truth point clouds. (without scaling back yet)
        rotations_pred (B, P, 3, 3): Estimated rotation matrices.
        translations_pred (B, P, 3): Estimated translation vectors. (without scaling back yet)
        points_per_part (B, P): Number of points in each part.
        anchor_part (B, P): Whether the part is an anchor part.
        matched_part_ids (B, P): Matched part ids per batch. If None, use the original part order.
        scale (B,): Scale of the point clouds. If None, use 1.0.
        cu_seqlens_batch (B + 1,): Cumulative sequence lengths for each batch. Required for dynamic batching.
        use_icp (bool): Whether to use ICP to refine transformation errors (but still with these sparse points). Default is False.

    Returns:
        rot_errors_mean (B,): Mean rotation errors per batch.
        trans_errors_mean (B,): Mean translation errors per batch.
    """
    device = pointclouds.device
    B, P = points_per_part.shape

    # Use the updated split_parts function that handles both batching formats
    parts_cond = split_parts(pointclouds, points_per_part, cu_seqlens_batch)
    parts_gt = split_parts(pointclouds_gt, points_per_part, cu_seqlens_batch)

    # Reshape scale to (B, 1, 1) for proper broadcasting with (B, P, 3) # not needed
    # translations_pred_scaled = translations_pred * scale.view(-1, 1, 1)

    # Re-order parts
    if matched_part_ids is not None:
        batch_idx = torch.arange(B, device=device)[:, None]
        rotations_pred = rotations_pred[batch_idx, matched_part_ids]
        translations_pred = translations_pred[batch_idx, matched_part_ids]

    if scale is None:
        scale = torch.ones(B, device=device)

    rot_errors = torch.zeros(B, P, device=device)
    trans_errors = torch.zeros(B, P, device=device)
    for b in range(B): # for each sample

        # Identify anchor part's transformations for the current batch
        anchor_idx = anchor_part[b].nonzero(as_tuple=False).squeeze(1)
        if anchor_idx.numel() > 0:
            # Assuming there's only one anchor part (change to the largest one, but for testing, we also disable the multi-anchor, so it's fine now)
            anchor_idx = anchor_idx[0]
            R_anchor_gt = rotations_gt[b, anchor_idx]  # (3, 3)
            t_anchor_gt = translations_gt[b, anchor_idx]  # (3,)
            R_anchor_pred = rotations_pred[b, anchor_idx]  # (3, 3)
            t_anchor_pred = translations_pred[b, anchor_idx]  # (3,)

            # Compute inverse of anchor transformations
            # Original: T_ref_part = (R_ref_part, t_ref_part)
            # Inverse: T_part_ref = (R_ref_part.T, -R_ref_part.T @ t_ref_part)
            R_anchor_gt_inv = R_anchor_gt.T
            t_anchor_gt_inv = -R_anchor_gt_inv @ t_anchor_gt

            R_anchor_pred_inv = R_anchor_pred.T
            t_anchor_pred_inv = -R_anchor_pred_inv @ t_anchor_pred
        else:
            # If no anchor part, use identity transformation
            R_anchor_gt_inv = torch.eye(3, device=device)
            t_anchor_gt_inv = torch.zeros(3, device=device)
            R_anchor_pred_inv = torch.eye(3, device=device)
            t_anchor_pred_inv = torch.zeros(3, device=device)

        for p in range(P): # for each non-anchor part
            if points_per_part[b, p] == 0 or anchor_part[b, p]:
                continue

            with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                if use_icp: # for the interchangable case, which does not apply for point cloud registration tasks
                    part_gt = parts_gt[b][p].unsqueeze(0)
                    part_cond = parts_cond[b][p]
                    part_transformed = (part_cond @ rotations_pred[b, p].T + translations_pred[b, p]).unsqueeze(0)
                    delta_transformation = iterative_closest_point(part_gt, part_transformed).RTs
                    
                    # Extract rotation and translation from ICP result
                    delta_R = delta_transformation.R[0]  # (3, 3)
                    delta_t = delta_transformation.T[0] * scale[b] # (3,)

                    # print('Using ICP !!!')

                else: # this is actually used now
                    # Get global transformations for the current part
                    R_gt_global = rotations_gt[b, p]  # (3, 3)
                    R_pred_global = rotations_pred[b, p]  # (3, 3)
                    t_gt_global = translations_gt[b, p]  # (3,)
                    t_pred_global = translations_pred[b, p]  # (3,)

                    # Compute relative transformations to the anchor part
                    R_gt_relative = R_anchor_gt_inv @ R_gt_global
                    t_gt_relative = R_anchor_gt_inv @ t_gt_global + t_anchor_gt_inv

                    R_pred_relative = R_anchor_pred_inv @ R_pred_global
                    t_pred_relative = R_anchor_pred_inv @ t_pred_global + t_anchor_pred_inv

                    # Compute rotation and translation differences in the relative frame
                    delta_R = R_gt_relative.T @ R_pred_relative  # Relative rotation
                    delta_t = (t_pred_relative - t_gt_relative) * scale[b]    # Translation difference (scale back to original scale in m)
                
                # Calculate rotation and translation errors from the delta transformation
                # Rotation error: angular difference using trace formula
                trace_R = torch.trace(delta_R)
                cos_theta = torch.clamp(0.5 * (trace_R - 1), -1.0, 1.0)
                rot_errors[b, p] = torch.rad2deg(torch.acos(cos_theta))
                
                # Translation error: L2 norm of translation difference
                trans_errors[b, p] = torch.norm(delta_t)
            
    # print("rot_errors (deg): ", rot_errors)
    # print("trans_errors (m): ", trans_errors)

    # Average over valid parts (excluding anchored parts)
    n_parts = ((points_per_part != 0) & (~anchor_part)).sum(dim=1)
    rot_errors_mean = rot_errors.sum(dim=1) / n_parts
    trans_errors_mean = trans_errors.sum(dim=1) / n_parts
    return rot_errors_mean, trans_errors_mean

def compute_transform_errors_direct(
    rotations_gt: torch.Tensor,
    translations_gt: torch.Tensor,
    rotations_pred: torch.Tensor,
    translations_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    matched_part_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
):
    """Compute the per-part rotation and translation errors directly without anchor normalization.
    
    This version compares GT and predicted transformations directly without computing relative 
    transformations to an anchor part. All parts are treated equally and included in the mean.
    
    The rotation error (RE) is computed using the angular difference (Rodrigues formula). 
    The translation error (TE) is computed using the L2 norm of the translation vectors.
    
    Note that the scale of the point clouds is considered in the computation of the translation errors.
    
    Args:
        rotations_gt (B, P, 3, 3): Ground truth rotation matrices.
        translations_gt (B, P, 3): Ground truth translation vectors (in scaled space).
        rotations_pred (B, P, 3, 3): Predicted rotation matrices.
        translations_pred (B, P, 3): Predicted translation vectors (in scaled space).
        points_per_part (B, P): Number of points in each part.
        matched_part_ids (B, P): Matched part ids per batch. If None, use the original part order.
        scale (B,): Scale of the point clouds. If None, use 1.0.
    
    Returns:
        rot_errors_mean (B,): Mean rotation errors per batch (over all valid parts).
        trans_errors_mean (B,): Mean translation errors per batch (over all valid parts).
    """
    device = rotations_gt.device
    B, P = points_per_part.shape
    
    # Re-order parts if matched_part_ids is provided
    if matched_part_ids is not None:
        batch_idx = torch.arange(B, device=device)[:, None]
        rotations_pred = rotations_pred[batch_idx, matched_part_ids]
        translations_pred = translations_pred[batch_idx, matched_part_ids]
    
    if scale is None:
        scale = torch.ones(B, device=device)
    
    rot_errors = torch.zeros(B, P, device=device)
    trans_errors = torch.zeros(B, P, device=device)
    
    for b in range(B):  # for each sample
        for p in range(P):  # for each part (including anchor parts)
            if points_per_part[b, p] == 0:
                continue
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                # Get GT and predicted transformations for this part
                R_gt = rotations_gt[b, p]  # (3, 3)
                t_gt = translations_gt[b, p]  # (3,) - in scaled space
                R_pred = rotations_pred[b, p]  # (3, 3)
                t_pred = translations_pred[b, p]  # (3,) - in scaled space
                
                # Compute rotation and translation differences directly
                # Rotation error: delta_R = R_gt^T @ R_pred (rotation from GT to predicted)
                delta_R = R_gt.T @ R_pred  # (3, 3)
                
                # Translation difference: scale back to meters
                delta_t = (t_pred - t_gt) * scale[b]  # (3,) - in meters
                
                # Calculate rotation error: angular difference using trace formula
                trace_R = torch.trace(delta_R)
                cos_theta = torch.clamp(0.5 * (trace_R - 1), -1.0, 1.0)
                rot_errors[b, p] = torch.rad2deg(torch.acos(cos_theta))
                
                # Translation error: L2 norm of translation difference
                trans_errors[b, p] = torch.norm(delta_t)
    
    # Average over all valid parts (including anchor parts)
    n_parts = (points_per_part != 0).sum(dim=1)
    rot_errors_mean = rot_errors.sum(dim=1) / n_parts
    trans_errors_mean = trans_errors.sum(dim=1) / n_parts
    return rot_errors_mean, trans_errors_mean


def compute_correspondence_rmse(
    source_gt: torch.Tensor,
    target_gt: torch.Tensor,
    source_pred: torch.Tensor,
    target_pred: torch.Tensor,
    distance_threshold: float = 0.1,
):
    """Compute RMSE between correspondences from predicted source and target point clouds.
    
    This function first finds valid correspondences between ground truth source and target point 
    clouds using proximity search with a distance threshold. Then, it uses those same correspondence 
    indices to compute the RMSE between the predicted source and target point clouds.
    
    Args:
        source_gt (N_source, 3): Ground truth source point cloud.
        target_gt (N_target, 3): Ground truth target point cloud.
        source_pred (N_source, 3): Predicted source point cloud.
        target_pred (N_target, 3): Predicted target point cloud.
        distance_threshold (float): Maximum distance for valid correspondences. Default is 0.01 meters.
    
    Returns:
        rmse (torch.Tensor): Root Mean Square Error between matched correspondences in predicted clouds.
        num_correspondences (int): Number of valid correspondences found.
        correspondence_ratio (float): Ratio of valid correspondences to total source points.
    """
    device = source_gt.device
    
    # Ensure point clouds are 2D tensors of shape (N, 3)
    def _ensure_2d(pc, name):
        if pc.dim() == 1:
            pc = pc.unsqueeze(0)
        if pc.dim() == 3:
            if pc.shape[0] != 1:
                raise ValueError(f"This function only works for a single pair of point clouds. "
                               f"{name} has shape (B, N, 3) with B > 1.")
            pc = pc.squeeze(0)
        return pc
    
    source_gt = _ensure_2d(source_gt, "source_gt")
    target_gt = _ensure_2d(target_gt, "target_gt")
    source_pred = _ensure_2d(source_pred, "source_pred")
    target_pred = _ensure_2d(target_pred, "target_pred")
    
    N_source = source_gt.shape[0]
    N_target = target_gt.shape[0]
    
    if N_source == 0 or N_target == 0:
        return torch.tensor(float('inf'), device=device), 0, 0.0
    
    # Verify that predicted clouds have the same number of points as GT clouds
    if source_pred.shape[0] != N_source:
        raise ValueError(f"source_pred must have the same number of points as source_gt. "
                        f"Got {source_pred.shape[0]} vs {N_source}.")
    if target_pred.shape[0] != N_target:
        raise ValueError(f"target_pred must have the same number of points as target_gt. "
                        f"Got {target_pred.shape[0]} vs {N_target}.")
    
    # Step 1: Find correspondences between GT source and GT target
    # Compute pairwise distances: (N_source, N_target)
    distances = torch.cdist(source_gt, target_gt, p=2)
    
    # Find nearest neighbor in target_gt for each point in source_gt
    min_distances, nearest_indices = torch.min(distances, dim=1)  # (N_source,)
    
    # Filter correspondences within threshold
    valid_mask = min_distances <= distance_threshold
    num_correspondences = valid_mask.sum().item()
    
    if num_correspondences == 0:
        return torch.tensor(float('inf'), device=device), 0, 0.0
    
    # Step 2: Use the same correspondence indices to compute RMSE between predicted clouds
    # Get corresponding points from predicted clouds
    valid_source_pred = source_pred[valid_mask]  # (num_correspondences, 3)
    valid_target_indices = nearest_indices[valid_mask]  # (num_correspondences,)
    valid_target_pred = target_pred[valid_target_indices]  # (num_correspondences, 3)
    
    # Compute RMSE over valid correspondences
    squared_errors = torch.sum((valid_source_pred - valid_target_pred) ** 2, dim=1)  # (num_correspondences,)
    rmse = torch.sqrt(torch.mean(squared_errors))
    
    correspondence_ratio = num_correspondences / N_source
    
    return rmse, num_correspondences, correspondence_ratio


def get_rotation_translation_from_transform(transform):
    """Extract rotation matrix and translation vector from 4x4 transformation matrix.
    
    Args:
        transform: 4x4 transformation matrix
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    R = transform[:3, :3]
    t = transform[:3, 3]
    return R, t


def compute_approximate_transform_error(rotation_error, translation_error, covariance):
    """Compute approximate transform error using rotation and translation errors with covariance matrix.
    
    Args:
        rotation_error: 3x3 rotation error matrix (relative rotation between GT and estimated)
        translation_error: 3x1 translation error vector (difference between GT and estimated)
        covariance: 6x6 covariance matrix (identity for standard RMSE)

        More information at http://redwood-data.org/indoor/registration.html
        This is commonly used by Predator, GeoTransformer, etc.
        
    Returns:
        Error value (scalar), but actually a squared error here, we need to take the square root to get the RMSE
    """
    # Convert rotation matrix to quaternion using scipy
    rot = Rotation.from_matrix(rotation_error)
    q = rot.as_quat()  # Returns [x, y, z, w] format
    # Reorder to [w, x, y, z] and take [x, y, z] (skip w)
    q_reordered = np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]
    er = np.concatenate([translation_error, q_reordered[1:]], axis=0)  # [t, x, y, z]
    p = er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]
    return p.item()


def compute_rigidity_rmse(
    pointclouds_input: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    rotations_pred: torch.Tensor,
    translations_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    cu_seqlens_batch: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    average_per_part: bool = False,
):
    """Compute rigidity RMSE by comparing generated point cloud parts with transformed input point cloud parts.
    
    For each object, this function:
    1. Transforms the input point cloud parts using the predicted rotations and translations
    2. Computes squared errors between the transformed input points and the predicted points
    3. Computes RMSE using one of two averaging methods:
       - If average_per_part=True: Computes RMSE per part, then averages across parts (equal weight per part)
       - If average_per_part=False: Averages squared errors over all points (across all parts) and takes square root
    
    This metric measures how well the generated point cloud matches the rigid transformation
    applied to the input point cloud, indicating the rigidity of the generation.
    
    Args:
        pointclouds_input (TP, 3) or (B, N, 3): Input point clouds (in scaled space).
        pointclouds_pred (TP, 3) or (B, N, 3): Predicted/generated point clouds (in scaled space).
        rotations_pred (B, P, 3, 3): Predicted rotation matrices.
        translations_pred (B, P, 3): Predicted translation vectors (in scaled space).
        points_per_part (B, P): Number of points in each part.
        cu_seqlens_batch (B + 1,): Cumulative sequence lengths for each batch. Required for dynamic batching.
        scales (B,): Scale factors. If provided, RMSE will be scaled to meters. If None, RMSE is in scaled space.
        average_per_part (bool): If True, averages RMSE per part first (equal weight per part).
                                 If False, averages over all points directly (weighted by points per part).
                                 Default: False.
    
    Returns:
        rigidity_rmse (B,): Per-object rigidity RMSE.
    """
    device = pointclouds_input.device
    B, P = points_per_part.shape
    
    # Split point clouds into parts
    parts_input = split_parts(pointclouds_input, points_per_part, cu_seqlens_batch)
    parts_pred = split_parts(pointclouds_pred, points_per_part, cu_seqlens_batch)
    
    # Initialize per-object rigidity RMSE
    rigidity_rmse_per_object = torch.zeros(B, device=device)
    
    for b in range(B):
        if average_per_part:
            # Method 1: Average per part first (equal weight per part)
            part_rmses = []
            
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                
                # Get input and predicted points for this part
                pts_input = parts_input[b][p]  # (N_p, 3)
                pts_pred = parts_pred[b][p]    # (N_p, 3)
                
                # Get predicted rotation and translation for this part
                R_pred = rotations_pred[b, p]  # (3, 3)
                t_pred = translations_pred[b, p]  # (3,)
                
                # Transform input points: x' = x @ R.T + t
                pts_transformed = pts_input @ R_pred.T + t_pred
                
                # Compute RMSE between transformed input and predicted points for this part
                squared_errors = torch.sum((pts_transformed - pts_pred) ** 2, dim=1)  # (N_p,)
                part_rmse = torch.sqrt(torch.mean(squared_errors))
                part_rmses.append(part_rmse)
            
            # Average RMSE across all parts for this object
            if len(part_rmses) > 0:
                rigidity_rmse_per_object[b] = torch.stack(part_rmses).mean()
            else:
                rigidity_rmse_per_object[b] = torch.tensor(float('inf'), device=device)
        else:
            # Method 2: Average over all points directly (weighted by points per part)
            all_squared_errors = []
            
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                
                # Get input and predicted points for this part
                pts_input = parts_input[b][p]  # (N_p, 3)
                pts_pred = parts_pred[b][p]    # (N_p, 3)
                
                # Get predicted rotation and translation for this part
                R_pred = rotations_pred[b, p]  # (3, 3)
                t_pred = translations_pred[b, p]  # (3,)
                
                # Transform input points: x' = x @ R.T + t
                pts_transformed = pts_input @ R_pred.T + t_pred
                
                # Compute squared errors between transformed input and predicted points
                squared_errors = torch.sum((pts_transformed - pts_pred) ** 2, dim=1)  # (N_p,)
                all_squared_errors.append(squared_errors)
            
            # Compute RMSE over all points (across all parts) for this object
            if len(all_squared_errors) > 0:
                all_squared_errors = torch.cat(all_squared_errors)  # Concatenate all squared errors
                rigidity_rmse_per_object[b] = torch.sqrt(torch.mean(all_squared_errors))
            else:
                rigidity_rmse_per_object[b] = torch.tensor(float('inf'), device=device)
    
    # Scale to meters if scales are provided
    if scales is not None:
        rigidity_rmse_per_object = rigidity_rmse_per_object * scales
    
    return rigidity_rmse_per_object


def compute_overlap_ratio(
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    cu_seqlens_batch: torch.Tensor | None = None,
    taus: list[float] = [0.005, 0.01, 0.02],
) -> torch.Tensor:
    """Compute overlap ratio among predicted per-part point clouds for multiple thresholds.

    For each tau in `taus`:
        OR_tau = |points having >=1 neighbor from a *different* part within distance <= tau|
                 / |total points|

    Args:
        pointclouds_pred: (TP, 3) Predicted/generated point clouds (in scaled space).
        points_per_part:  (B, P) Number of points in each part.
        cu_seqlens_batch: (B + 1,) Cumulative sequence lengths per batch item; if None,
                          inferred from points_per_part.
        taus: Sequence of distance thresholds. If a single float is given, it is wrapped.

    Returns:
        (T, B): Overlap ratios for each tau.
    """
    device = pointclouds_pred.device
    base_dtype = pointclouds_pred.dtype
    B, P = points_per_part.shape

    if isinstance(taus, (float, int)):
        tau_list = [float(taus)]
    else:
        tau_list = [float(t) for t in taus]
    T = len(tau_list)
    cu_seqlens_batch = cu_seqlens_batch.to(device=device, dtype=torch.long)

    ids = ppp_to_ids(points_per_part)  # ids[b][:N]
    ratios = torch.zeros(T, B, device=device, dtype=base_dtype)

    # Use float32 for distance computations for stability
    CHUNK = 1024

    for b in range(B):
        start = int(cu_seqlens_batch[b].item())
        end   = int(cu_seqlens_batch[b + 1].item())
        if end <= start:
            continue

        pred_points = pointclouds_pred[start:end]                                     # (N, 3)
        N = pred_points.shape[0]
        part_idx = ids[b][:N].to(device=device)                                       # (N,)

        if N <= 1 or (part_idx.unique().numel() <= 1):
            continue

        min_other_dists = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
        pred_points_f = pred_points.to(torch.float32)
        for i0 in range(0, N, CHUNK):
            i1 = min(i0 + CHUNK, N)
            cross_part_mask = (part_idx[i0:i1].unsqueeze(1) != part_idx.unsqueeze(0))  # (m, N)
            dists = torch.cdist(pred_points_f[i0:i1], pred_points_f, p=2)              # (m, N)
            dists.masked_fill_(~cross_part_mask, float("inf"))
            chunk_min = dists.min(dim=1).values                                        # (m,)
            min_other_dists[i0:i1] = chunk_min

        for t_idx, tau in enumerate(tau_list):
            flags = (min_other_dists <= float(tau))
            ratios[t_idx, b] = flags.float().mean().to(base_dtype)

    return ratios


def compute_ecdf(
    rotation_errors: torch.Tensor,
    translation_errors: torch.Tensor,
    r_splits: list[float] = [3, 5, 10, 30, 45],
    t_splits: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75],
) -> tuple[torch.Tensor, float, float, torch.Tensor, float, float]:
    """Compute Empirical Cumulative Distribution Function (ECDF) for rotation and translation errors.
    
    Args:
        rotation_errors: Tensor of shape (B,) with rotation errors in degrees.
        translation_errors: Tensor of shape (B,) with translation errors in meters.
        r_splits: List of rotation error thresholds (in degrees) for ECDF computation.
        t_splits: List of translation error thresholds (in meters) for ECDF computation.
    
    Returns:
        A tuple containing:
            - ecdf_r: Tensor of shape (len(r_splits),) with ECDF values for rotation errors.
            - mean_r: Mean rotation error (float).
            - med_r: Median rotation error (float).
            - ecdf_t: Tensor of shape (len(t_splits),) with ECDF values for translation errors.
            - mean_t: Mean translation error (float).
            - med_t: Median translation error (float).
    """
    # Convert to numpy for statistics computation
    rotation_errors_np = rotation_errors.detach().cpu().numpy()
    translation_errors_np = translation_errors.detach().cpu().numpy()
    
    # Compute ECDF for rotations: proportion of errors below each threshold
    ecdf_r = torch.tensor([np.mean(rotation_errors_np < rthresh) for rthresh in r_splits], 
                          dtype=torch.float32, device=rotation_errors.device)
    mean_r = float(np.mean(rotation_errors_np))
    med_r = float(np.median(rotation_errors_np))
    
    # Compute ECDF for translations: proportion of errors below each threshold
    ecdf_t = torch.tensor([np.mean(translation_errors_np < tthresh) for tthresh in t_splits],
                          dtype=torch.float32, device=translation_errors.device)
    mean_t = float(np.mean(translation_errors_np))
    med_t = float(np.median(translation_errors_np))
    
    return ecdf_r, mean_r, med_r, ecdf_t, mean_t, med_t