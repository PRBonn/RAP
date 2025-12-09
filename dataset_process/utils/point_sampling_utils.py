import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
import torch # For torch.manual_seed in _allocate_fps_points
from pytorch3d.ops import sample_farthest_points
import open3d as o3d

logger = logging.getLogger(__name__)

def calculate_voxel_coverage(points: np.ndarray, voxel_size: float) -> int:
    """
    Calculate the number of unique voxels occupied by the point cloud.
    
    Args:
        points: Point cloud array (N, 3)
        voxel_size: Voxel size in meters
        
    Returns:
        Number of unique voxels occupied
    """
    if len(points) == 0:
        return 0
    
    # Convert points to voxel coordinates
    voxel_coords = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels = np.unique(voxel_coords, axis=0)
    
    return len(unique_voxels)

def calculate_adaptive_sample_count_per_part(
    parts_points: List[np.ndarray], 
    voxel_size: float, 
    voxel_ratio: float, 
    min_points_per_part: int, 
    max_sample_points: int
) -> List[int]:
    """
    Calculate the number of sample points for each part based on its occupied voxels.
    
    Args:
        parts_points: List of point cloud arrays for each part
        voxel_size: Voxel size in meters for spatial coverage calculation
        voxel_ratio: Ratio of occupied voxels to sample points
        min_points_per_part: Minimum number of points each part should have
        max_sample_points: Maximum number of sample points for voxel_adaptive method
        
    Returns:
        List of sample points to allocate for each part
    """
    if not parts_points:
        return []
    
    sample_counts_per_part = []
    
    for i, points in enumerate(parts_points):
        if len(points) == 0:
            sample_counts_per_part.append(0)
            continue
        
        # Calculate voxel coverage for this part
        voxel_coverage = calculate_voxel_coverage(points, voxel_size)
        
        # Calculate adaptive sample count for this part
        adaptive_sample_count = int(voxel_coverage * voxel_ratio)
        
        # Apply min/max constraints for this part
        adaptive_sample_count = max(min_points_per_part, adaptive_sample_count)
        # Don't exceed available points in this part
        adaptive_sample_count = min(len(points), adaptive_sample_count)
        # Apply global max constraint
        adaptive_sample_count = min(max_sample_points, adaptive_sample_count)
        
        sample_counts_per_part.append(adaptive_sample_count)
        
        logger.debug(f"Part {i}: {voxel_coverage} occupied voxels -> {adaptive_sample_count} sample points "
                    f"(ratio: {voxel_ratio}, voxel_size: {voxel_size}m)")
    
    total_sample_points = sum(sample_counts_per_part)
    logger.debug(f"Voxel-adaptive per-part sampling: total {total_sample_points} sample points across {len(parts_points)} parts")
    
    return sample_counts_per_part

def allocate_fps_points(
    parts_data: Union[List[int], List[np.ndarray]], 
    allocation_method: str,
    num_points: int,
    min_points_per_part: int,
    voxel_size: float,
    voxel_ratio: float,
    total_sample_points: Optional[List[int]] = None # for voxel_adaptive method
) -> np.ndarray:
    """
    Allocate FPS points to parts proportionally with minimum constraints.
    
    Args:
        parts_data: Either list of point counts (for point_count method) or 
                   list of point arrays (for spatial_coverage and voxel_adaptive methods)
        allocation_method: Method for allocating FPS points ('point_count', 'spatial_coverage', or 'voxel_adaptive')
        num_points: Total number of sample points to allocate (for point_count and spatial_coverage methods)
        min_points_per_part: Minimum number of points each part should have after FPS
        voxel_size: Voxel size in meters for spatial coverage calculation (used in spatial_coverage and voxel_adaptive)
        voxel_ratio: Ratio of occupied voxels to sample points for voxel_adaptive method
        total_sample_points: Total number of sample points to allocate for voxel_adaptive (per-part counts)
        
    Returns:
        Array of target points to sample from each part
    """
    if allocation_method == 'point_count':
        if isinstance(parts_data[0], np.ndarray):
            pts_per_part = np.array([len(part) for part in parts_data])
        else:
            pts_per_part = np.array(parts_data)
        return _allocate_by_point_count_logic(pts_per_part, num_points, min_points_per_part)
    
    elif allocation_method == 'spatial_coverage':
        if isinstance(parts_data[0], np.ndarray):
            coverage_per_part = np.array([calculate_voxel_coverage(part, voxel_size) for part in parts_data])
            pts_per_part = np.array([len(part) for part in parts_data])
        else:
            raise ValueError("spatial_coverage allocation method requires point arrays, not point counts")
        return _allocate_by_spatial_coverage_logic(coverage_per_part, pts_per_part, num_points, min_points_per_part)
    
    elif allocation_method == 'voxel_adaptive':
        if isinstance(parts_data[0], np.ndarray):
            if total_sample_points is None:
                raise ValueError("voxel_adaptive allocation method requires total_sample_points parameter (per-part sample counts)")
            
            if not isinstance(total_sample_points, list):
                raise ValueError("voxel_adaptive allocation method requires total_sample_points to be a list of per-part counts")
            
            per_part_sample_counts = np.array(total_sample_points)
            pts_per_part = np.array([len(part) for part in parts_data])
            
            target_per_part = np.minimum(per_part_sample_counts, pts_per_part)
            
            logger.debug(f"Voxel-adaptive per-part allocation (voxel_size={voxel_size}m, ratio={voxel_ratio}):")
            for i, (requested_count, available_count, final_count) in enumerate(zip(per_part_sample_counts, pts_per_part, target_per_part)):
                logger.debug(f"  Part {i}: requested {requested_count}, available {available_count} -> {final_count} sampled")
            
            return target_per_part
        else:
            raise ValueError("voxel_adaptive allocation method requires point arrays, not point counts")
    
    else:
        raise ValueError(f"Unknown allocation method: {allocation_method}")

def _allocate_by_point_count_logic(pts_per_part: np.ndarray, num_points: int, min_points_per_part: int) -> np.ndarray:
    """
    Allocate FPS points based on point count (original method) - internal logic.
    """
    n_parts = len(pts_per_part)
    
    min_per_part = np.minimum(min_points_per_part, pts_per_part)
    total_min_points = min_per_part.sum()
    
    if total_min_points > num_points:
        logger.warning(f"Minimum points per part ({min_points_per_part}) * {n_parts} parts = {total_min_points} "
                      f"exceeds target ({num_points}). Scaling down.")
        scale_factor = num_points / total_min_points
        min_per_part = np.maximum(1, np.round(min_per_part * scale_factor).astype(int))
        total_min_points = min_per_part.sum()
    
    target_per_part = min_per_part.copy()
    remaining_points = num_points - total_min_points
    
    if remaining_points > 0:
        remaining_capacity = pts_per_part - min_per_part
        total_capacity = remaining_capacity.sum()
        
        if total_capacity > 0:
            extra = np.round(remaining_capacity * remaining_points / total_capacity).astype(int)
            target_per_part = np.minimum(target_per_part + extra, pts_per_part)
            
            diff = num_points - target_per_part.sum()
            while diff != 0 and ((diff > 0 and (target_per_part < pts_per_part).sum() > 0) or 
                               (diff < 0 and (target_per_part > min_per_part).sum() > 0)):
                if diff > 0:
                    valid_mask = target_per_part < pts_per_part
                else:
                    valid_mask = target_per_part > min_per_part
                
                if valid_mask.sum() > 0:
                    idx = np.random.choice(np.where(valid_mask)[0])
                    target_per_part[idx] += 1 if diff > 0 else -1
                    diff -= 1 if diff > 0 else -1
                else:
                    break
    
    return target_per_part

def _allocate_by_spatial_coverage_logic(coverage_per_part: np.ndarray, pts_per_part: np.ndarray, num_points: int, min_points_per_part: int) -> np.ndarray:
    """
    Allocate FPS points based on spatial coverage (voxel count) - internal logic.
    """
    n_parts = len(coverage_per_part)
    
    min_per_part = np.minimum(min_points_per_part, pts_per_part)
    total_min_points = min_per_part.sum()
    
    if total_min_points > num_points:
        logger.warning(f"Minimum points per part ({min_points_per_part}) * {n_parts} parts = {total_min_points} "
                      f"exceeds target ({num_points}). Scaling down.")
        scale_factor = num_points / total_min_points
        min_per_part = np.maximum(1, np.round(min_per_part * scale_factor).astype(int))
        total_min_points = min_per_part.sum()
    
    target_per_part = min_per_part.copy()
    remaining_points = num_points - total_min_points
    
    if remaining_points > 0:
        remaining_capacity = pts_per_part - min_per_part
        total_coverage = coverage_per_part.sum()
        
        if total_coverage > 0:
            coverage_proportion = coverage_per_part / total_coverage
            extra_by_coverage = np.round(coverage_proportion * remaining_points).astype(int)
            extra_by_coverage = np.minimum(extra_by_coverage, remaining_capacity)
            target_per_part += extra_by_coverage
            
            diff = num_points - target_per_part.sum()
            iteration_count = 0
            max_iterations = remaining_points
            
            while diff != 0 and iteration_count < max_iterations:
                if diff > 0 and (target_per_part < pts_per_part).sum() > 0:
                    valid_mask = target_per_part < pts_per_part
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        coverage_weights = coverage_per_part[valid_indices]
                        if coverage_weights.sum() > 0:
                            coverage_weights = coverage_weights / coverage_weights.sum()
                            idx = valid_indices[np.random.choice(len(valid_indices), p=coverage_weights)]
                        else:
                            idx = np.random.choice(valid_indices)
                        target_per_part[idx] += 1
                        diff -= 1
                elif diff < 0 and (target_per_part > min_per_part).sum() > 0:
                    valid_mask = target_per_part > min_per_part
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        coverage_weights = coverage_per_part[valid_indices]
                        if coverage_weights.sum() > 0:
                            inv_weights = 1.0 / (coverage_weights + 1e-8)
                            inv_weights = inv_weights / inv_weights.sum()
                            idx = valid_indices[np.random.choice(len(valid_indices), p=inv_weights)]
                        else:
                            idx = np.random.choice(valid_indices)
                        target_per_part[idx] -= 1
                        diff += 1
                else:
                    break
                
                iteration_count += 1
            
            if iteration_count >= max_iterations:
                logger.warning(f"Fine-tuning reached maximum iterations ({max_iterations}), final diff: {diff}")
    
    return target_per_part

def apply_batched_fps(
    batch_augmented_tensor: torch.Tensor,
    batch_lengths_tensor: torch.Tensor,
    batch_k_tensor: torch.Tensor,
    global_seed: int,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply batched FPS using PyTorch3D.

    Args:
        batch_augmented_tensor: Batched augmented point clouds (N, max_points, 3)
        batch_lengths_tensor: Lengths of point clouds in the batch (N,)
        batch_k_tensor: Number of points to sample for each point cloud (N,)
        global_seed: Global random seed for reproducible sampling
        device: Device to perform computation on

    Returns:
        Tuple of (list of sampled augmented parts, list of sampled indices)
    """
    try:
        # Set the FPS-specific random seed for reproducible sampling
        torch.manual_seed(global_seed)
        
        # PyTorch3D FPS requires a contiguous tensor, ensure it
        batch_augmented_tensor = batch_augmented_tensor.contiguous()
        
        sampled_points, indices_tensor = sample_farthest_points(
            batch_augmented_tensor.to(device), 
            lengths=batch_lengths_tensor.to(device),
            K=batch_k_tensor.to(device), 
            random_start_point=True
        )

        # Extract sampled points using indices (keep as tensors to avoid redundant conversions)
        sampled_augmented_parts = []
        
        for i, (k_i, indices_i) in enumerate(zip(batch_k_tensor, indices_tensor)):
            # Get valid indices (not padded) - keep as tensor
            valid_indices = indices_i[:k_i]
            sampled_augmented_parts.append(batch_augmented_tensor[i][valid_indices])
        
        return sampled_augmented_parts, indices_tensor

    except Exception as e:
        logger.error(f"PyTorch3D batched FPS failed: {e}")
        raise RuntimeError(f"PyTorch3D FPS failed: {e}") from e 