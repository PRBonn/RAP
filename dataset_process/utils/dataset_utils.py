"""
General utility functions for dataset processing.

This module contains common functions that can be used across different dataset loaders,
such as point cloud processing, normal estimation, and downsampling.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Union, List
import logging
import torch
import os

# Import necessary modules from other utils files
from .io_utils import get_dataset_name

logger = logging.getLogger(__name__)


def random_downsample_points(points: np.ndarray, max_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly downsample a point cloud to a maximum number of points.
    
    Args:
        points: Input point cloud (N, 3) or (N, 4) with intensity
        max_points: Maximum number of points to keep
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Downsampled point cloud
    """
    if len(points) <= max_points:
        return points
    
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.choice(len(points), max_points, replace=False)
    downsampled_points = points[indices]
    
    # logger.debug(f"Randomly downsampled {len(points)} to {len(downsampled_points)} points")
    
    return downsampled_points


def farthest_point_sampling(points: np.ndarray, num_samples: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform farthest point sampling (FPS) on a point cloud using PyTorch3D when available.

    Note: this is still super slow when the input point cloud is too large.
    
    Args:
        points: Input point cloud (N, 3)
        num_samples: Number of points to sample
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (sampled_points, sampled_indices)
        - sampled_points: The sampled points (num_samples, 3)
        - sampled_indices: The indices of sampled points in original array (num_samples,)
    """
    if len(points) == 0:
        return np.array([]), np.array([], dtype=int)
    
    if num_samples >= len(points):
        return points.copy(), np.arange(len(points))
    
    if seed is not None:
        np.random.seed(seed)
        if PYTORCH3D_AVAILABLE:
            torch.manual_seed(seed)
    
    if PYTORCH3D_AVAILABLE:
        # Use PyTorch3D's efficient implementation
        try:
            # Determine device (use GPU if available for large point clouds)
            device = torch.device('cuda' if torch.cuda.is_available() and len(points) > 10000 else 'cpu')
            
            # Convert to torch tensor
            points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)  # Add batch dimension
            
            # Sample farthest points
            sampled_points_tensor, sampled_indices_tensor = sample_farthest_points(
                points_tensor, K=num_samples, random_start_point=True
            )
            
            # Convert back to numpy
            sampled_points = sampled_points_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
            sampled_indices = sampled_indices_tensor.squeeze(0).cpu().numpy().astype(int)
            
            device_str = f" on {device}" if device.type == 'cuda' else ""
            logger.info(f"FPS (PyTorch3D{device_str}) downsampled {len(points)} to {len(sampled_points)} points")
            
            return sampled_points, sampled_indices
        
        except Exception as e:
            logger.warning(f"PyTorch3D FPS failed ({e}), falling back to NumPy implementation")
            # Fall through to NumPy implementation
    
    # Fallback NumPy implementation
    logger.debug("Using NumPy FPS implementation (PyTorch3D not available)")
    
    num_points = len(points)
    sampled_indices = []
    distances = np.full(num_points, np.inf)
    
    # Start with a random point
    current_idx = np.random.randint(0, num_points)
    sampled_indices.append(current_idx)
    
    for _ in range(1, num_samples):
        # Update distances to the current point
        current_point = points[current_idx]
        current_distances = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, current_distances)
        
        # Select the farthest point
        current_idx = np.argmax(distances)
        sampled_indices.append(current_idx)
    
    sampled_indices = np.array(sampled_indices)
    sampled_points = points[sampled_indices]
    
    logger.debug(f"FPS (NumPy) downsampled {len(points)} to {len(sampled_points)} points")
    
    return sampled_points, sampled_indices


def farthest_point_sampling_with_normals(points: np.ndarray, 
                                        normals: Optional[np.ndarray] = None, 
                                        num_samples: int = 1000, 
                                        seed: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform farthest point sampling on a point cloud with optional normals.
    
    Args:
        points: Input point cloud (N, 3)
        normals: Optional normal vectors (N, 3)
        num_samples: Number of points to sample
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (sampled_points, sampled_normals)
        - sampled_points: The sampled points (num_samples, 3)
        - sampled_normals: The sampled normals (num_samples, 3) or None if normals is None
    """
    sampled_points, sampled_indices = farthest_point_sampling(points, num_samples, seed)
    
    if normals is not None and len(normals) > 0:
        sampled_normals = normals[sampled_indices]
        return sampled_points, sampled_normals
    else:
        return sampled_points, None


def downsample_points(points: np.ndarray, 
                     normals: Optional[np.ndarray] = None,
                     method: str = "voxel",
                     voxel_size: float = 0.1,
                     num_points: Optional[int] = None,
                     seed: Optional[int] = None,
                     use_torch: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unified function for downsampling point clouds using different methods.
    
    Args:
        points: Input point cloud (N, 3)
        normals: Optional normal vectors (N, 3)
        method: Downsampling method ("voxel", "fps", or "random")
        voxel_size: Size of voxels for voxel downsampling
        num_points: Target number of points (required for FPS and random methods)
        seed: Random seed for reproducibility (optional)
        use_torch: If True and method is "voxel", use torch-based voxel downsampling for speedup (default: False)
    
    Returns:
        Tuple of (downsampled_points, downsampled_normals)
        - downsampled_points: The downsampled points
        - downsampled_normals: The downsampled normals (or None if normals is None)
    """
    if len(points) == 0:
        return points, normals
    
    if method == "voxel":
        if use_torch:
            # Use torch-based voxel downsampling for speedup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            points_tensor = torch.from_numpy(points).float().to(device)
            
            # Get downsampled indices using torch
            downsampled_indices_tensor = voxel_down_sample_torch(points_tensor, voxel_size)
            downsampled_indices = downsampled_indices_tensor.cpu().numpy()
            
            # Apply indices to points and normals
            downsampled_points = points[downsampled_indices]
            downsampled_normals = normals[downsampled_indices] if normals is not None else None
        else:
            # Use numpy-based voxel downsampling (original implementation)
            downsampled_indices = _get_voxel_downsample_indices(points, voxel_size)
            downsampled_points = points[downsampled_indices]
            downsampled_normals = normals[downsampled_indices] if normals is not None else None
        
    elif method == "fps":
        if num_points is None:
            raise ValueError("num_points must be specified for FPS downsampling")
        downsampled_points, downsampled_normals = farthest_point_sampling_with_normals(
            points, normals, num_points, seed
        )
        
    elif method == "random":
        if num_points is None:
            raise ValueError("num_points must be specified for random downsampling")
        if len(points) <= num_points:
            downsampled_points = points.copy()
            downsampled_normals = normals.copy() if normals is not None else None
        else:
            if seed is not None:
                np.random.seed(seed)
            indices = np.random.choice(len(points), num_points, replace=False)
            downsampled_points = points[indices]
            downsampled_normals = normals[indices] if normals is not None else None
            
    else:
        raise ValueError(f"Unknown downsampling method: {method}. Choose from 'voxel', 'fps', or 'random'")
    
    method_str = f"{method} method"
    if method == "voxel" and use_torch:
        method_str = f"{method} method (torch-based)"
    logger.debug(f"Downsampled {len(points)} to {len(downsampled_points)} points using {method_str}")
    
    return downsampled_points, downsampled_normals


def _get_voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Get indices of points after voxel downsampling.
    
    Args:
        points: Input point cloud (N, 3)
        voxel_size: Size of voxels for downsampling
    
    Returns:
        Indices of downsampled points
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    
    # Convert to voxel coordinates
    voxel_coords = np.floor(points / voxel_size).astype(int)
    
    # Create unique voxel keys and track which points belong to each voxel
    voxel_to_points = {}
    for i, voxel_coord in enumerate(voxel_coords):
        voxel_key = tuple(voxel_coord)
        if voxel_key not in voxel_to_points:
            voxel_to_points[voxel_key] = []
        voxel_to_points[voxel_key].append(i)
    
    # For each voxel, find the point closest to the voxel center
    downsampled_indices = []
    for voxel_key, point_indices in voxel_to_points.items():
        if len(point_indices) == 1:
            # Only one point in this voxel, keep it
            downsampled_indices.append(point_indices[0])
        else:
            # Multiple points in this voxel, find the one closest to voxel center
            voxel_center = (np.array(voxel_key) + 0.5) * voxel_size
            voxel_points = points[point_indices]
            
            # Calculate distances to voxel center
            distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
            
            # Find the point with minimum distance
            closest_idx = np.argmin(distances)
            downsampled_indices.append(point_indices[closest_idx])
    
    return np.array(downsampled_indices)

def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`

    Reference: Louis Wiesmann
    """
    _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1) ** 0.5
    dist = (
        dist / dist.max() * (_quantization - 1)
    ).long()  # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx


def estimate_point_normals_open3d(points: np.ndarray, k_neighbors: int = 20, radius: float = 0.5) -> np.ndarray:
    """
    Estimate point-wise normals using Open3D.
    
    Args:
        points: Input point cloud (N, 3)
        k_neighbors: Number of neighbors for normal estimation (default: 20)
        radius: Search radius for normal estimation (default: 0.5m)
    
    Returns:
        Normal vectors (N, 3) with unit length
    """
    if len(points) == 0:
        return np.array([])
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    # Use both k-neighbors and radius for robust estimation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k_neighbors)
    )
    
    # Orient normals consistently (optional but recommended)
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
    
    # Get normals as numpy array
    normals = np.asarray(pcd.normals)
    
    logger.debug(f"Estimated normals for {len(points)} points")
    
    return normals


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform points using 4x4 transformation matrix.
    
    Args:
        points: Input point cloud (N, 3)
        transform: 4x4 transformation matrix
    
    Returns:
        Transformed point cloud (N, 3)
    """
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply transformation
    transformed_points = (transform @ points_homo.T).T
    return transformed_points[:, :3]


def transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform normals using 4x4 transformation matrix.
    Normals are transformed using the rotation part of the transformation matrix.
    
    Args:
        normals: Normal vectors (N, 3)
        transform: 4x4 transformation matrix
    
    Returns:
        Transformed normal vectors (N, 3)
    """
    if len(normals) == 0:
        return normals
    
    # Extract rotation matrix (3x3 upper-left part)
    rotation_matrix = transform[:3, :3]
    
    # Transform normals using rotation matrix only
    # Note: We don't add homogeneous coordinate for normals
    transformed_normals = (rotation_matrix @ normals.T).T
    
    # Normalize the transformed normals
    norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-8, 1.0, norms)
    transformed_normals = transformed_normals / norms
    
    return transformed_normals


def voxel_downsample_points(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Downsample points using voxel grid, keeping the actual points closest to voxel centers.
    
    Args:
        points: Input point cloud (N, 3)
        voxel_size: Size of voxels for downsampling
    
    Returns:
        Downsampled point cloud with actual points (not voxel centers)
        
    Note:
        This function is kept for backward compatibility. Consider using downsample_points() for more options.
    """
    downsampled_points, _ = downsample_points(points, method="voxel", voxel_size=voxel_size)
    logger.debug(f"Voxel downsampled {len(points)} to {len(downsampled_points)} points")
    return downsampled_points


def voxel_downsample_points_open3d(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Downsample points using Open3D's voxel downsampling (returns voxel centers).
    
    Args:
        points: Input point cloud (N, 3)
        voxel_size: Size of voxels for downsampling
    
    Returns:
        Downsampled point cloud with voxel center points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Apply voxel downsampling (returns voxel centers)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points)


def get_pose_center(pose: np.ndarray) -> np.ndarray:
    """
    Extract the center position from a 4x4 pose matrix.
    
    Args:
        pose: 4x4 pose matrix
    
    Returns:
        Center position (3,) as numpy array
    """
    return pose[:3, 3]


def filter_keyframes_by_motion(poses: List[np.ndarray], 
                              translation_threshold: float = 0.5,
                              rotation_threshold_degrees: float = 5.0,
                              min_frame_interval: int = 1) -> List[int]:
    """
    Filter poses to extract keyframes based on translation and rotation thresholds.
    
    This function identifies keyframes by comparing the motion between consecutive poses.
    A frame is selected as a keyframe if either:
    1. The translation distance from the last keyframe exceeds translation_threshold, OR
    2. The rotation angle from the last keyframe exceeds rotation_threshold_degrees
    
    Args:
        poses: List of 4x4 pose matrices
        translation_threshold: Minimum translation distance in meters to trigger keyframe selection (default: 0.5m)
        rotation_threshold_degrees: Minimum rotation angle in degrees to trigger keyframe selection (default: 5.0°)
        min_frame_interval: Minimum number of frames between keyframes (default: 1, meaning consecutive frames allowed)
    
    Returns:
        List of keyframe indices (0-based)
    """
    if poses is None or len(poses) == 0:
        return []
    
    if len(poses) == 1:
        return [0]
    
    # Convert rotation threshold from degrees to radians
    rotation_threshold_rad = np.radians(rotation_threshold_degrees)
    
    keyframe_indices = [0]  # Always include the first frame
    last_keyframe_pose = poses[0]
    last_keyframe_idx = 0
    
    for i in range(1, len(poses)):
        current_pose = poses[i]
        
        # Check minimum frame interval
        if i - last_keyframe_idx < min_frame_interval:
            continue
        
        # Calculate translation distance
        translation_distance = np.linalg.norm(
            get_pose_center(current_pose) - get_pose_center(last_keyframe_pose)
        )
        
        # Calculate rotation angle difference
        rotation_angle_rad = calculate_rotation_angle_between_poses(current_pose, last_keyframe_pose)
        
        # Check if either threshold is exceeded
        if (translation_distance >= translation_threshold or 
            rotation_angle_rad >= rotation_threshold_rad):
            keyframe_indices.append(i)
            last_keyframe_pose = current_pose
            last_keyframe_idx = i
    
    # Always include the last frame if it's not already included
    if keyframe_indices[-1] != len(poses) - 1:
        keyframe_indices.append(len(poses) - 1)
    
    logger.info(f"Selected {len(keyframe_indices)} keyframes from {len(poses)} total frames "
               f"(translation_threshold: {translation_threshold}m, "
               f"rotation_threshold: {rotation_threshold_degrees:.1f}°)")
    
    return keyframe_indices


def calculate_rotation_angle_between_poses(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """
    Calculate the rotation angle between two 4x4 pose matrices.
    
    Args:
        pose1: First 4x4 pose matrix
        pose2: Second 4x4 pose matrix
    
    Returns:
        Rotation angle in radians (0 to π)
    """
    # Extract rotation matrices
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Calculate relative rotation
    R_rel = R1.T @ R2
    
    # Calculate rotation angle using trace formula
    # For rotation matrix R, the angle θ satisfies: trace(R) = 1 + 2*cos(θ)
    trace = np.trace(R_rel)
    
    # Clamp trace to valid range [-1, 3] to handle numerical errors
    trace = np.clip(trace, -1.0, 3.0)
    
    # Calculate angle: cos(θ) = (trace - 1) / 2
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range for arccos
    
    angle = np.arccos(cos_angle)
    
    return angle


def filter_poses_and_data_by_keyframes(poses: List[np.ndarray],
                                      data_list: Optional[List] = None,
                                      translation_threshold: float = 0.5,
                                      rotation_threshold_degrees: float = 5.0,
                                      min_frame_interval: int = 1) -> Tuple[List[np.ndarray], List[int], Optional[List]]:
    """
    Filter poses and associated data by keyframes based on motion thresholds.
    
    This is a convenience function that combines keyframe filtering with data filtering.
    
    Args:
        poses: List of 4x4 pose matrices
        data_list: Optional list of data (e.g., point clouds, timestamps) to filter alongside poses
        translation_threshold: Minimum translation distance in meters to trigger keyframe selection
        rotation_threshold_degrees: Minimum rotation angle in degrees to trigger keyframe selection  
        min_frame_interval: Minimum number of frames between keyframes
    
    Returns:
        Tuple of (filtered_poses, keyframe_indices, filtered_data_list)
        - filtered_poses: List of keyframe poses
        - keyframe_indices: List of original indices of selected keyframes
        - filtered_data_list: Filtered data list (None if data_list was None)
    """
    keyframe_indices = filter_keyframes_by_motion(
        poses, translation_threshold, rotation_threshold_degrees, min_frame_interval
    )
    
    # Filter poses
    filtered_poses = [poses[i] for i in keyframe_indices]
    
    # Filter data if provided
    filtered_data_list = None
    if data_list is not None:
        if len(data_list) != len(poses):
            logger.warning(f"Data list length ({len(data_list)}) doesn't match poses length ({len(poses)})")
        else:
            filtered_data_list = [data_list[i] for i in keyframe_indices]
    
    return filtered_poses, keyframe_indices, filtered_data_list


def calculate_point_cloud_overlap_ratio_fast(points1: np.ndarray, 
                                            points2: np.ndarray, 
                                            voxel_size: float = 2.0,
                                            max_points_per_cloud: int = 20000) -> float:
    """
    Fast approximate overlap calculation using downsampling and voxelization.
    
    Args:
        points1: First point cloud (N, 3)
        points2: Second point cloud (M, 3)
        voxel_size: Size of voxels for discretization
        max_points_per_cloud: Maximum points to use per cloud for speed
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # Downsample if too many points
    if len(points1) > max_points_per_cloud:
        indices = np.random.choice(len(points1), max_points_per_cloud, replace=False)
        points1 = points1[indices]
    
    if len(points2) > max_points_per_cloud:
        indices = np.random.choice(len(points2), max_points_per_cloud, replace=False)
        points2 = points2[indices]
    
    # Voxelize both point clouds
    def voxelize_points(points):
        # Convert to voxel coordinates
        voxel_coords = np.floor(points / voxel_size).astype(int)
        # Create unique voxel keys
        voxel_keys = set(map(tuple, voxel_coords))
        return voxel_keys
    
    voxels1 = voxelize_points(points1)
    voxels2 = voxelize_points(points2)
    
    # Calculate intersection and union
    intersection = len(voxels1.intersection(voxels2))
    union = len(voxels1.union(voxels2))
    
    if union == 0:
        return 0.0
    
    overlap_ratio = intersection / union
    return overlap_ratio

def sample_truncated_gaussian(min_val: int, max_val: int, mean: float, std: float, max_attempts: int = 100) -> int:
    """
    Sample from a truncated Gaussian distribution bounded by min_val and max_val.
    
    Args:
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        mean: Mean of the Gaussian distribution
        std: Standard deviation of the Gaussian distribution
        max_attempts: Maximum attempts to find a valid sample
    
    Returns:
        Sampled integer value within [min_val, max_val]
    """
    for _ in range(max_attempts):
        # Sample from normal distribution
        sample = np.random.normal(mean, std)
        # Round to nearest integer
        sample_int = int(round(sample))
        
        # Check if within bounds
        if min_val <= sample_int <= max_val:
            return sample_int
    
    # If we can't find a valid sample, fall back to uniform distribution
    logger.warning(f"Could not sample from truncated Gaussian with mean={mean}, std={std}, bounds=[{min_val}, {max_val}]. Using uniform distribution.")
    import random
    return random.randint(min_val, max_val) 


def deskewing(
    points: np.ndarray, 
    ts: np.ndarray, 
    pose: np.ndarray, 
    ts_mid_pose: float = 0.5
) -> np.ndarray:
    """
    Deskew a batch of points at timestamp ts by a relative transformation matrix.
    PyTorch implementation using roma library for faster SLERP.
    
    Args:
        points: Point cloud with shape (N, 3) or (N, 4) if including intensity
        ts: Timestamps for each point, shape (N,) with values in range [0, 1]
        pose: Transformation matrix from current to last frame, shape (4, 4)
        ts_mid_pose: Timestamp corresponding to the pose (default: 0.5)
        
    Returns:
        Deskewed point cloud with same shape as input points
    """
    if ts is None:
        return points  # no deskewing

    try:
        import torch
        import roma
    except ImportError:
        raise ImportError("PyTorch and roma are required for fast deskewing. Install with: pip install torch roma")

    # Convert inputs to torch tensors
    points_torch = torch.from_numpy(points).float()
    ts_torch = torch.from_numpy(ts).float()
    pose_torch = torch.from_numpy(pose).float()
    
    # Ensure ts is 1D
    ts_torch = ts_torch.squeeze(-1)

    # Normalize the tensor to the range [0, 1]
    # NOTE: you need to figure out the begin and end of a frame because
    # sometimes there's only partial measurements, some part are blocked by some occlusions
    min_ts = torch.min(ts_torch)
    max_ts = torch.max(ts_torch)
    
    # Avoid division by zero
    if max_ts - min_ts > 1e-8:
        ts_torch = (ts_torch - min_ts) / (max_ts - min_ts)
    else:
        # If all timestamps are the same, set them to 0.5
        ts_torch = torch.full_like(ts_torch, 0.5)

    # this is related to: https://github.com/PRBonn/kiss-icp/issues/299
    ts_torch -= ts_mid_pose 

    # Use roma for fast SLERP
    rotmat_slerp = roma.rotmat_slerp(
        torch.eye(3).to(points_torch), pose_torch[:3, :3].to(points_torch), ts_torch
    )

    tran_lerp = ts_torch[:, None] * pose_torch[:3, 3].to(points_torch)

    points_deskewed_torch = points_torch.clone()
    points_deskewed_torch[:, :3] = (rotmat_slerp @ points_torch[:, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    # Convert back to numpy
    points_deskewed = points_deskewed_torch.cpu().numpy()
    
    return points_deskewed
    

def get_global_transformation_matrix(sequence_name: str) -> np.ndarray:
    """Get global transformation matrix based on sequence name."""
    if sequence_name.startswith('7-scenes') or \
       sequence_name.startswith('bundlefusion') or \
       sequence_name.startswith('rgbd-scenes') or \
       sequence_name.startswith('sun3d'): # for the test set
        # Transformation for 7-scenes, bundlefusion, rgbd-scenes
        return np.array([[0, 0, 1],
                         [-1, 0, 0],
                         [0, -1, 0]], dtype=np.float32)
    # elif sequence_name.startswith('sun3d'):
    #     # Transformation for sun3d
    #     return np.array([[0, 0, -1],
    #                      [1, 0, 0],
    #                      [0, 1, 0]], dtype=np.float32)
    elif sequence_name.startswith('analysis-by-synthesis'):
        # Identity transformation for analysis-by-synthesis
        return np.eye(3, dtype=np.float32)
    else:
        # Default to identity for other sequences
        return np.eye(3, dtype=np.float32) 


def save_num_points_to_folder(output_dir: str, 
                             dataset_name: str, 
                             sample_num_points: List[int], 
                             sample_dirs: List[Tuple[str, str]]):
    """Save num_points data to folder structure following precompute_num_points.py format."""
    num_points_dir = os.path.join(output_dir, "num_points")
    os.makedirs(num_points_dir, exist_ok=True)
    
    # For folder structure, we need to map samples to train/val splits
    # We'll try to read the data_split files to determine which samples belong to which split
    data_split_dir = os.path.join(output_dir, 'data_split')
    
    # Create sample path to num_points mapping
    sample_to_num_points = {}
    all_num_points = []
    
    def normalize_path(path: str) -> str:
        """Normalize path by removing dataset prefix and _processed suffix for matching."""
        # Remove dataset name prefix if present
        if path.startswith(f"{dataset_name}/"):
            path = path[len(f"{dataset_name}/"):]
        # Remove _processed suffix if present
        if path.endswith("_processed"):
            path = path[:-len("_processed")]
        return path
    
    for i, (relative_path, _) in enumerate(sample_dirs):
        num_points = sample_num_points[i] if i < len(sample_num_points) else 0
        normalized_path = normalize_path(relative_path)
        sample_to_num_points[normalized_path] = num_points
        all_num_points.append(num_points)
    
    # Create split-specific num_points lists
    split_num_points = {}
    
    # Read all split files and create corresponding num_points lists
    if os.path.exists(data_split_dir):
        for split_file_name in os.listdir(data_split_dir):
            if split_file_name.endswith('.txt'):
                split_name = split_file_name.replace('.txt', '')
                split_file_path = os.path.join(data_split_dir, split_file_name)
                
                with open(split_file_path, 'r') as f:
                    samples = [line.strip() for line in f if line.strip()]
                
                # Create num_points list for this split
                split_points = []
                for sample in samples:
                    normalized_sample = normalize_path(sample)
                    if normalized_sample in sample_to_num_points:
                        split_points.append(sample_to_num_points[normalized_sample])
                    else:
                        logger.warning(f"Sample {sample} (normalized: {normalized_sample}) not found in processed data for {split_name} split")
                        split_points.append(0)
                
                if split_points:
                    split_num_points[split_name] = split_points
    
    # Always include 'all' split
    split_num_points['all'] = all_num_points
    
    # Save all split files
    for split_name, num_points_data in split_num_points.items():
        if num_points_data:
            split_file = os.path.join(num_points_dir, f'{split_name}.txt')
            with open(split_file, 'w') as f:
                for num_point in num_points_data:
                    f.write(f"{num_point}\n")
            logger.info(f"Saved num_points for {split_name} split: {split_file} ({len(num_points_data)} samples)")
            logger.debug(f"{split_name} num_points stats: min={min(num_points_data)}, max={max(num_points_data)}, mean={np.mean(num_points_data):.1f}") 

def save_points_to_ply(points: np.ndarray, ply_path: str, normals: Optional[np.ndarray] = None) -> None:
    """Save points to a PLY file using Open3D."""
    if o3d is None:
        raise ImportError("open3d is required. Install with: pip install open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64, copy=False))
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False) 

