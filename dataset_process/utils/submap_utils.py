#!/usr/bin/env python3
"""
Submap Utilities for Training Sample Generation

This module contains functions for creating, validating, and selecting submaps
for training sample generation.
"""

import numpy as np
import random
import logging
from typing import List, Tuple, Optional, Dict
from . import dataset_utils

logger = logging.getLogger(__name__)

def get_default_num_samples(sequence: str, frame_count: int, data_loader, sample_count_multiplier: float = 0.1) -> int:
    """Automatically determine the default number of samples (K) for a sequence."""
    has_loop = sequence in data_loader.get_loop_closure_sequences()
    base_multiplier = sample_count_multiplier if has_loop else sample_count_multiplier / 2
    calculated_samples = int(frame_count * base_multiplier)
    
    logger.info(f"Sequence {sequence}: {frame_count} frames, {'has' if has_loop else 'no'} loop closure -> {calculated_samples} samples")
    return calculated_samples

def create_submap_from_frames(points_list: List[np.ndarray], 
                             poses_list: List[np.ndarray], 
                             start_idx: int, 
                             num_frames: int,
                             normals_list: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create a submap from multiple adjacent frames, including point normals if available."""
    all_points, all_normals = [], []
    
    for i in range(start_idx, min(start_idx + num_frames, len(points_list))):
        # Transform points to world coordinates
        world_points = dataset_utils.transform_points(points_list[i], poses_list[i]) if i < len(poses_list) else points_list[i]
        all_points.append(world_points)
        
        # Handle normals if available
        if normals_list and i < len(normals_list) and normals_list[i] is not None:
            world_normals = dataset_utils.transform_normals(normals_list[i], poses_list[i]) if i < len(poses_list) else normals_list[i]
            all_normals.append(world_normals)
    
    if not all_points:
        return np.array([]), None
    
    combined_points = np.vstack(all_points)
    combined_normals = np.vstack(all_normals) if all_normals else None
    
    return combined_points, combined_normals

def check_submap_validity_fast(selected_indices: List[int],
                              submap_boundaries: List[Tuple],
                              submap_centers: List[np.ndarray],
                              frame_ids: List,
                              min_spatial_threshold: float,
                              max_spatial_threshold: float,
                              min_frame_interval: int = 0,
                              max_frame_interval: Optional[int] = None) -> bool:
    """
    Check fast validity criteria (frame interval and spatial distance).
    These checks are fast and should not count as attempts.
    """
    n = len(selected_indices)
    
    # Check frame interval criteria (fast operation, not counted as attempt)
    if min_frame_interval > 0 or max_frame_interval is not None:
        for i in range(n):
            for j in range(i + 1, n):
                start1, _ = submap_boundaries[selected_indices[i]]
                start2, _ = submap_boundaries[selected_indices[j]]
                
                # Handle both integer and string frame IDs
                try:
                    # Try to convert to integers for interval calculation
                    start1_int = int(start1) if isinstance(start1, str) else start1
                    start2_int = int(start2) if isinstance(start2, str) else start2
                    frame_interval = abs(start1_int - start2_int)
                except (ValueError, TypeError):
                    # If conversion fails, skip frame interval check for string frame IDs
                    frame_interval = float('inf')
                
                if min_frame_interval > 0 and frame_interval < min_frame_interval:
                    # Frame interval check is fast, so we don't count it as an attempt
                    logger.debug(f"Frame interval {frame_interval} < {min_frame_interval} [submap {selected_indices[i]} - submap {selected_indices[j]}]")
                    return False
                
                if max_frame_interval is not None and frame_interval > max_frame_interval:
                    # Frame interval check is fast, so we don't count it as an attempt
                    logger.debug(f"Frame interval {frame_interval} > {max_frame_interval} [submap {selected_indices[i]} - submap {selected_indices[j]}]")
                    return False
    
    # Check spatial distances (fast operation, not counted as attempt)
    for i in range(n):
        for j in range(i + 1, n):
            spatial_dist = np.linalg.norm(submap_centers[selected_indices[i]] - submap_centers[selected_indices[j]])
            if not (min_spatial_threshold <= spatial_dist <= max_spatial_threshold):
                return False
    
    return True

def check_submap_validity(selected_indices: List[int],
                          submap_boundaries: List[Tuple],
                          submap_centers: List[np.ndarray],
                          points_list: List[np.ndarray],
                          poses: List[np.ndarray],
                          frame_ids: List,
                          min_spatial_threshold: float,
                          max_spatial_threshold: float,
                          min_overlap_ratio: float,
                          max_overlap_ratio: float,
                          overlap_method: str,
                          min_frame_interval: int = 0,
                          overlap_voxel_size: float = 2.0,
                          attempt: int = -1) -> bool:
    """
    Check if a set of selected submaps meets all validity criteria.
    
    Note: Only the overlap calculation (expensive operation) is counted as an attempt.
    Frame interval and spatial distance checks are fast and not counted.
    """
    n = len(selected_indices)
    
    # Check overlap using Union-Find (expensive operation, counted as attempt)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check all pairs for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            idx1, idx2 = selected_indices[i], selected_indices[j]
            start_frame_id1, end_frame_id1 = submap_boundaries[idx1]
            start_frame_id2, end_frame_id2 = submap_boundaries[idx2]
            
            # Convert frame IDs to array indices
            start_idx1 = frame_ids.index(start_frame_id1)
            end_idx1 = frame_ids.index(end_frame_id1) + 1  # end_frame_id is inclusive
            start_idx2 = frame_ids.index(start_frame_id2)
            end_idx2 = frame_ids.index(end_frame_id2) + 1  # end_frame_id is inclusive
            
            # Create submaps and calculate overlap
            submap1, _ = create_submap_from_frames(points_list, poses, start_idx1, end_idx1 - start_idx1)
            submap2, _ = create_submap_from_frames(points_list, poses, start_idx2, end_idx2 - start_idx2)
            
            overlap_ratio = dataset_utils.calculate_point_cloud_overlap_ratio_fast(submap1, submap2, voxel_size=overlap_voxel_size)
            
            if attempt >= 0:
                logger.debug(f"overlap_ratio for attempt {attempt} [submap {idx1} - submap {idx2}]: {overlap_ratio}")
            
            if min_overlap_ratio <= overlap_ratio <= max_overlap_ratio:
                union(i, j)
    
    # Check if all submaps are connected
    root = find(0)
    return all(find(i) == root for i in range(n))

def generate_submap_boundaries_for_sample(frame_ids: List,
                                         min_frames_per_submap: int,
                                         max_frames_per_submap: int,
                                         random_drop_to_single_frame: bool = False) -> List[Tuple]:
    """
    Generate submap boundaries for a single sample by randomly selecting frame ranges.
    Each submap is created from consecutive frames with no overlap within the sample.
    Uses frame_ids instead of array indices for boundaries.
    
    Args:
        frame_ids: List of frame IDs in chronological order
        min_frames_per_submap: Minimum number of frames per submap
        max_frames_per_submap: Maximum number of frames per submap
        random_drop_to_single_frame: If True, randomly select one submap and reduce it to a single frame
        
    Returns:
        List of (start_frame_id, end_frame_id) tuples for each submap
    """
    submap_boundaries = []
    start_idx = 0
    
    # Calculate mean and std for truncated Gaussian distribution
    # Bias toward minimum value but allow reaching maximum
    mean = min_frames_per_submap + (max_frames_per_submap - min_frames_per_submap) * 0.2  # 20% toward min
    std = (max_frames_per_submap - min_frames_per_submap) * 0.35  # 35% of range as std
    
    # Generate submap boundaries as usual
    while start_idx < len(frame_ids):
        # Use truncated Gaussian to select number of frames for this submap
        frames_this_submap = dataset_utils.sample_truncated_gaussian(
            min_frames_per_submap, 
            max_frames_per_submap, 
            mean, 
            std
        )
        
        end_idx = min(start_idx + frames_this_submap, len(frame_ids))
        
        # Use frame IDs instead of array indices
        start_frame_id = frame_ids[start_idx]
        end_frame_id = frame_ids[end_idx - 1]  # end_idx is exclusive, so use end_idx-1
        
        submap_boundaries.append((start_frame_id, end_frame_id))
        start_idx = end_idx
    
    # If random_drop_to_single_frame is enabled, randomly select one submap and reduce it to a single frame
    if random_drop_to_single_frame and len(submap_boundaries) > 0:
        # Randomly select one submap index
        selected_submap_idx = random.randint(0, len(submap_boundaries) - 1)
        start_frame_id, end_frame_id = submap_boundaries[selected_submap_idx]
        
        # Find the array indices for this submap
        start_array_idx = frame_ids.index(start_frame_id)
        end_array_idx = frame_ids.index(end_frame_id) + 1  # end_frame_id is inclusive
        
        # Randomly select one frame from this submap
        selected_frame_idx = random.randint(start_array_idx, end_array_idx - 1)
        selected_frame_id = frame_ids[selected_frame_idx]
        
        # Update the submap boundary to use only the selected frame
        submap_boundaries[selected_submap_idx] = (selected_frame_id, selected_frame_id)
    
    return submap_boundaries

def select_spatially_close_submaps(submap_boundaries: List[Tuple],
                                  submap_centers: List[np.ndarray],
                                  poses: List[np.ndarray],
                                  points_list: List[np.ndarray],
                                  frame_ids: List,
                                  min_spatial_threshold: float,
                                  max_spatial_threshold: float,
                                  min_submaps_per_sample: int,
                                  max_submaps_per_sample: int,
                                  min_overlap_ratio: float = 0.01,
                                  max_overlap_ratio: float = 0.8,
                                  overlap_method: str = "fast",
                                  min_frame_interval: int = 0,
                                  max_frame_interval: Optional[int] = None,
                                  overlap_voxel_size: float = 2.0,
                                  max_attempts: int = 50) -> List[int]:
    """From a list of submaps, select a random subset that are spatially close."""
    num_submaps = len(submap_boundaries)
    
    if num_submaps < min_submaps_per_sample:
        return []
    
    # Try different numbers of submaps, starting from a random number K between min and max, then K-1, K-2, etc.
    max_possible_submaps = min(max_submaps_per_sample, num_submaps)
    
    # Pick a random starting number K between min and max
    random_start = random.randint(min_submaps_per_sample, max_possible_submaps)
    
    # Try from random_start down to min_submaps_per_sample
    for target_num in range(random_start, min_submaps_per_sample - 1, -1):
        logger.debug(f"Trying with {target_num} submaps")
        
        for attempt in range(max_attempts):
            selected_indices = random.sample(range(num_submaps), target_num)
            
            # First do fast checks (frame interval and spatial distance) - these don't count as attempts
            if not check_submap_validity_fast(selected_indices, submap_boundaries, submap_centers, 
                                            frame_ids, min_spatial_threshold, max_spatial_threshold, 
                                            min_frame_interval, max_frame_interval):
                continue
            
            # Then do expensive overlap check - this counts as an attempt
            if check_submap_validity(selected_indices, submap_boundaries, submap_centers, 
                                   points_list, poses, frame_ids, min_spatial_threshold, max_spatial_threshold,
                                   min_overlap_ratio, max_overlap_ratio, overlap_method, 
                                   min_frame_interval, overlap_voxel_size, attempt if target_num == random_start else -1):
                return selected_indices
    
    return []

def validate_no_overlap(submap_meta: List[Dict]) -> bool:
    """Validate that submaps in a group don't have overlapping frames."""
    for i, meta_i in enumerate(submap_meta):
        for j, meta_j in enumerate(submap_meta[i+1:], i+1):
            start_i, end_i = meta_i['start_frame'], meta_i['end_frame']
            start_j, end_j = meta_j['start_frame'], meta_j['end_frame']
            
            # Handle both integer and string frame IDs
            try:
                # Try to convert to integers for comparison
                start_i_int = int(start_i) if isinstance(start_i, str) else start_i
                end_i_int = int(end_i) if isinstance(end_i, str) else end_i
                start_j_int = int(start_j) if isinstance(start_j, str) else start_j
                end_j_int = int(end_j) if isinstance(end_j, str) else end_j
                
                if not (end_i_int < start_j_int or end_j_int < start_i_int):
                    logger.warning(f"Overlap detected: submap {i} ({start_i}-{end_i}) and submap {j} ({start_j}-{end_j})")
                    return False
            except (ValueError, TypeError):
                # If conversion fails, skip overlap validation for string frame IDs
                # This is acceptable for datasets like NCLT where frame IDs are timestamps
                pass
    
    return True 