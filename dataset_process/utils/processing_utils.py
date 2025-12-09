#!/usr/bin/env python3
"""
Processing Utilities for Training Sample Generation

This module contains functions for processing different types of datasets,
including NSS, sequence-based processing, and validation functions.
"""

import os
import numpy as np
import open3d as o3d
import json
import logging
import random
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from . import dataset_utils
from .dataset_utils import downsample_points
from .submap_utils import (
    create_submap_from_frames, 
    generate_submap_boundaries_for_sample,
    select_spatially_close_submaps,
    validate_no_overlap
)
import torch
from .io_utils import save_training_sample

logger = logging.getLogger(__name__)

def _load_threedmatch_gt_log(gt_path: str) -> Dict[str, np.ndarray]:
    """
    Load ground truth transformation log file for ThreeDMatch test dataset.
    This is a copy of the loadlog function from threedmatch_test.py to avoid import issues.
    """
    log_file = os.path.join(gt_path, 'gt.log')
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Ground truth log file not found: {log_file}")
    
    with open(log_file) as f:
        content = f.readlines()
    
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans
    
    return result

def _build_threedmatch_transformation_graph(sequence: str, data_loader, benchmark: str = "3DMatch") -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build a transformation graph from ThreeDMatch test ground truth pairs.
    
    Returns a dictionary mapping (src_id, tgt_id) -> transformation matrix.
    """
    # Get ground truth file path
    if hasattr(data_loader, 'data_root'):
        data_root = data_loader.data_root
    elif hasattr(data_loader, 'threedmatch_loader'):
        data_root = data_loader.threedmatch_loader.data_root
    else:
        logger.warning("Cannot determine data_root for ThreeDMatch test, returning empty transformation graph")
        return {}
    
    if benchmark == "3DMatch":
        gt_path = os.path.join(data_root, "test", "3DMatch", "gt_result", sequence)
    elif benchmark == "3DLoMatch":
        gt_path = os.path.join(data_root, "test", "3DLoMatch", sequence)
    else:
        logger.warning(f"Unknown benchmark: {benchmark}, returning empty transformation graph")
        return {}
    
    try:
        gt_log = _load_threedmatch_gt_log(gt_path)
    except Exception as e:
        logger.warning(f"Failed to load ground truth for sequence {sequence}: {e}")
        return {}
    
    # Build transformation graph: (src_id, tgt_id) -> transformation
    transform_graph = {}
    for key, transformation in gt_log.items():
        id1, id2 = key.split("_")
        src_id = f"cloud_bin_{id1}"
        tgt_id = f"cloud_bin_{id2}"
        transform_graph[(src_id, tgt_id)] = transformation
        # Also store reverse transformation
        transform_graph[(tgt_id, src_id)] = np.linalg.inv(transformation)
    
    logger.debug(f"Built transformation graph with {len(gt_log)} pairs for sequence {sequence}")
    return transform_graph

def _find_transformation_path(src_fragment: str, tgt_fragment: str, transform_graph: Dict[Tuple[str, str], np.ndarray], max_depth: int = 3) -> Optional[np.ndarray]:
    """
    Find a transformation path from src_fragment to tgt_fragment using BFS.
    Returns the combined transformation matrix or None if no path found.
    """
    if src_fragment == tgt_fragment:
        return np.eye(4, dtype=np.float32)
    
    # BFS to find shortest path
    from collections import deque
    queue = deque([(src_fragment, np.eye(4, dtype=np.float32))])
    visited = {src_fragment}
    
    for depth in range(max_depth):
        next_queue = deque()
        while queue:
            current_fragment, current_transform = queue.popleft()
            
            # Check all neighbors
            for (src_id, tgt_id), transform in transform_graph.items():
                if src_id == current_fragment and tgt_id not in visited:
                    new_transform = current_transform @ transform
                    if tgt_id == tgt_fragment:
                        return new_transform
                    visited.add(tgt_id)
                    next_queue.append((tgt_id, new_transform))
        
        queue = next_queue
        if not queue:
            break
    
    return None

def _transform_threedmatch_submaps_to_common_coordinate(submaps: List[np.ndarray], 
                                                       submap_meta: List[Dict],
                                                       frame_ids: List[str],
                                                       sequence: str,
                                                       data_loader,
                                                       benchmark: str = "3DMatch",
                                                       submap_normals: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], List[Dict]]:
    """
    Transform all ThreeDMatch test submaps to the first submap's coordinate system.
    Uses optimized poses from pose graph optimization directly.
    Removes fragments that don't have optimized poses.
    
    Args:
        submaps: List of submap point clouds
        submap_meta: List of submap metadata
        frame_ids: List of frame IDs (fragment names)
        sequence: Sequence name
        data_loader: Data loader instance (may have threedmatch_loader attribute or be the loader itself)
        benchmark: Benchmark type
        submap_normals: Optional list of normals
        
    Returns:
        Tuple of (transformed submaps, transformed normals, filtered submap_meta)
    """
    if len(submaps) <= 1:
        return submaps, submap_normals, submap_meta
    
    # Get the actual ThreeDMatch loader (might be wrapped)
    threedmatch_loader = data_loader
    if hasattr(data_loader, 'threedmatch_loader'):
        threedmatch_loader = data_loader.threedmatch_loader
    
    # Extract fragment names from frame IDs
    # Frame IDs are in format: "{sequence}_{fragment_name}"
    fragment_names = []
    for meta in submap_meta:
        # Get the fragment name from the random_frame_id or start_frame
        frame_id = meta.get('random_frame_id') or meta.get('start_frame')
        if isinstance(frame_id, str):
            # Extract fragment name (e.g., "cloud_bin_0" from "sequence_cloud_bin_0")
            # Frame ID format: "{sequence}_{fragment_name}"
            parts = frame_id.split('_', 1)  # Split only on first underscore
            if len(parts) == 2:
                fragment_name = parts[1]  # Get everything after sequence name
            else:
                # Fallback: try to extract cloud_bin_X pattern
                parts = frame_id.split('_')
                if len(parts) >= 3 and 'bin' in parts:
                    # Find 'bin' index and reconstruct
                    bin_idx = parts.index('bin')
                    if bin_idx > 0 and bin_idx < len(parts) - 1:
                        fragment_name = f"{parts[bin_idx-1]}_{parts[bin_idx]}_{parts[bin_idx+1]}"
                    else:
                        fragment_name = frame_id
                else:
                    fragment_name = frame_id
        else:
            fragment_name = str(frame_id)
        fragment_names.append(fragment_name)
    
    # Get optimized poses directly from ThreeDMatch loader
    if not hasattr(threedmatch_loader, '_optimized_poses') or not threedmatch_loader._optimized_poses:
        logger.warning(f"No optimized poses available in data loader for sequence {sequence}, skipping coordinate alignment")
        return submaps, submap_normals, submap_meta
    
    optimized_poses = threedmatch_loader._optimized_poses
    
    # Filter out fragments without optimized poses
    valid_indices = []
    for i, fragment_name in enumerate(fragment_names):
        if fragment_name in optimized_poses:
            valid_indices.append(i)
        else:
            logger.warning(f"Fragment {fragment_name} not found in optimized poses, removing from sample")
    
    if len(valid_indices) == 0:
        logger.error(f"No fragments with optimized poses found for sequence {sequence}")
        return [], [], []
    
    if len(valid_indices) < len(submaps):
        logger.info(f"Filtered {len(submaps) - len(valid_indices)} fragments without optimized poses, keeping {len(valid_indices)} fragments")
    
    # Use first valid submap as reference
    ref_idx = valid_indices[0]
    ref_fragment = fragment_names[ref_idx]
    ref_pose = optimized_poses[ref_fragment]
    
    transformed_submaps = [submaps[ref_idx].copy()]
    transformed_normals = [submap_normals[ref_idx].copy() if submap_normals and submap_normals[ref_idx] is not None else None]
    filtered_meta = [submap_meta[ref_idx].copy()]
    
    # Transform all other valid submaps to reference coordinate system
    for i in valid_indices[1:]:
        fragment_name = fragment_names[i]
        fragment_pose = optimized_poses[fragment_name]
        
        # Compute transformation from fragment to reference coordinate system
        # The optimized poses transform points FROM fragment TO world (anchor frame): point_world = pose @ point_fragment
        # So: ref_pose transforms ref -> world, fragment_pose transforms fragment -> world
        # To transform points FROM fragment TO ref:
        #   point_world = fragment_pose @ point_fragment
        #   point_world = ref_pose @ point_ref
        #   Therefore: ref_pose @ point_ref = fragment_pose @ point_fragment
        #   So: point_ref = inv(ref_pose) @ fragment_pose @ point_fragment
        #   Therefore: T_fragment_to_ref = inv(ref_pose) @ fragment_pose
        transform = np.linalg.inv(ref_pose) @ fragment_pose
        
        # Transform points
        transformed_points = dataset_utils.transform_points(submaps[i], transform)
        transformed_submaps.append(transformed_points)
        
        # Transform normals if available
        if submap_normals and submap_normals[i] is not None:
            transformed_normal = dataset_utils.transform_normals(submap_normals[i], transform)
            transformed_normals.append(transformed_normal)
        else:
            transformed_normals.append(None)
        
        filtered_meta.append(submap_meta[i].copy())
        logger.debug(f"Transformed fragment {fragment_name} to reference {ref_fragment} coordinate system using optimized poses")
    
    return transformed_submaps, transformed_normals, filtered_meta

def _calculate_statistics(submap_counts, submap_frame_counts, temporal_differences, spatial_differences):
    """Calculate statistics for the generated samples."""
    def safe_stats(data_list, prefix):
        if not data_list:
            return {f'{prefix}_mean': 0, f'{prefix}_std': 0, f'{prefix}_min': 0, f'{prefix}_max': 0}
        return {
            f'{prefix}_mean': np.mean(data_list),
            f'{prefix}_std': np.std(data_list),
            f'{prefix}_min': min(data_list),
            f'{prefix}_max': max(data_list)
        }
    
    stats = {'num_samples': len(submap_counts)}
    stats.update(safe_stats(submap_counts, 'submap_count'))
    stats.update(safe_stats(submap_frame_counts, 'frames_per_submap'))
    stats.update(safe_stats(temporal_differences, 'temporal_difference'))
    stats.update(safe_stats(spatial_differences, 'spatial_difference'))
    
    # Add distributions
    stats['submap_count_distribution'] = {str(count): submap_counts.count(count) for count in set(submap_counts)}
    stats['frames_per_submap_distribution'] = {str(count): submap_frame_counts.count(count) for count in set(submap_frame_counts)}
    
    return stats

def process_nss_dataset(data_loader,
                       output_dir: str,
                       annotation_split: str = 'original',
                       split_type: str = 'train',
                       max_samples: Optional[int] = None,
                       voxel_size: float = 0.1,
                       downsample_method: str = "voxel",
                       num_points_downsample: Optional[int] = None,
                       min_overlap_ratio: float = 0.1,
                       max_overlap_ratio: float = 0.8,
                       filter_by_building: Optional[List[int]] = None,
                       filter_by_stage: Optional[List[int]] = None,
                       same_stage_only: bool = False,
                       cross_stage_only: bool = False) -> Tuple[int, Dict]:
    """
    Process NSS dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: NSS data loader instance
        output_dir: Output directory for training samples
        annotation_split: NSS annotation split ('original', 'cross_area', 'cross_stage')
        split_type: Split type ('train', 'val')
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter
        max_overlap_ratio: Maximum overlap ratio filter
        filter_by_building: List of building IDs to include (None for all)
        filter_by_stage: List of stage IDs to include (None for all)
        same_stage_only: Only include same-stage pairs
        cross_stage_only: Only include cross-stage pairs
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing NSS dataset directly (annotation_split={annotation_split}, split_type={split_type})")
    
    # Initialize NSS data loader with specified parameters
    if hasattr(data_loader, 'nss_loader'):
        # If we're using the sequence interface, get the underlying NSS loader
        nss_loader = data_loader.nss_loader
    else:
        # Create a new NSS loader directly
        from ..data_loaders import NSSDataLoader
        nss_loader = NSSDataLoader(
            data_root=data_loader.data_root if hasattr(data_loader, 'data_root') else './dataset/NSS',
            annotation_split=annotation_split,
            split_type=split_type,
            random_downsample=getattr(data_loader, 'random_downsample', False),
            max_points_per_frame=getattr(data_loader, 'max_points_per_frame', 10000),
            estimate_normals=getattr(data_loader, 'estimate_normals', False)
        )
    
    # Set the sequence (split_type) in the NSS loader
    nss_loader.set_sequence(split_type)
    
    logger.info(f"NSS dataset loaded: {len(nss_loader)} pairs available")
    
    # Apply filters to get valid pair indices
    valid_indices = list(range(len(nss_loader)))
    
    # Filter by overlap ratio
    if min_overlap_ratio > 0 or max_overlap_ratio < 1.0:
        logger.info(f"Filtering by overlap ratio: {min_overlap_ratio:.3f} - {max_overlap_ratio:.3f}")
        overlap_filtered = []
        for idx in valid_indices:
            pair_data = nss_loader[idx]
            overlap = pair_data['overlap']
            if min_overlap_ratio <= overlap <= max_overlap_ratio:
                overlap_filtered.append(idx)
        valid_indices = overlap_filtered
        logger.info(f"After overlap filtering: {len(valid_indices)} pairs")
    
    # Filter by building
    if filter_by_building is not None:
        logger.info(f"Filtering by buildings: {filter_by_building}")
        building_filtered = []
        for idx in valid_indices:
            pair_data = nss_loader[idx]
            if pair_data['building'] in filter_by_building:
                building_filtered.append(idx)
        valid_indices = building_filtered
        logger.info(f"After building filtering: {len(valid_indices)} pairs")
    
    # Filter by stage
    if filter_by_stage is not None:
        logger.info(f"Filtering by stages: {filter_by_stage}")
        stage_filtered = []
        for idx in valid_indices:
            pair_data = nss_loader[idx]
            if (pair_data['source_stage'] in filter_by_stage or 
                pair_data['target_stage'] in filter_by_stage):
                stage_filtered.append(idx)
        valid_indices = stage_filtered
        logger.info(f"After stage filtering: {len(valid_indices)} pairs")
    
    # Filter by same/cross stage
    if same_stage_only:
        logger.info("Filtering for same-stage pairs only")
        same_stage_indices = nss_loader.get_same_stage_pairs()
        valid_indices = [idx for idx in valid_indices if idx in same_stage_indices]
        logger.info(f"After same-stage filtering: {len(valid_indices)} pairs")
    elif cross_stage_only:
        logger.info("Filtering for cross-stage pairs only")
        cross_stage_indices = nss_loader.get_cross_stage_pairs()
        valid_indices = [idx for idx in valid_indices if idx in cross_stage_indices]
        logger.info(f"After cross-stage filtering: {len(valid_indices)} pairs")
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} NSS pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    overlap_values = []
    temporal_change_ratios = []
    curvature_values = []
    building_counts = {}
    stage_combinations = {}
    same_stage_count = 0
    cross_stage_count = 0
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing NSS pairs")):
        try:
            # Get pair data
            pair_data = nss_loader[pair_idx]
            
            # Create sample directory (directly in output_dir for NSS)
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract spot names from filenames
            source_filename = pair_data['source_file']
            target_filename = pair_data['target_file']
            
            # Extract spot names (e.g., "Spot148" from "Bldg2_Stage2_Spot148.450.ply")
            def extract_spot_name(filename):
                import re
                match = re.search(r'Spot(\d+)', filename)
                return f"Spot{match.group(1)}" if match else "SpotUnknown"
            
            source_spot = extract_spot_name(source_filename)
            target_spot = extract_spot_name(target_filename)
            
            # Process source point cloud
            source_points = pair_data['source_points']
            source_normals = pair_data['source_normals']
            
            # Process target point cloud  
            target_points = pair_data['target_points']
            target_normals = pair_data['target_normals']
            
            # Apply transformation to align source to target coordinate system
            # The transformation matrix transforms source points to target coordinate system
            transformation_matrix = pair_data['transformation']
            
            # Transform source points to target coordinate system
            if len(source_points) > 0:
                # Convert to homogeneous coordinates
                source_homogeneous = np.hstack([source_points, np.ones((len(source_points), 1))])
                # Apply transformation
                source_transformed_homogeneous = (transformation_matrix @ source_homogeneous.T).T
                # Convert back to 3D coordinates
                source_points_aligned = source_transformed_homogeneous[:, :3]
                
                # Transform normals if available (only rotation part)
                if source_normals is not None:
                    rotation_matrix = transformation_matrix[:3, :3]
                    source_normals_aligned = (rotation_matrix @ source_normals.T).T
                else:
                    source_normals_aligned = None
            else:
                source_points_aligned = source_points
                source_normals_aligned = source_normals
            
            logger.debug(f"Applied transformation to align source {source_spot} to target {target_spot}")
            
            # For NSS dataset, preserve original point cloud resolution (no downsampling)
            source_downsampled = source_points_aligned
            source_normals_downsampled = source_normals_aligned
            target_downsampled = target_points
            target_normals_downsampled = target_normals
            
            logger.debug(f"NSS: Preserved original resolution - Source: {len(source_downsampled)} points, Target: {len(target_downsampled)} points")
            
            # Save source point cloud (now aligned to target coordinate system)
            source_filename = f"sample_{sample_idx:06d}_source_Bldg{pair_data['building']}_Stage{pair_data['source_stage']}_{source_spot}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled)
            if source_normals_downsampled is not None:
                source_pcd.normals = o3d.utility.Vector3dVector(source_normals_downsampled)
            o3d.io.write_point_cloud(source_filepath, source_pcd, write_ascii=False)
            
            # Save target point cloud
            target_filename = f"sample_{sample_idx:06d}_target_Bldg{pair_data['building']}_Stage{pair_data['target_stage']}_{target_spot}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled)
            if target_normals_downsampled is not None:
                target_pcd.normals = o3d.utility.Vector3dVector(target_normals_downsampled)
            o3d.io.write_point_cloud(target_filepath, target_pcd, write_ascii=False)
            
            # Note: No transformation files saved for NSS since point clouds are already aligned
            
            # Save metadata
            metadata = {
                'pair_id': pair_idx,
                'sample_id': sample_idx,
                'building': pair_data['building'],
                'source_stage': pair_data['source_stage'],
                'target_stage': pair_data['target_stage'],
                'source_spot': source_spot,
                'target_spot': target_spot,
                'same_stage': pair_data['same_stage'],
                'overlap': pair_data['overlap'],
                'temporal_change_ratio': pair_data['temporal_change_ratio'],
                'curvature': pair_data['curvature'],
                'source_file': pair_data['source_file'],
                'target_file': pair_data['target_file'],
                'source_points_original': len(source_points),
                'target_points_original': len(target_points),
                'source_points_final': len(source_downsampled),
                'target_points_final': len(target_downsampled),
                'transformation_applied': True,
                'coordinate_system': 'target_aligned',  # Source transformed to target coordinate system
                'downsampling_applied': False,  # NSS preserves original resolution
                'downsample_method': 'none',
                'voxel_size': None,
                'num_points_target': None
            }
            
            metadata_filepath = os.path.join(sample_dir, f"metadata_sample_{sample_idx:06d}.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Collect statistics
            overlap_values.append(pair_data['overlap'])
            temporal_change_ratios.append(pair_data['temporal_change_ratio'])
            curvature_values.append(pair_data['curvature'])
            
            building = pair_data['building']
            building_counts[building] = building_counts.get(building, 0) + 1
            
            stage_combo = (pair_data['source_stage'], pair_data['target_stage'])
            stage_combinations[stage_combo] = stage_combinations.get(stage_combo, 0) + 1
            
            if pair_data['same_stage']:
                same_stage_count += 1
            else:
                cross_stage_count += 1
            
            logger.debug(f"Processed sample {sample_idx}: {pair_data['source_file']} -> {pair_data['target_file']}")
            
        except Exception as e:
            logger.error(f"Error processing pair {pair_idx}: {e}")
            continue
    
    # Calculate statistics
    num_samples_generated = len(valid_indices)
    
    stats = {
        'num_samples': num_samples_generated,
        'annotation_split': annotation_split,
        'split_type': split_type,
        'overlap': {
            'mean': np.mean(overlap_values) if overlap_values else 0,
            'std': np.std(overlap_values) if overlap_values else 0,
            'min': min(overlap_values) if overlap_values else 0,
            'max': max(overlap_values) if overlap_values else 0
        },
        'temporal_change_ratio': {
            'mean': np.mean(temporal_change_ratios) if temporal_change_ratios else 0,
            'std': np.std(temporal_change_ratios) if temporal_change_ratios else 0,
            'min': min(temporal_change_ratios) if temporal_change_ratios else 0,
            'max': max(temporal_change_ratios) if temporal_change_ratios else 0
        },
        'curvature': {
            'mean': np.mean(curvature_values) if curvature_values else 0,
            'std': np.std(curvature_values) if curvature_values else 0,
            'min': min(curvature_values) if curvature_values else 0,
            'max': max(curvature_values) if curvature_values else 0
        },
        'building_distribution': building_counts,
        'stage_combinations': {f"{s[0]}->{s[1]}": count for s, count in stage_combinations.items()},
        'same_stage_pairs': same_stage_count,
        'cross_stage_pairs': cross_stage_count,
        'processing_method': 'direct_pairs',
        'downsampling_applied': False,
        'downsample_method': 'none',
        'voxel_size': None,
        'num_points_downsample': None
    }
    
    logger.info(f"NSS processing complete: {num_samples_generated} samples generated")
    logger.info(f"Same-stage pairs: {same_stage_count}, Cross-stage pairs: {cross_stage_count}")
    logger.info(f"Average overlap: {stats['overlap']['mean']:.3f} ± {stats['overlap']['std']:.3f}")
    logger.info(f"Building distribution: {building_counts}")
    
    return num_samples_generated, stats


def process_threedmatch_test_dataset(data_loader,
                                   output_dir: str,
                                   sequence_name: str,
                                   benchmark: str = "3DMatch",
                                   max_samples: Optional[int] = None,
                                   voxel_size: float = 0.05,
                                   downsample_method: str = "voxel",
                                   num_points_downsample: Optional[int] = None,
                                   min_overlap_ratio: float = 0.0,
                                   max_overlap_ratio: float = 1.0) -> Tuple[int, Dict]:
    """
    Process ThreeDMatch test dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: ThreeDMatch test data loader instance
        output_dir: Output directory for training samples
        sequence_name: Name of the test sequence
        benchmark: Benchmark type ("3DMatch" or "3DLoMatch")
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter (not used for ThreeDMatch)
        max_overlap_ratio: Maximum overlap ratio filter (not used for ThreeDMatch)
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing ThreeDMatch test dataset directly (sequence={sequence_name}, benchmark={benchmark})")
    
    # Get the underlying ThreeDMatch loader
    if hasattr(data_loader, 'threedmatch_loader'):
        # If we're using the sequence interface, get the underlying ThreeDMatch loader
        threedmatch_loader = data_loader.threedmatch_loader
    else:
        # Use the data loader directly if it's already a ThreeDMatch loader
        from ..data_loaders import ThreeDMatchTestDataLoader
        threedmatch_loader = ThreeDMatchTestDataLoader(
            data_root=data_loader.data_root if hasattr(data_loader, 'data_root') else './dataset/ThreeDMatch',
            benchmark=benchmark,
            voxel_size=voxel_size,
            max_points_per_frame=getattr(data_loader, 'max_points_per_frame', 50000),
            mode="pair"  # Use pair mode for direct pair processing
        )
    
    # Set the sequence in pair mode
    threedmatch_loader.mode = "pair"
    threedmatch_loader.set_sequence(sequence_name)
    
    logger.info(f"ThreeDMatch test dataset loaded: {len(threedmatch_loader)} pairs available")
    
    # Get all valid pair indices
    valid_indices = list(range(len(threedmatch_loader)))
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} ThreeDMatch test pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    transformation_distances = []
    rotation_angles = []
    source_point_counts = []
    target_point_counts = []
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing ThreeDMatch test pairs")):
        try:
            # Get pair data
            pair_data = threedmatch_loader[pair_idx]
            
            # Create sample directory (directly in output_dir)
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract source and target data
            source_points = pair_data['src_points']
            target_points = pair_data['tgt_points']
            transformation = np.linalg.inv(pair_data['transformation'])
            src_id = pair_data['src_id']
            tgt_id = pair_data['tgt_id']
            
            # Get global transformation matrix for this sequence
            global_transform_3x3 = dataset_utils.get_global_transformation_matrix(sequence_name)
            # Convert 3x3 rotation matrix to 4x4 transformation matrix
            global_transform = np.eye(4, dtype=np.float32)
            global_transform[:3, :3] = global_transform_3x3
            
            logger.debug(f"Using global transformation for sequence {sequence_name}: {global_transform_3x3}")
            
            # Apply transformation to align source to target coordinate system
            # The transformation matrix transforms source points to target coordinate system
            if len(source_points) > 0:
                # Convert to homogeneous coordinates
                source_homogeneous = np.hstack([source_points, np.ones((len(source_points), 1))])
                # Apply transformation
                source_transformed_homogeneous = (transformation @ source_homogeneous.T).T
                # Convert back to 3D coordinates
                source_points_aligned = source_transformed_homogeneous[:, :3]
                
                # Apply global transformation to source points
                source_points_aligned = (global_transform_3x3 @ source_points_aligned.T).T
            else:
                source_points_aligned = source_points
            
            # Apply global transformation to target points as well
            if len(target_points) > 0:
                target_points_aligned = (global_transform_3x3 @ target_points.T).T
            else:
                target_points_aligned = target_points
            
            logger.debug(f"Applied transformation and global transform to align source {src_id} to target {tgt_id} coordinate system")
            
            # Apply downsampling if requested
            source_downsampled, _ = downsample_points(
                source_points_aligned, None, downsample_method, voxel_size, num_points_downsample
            )
            target_downsampled, _ = downsample_points(
                target_points_aligned, None, downsample_method, voxel_size, num_points_downsample
            )
            
            logger.debug(f"ThreeDMatch: Source: {len(source_points)} -> {len(source_downsampled)} points, "
                        f"Target: {len(target_points)} -> {len(target_downsampled)} points")
            
            # Save source point cloud
            source_filename = f"sample_{sample_idx:06d}_source_{src_id}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled)
            o3d.io.write_point_cloud(source_filepath, source_pcd, write_ascii=False)
            
            # Save target point cloud
            target_filename = f"sample_{sample_idx:06d}_target_{tgt_id}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled)
            o3d.io.write_point_cloud(target_filepath, target_pcd, write_ascii=False)
            
            # Save transformation matrix (original, for reference)
            transformation_filepath = os.path.join(sample_dir, f"transformation_original_{sample_idx:06d}.txt")
            np.savetxt(transformation_filepath, transformation, fmt='%.6f')
            
            # Save identity matrix (current state after alignment)
            identity_filepath = os.path.join(sample_dir, f"transformation_current_{sample_idx:06d}.txt")
            np.savetxt(identity_filepath, np.eye(4), fmt='%.6f')
            
            # Save metadata
            translation_distance = np.linalg.norm(transformation[:3, 3])
            rotation_matrix = transformation[:3, :3]
            rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
            
            metadata = {
                'pair_id': pair_idx,
                'sample_id': sample_idx,
                'sequence': sequence_name,
                'benchmark': benchmark,
                'src_id': src_id,
                'tgt_id': tgt_id,
                'src_fragment': src_id,
                'tgt_fragment': tgt_id,
                'source_points_original': len(source_points),
                'target_points_original': len(target_points),
                'source_points_final': len(source_downsampled),
                'target_points_final': len(target_downsampled),
                'transformation_matrix_original': transformation.tolist(),
                'transformation_matrix_current': np.eye(4).tolist(),  # Identity after alignment
                'translation_distance': float(translation_distance),
                'rotation_angle_rad': float(rotation_angle),
                'rotation_angle_deg': float(np.degrees(rotation_angle)),
                'transformation_applied': True,
                'coordinate_system': 'target_aligned',  # Source transformed to target coordinate system
                'global_transformation_applied': True,
                'global_transformation_matrix': global_transform.tolist(),
                'transformation_files': {
                    'original': f"transformation_original_{sample_idx:06d}.txt",
                    'current': f"transformation_current_{sample_idx:06d}.txt"
                },
                'downsampling_applied': downsample_method != 'none',
                'downsample_method': downsample_method,
                'voxel_size': voxel_size if downsample_method == 'voxel' else None,
                'num_points_target': num_points_downsample if downsample_method in ['fps', 'random'] else None
            }
            
            metadata_filepath = os.path.join(sample_dir, f"metadata_sample_{sample_idx:06d}.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Collect statistics
            transformation_distances.append(translation_distance)
            rotation_angles.append(np.degrees(rotation_angle))
            source_point_counts.append(len(source_downsampled))
            target_point_counts.append(len(target_downsampled))
            
            logger.debug(f"Processed sample {sample_idx}: {src_id} -> {tgt_id}")
            
        except Exception as e:
            logger.error(f"Error processing pair {pair_idx}: {e}")
            continue
    
    # Calculate statistics
    num_samples_generated = len(valid_indices)
    
    stats = {
        'num_samples': num_samples_generated,
        'sequence': sequence_name,
        'benchmark': benchmark,
        'translation_distance': {
            'mean': np.mean(transformation_distances) if transformation_distances else 0,
            'std': np.std(transformation_distances) if transformation_distances else 0,
            'min': min(transformation_distances) if transformation_distances else 0,
            'max': max(transformation_distances) if transformation_distances else 0
        },
        'rotation_angle_deg': {
            'mean': np.mean(rotation_angles) if rotation_angles else 0,
            'std': np.std(rotation_angles) if rotation_angles else 0,
            'min': min(rotation_angles) if rotation_angles else 0,
            'max': max(rotation_angles) if rotation_angles else 0
        },
        'source_point_count': {
            'mean': np.mean(source_point_counts) if source_point_counts else 0,
            'std': np.std(source_point_counts) if source_point_counts else 0,
            'min': min(source_point_counts) if source_point_counts else 0,
            'max': max(source_point_counts) if source_point_counts else 0
        },
        'target_point_count': {
            'mean': np.mean(target_point_counts) if target_point_counts else 0,
            'std': np.std(target_point_counts) if target_point_counts else 0,
            'min': min(target_point_counts) if target_point_counts else 0,
            'max': max(target_point_counts) if target_point_counts else 0
        },
        'processing_method': 'direct_pairs',
        'downsampling_applied': downsample_method != 'none',
        'downsample_method': downsample_method,
        'voxel_size': voxel_size if downsample_method == 'voxel' else None,
        'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None
    }
    
    logger.info(f"ThreeDMatch test processing complete: {num_samples_generated} samples generated")
    logger.info(f"Average translation distance: {stats['translation_distance']['mean']:.3f} ± {stats['translation_distance']['std']:.3f} m")
    logger.info(f"Average rotation angle: {stats['rotation_angle_deg']['mean']:.1f} ± {stats['rotation_angle_deg']['std']:.1f} degrees")
    logger.info(f"Average point counts - Source: {stats['source_point_count']['mean']:.0f}, Target: {stats['target_point_count']['mean']:.0f}")
    
    return num_samples_generated, stats


def process_kitti_benchmark_dataset(data_loader,
                                   output_dir: str,
                                   sequence_name: str,
                                   max_samples: Optional[int] = None,
                                   voxel_size: float = 0.25,
                                   downsample_method: str = "voxel",
                                   num_points_downsample: Optional[int] = None,
                                   min_overlap_ratio: float = 0.0,
                                   max_overlap_ratio: float = 1.0) -> Tuple[int, Dict]:
    """
    Process KITTI benchmark dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: KITTI data loader instance in benchmark mode
        output_dir: Output directory for training samples
        sequence_name: Name of the KITTI sequence (e.g., '08')
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter (not used for KITTI benchmark)
        max_overlap_ratio: Maximum overlap ratio filter (not used for KITTI benchmark)
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing benchmark dataset directly (sequence={sequence_name})")
    logger.debug(f"Data loader type: {type(data_loader).__name__}")
    logger.debug(f"Data loader benchmark_mode: {getattr(data_loader, 'benchmark_mode', 'Not set')}")
    
    # Ensure the data loader is in benchmark mode
    if not getattr(data_loader, 'benchmark_mode', False):
        logger.warning("KITTI data loader is not in benchmark mode, attempting to enable it")
        # Try to set benchmark mode if it's a KITTI loader
        if hasattr(data_loader, '__class__') and 'KITTI' in data_loader.__class__.__name__:
            data_loader.benchmark_mode = True
            logger.info("Enabled benchmark mode for KITTI data loader")
        else:
            logger.error(f"Data loader class: {data_loader.__class__.__name__}")
            logger.error(f"Data loader attributes: {[attr for attr in dir(data_loader) if not attr.startswith('_')]}")
            raise ValueError("KITTI data loader must be in benchmark mode for direct pairs processing")
    
    # Set the sequence in benchmark mode
    data_loader.set_sequence(sequence_name)
    
    logger.info(f"KITTI benchmark dataset loaded: {len(data_loader)} pairs available")
    
    # Get all valid pair indices
    valid_indices = list(range(len(data_loader)))
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} KITTI benchmark pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    transformation_distances = []
    rotation_angles = []
    source_point_counts = []
    target_point_counts = []
    frame_id_differences = []
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing benchmark pairs")):
        try:
            # Get pair data
            pair_data = data_loader[pair_idx]
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract pair data
            source_points = pair_data['points1']
            target_points = pair_data['points2']
            source_pose = pair_data['pose1']
            target_pose = pair_data['pose2']
            source_normals = pair_data.get('normals1')
            target_normals = pair_data.get('normals2')
            frame_id1 = pair_data['frame_id1']
            frame_id2 = pair_data['frame_id2']
            sequence = pair_data['sequence']
            pair_name = pair_data['pair_name']
            
            # Calculate relative transformation (from source to target)
            relative_transform = np.linalg.inv(target_pose) @ source_pose
            
            # Calculate statistics
            translation = relative_transform[:3, 3]
            translation_distance = np.linalg.norm(translation)
            transformation_distances.append(translation_distance)
            
            # Calculate rotation angle
            rotation_matrix = relative_transform[:3, :3]
            trace = np.trace(rotation_matrix)
            # Clamp trace to valid range for arccos
            trace = np.clip(trace, -1.0, 3.0)
            rotation_angle = np.arccos((trace - 1) / 2)
            rotation_angles.append(rotation_angle)
            
            # Frame ID difference
            frame_diff = abs(frame_id2 - frame_id1)
            frame_id_differences.append(frame_diff)
            
            logger.debug(f"KITTI benchmark pair {pair_name}: "
                        f"translation={translation_distance:.3f}m, "
                        f"rotation={np.degrees(rotation_angle):.1f}°, "
                        f"frame_diff={frame_diff}")
            logger.debug(f"Point clouds transformed to global coordinates using poses")
            logger.debug(f"Source pose: {source_pose[:3, 3]} (translation)")
            logger.debug(f"Target pose: {target_pose[:3, 3]} (translation)")
            
            # Apply downsampling if requested
            source_downsampled, source_normals_downsampled = downsample_points(
                source_points, source_normals, downsample_method, voxel_size, num_points_downsample
            )
            target_downsampled, target_normals_downsampled = downsample_points(
                target_points, target_normals, downsample_method, voxel_size, num_points_downsample
            )
            
            logger.debug(f"KITTI benchmark: Source: {len(source_points)} -> {len(source_downsampled)} points, "
                        f"Target: {len(target_points)} -> {len(target_downsampled)} points")
            
            # Update point count statistics
            source_point_counts.append(len(source_downsampled))
            target_point_counts.append(len(target_downsampled))
            
            # Save source point cloud
            source_filename = f"sample_{sample_idx:06d}_source_{sequence}_{frame_id1:06d}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            # Create source point cloud
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled.astype(np.float64))
            if source_normals_downsampled is not None:
                source_pcd.normals = o3d.utility.Vector3dVector(source_normals_downsampled.astype(np.float64))
            
            # Save source point cloud
            o3d.io.write_point_cloud(source_filepath, source_pcd)
            
            # Save target point cloud
            target_filename = f"sample_{sample_idx:06d}_target_{sequence}_{frame_id2:06d}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            # Create target point cloud
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled.astype(np.float64))
            if target_normals_downsampled is not None:
                target_pcd.normals = o3d.utility.Vector3dVector(target_normals_downsampled.astype(np.float64))
            
            # Save target point cloud
            o3d.io.write_point_cloud(target_filepath, target_pcd)
            
            # Create sample metadata
            sample_metadata = {
                'sample_id': f"sample_{sample_idx:06d}",
                'sequence': sequence,
                'source_frame_id': int(frame_id1),
                'target_frame_id': int(frame_id2),
                'frame_id_difference': int(frame_diff),
                'pair_name': pair_name,
                'source_file': source_filename,
                'target_file': target_filename,
                'source_pose': source_pose.tolist(),
                'target_pose': target_pose.tolist(),
                'relative_transformation': relative_transform.tolist(),
                'translation_distance': float(translation_distance),
                'rotation_angle_rad': float(rotation_angle),
                'rotation_angle_deg': float(np.degrees(rotation_angle)),
                'source_point_count': len(source_downsampled),
                'target_point_count': len(target_downsampled),
                'source_point_count_original': len(source_points),
                'target_point_count_original': len(target_points),
                'downsample_method': downsample_method,
                'voxel_size': voxel_size if downsample_method == 'voxel' else None,
                'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None,
                'has_normals': source_normals is not None and target_normals is not None,
                'coordinate_frame': 'global',
                'transformation_applied': True,
                'note': 'Point clouds have been transformed to global coordinates using their respective poses'
            }
            
            # Save sample metadata
            metadata_filepath = os.path.join(sample_dir, "metadata.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(sample_metadata, f, indent=2)
            
            logger.debug(f"Saved benchmark sample {sample_idx:06d}: {pair_name}")
            
        except Exception as e:
            logger.error(f"Error processing benchmark pair {pair_idx}: {e}")
            continue
    
    # Calculate final statistics
    num_samples_generated = len([f for f in os.listdir(output_dir) if f.startswith('sample_')])
    
    stats = {
        'num_samples': num_samples_generated,
        'num_pairs_processed': len(valid_indices),
        'translation_distance': {
            'mean': np.mean(transformation_distances) if transformation_distances else 0,
            'std': np.std(transformation_distances) if transformation_distances else 0,
            'min': min(transformation_distances) if transformation_distances else 0,
            'max': max(transformation_distances) if transformation_distances else 0
        },
        'rotation_angle_deg': {
            'mean': np.degrees(np.mean(rotation_angles)) if rotation_angles else 0,
            'std': np.degrees(np.std(rotation_angles)) if rotation_angles else 0,
            'min': np.degrees(min(rotation_angles)) if rotation_angles else 0,
            'max': np.degrees(max(rotation_angles)) if rotation_angles else 0
        },
        'frame_id_difference': {
            'mean': np.mean(frame_id_differences) if frame_id_differences else 0,
            'std': np.std(frame_id_differences) if frame_id_differences else 0,
            'min': min(frame_id_differences) if frame_id_differences else 0,
            'max': max(frame_id_differences) if frame_id_differences else 0
        },
        'source_point_count': {
            'mean': np.mean(source_point_counts) if source_point_counts else 0,
            'std': np.std(source_point_counts) if source_point_counts else 0,
            'min': min(source_point_counts) if source_point_counts else 0,
            'max': max(source_point_counts) if source_point_counts else 0
        },
        'target_point_count': {
            'mean': np.mean(target_point_counts) if target_point_counts else 0,
            'std': np.std(target_point_counts) if target_point_counts else 0,
            'min': min(target_point_counts) if target_point_counts else 0,
            'max': max(target_point_counts) if target_point_counts else 0
        },
        'processing_method': 'direct_pairs',
        'downsampling_applied': downsample_method != 'none',
        'downsample_method': downsample_method,
        'voxel_size': voxel_size if downsample_method == 'voxel' else None,
        'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None
    }
    
    logger.info(f"KITTI benchmark processing complete: {num_samples_generated} samples generated")
    logger.info(f"Average translation distance: {stats['translation_distance']['mean']:.3f} ± {stats['translation_distance']['std']:.3f} m")
    logger.info(f"Average rotation angle: {stats['rotation_angle_deg']['mean']:.1f} ± {stats['rotation_angle_deg']['std']:.1f} degrees")
    logger.info(f"Average frame ID difference: {stats['frame_id_difference']['mean']:.1f} ± {stats['frame_id_difference']['std']:.1f}")
    logger.info(f"Average point counts - Source: {stats['source_point_count']['mean']:.0f}, Target: {stats['target_point_count']['mean']:.0f}")
    
    return num_samples_generated, stats


def process_mit_benchmark_dataset(data_loader,
                                   output_dir: str,
                                   sequence_name: str,
                                   max_samples: Optional[int] = None,
                                   voxel_size: float = 0.25,
                                   downsample_method: str = "voxel",
                                   num_points_downsample: Optional[int] = None,
                                   min_overlap_ratio: float = 0.0,
                                   max_overlap_ratio: float = 1.0) -> Tuple[int, Dict]:
    """
    Process MIT benchmark dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: MIT data loader instance in benchmark mode
        output_dir: Output directory for training samples
        sequence_name: Name of the MIT sequence (e.g., 'acl_jackal')
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter (not used for MIT benchmark)
        max_overlap_ratio: Maximum overlap ratio filter (not used for MIT benchmark)
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing MIT benchmark dataset directly (sequence={sequence_name})")
    logger.debug(f"Data loader type: {type(data_loader).__name__}")
    logger.debug(f"Data loader benchmark_mode: {getattr(data_loader, 'benchmark_mode', 'Not set')}")
    
    # Ensure the data loader is in benchmark mode
    if not getattr(data_loader, 'benchmark_mode', False):
        logger.warning("MIT data loader is not in benchmark mode, attempting to enable it")
        # Try to set benchmark mode if it's a MIT loader
        if hasattr(data_loader, '__class__') and 'MIT' in data_loader.__class__.__name__:
            data_loader.benchmark_mode = True
            logger.info("Enabled benchmark mode for MIT data loader")
        else:
            logger.error(f"Data loader class: {data_loader.__class__.__name__}")
            logger.error(f"Data loader attributes: {[attr for attr in dir(data_loader) if not attr.startswith('_')]}")
            raise ValueError("MIT data loader must be in benchmark mode for direct pairs processing")
    
    # Set the sequence in benchmark mode
    data_loader.set_sequence(sequence_name)
    
    logger.info(f"MIT benchmark dataset loaded: {len(data_loader)} pairs available")
    
    # Get all valid pair indices
    valid_indices = list(range(len(data_loader)))
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} MIT benchmark pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    transformation_distances = []
    rotation_angles = []
    source_point_counts = []
    target_point_counts = []
    frame_id_differences = []
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing benchmark pairs")):
        try:
            # Get pair data
            pair_data = data_loader[pair_idx]
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract pair data
            source_points = pair_data['points1']
            target_points = pair_data['points2']
            source_pose = pair_data['pose1']
            target_pose = pair_data['pose2']
            source_normals = pair_data.get('normals1')
            target_normals = pair_data.get('normals2')
            frame_id1 = pair_data['frame_id1']
            frame_id2 = pair_data['frame_id2']
            sequence = pair_data['sequence']
            pair_name = pair_data['pair_name']
            
            # Calculate relative transformation (from source to target)
            relative_transform = np.linalg.inv(target_pose) @ source_pose
            
            # Calculate statistics
            translation = relative_transform[:3, 3]
            translation_distance = np.linalg.norm(translation)
            transformation_distances.append(translation_distance)
            
            # Calculate rotation angle
            rotation_matrix = relative_transform[:3, :3]
            trace = np.trace(rotation_matrix)
            # Clamp trace to valid range for arccos
            trace = np.clip(trace, -1.0, 3.0)
            rotation_angle = np.arccos((trace - 1) / 2)
            rotation_angles.append(rotation_angle)
            
            # Frame ID difference
            frame_diff = abs(frame_id2 - frame_id1)
            frame_id_differences.append(frame_diff)
            
            logger.debug(f"MIT benchmark pair {pair_name}: "
                        f"translation={translation_distance:.3f}m, "
                        f"rotation={np.degrees(rotation_angle):.1f}°, "
                        f"frame_diff={frame_diff}")
            logger.debug(f"Point clouds transformed to global coordinates using poses")
            logger.debug(f"Source pose: {source_pose[:3, 3]} (translation)")
            logger.debug(f"Target pose: {target_pose[:3, 3]} (translation)")
            
            # Apply downsampling if requested
            source_downsampled, source_normals_downsampled = downsample_points(
                source_points, source_normals, downsample_method, voxel_size, num_points_downsample
            )
            target_downsampled, target_normals_downsampled = downsample_points(
                target_points, target_normals, downsample_method, voxel_size, num_points_downsample
            )
            
            logger.debug(f"MIT benchmark: Source: {len(source_points)} -> {len(source_downsampled)} points, "
                        f"Target: {len(target_points)} -> {len(target_downsampled)} points")
            
            # Update point count statistics
            source_point_counts.append(len(source_downsampled))
            target_point_counts.append(len(target_downsampled))
            
            # Save source point cloud
            source_filename = f"frame_{frame_id1:06d}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            # Create source point cloud
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled.astype(np.float64))
            if source_normals_downsampled is not None:
                source_pcd.normals = o3d.utility.Vector3dVector(source_normals_downsampled.astype(np.float64))
            
            # Save source point cloud
            o3d.io.write_point_cloud(source_filepath, source_pcd)
            
            # Save target point cloud
            target_filename = f"frame_{frame_id2:06d}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            # Create target point cloud
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled.astype(np.float64))
            if target_normals_downsampled is not None:
                target_pcd.normals = o3d.utility.Vector3dVector(target_normals_downsampled.astype(np.float64))
            
            # Save target point cloud
            o3d.io.write_point_cloud(target_filepath, target_pcd)
            
            # Create sample metadata
            sample_metadata = {
                'sample_id': f"sample_{sample_idx:05d}",
                'sequence': sequence,
                'source_frame_id': int(frame_id1),
                'target_frame_id': int(frame_id2),
                'frame_id_difference': int(frame_diff),
                'pair_name': pair_name,
                'source_file': source_filename,
                'target_file': target_filename,
                'source_pose': source_pose.tolist(),
                'target_pose': target_pose.tolist(),
                'relative_transformation': relative_transform.tolist(),
                'translation_distance': float(translation_distance),
                'rotation_angle_rad': float(rotation_angle),
                'rotation_angle_deg': float(np.degrees(rotation_angle)),
                'source_point_count': len(source_downsampled),
                'target_point_count': len(target_downsampled),
                'source_point_count_original': len(source_points),
                'target_point_count_original': len(target_points),
                'downsample_method': downsample_method,
                'voxel_size': voxel_size if downsample_method == 'voxel' else None,
                'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None,
                'has_normals': source_normals is not None and target_normals is not None,
                'coordinate_frame': 'global',
                'transformation_applied': True,
                'note': 'Point clouds have been transformed to global coordinates using their respective poses'
            }
            
            # Save sample metadata
            metadata_filepath = os.path.join(sample_dir, "metadata.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(sample_metadata, f, indent=2)
            
            logger.debug(f"Saved benchmark sample {sample_idx:05d}: {pair_name}")
            
        except Exception as e:
            logger.error(f"Error processing benchmark pair {pair_idx}: {e}")
            continue
    
    # Calculate final statistics
    num_samples_generated = len([f for f in os.listdir(output_dir) if f.startswith('sample_')])
    
    stats = {
        'num_samples': num_samples_generated,
        'num_pairs_processed': len(valid_indices),
        'translation_distance': {
            'mean': np.mean(transformation_distances) if transformation_distances else 0,
            'std': np.std(transformation_distances) if transformation_distances else 0,
            'min': min(transformation_distances) if transformation_distances else 0,
            'max': max(transformation_distances) if transformation_distances else 0
        },
        'rotation_angle_deg': {
            'mean': np.degrees(np.mean(rotation_angles)) if rotation_angles else 0,
            'std': np.degrees(np.std(rotation_angles)) if rotation_angles else 0,
            'min': np.degrees(min(rotation_angles)) if rotation_angles else 0,
            'max': np.degrees(max(rotation_angles)) if rotation_angles else 0
        },
        'frame_id_difference': {
            'mean': np.mean(frame_id_differences) if frame_id_differences else 0,
            'std': np.std(frame_id_differences) if frame_id_differences else 0,
            'min': min(frame_id_differences) if frame_id_differences else 0,
            'max': max(frame_id_differences) if frame_id_differences else 0
        },
        'source_point_count': {
            'mean': np.mean(source_point_counts) if source_point_counts else 0,
            'std': np.std(source_point_counts) if source_point_counts else 0,
            'min': min(source_point_counts) if source_point_counts else 0,
            'max': max(source_point_counts) if source_point_counts else 0
        },
        'target_point_count': {
            'mean': np.mean(target_point_counts) if target_point_counts else 0,
            'std': np.std(target_point_counts) if target_point_counts else 0,
            'min': min(target_point_counts) if target_point_counts else 0,
            'max': max(target_point_counts) if target_point_counts else 0
        },
        'processing_method': 'direct_pairs',
        'downsampling_applied': downsample_method != 'none',
        'downsample_method': downsample_method,
        'voxel_size': voxel_size if downsample_method == 'voxel' else None,
        'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None
    }
    
    logger.info(f"MIT benchmark processing complete: {num_samples_generated} samples generated")
    logger.info(f"Average translation distance: {stats['translation_distance']['mean']:.3f} ± {stats['translation_distance']['std']:.3f} m")
    logger.info(f"Average rotation angle: {stats['rotation_angle_deg']['mean']:.1f} ± {stats['rotation_angle_deg']['std']:.1f} degrees")
    logger.info(f"Average frame ID difference: {stats['frame_id_difference']['mean']:.1f} ± {stats['frame_id_difference']['std']:.1f}")
    logger.info(f"Average point counts - Source: {stats['source_point_count']['mean']:.0f}, Target: {stats['target_point_count']['mean']:.0f}")
    
    return num_samples_generated, stats


def process_tiers_benchmark_dataset(data_loader,
                                   output_dir: str,
                                   sequence_name: str,
                                   max_samples: Optional[int] = None,
                                   voxel_size: float = 0.25,
                                   downsample_method: str = "voxel",
                                   num_points_downsample: Optional[int] = None,
                                   min_overlap_ratio: float = 0.0,
                                   max_overlap_ratio: float = 1.0) -> Tuple[int, Dict]:
    """
    Process TIERS benchmark dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: TIERS data loader instance in benchmark mode
        output_dir: Output directory for training samples
        sequence_name: Name of the TIERS sequence (e.g., 'tiers_indoor11')
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter (not used for TIERS benchmark)
        max_overlap_ratio: Maximum overlap ratio filter (not used for TIERS benchmark)
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing TIERS benchmark dataset directly (sequence={sequence_name})")
    logger.debug(f"Data loader type: {type(data_loader).__name__}")
    logger.debug(f"Data loader benchmark_mode: {getattr(data_loader, 'benchmark_mode', 'Not set')}")
    
    # Ensure the data loader is in benchmark mode
    if not getattr(data_loader, 'benchmark_mode', False):
        logger.warning("TIERS data loader is not in benchmark mode, attempting to enable it")
        # Try to set benchmark mode if it's a TIERS loader
        if hasattr(data_loader, '__class__') and 'TIERS' in data_loader.__class__.__name__:
            data_loader.benchmark_mode = True
            logger.info("Enabled benchmark mode for TIERS data loader")
        else:
            logger.error(f"Data loader class: {data_loader.__class__.__name__}")
            logger.error(f"Data loader attributes: {[attr for attr in dir(data_loader) if not attr.startswith('_')]}")
            raise ValueError("TIERS data loader must be in benchmark mode for direct pairs processing")
    
    # Set the sequence in benchmark mode
    data_loader.set_sequence(sequence_name)
    
    logger.info(f"TIERS benchmark dataset loaded: {len(data_loader)} pairs available")
    
    # Get all valid pair indices
    valid_indices = list(range(len(data_loader)))
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} TIERS benchmark pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    transformation_distances = []
    rotation_angles = []
    source_point_counts = []
    target_point_counts = []
    frame_id_differences = []
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing benchmark pairs")):
        try:
            # Get pair data
            pair_data = data_loader[pair_idx]
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract pair data
            source_points = pair_data['points1']
            target_points = pair_data['points2']
            source_pose = pair_data['pose1']
            target_pose = pair_data['pose2']
            source_normals = pair_data.get('normals1')
            target_normals = pair_data.get('normals2')
            frame_id1 = pair_data['frame_id1']
            frame_id2 = pair_data['frame_id2']
            sequence = pair_data['sequence']
            sensor = pair_data.get('sensor', 'unknown')
            pair_name = pair_data['pair_name']
            
            # Calculate relative transformation (from source to target)
            relative_transform = np.linalg.inv(target_pose) @ source_pose
            
            # Calculate statistics
            translation = relative_transform[:3, 3]
            translation_distance = np.linalg.norm(translation)
            transformation_distances.append(translation_distance)
            
            # Calculate rotation angle
            rotation_matrix = relative_transform[:3, :3]
            trace = np.trace(rotation_matrix)
            # Clamp trace to valid range for arccos
            trace = np.clip(trace, -1.0, 3.0)
            rotation_angle = np.arccos((trace - 1) / 2)
            rotation_angles.append(rotation_angle)
            
            # Frame ID difference
            frame_diff = abs(frame_id2 - frame_id1)
            frame_id_differences.append(frame_diff)
            
            logger.debug(f"TIERS benchmark pair {pair_name}: "
                        f"translation={translation_distance:.3f}m, "
                        f"rotation={np.degrees(rotation_angle):.1f}°, "
                        f"frame_diff={frame_diff}")
            logger.debug(f"Point clouds transformed to global coordinates using poses")
            logger.debug(f"Source pose: {source_pose[:3, 3]} (translation)")
            logger.debug(f"Target pose: {target_pose[:3, 3]} (translation)")
            
            # Apply downsampling if requested
            source_downsampled, source_normals_downsampled = downsample_points(
                source_points, source_normals, downsample_method, voxel_size, num_points_downsample
            )
            target_downsampled, target_normals_downsampled = downsample_points(
                target_points, target_normals, downsample_method, voxel_size, num_points_downsample
            )
            
            logger.debug(f"TIERS benchmark: Source: {len(source_points)} -> {len(source_downsampled)} points, "
                        f"Target: {len(target_points)} -> {len(target_downsampled)} points")
            
            # Update point count statistics
            source_point_counts.append(len(source_downsampled))
            target_point_counts.append(len(target_downsampled))
            
            # Save source point cloud
            source_filename = f"frame_{frame_id1:06d}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            # Create source point cloud
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled.astype(np.float64))
            if source_normals_downsampled is not None:
                source_pcd.normals = o3d.utility.Vector3dVector(source_normals_downsampled.astype(np.float64))
            
            # Save source point cloud
            o3d.io.write_point_cloud(source_filepath, source_pcd)
            
            # Save target point cloud
            target_filename = f"frame_{frame_id2:06d}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            # Create target point cloud
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled.astype(np.float64))
            if target_normals_downsampled is not None:
                target_pcd.normals = o3d.utility.Vector3dVector(target_normals_downsampled.astype(np.float64))
            
            # Save target point cloud
            o3d.io.write_point_cloud(target_filepath, target_pcd)
            
            # Create sample metadata
            sample_metadata = {
                'sample_id': f"sample_{sample_idx:05d}",
                'sequence': sequence,
                'sensor': sensor,
                'source_frame_id': int(frame_id1),
                'target_frame_id': int(frame_id2),
                'frame_id_difference': int(frame_diff),
                'pair_name': pair_name,
                'source_file': source_filename,
                'target_file': target_filename,
                'source_pose': source_pose.tolist(),
                'target_pose': target_pose.tolist(),
                'relative_transformation': relative_transform.tolist(),
                'translation_distance': float(translation_distance),
                'rotation_angle_rad': float(rotation_angle),
                'rotation_angle_deg': float(np.degrees(rotation_angle)),
                'source_point_count': len(source_downsampled),
                'target_point_count': len(target_downsampled),
                'source_point_count_original': len(source_points),
                'target_point_count_original': len(target_points),
                'downsample_method': downsample_method,
                'voxel_size': voxel_size if downsample_method == 'voxel' else None,
                'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None,
                'has_normals': source_normals is not None and target_normals is not None,
                'coordinate_frame': 'global',
                'transformation_applied': True,
                'note': 'Point clouds have been transformed to global coordinates using their respective poses'
            }
            
            # Save sample metadata
            metadata_filepath = os.path.join(sample_dir, "metadata.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(sample_metadata, f, indent=2)
            
            logger.debug(f"Saved benchmark sample {sample_idx:05d}: {pair_name}")
            
        except Exception as e:
            logger.error(f"Error processing benchmark pair {pair_idx}: {e}")
            continue
    
    # Calculate final statistics
    num_samples_generated = len([f for f in os.listdir(output_dir) if f.startswith('sample_')])
    
    stats = {
        'num_samples': num_samples_generated,
        'num_pairs_processed': len(valid_indices),
        'translation_distance': {
            'mean': np.mean(transformation_distances) if transformation_distances else 0,
            'std': np.std(transformation_distances) if transformation_distances else 0,
            'min': min(transformation_distances) if transformation_distances else 0,
            'max': max(transformation_distances) if transformation_distances else 0
        },
        'rotation_angle_deg': {
            'mean': np.degrees(np.mean(rotation_angles)) if rotation_angles else 0,
            'std': np.degrees(np.std(rotation_angles)) if rotation_angles else 0,
            'min': np.degrees(min(rotation_angles)) if rotation_angles else 0,
            'max': np.degrees(max(rotation_angles)) if rotation_angles else 0
        },
        'frame_id_difference': {
            'mean': np.mean(frame_id_differences) if frame_id_differences else 0,
            'std': np.std(frame_id_differences) if frame_id_differences else 0,
            'min': min(frame_id_differences) if frame_id_differences else 0,
            'max': max(frame_id_differences) if frame_id_differences else 0
        },
        'source_point_count': {
            'mean': np.mean(source_point_counts) if source_point_counts else 0,
            'std': np.std(source_point_counts) if source_point_counts else 0,
            'min': min(source_point_counts) if source_point_counts else 0,
            'max': max(source_point_counts) if source_point_counts else 0
        },
        'target_point_count': {
            'mean': np.mean(target_point_counts) if target_point_counts else 0,
            'std': np.std(target_point_counts) if target_point_counts else 0,
            'min': min(target_point_counts) if target_point_counts else 0,
            'max': max(target_point_counts) if target_point_counts else 0
        },
        'processing_method': 'direct_pairs',
        'downsampling_applied': downsample_method != 'none',
        'downsample_method': downsample_method,
        'voxel_size': voxel_size if downsample_method == 'voxel' else None,
        'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None
    }
    
    logger.info(f"TIERS benchmark processing complete: {num_samples_generated} samples generated")
    logger.info(f"Average translation distance: {stats['translation_distance']['mean']:.3f} ± {stats['translation_distance']['std']:.3f} m")
    logger.info(f"Average rotation angle: {stats['rotation_angle_deg']['mean']:.1f} ± {stats['rotation_angle_deg']['std']:.1f} degrees")
    logger.info(f"Average frame ID difference: {stats['frame_id_difference']['mean']:.1f} ± {stats['frame_id_difference']['std']:.1f}")
    logger.info(f"Average point counts - Source: {stats['source_point_count']['mean']:.0f}, Target: {stats['target_point_count']['mean']:.0f}")
    
    return num_samples_generated, stats


def process_waymo_benchmark_dataset(data_loader,
                                   output_dir: str,
                                   sequence_name: str,
                                   max_samples: Optional[int] = None,
                                   voxel_size: float = 0.25,
                                   downsample_method: str = "voxel",
                                   num_points_downsample: Optional[int] = None,
                                   min_overlap_ratio: float = 0.0,
                                   max_overlap_ratio: float = 1.0) -> Tuple[int, Dict]:
    """
    Process Waymo benchmark dataset directly without sequence-based submap generation.
    Each pair becomes a training sample with source and target point clouds.
    
    Args:
        data_loader: Waymo data loader instance in benchmark mode
        output_dir: Output directory for training samples
        sequence_name: Name of the Waymo sequence (e.g., '14737335824319407706_1980_000_2000_000')
        max_samples: Maximum number of samples to process (None for all)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_overlap_ratio: Minimum overlap ratio filter (not used for Waymo benchmark)
        max_overlap_ratio: Maximum overlap ratio filter (not used for Waymo benchmark)
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing Waymo benchmark dataset directly (sequence={sequence_name})")
    logger.debug(f"Data loader type: {type(data_loader).__name__}")
    logger.debug(f"Data loader benchmark_mode: {getattr(data_loader, 'benchmark_mode', 'Not set')}")
    
    # Ensure the data loader is in benchmark mode
    if not getattr(data_loader, 'benchmark_mode', False):
        logger.warning("Waymo data loader is not in benchmark mode, attempting to enable it")
        # Try to set benchmark mode if it's a Waymo loader
        if hasattr(data_loader, '__class__') and 'Waymo' in data_loader.__class__.__name__:
            data_loader.benchmark_mode = True
            logger.info("Enabled benchmark mode for Waymo data loader")
        else:
            logger.error(f"Data loader class: {data_loader.__class__.__name__}")
            logger.error(f"Data loader attributes: {[attr for attr in dir(data_loader) if not attr.startswith('_')]}")
            raise ValueError("Waymo data loader must be in benchmark mode for direct pairs processing")
    
    # Set the sequence in benchmark mode
    data_loader.set_sequence(sequence_name)
    
    logger.info(f"Waymo benchmark dataset loaded: {len(data_loader)} pairs available")
    
    # Get all valid pair indices
    valid_indices = list(range(len(data_loader)))
    
    # Limit number of samples if specified
    if max_samples is not None and len(valid_indices) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (randomly selected)")
        valid_indices = random.sample(valid_indices, max_samples)
    
    logger.info(f"Processing {len(valid_indices)} Waymo benchmark pairs as training samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    transformation_distances = []
    rotation_angles = []
    source_point_counts = []
    target_point_counts = []
    frame_id_differences = []
    
    # Process each valid pair
    for sample_idx, pair_idx in enumerate(tqdm(valid_indices, desc="Processing Waymo benchmark pairs")):
        try:
            # Get pair data
            pair_data = data_loader[pair_idx]
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract pair data
            source_points = pair_data['points1']
            target_points = pair_data['points2']
            source_pose = pair_data['pose1']
            target_pose = pair_data['pose2']
            source_normals = pair_data.get('normals1')
            target_normals = pair_data.get('normals2')
            frame_id1 = pair_data['frame_id1']
            frame_id2 = pair_data['frame_id2']
            sequence = pair_data['sequence']
            pair_name = pair_data['pair_name']
            
            # Calculate relative transformation (from source to target)
            relative_transform = np.linalg.inv(target_pose) @ source_pose
            
            # Calculate statistics
            translation = relative_transform[:3, 3]
            translation_distance = np.linalg.norm(translation)
            transformation_distances.append(translation_distance)
            
            # Calculate rotation angle
            rotation_matrix = relative_transform[:3, :3]
            trace = np.trace(rotation_matrix)
            # Clamp trace to valid range for arccos
            trace = np.clip(trace, -1.0, 3.0)
            rotation_angle = np.arccos((trace - 1) / 2)
            rotation_angles.append(rotation_angle)
            
            # Frame ID difference
            frame_diff = abs(frame_id2 - frame_id1)
            frame_id_differences.append(frame_diff)
            
            logger.debug(f"Waymo benchmark pair {pair_name}: "
                        f"translation={translation_distance:.3f}m, "
                        f"rotation={np.degrees(rotation_angle):.1f}°, "
                        f"frame_diff={frame_diff}")
            logger.debug(f"Point clouds transformed to global coordinates using poses")
            logger.debug(f"Source pose: {source_pose[:3, 3]} (translation)")
            logger.debug(f"Target pose: {target_pose[:3, 3]} (translation)")
            
            # Apply downsampling if requested
            source_downsampled, source_normals_downsampled = downsample_points(
                source_points, source_normals, downsample_method, voxel_size, num_points_downsample
            )
            target_downsampled, target_normals_downsampled = downsample_points(
                target_points, target_normals, downsample_method, voxel_size, num_points_downsample
            )
            
            logger.debug(f"Waymo benchmark: Source: {len(source_points)} -> {len(source_downsampled)} points, "
                        f"Target: {len(target_points)} -> {len(target_downsampled)} points")
            
            # Update point count statistics
            source_point_counts.append(len(source_downsampled))
            target_point_counts.append(len(target_downsampled))
            
            # Save source point cloud
            source_filename = f"sample_{sample_idx:06d}_source_{sequence}_{frame_id1:06d}.ply"
            source_filepath = os.path.join(sample_dir, source_filename)
            
            # Create source point cloud
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_downsampled.astype(np.float64))
            if source_normals_downsampled is not None:
                source_pcd.normals = o3d.utility.Vector3dVector(source_normals_downsampled.astype(np.float64))
            
            # Save source point cloud
            o3d.io.write_point_cloud(source_filepath, source_pcd)
            
            # Save target point cloud
            target_filename = f"sample_{sample_idx:06d}_target_{sequence}_{frame_id2:06d}.ply"
            target_filepath = os.path.join(sample_dir, target_filename)
            
            # Create target point cloud
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_downsampled.astype(np.float64))
            if target_normals_downsampled is not None:
                target_pcd.normals = o3d.utility.Vector3dVector(target_normals_downsampled.astype(np.float64))
            
            # Save target point cloud
            o3d.io.write_point_cloud(target_filepath, target_pcd)
            
            # Create sample metadata
            sample_metadata = {
                'sample_id': f"sample_{sample_idx:06d}",
                'sequence': sequence,
                'source_frame_id': int(frame_id1),
                'target_frame_id': int(frame_id2),
                'frame_id_difference': int(frame_diff),
                'pair_name': pair_name,
                'source_file': source_filename,
                'target_file': target_filename,
                'source_pose': source_pose.tolist(),
                'target_pose': target_pose.tolist(),
                'relative_transformation': relative_transform.tolist(),
                'translation_distance': float(translation_distance),
                'rotation_angle_rad': float(rotation_angle),
                'rotation_angle_deg': float(np.degrees(rotation_angle)),
                'source_point_count': len(source_downsampled),
                'target_point_count': len(target_downsampled),
                'source_point_count_original': len(source_points),
                'target_point_count_original': len(target_points),
                'downsample_method': downsample_method,
                'voxel_size': voxel_size if downsample_method == 'voxel' else None,
                'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None,
                'has_normals': source_normals is not None and target_normals is not None,
                'coordinate_frame': 'global',
                'transformation_applied': True,
                'note': 'Point clouds have been transformed to global coordinates using their respective poses'
            }
            
            # Save sample metadata
            metadata_filepath = os.path.join(sample_dir, "metadata.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(sample_metadata, f, indent=2)
            
            logger.debug(f"Saved Waymo benchmark sample {sample_idx:06d}: {pair_name}")
            
        except Exception as e:
            logger.error(f"Error processing Waymo benchmark pair {pair_idx}: {e}")
            continue
    
    # Calculate final statistics
    num_samples_generated = len([f for f in os.listdir(output_dir) if f.startswith('sample_')])
    
    stats = {
        'num_samples': num_samples_generated,
        'num_pairs_processed': len(valid_indices),
        'translation_distance': {
            'mean': np.mean(transformation_distances) if transformation_distances else 0,
            'std': np.std(transformation_distances) if transformation_distances else 0,
            'min': min(transformation_distances) if transformation_distances else 0,
            'max': max(transformation_distances) if transformation_distances else 0
        },
        'rotation_angle_deg': {
            'mean': np.degrees(np.mean(rotation_angles)) if rotation_angles else 0,
            'std': np.degrees(np.std(rotation_angles)) if rotation_angles else 0,
            'min': np.degrees(min(rotation_angles)) if rotation_angles else 0,
            'max': np.degrees(max(rotation_angles)) if rotation_angles else 0
        },
        'frame_id_difference': {
            'mean': np.mean(frame_id_differences) if frame_id_differences else 0,
            'std': np.std(frame_id_differences) if frame_id_differences else 0,
            'min': min(frame_id_differences) if frame_id_differences else 0,
            'max': max(frame_id_differences) if frame_id_differences else 0
        },
        'source_point_count': {
            'mean': np.mean(source_point_counts) if source_point_counts else 0,
            'std': np.std(source_point_counts) if source_point_counts else 0,
            'min': min(source_point_counts) if source_point_counts else 0,
            'max': max(source_point_counts) if source_point_counts else 0
        },
        'target_point_count': {
            'mean': np.mean(target_point_counts) if target_point_counts else 0,
            'std': np.std(target_point_counts) if target_point_counts else 0,
            'min': min(target_point_counts) if target_point_counts else 0,
            'max': max(target_point_counts) if target_point_counts else 0
        },
        'processing_method': 'direct_pairs',
        'downsampling_applied': downsample_method != 'none',
        'downsample_method': downsample_method,
        'voxel_size': voxel_size if downsample_method == 'voxel' else None,
        'num_points_downsample': num_points_downsample if downsample_method in ['fps', 'random'] else None
    }
    
    logger.info(f"Waymo benchmark processing complete: {num_samples_generated} samples generated")
    logger.info(f"Average translation distance: {stats['translation_distance']['mean']:.3f} ± {stats['translation_distance']['std']:.3f} m")
    logger.info(f"Average rotation angle: {stats['rotation_angle_deg']['mean']:.1f} ± {stats['rotation_angle_deg']['std']:.1f} degrees")
    logger.info(f"Average frame ID difference: {stats['frame_id_difference']['mean']:.1f} ± {stats['frame_id_difference']['std']:.1f}")
    logger.info(f"Average point counts - Source: {stats['source_point_count']['mean']:.0f}, Target: {stats['target_point_count']['mean']:.0f}")
    
    return num_samples_generated, stats


def process_sequence_with_loader(data_loader,
                                sequence: str,
                                output_dir: str,
                                num_samples_to_generate: int = 100,
                                min_frames_per_submap: int = 10,
                                max_frames_per_submap: int = 200,
                                min_spatial_threshold: float = 10.0,
                                max_spatial_threshold: float = 200.0,
                                min_submaps_per_sample: int = 2,
                                max_submaps_per_sample: int = 10,
                                voxel_size: float = 0.5,
                                downsample_method: str = "voxel",
                                num_points_downsample: Optional[int] = None,
                                start_frame: int = 0,
                                end_frame: Optional[int] = None,
                                max_frames_per_sequence: Optional[int] = None,
                                min_overlap_ratio: float = 0.1,
                                max_overlap_ratio: float = 0.8,
                                overlap_method: str = "fast",
                                min_frame_interval: int = 0,
                                max_frame_interval: Optional[int] = None,
                                overlap_voxel_size: float = 2.0,
                                max_attempts: int = 50,
                                enable_deskewing: bool = False,
                                random_drop_to_single_frame: bool = False) -> Tuple[int, Dict]:
    """
    Process a single sequence to generate training samples using the data loader.
    
    Args:
        random_drop_to_single_frame: If True, randomly select one submap in each sample and reduce it to a single frame.
                                   Other submaps can still vary within min/max_frames_per_submap range.
    
    Returns:
        Number of training samples generated
    """
    logger.info(f"Loading sequence {sequence} using data loader...")
    
    # Set the sequence in the data loader
    data_loader.set_sequence(
        sequence=sequence,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
     # Apply frame limiting if specified
    if max_frames_per_sequence is not None and len(data_loader) > max_frames_per_sequence:
        logger.info(f"Sequence has {len(data_loader)} frames, limiting to {max_frames_per_sequence} frames")
        
        # Randomly sample frames
        frame_indices = list(range(len(data_loader)))
        selected_indices = sorted(random.sample(frame_indices, max_frames_per_sequence))
        
        # Load data for selected frames
        frame_data_list = [data_loader[idx] for idx in tqdm(selected_indices, desc="Loading selected frames")]
        
        logger.info(f"Randomly sampled {len(selected_indices)} frames from indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        # Store original indices for deskewing
        original_indices = selected_indices
    else:
        # Load all frames
        frame_data_list = [data_loader[i] for i in tqdm(range(len(data_loader)), desc="Loading all frames")]
        original_indices = None
    
    
    # Extract poses, points, and normals
    poses = [frame_data['pose'] for frame_data in frame_data_list]
    points_list = [frame_data['points'] for frame_data in frame_data_list]
    normals_list = [frame_data.get('normals') for frame_data in frame_data_list]
    ts_list = [frame_data.get('timestamps') for frame_data in frame_data_list]
    frame_ids = [frame_data['frame_id'] for frame_data in frame_data_list]

    if 'prev_pose' in frame_data_list[0]:
        prev_poses = [frame_data['prev_pose'] for frame_data in frame_data_list]
    else:
        prev_poses = None
    
    # Apply deskewing to points if enabled and timestamps are available
    if enable_deskewing and any(ts is not None for ts in ts_list):
        logger.info("Applying deskewing to point clouds...")
        deskewed_points_list = []
        
        for i in tqdm(range(len(points_list)), desc="Deskewing point clouds"):
            if ts_list[i] is not None and i > 0:
                # Calculate relative pose from previous frame to current frame
                # Use original indices if frame limiting was applied
                if original_indices is not None:
                    # Get the original frame indices for current and previous frame
                    current_original_idx = original_indices[i]
                    prev_original_idx = original_indices[i-1]
                    
                    # Load the actual previous frame from the data loader
                    prev_frame_data = data_loader[prev_original_idx]
                    prev_pose = prev_frame_data['pose']

                else:
                    if prev_poses is not None:
                        prev_pose = prev_poses[i]
                    else:
                        prev_pose = poses[i-1]
                    # prev_pose = poses[i-1]                        

                current_pose = poses[i]     
            
                # Calculate relative pose
                relative_pose = np.linalg.inv(prev_pose) @ current_pose
                
                # Apply deskewing
                deskewed_points = dataset_utils.deskewing(
                    points=points_list[i],
                    ts=ts_list[i],
                    pose=relative_pose,
                    ts_mid_pose=0.5
                )

                deskewed_points_list.append(deskewed_points)
                logger.debug(f"Applied deskewing to frame {i} (original idx: {original_indices[i] if original_indices else i}) with {len(deskewed_points)} points")
            else:
                # No deskewing for first frame or frames without timestamps
                deskewed_points_list.append(points_list[i])
        
        # Replace original points with deskewed points
        points_list = deskewed_points_list
        logger.info("Deskewing completed")
    elif enable_deskewing:
        logger.info("Deskewing enabled but no timestamps available, skipping deskewing")
    else:
        logger.info("Deskewing disabled")
    
    logger.info(f"Final data: {len(poses)} poses and {len(points_list)} point clouds")
    
    # Generate K training samples
    logger.info(f"Generating {num_samples_to_generate} training samples...")
    sample_idx = 0
    
    # Statistics collection
    submap_counts = []
    submap_frame_counts = []
    temporal_differences = []
    spatial_differences = []
    
    for sample_num in range(num_samples_to_generate):
        logger.info(f"Generating sample {sample_num + 1}/{num_samples_to_generate}")
        
        # Generate submap boundaries for this sample (non-overlapping within sample)
        sample_submap_boundaries = generate_submap_boundaries_for_sample(
            frame_ids, 
            min_frames_per_submap, 
            max_frames_per_submap,
            random_drop_to_single_frame=random_drop_to_single_frame
        )
        
        # Calculate submap centers for this sample
        sample_submap_centers = []
        for start_frame_id, end_frame_id in sample_submap_boundaries:
            # Find array indices for the frame IDs
            start_idx = frame_ids.index(start_frame_id)
            end_idx = frame_ids.index(end_frame_id) + 1  # end_frame_id is inclusive
            
            centers = [dataset_utils.get_pose_center(poses[i]) for i in range(start_idx, end_idx)]
            submap_center = np.mean(centers, axis=0)
            sample_submap_centers.append(submap_center)
        
        # Select spatially close submaps from this sample
        selected_submap_indices = select_spatially_close_submaps(
            sample_submap_boundaries,
            sample_submap_centers,
            poses,
            points_list,
            frame_ids,
            min_spatial_threshold,
            max_spatial_threshold,
            min_submaps_per_sample,
            max_submaps_per_sample,
            min_overlap_ratio=min_overlap_ratio,
            max_overlap_ratio=max_overlap_ratio,
            overlap_method=overlap_method,
            min_frame_interval=min_frame_interval,
            max_frame_interval=max_frame_interval,
            overlap_voxel_size=overlap_voxel_size,
            max_attempts=max_attempts
        )
        
        if not selected_submap_indices:
            logger.warning(f"Could not find spatially close submaps for sample {sample_num + 1}, skipping")
            continue
        
        # Create submaps for this sample
        submaps = []
        submap_meta = []
        submap_normals = []
        
        for local_idx, global_submap_idx in enumerate(selected_submap_indices):
            start_frame_id, end_frame_id = sample_submap_boundaries[global_submap_idx]
            
            # Find array indices for the frame IDs
            start_idx = frame_ids.index(start_frame_id)
            end_idx = frame_ids.index(end_frame_id) + 1  # end_frame_id is inclusive
            frames_this_submap = end_idx - start_idx
            
            # Randomly select a frame within this submap to represent the pose
            if frames_this_submap > 0:
                rand_frame_idx = random.randint(start_idx, end_idx - 1)
                rand_pose = poses[rand_frame_idx]
                rand_frame_id = frame_ids[rand_frame_idx]
            else:
                rand_pose = None
                rand_frame_id = None

            submap_points, submap_normals_raw = create_submap_from_frames(
                points_list, poses, start_idx, frames_this_submap, normals_list
            )
            
            if len(submap_points) > 0:
                # Downsample submap using the specified method
                downsampled_points, downsampled_normals = dataset_utils.downsample_points(
                    points=submap_points,
                    normals=submap_normals_raw,
                    method=downsample_method,
                    voxel_size=voxel_size,
                    num_points=num_points_downsample
                )
                submaps.append(downsampled_points)
                submap_normals.append(downsampled_normals)
                
                # Store meta information
                meta = {
                    'start_frame': start_frame_id,
                    'end_frame': end_frame_id,  # End frame is inclusive
                    'submap_idx': global_submap_idx,
                    'local_submap_idx': local_idx,
                    'frames_this_submap': frames_this_submap,
                    'random_frame_id': rand_frame_id,
                    'pose_matrix': rand_pose if rand_pose is not None else None
                }
                submap_meta.append(meta)
        
        # Validate no overlap between submaps (double-check)
        if len(submap_meta) > 1:
            if not validate_no_overlap(submap_meta):
                logger.warning(f"Skipping sample {sample_num + 1} due to overlapping submaps")
                continue
        
        # Save training sample if we have enough valid submaps
        if len(submaps) >= min_submaps_per_sample:
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
            # Extract sequence name from output_dir path
            # sequence_name = os.path.basename(os.path.normpath(output_dir))
            # Get global transformation matrix
            global_transform = dataset_utils.get_global_transformation_matrix(sequence)
            
            # ThreeDMatch test set now uses the same logic as train set (poses from .pose.npy files)
            # No special coordinate transformation needed - poses are already correct
            
            save_training_sample(submaps, submap_meta, sample_dir, sample_idx, 
                                 os.path.basename(os.path.normpath(sequence)), submap_normals, global_transform_matrix=global_transform)
            
            # Collect statistics for this sample
            submap_counts.append(len(submaps))
            
            # Collect frame counts for each submap
            for meta in submap_meta:
                submap_frame_counts.append(meta.get('frames_this_submap', 0))
            
            # Calculate temporal and spatial differences between submaps
            if len(selected_submap_indices) > 1:
                # Temporal differences
                for i in range(len(selected_submap_indices)):
                    for j in range(i + 1, len(selected_submap_indices)):
                        start_i, _ = sample_submap_boundaries[selected_submap_indices[i]]
                        start_j, _ = sample_submap_boundaries[selected_submap_indices[j]]
                        
                        # Handle both integer and string frame IDs
                        try:
                            # Try to convert to integers for temporal difference calculation
                            start_i_int = int(start_i) if isinstance(start_i, str) else start_i
                            start_j_int = int(start_j) if isinstance(start_j, str) else start_j
                            temp_diff = abs(start_i_int - start_j_int)
                        except (ValueError, TypeError):
                            # If conversion fails, skip temporal difference for string frame IDs
                            temp_diff = 0
                        
                        temporal_differences.append(temp_diff)
                
                # Spatial differences
                sample_centers = [sample_submap_centers[idx] for idx in selected_submap_indices]
                for i in range(len(sample_centers)):
                    for j in range(i + 1, len(sample_centers)):
                        spatial_diff = np.linalg.norm(sample_centers[i] - sample_centers[j])
                        spatial_differences.append(spatial_diff)
            
            sample_idx += 1
            logger.info(f"Generated sample {sample_idx} with {len(submaps)} submaps: saved to {sample_dir}")
        else:
            logger.warning(f"Sample {sample_num + 1} has insufficient submaps ({len(submaps)} < {min_submaps_per_sample})")
    
    logger.info(f"Generated {sample_idx} training samples out of {num_samples_to_generate} attempts")
    
    # Calculate statistics
    stats = _calculate_statistics(submap_counts, submap_frame_counts, temporal_differences, spatial_differences)
    
    return sample_idx, stats 

def _generate_connected_groups_from_pose_graph(edges: List[Dict], node_info: List[Dict],
                                               num_groups: int, min_group_size: int, max_group_size: int,
                                               min_overlap_ratio: float = 0.01, max_overlap_ratio: float = 0.8,
                                               max_attempts: int = 50, same_stage_only: bool = False) -> List[List[int]]:
    """
    Generate connected groups of nodes from pose graph edges.
    
    Args:
        edges: List of edge dictionaries with source_id, target_id, overlap_ratio
        node_info: List of node information dictionaries
        num_groups: Number of groups to generate
        min_group_size: Minimum number of nodes per group
        max_group_size: Maximum number of nodes per group
        min_overlap_ratio: Minimum overlap ratio for edges to be considered
        max_overlap_ratio: Maximum overlap ratio for edges to be considered
        max_attempts: Maximum attempts to find valid groups
        same_stage_only: If True, each group will only contain nodes from the same stage
        
    Returns:
        List of groups, where each group is a list of node indices
    """
    if not edges or not node_info:
        return []
    
    # Build adjacency list from edges with valid overlap ratios
    adjacency = defaultdict(set)
    node_id_to_idx = {node['id']: idx for idx, node in enumerate(node_info)}
    
    for edge in edges:
        overlap_ratio = edge.get('overlap_ratio', 0.0)
        if min_overlap_ratio <= overlap_ratio <= max_overlap_ratio:
            source_idx = node_id_to_idx.get(edge['source_id'])
            target_idx = node_id_to_idx.get(edge['target_id'])
            if source_idx is not None and target_idx is not None:
                # If same_stage_only is enabled, only add edges between nodes of the same stage
                if same_stage_only:
                    source_stage = node_info[source_idx]['stage']
                    target_stage = node_info[target_idx]['stage']
                    if source_stage == target_stage:
                        adjacency[source_idx].add(target_idx)
                        adjacency[target_idx].add(source_idx)
                else:
                    adjacency[source_idx].add(target_idx)
                    adjacency[target_idx].add(source_idx)
    
    if not adjacency:
        logger.warning("No valid edges found for group generation")
        return []
    
    total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    logger.debug(f"Built adjacency graph with {len(adjacency)} connected nodes out of {len(node_info)} total nodes")
    logger.debug(f"Adjacency graph has {total_edges} valid edges")
    
    # Check connectivity distribution
    connectivity_stats = {}
    for node, neighbors in adjacency.items():
        degree = len(neighbors)
        connectivity_stats[degree] = connectivity_stats.get(degree, 0) + 1
    logger.debug(f"Node connectivity distribution: {dict(sorted(connectivity_stats.items()))}")
    
    # Generate groups using all available connected nodes
    # The same_stage_only constraint is already enforced in the adjacency graph construction above
    available_nodes = [node for node in range(len(node_info)) if node in adjacency]
    
    if same_stage_only:
        logger.debug("Same-stage constraint: Groups will only contain nodes from the same stage (enforced via adjacency)")
    
    groups = _generate_groups_from_node_list(
        available_nodes, adjacency, num_groups, min_group_size, max_group_size, max_attempts, node_info, same_stage_only
    )
    
    logger.info(f"Generated {len(groups)} connected groups from pose graph")
    return groups


def _generate_groups_from_node_list(available_nodes: List[int], adjacency: dict, 
                                   num_groups: int, min_group_size: int, max_group_size: int, 
                                   max_attempts: int, node_info: List[Dict] = None, same_stage_only: bool = False) -> List[List[int]]:
    """
    Generate connected groups from a specific list of nodes.
    
    Args:
        available_nodes: List of node indices to generate groups from
        adjacency: Adjacency dictionary for the graph
        num_groups: Number of groups to generate
        min_group_size: Minimum group size
        max_group_size: Maximum group size
        max_attempts: Maximum attempts per group
        node_info: List of node information (needed for same_stage_only)
        same_stage_only: If True, enforce same-stage constraint during group expansion
        
    Returns:
        List of generated groups
    """
    groups = []
    used_nodes = set()
    
    # Log stage distribution if same_stage_only is enabled
    if same_stage_only and node_info:
        stage_distribution = defaultdict(int)
        stage_connectivity = defaultdict(int)
        for node_idx in available_nodes:
            stage = node_info[node_idx]['stage']
            stage_distribution[stage] += 1
            stage_connectivity[stage] += len(adjacency.get(node_idx, []))
        logger.debug(f"Available nodes by stage: {dict(stage_distribution)}")
        logger.debug(f"Total connections by stage: {dict(stage_connectivity)}")
        
        # Check if each stage has enough nodes for minimum group size
        for stage, count in stage_distribution.items():
            if count < min_group_size:
                logger.warning(f"Stage {stage} has only {count} nodes, less than min_group_size {min_group_size}")
    
    logger.debug(f"Starting group generation: target={num_groups}, available_nodes={len(available_nodes)}, min_size={min_group_size}, max_size={max_group_size}")
    
    for group_idx in range(num_groups):
        best_group = None
        
        for attempt in range(max_attempts):
            # Determine group size for this attempt
            available_for_group = [node for node in available_nodes if node not in used_nodes]
            if len(available_for_group) < min_group_size:
                break  # Not enough nodes left
                
            target_group_size = random.randint(min_group_size, min(max_group_size, len(available_for_group)))
            
            # Start with a random available node that has connections
            available_connected_nodes = [node for node in available_for_group if node in adjacency]
            if not available_connected_nodes:
                break  # No more connected nodes available
                
            start_node = random.choice(available_connected_nodes)
            current_group = {start_node}
            
            # Grow the group using BFS-like approach
            candidates = list(adjacency[start_node] & set(available_for_group) - used_nodes)
            
            # If same_stage_only is enabled, filter candidates to same stage as start_node
            if same_stage_only and node_info:
                start_stage = node_info[start_node]['stage']
                candidates = [c for c in candidates if node_info[c]['stage'] == start_stage]
            
            while len(current_group) < target_group_size and candidates:
                # Choose next node that connects to the current group
                next_node = random.choice(candidates)
                current_group.add(next_node)
                
                # Add new candidates from this node
                new_candidates = adjacency[next_node] & set(available_for_group) - current_group - used_nodes
                
                # If same_stage_only is enabled, filter new candidates to same stage
                if same_stage_only and node_info:
                    current_stage = node_info[next_node]['stage']
                    new_candidates = {c for c in new_candidates if node_info[c]['stage'] == current_stage}
                
                candidates.extend(new_candidates)
                candidates = list(set(candidates) - current_group)  # Remove duplicates and already selected
                
                if not candidates and len(current_group) < target_group_size:
                    # Try to find more candidates from any node in current group
                    for group_node in current_group:
                        new_candidates = adjacency[group_node] & set(available_for_group) - current_group - used_nodes
                        
                        # If same_stage_only is enabled, filter candidates to same stage
                        if same_stage_only and node_info:
                            group_stage = node_info[group_node]['stage']
                            new_candidates = {c for c in new_candidates if node_info[c]['stage'] == group_stage}
                        
                        candidates.extend(new_candidates)
                    candidates = list(set(candidates))
            
            # Check if this group is valid (connected and meets size requirements)
            if len(current_group) >= min_group_size and _is_group_connected(current_group, adjacency):
                # Additional validation for same_stage_only constraint
                if same_stage_only and node_info:
                    stages_in_group = {node_info[node]['stage'] for node in current_group}
                    if len(stages_in_group) == 1:
                        best_group = list(current_group)
                        break
                    else:
                        logger.debug(f"Group validation failed: found {len(stages_in_group)} stages in group (expected 1)")
                else:
                    best_group = list(current_group)
                    break
        
        if best_group:
            groups.append(best_group)
            used_nodes.update(best_group)
            
            # Log group stage information for debugging
            if same_stage_only and node_info:
                group_stages = {node_info[node]['stage'] for node in best_group}
                logger.debug(f"Generated group {group_idx + 1} with {len(best_group)} nodes from stage(s): {group_stages}")
            else:
                logger.debug(f"Generated group {group_idx + 1} with {len(best_group)} nodes")
        else:
            break  # Could not generate more valid groups
    
    return groups


def _is_group_connected(group: set, adjacency: dict) -> bool:
    """Check if a group of nodes is connected."""
    if len(group) <= 1:
        return True
    
    # BFS to check connectivity
    group_list = list(group)
    visited = {group_list[0]}
    queue = [group_list[0]]
    
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor in group and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(group)


def process_nss_multi_dataset(data_loader,
                             output_dir: str,
                             split_type: str = 'train',
                             max_samples: Optional[int] = None,
                             sample_count_multiplier: float = 1.0,
                             voxel_size: float = 0.1,
                             downsample_method: str = "voxel",
                             num_points_downsample: Optional[int] = None,
                             min_graph_size: int = 10,
                             max_graph_size: int = 200,
                             min_submaps_per_sample: int = 2,
                             max_submaps_per_sample: int = 10,
                             filter_outliers: bool = True,
                             same_stage_only: bool = False,
                             preferred_stage: Optional[int] = None,
                             generate_groups: bool = False,
                             min_overlap_ratio: float = 0.01,
                             max_overlap_ratio: float = 0.8,
                             overlap_method: str = "fast",
                             overlap_voxel_size: float = 2.0,
                             max_attempts: int = 50) -> Tuple[int, Dict]:
    """
    Process NSS multiway dataset from pose graphs.
    Each pose graph becomes a training sample with multiple point clouds.
    
    Args:
        data_loader: NSS multi data loader instance (sequence interface)
        output_dir: Output directory for training samples
        split_type: Split type ('train', 'val', 'test')
        max_samples: Maximum number of samples to process (None for all)
        sample_count_multiplier: Multiplier for number of groups to generate per graph (when generate_groups=True)
        voxel_size: Voxel size for downsampling
        downsample_method: Downsampling method ('voxel', 'fps', 'random')
        num_points_downsample: Number of points for fps/random downsampling
        min_graph_size: Minimum number of nodes in pose graph
        max_graph_size: Maximum number of nodes in pose graph
        min_submaps_per_sample: Minimum number of submaps per sample (when generate_groups=True)
        max_submaps_per_sample: Maximum number of submaps per sample (when generate_groups=True)
        filter_outliers: Whether to filter outlier nodes
        same_stage_only: Whether to only extract point clouds from the same stage
        preferred_stage: Preferred stage to extract (if same_stage_only=True)
        generate_groups: Whether to generate multiple connected groups from each pose graph
        min_overlap_ratio: Minimum overlap ratio for edges to be considered valid
        max_overlap_ratio: Maximum overlap ratio for edges to be considered valid
        overlap_method: Method for calculating overlap (not used in NSS multi)
        overlap_voxel_size: Voxel size for overlap calculation (not used in NSS multi)
        max_attempts: Maximum attempts to generate valid groups
        
    Returns:
        Tuple of (number of samples generated, statistics)
    """
    logger.info(f"Processing NSS multiway dataset (split_type={split_type})")
    logger.info("NSS multiway processing: Preserving original point cloud quality (no downsampling applied)")
    logger.info("NSS multiway processing: Applying global transformations to align point clouds to common coordinate system")
    
    if generate_groups:
        logger.info(f"NSS multiway processing: Generating multiple connected groups from each pose graph")
        logger.info(f"NSS multiway processing: Groups per graph = int({sample_count_multiplier} * graph_size)")
        logger.info(f"NSS multiway processing: Group size range = {min_submaps_per_sample}-{max_submaps_per_sample} nodes")
        if same_stage_only:
            logger.info("NSS multiway processing: Each group will contain only point clouds from the same stage")
        else:
            logger.info("NSS multiway processing: Groups can contain point clouds from different stages")
    else:
        logger.info("NSS multiway processing: Using entire pose graphs as single training samples")
    
    if same_stage_only:
        if preferred_stage is not None:
            logger.info(f"NSS multiway processing: Extracting only same-stage point clouds (preferred stage: {preferred_stage})")
        else:
            logger.info("NSS multiway processing: Extracting only same-stage point clouds (using most common stage per graph)")
    
    # Set sequence in data loader
    data_loader.set_sequence(split_type)
    total_graphs = len(data_loader)
    
    if total_graphs == 0:
        logger.warning(f"No pose graphs found for split {split_type}")
        return 0, {}
    
    # Determine number of samples to process
    if max_samples is None:
        num_samples_to_process = total_graphs
    else:
        num_samples_to_process = min(max_samples, total_graphs)
    
    logger.info(f"Processing {num_samples_to_process} pose graphs (out of {total_graphs} available)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics collection
    graph_sizes = []
    node_counts = []
    building_counts = {}
    stage_combinations = {}
    same_stage_edges = 0
    cross_stage_edges = 0
    edge_overlap_ratios = []
    
    num_samples_generated = 0
    
    # Process each pose graph
    for graph_idx in tqdm(range(num_samples_to_process), desc="Processing NSS multiway graphs"):
        try:
            # Get pose graph data from sequence interface
            frame_data = data_loader[graph_idx]
            # Extract the actual graph data from the interface
            graph_data = frame_data['_nss_multi_graph_data']
            
            # Extract graph info first to check size before creating directories
            graph_name = graph_data['graph_name']
            graph_size = graph_data['graph_size']
            original_size = graph_data['original_size']
            
            # Check graph size constraints early to avoid creating unnecessary directories
            if original_size < min_graph_size:
                logger.debug(f"Skipping graph {graph_name}: size {original_size} < min_graph_size {min_graph_size}")
                continue
            if original_size > max_graph_size:
                logger.debug(f"Skipping graph {graph_name}: size {original_size} > max_graph_size {max_graph_size}")
                continue
            
            # Now extract the rest of the data
            point_clouds = graph_data['point_clouds']
            normals_list = graph_data['normals']
            global_transforms = graph_data['global_transforms']
            node_info = graph_data['node_info']
            edges = graph_data['edges']
            
            # When using group generation with same_stage_only, we don't pre-filter by stage
            # The stage filtering will happen during group generation
            if same_stage_only and not generate_groups and node_info:
                # Only apply stage pre-filtering when NOT using group generation (original behavior)
                # Get all stages in this graph
                stages_in_graph = [node['stage'] for node in node_info]
                stage_counts = {}
                for stage in stages_in_graph:
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
                # Log stage distribution for debugging
                logger.debug(f"Graph {graph_name}: Stage distribution = {stage_counts}")
                
                # Determine which stage to use
                if preferred_stage is not None and preferred_stage in stages_in_graph:
                    selected_stage = preferred_stage
                    logger.debug(f"Using preferred stage {preferred_stage} for graph {graph_name}")
                else:
                    # Use the most common stage (in case of ties, prefer lower stage numbers)
                    max_count = max(stage_counts.values())
                    stages_with_max_count = [stage for stage, count in stage_counts.items() if count == max_count]
                    selected_stage = min(stages_with_max_count)  # Prefer lower stage numbers in case of ties
                    
                    if len(stages_with_max_count) > 1:
                        logger.debug(f"Graph {graph_name}: Tie between stages {stages_with_max_count} (count: {max_count}), selecting stage {selected_stage}")
                    
                    if preferred_stage is not None:
                        logger.debug(f"Preferred stage {preferred_stage} not available in graph {graph_name}, using most common stage {selected_stage} (count: {max_count})")
                    else:
                        logger.debug(f"Using most common stage {selected_stage} (count: {max_count}) for graph {graph_name}")
                
                # Filter all data to only include nodes from the selected stage
                filtered_indices = [i for i, node in enumerate(node_info) if node['stage'] == selected_stage]
                
                if not filtered_indices:
                    logger.warning(f"No nodes found for stage {selected_stage} in graph {graph_name}, skipping")
                    continue
                
                # Apply filtering
                point_clouds = [point_clouds[i] for i in filtered_indices]
                normals_list = [normals_list[i] for i in filtered_indices]
                global_transforms = [global_transforms[i] for i in filtered_indices]
                node_info = [node_info[i] for i in filtered_indices]
                
                # Filter edges to only include edges between selected stage nodes
                selected_node_ids = {node_info[i]['id'] for i in range(len(filtered_indices))}
                edges = [edge for edge in edges if edge['source_id'] in selected_node_ids and edge['target_id'] in selected_node_ids]
                
                # Update graph size
                graph_size = len(point_clouds)
                
                logger.debug(f"Filtered graph {graph_name} to {graph_size} nodes from stage {selected_stage} (original: {original_size})")
                
                # Skip if filtered graph is too small
                if graph_size < min_graph_size:
                    logger.debug(f"Filtered graph {graph_name} has only {graph_size} nodes (< {min_graph_size}), skipping")
                    continue
            elif same_stage_only and generate_groups and node_info:
                # For group generation with same_stage_only, log available stages but don't pre-filter
                stages_in_graph = [node['stage'] for node in node_info]
                stage_counts = {}
                for stage in stages_in_graph:
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                logger.debug(f"Graph {graph_name}: Available stages for group generation = {stage_counts}")
                logger.debug("Stage filtering will be applied during group generation (each group will contain only same-stage nodes)")
            
            # Collect statistics
            graph_sizes.append(original_size)
            node_counts.append(graph_size)
            
            # Building and stage statistics
            for node in node_info:
                building = node['building']
                stage = node['stage']
                
                building_counts[building] = building_counts.get(building, 0) + 1
                
                # Count stage combinations in edges
                for edge in edges:
                    if edge['source_id'] == node['id']:
                        # Find target node
                        target_node = next((n for n in node_info if n['id'] == edge['target_id']), None)
                        if target_node:
                            stage_pair = (stage, target_node['stage'])
                            stage_combinations[stage_pair] = stage_combinations.get(stage_pair, 0) + 1
                            
                            if edge.get('same_stage', False):
                                same_stage_edges += 1
                            else:
                                cross_stage_edges += 1
                                
                            if 'overlap_ratio' in edge:
                                edge_overlap_ratios.append(edge['overlap_ratio'])
            
            # Generate samples - either entire graph or multiple groups
            if generate_groups:
                # Generate multiple connected groups from this pose graph
                num_groups_to_generate = max(1, int(sample_count_multiplier * graph_size))
                logger.debug(f"Generating {num_groups_to_generate} groups from graph {graph_name} (size: {graph_size})")
                
                groups = _generate_connected_groups_from_pose_graph(
                    edges=edges,
                    node_info=node_info,
                    num_groups=num_groups_to_generate,
                    min_group_size=min_submaps_per_sample,
                    max_group_size=max_submaps_per_sample,
                    min_overlap_ratio=min_overlap_ratio,
                    max_overlap_ratio=max_overlap_ratio,
                    max_attempts=max_attempts,
                    same_stage_only=same_stage_only
                )
                
                # Process each group as a separate training sample
                for group_idx, group_node_indices in enumerate(groups):
                    sample_dir = os.path.join(output_dir, f"sample_{num_samples_generated:06d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    group_point_clouds = [point_clouds[i] for i in group_node_indices]
                    group_normals = [normals_list[i] for i in group_node_indices]
                    group_transforms = [global_transforms[i] for i in group_node_indices]
                    group_node_info = [node_info[i] for i in group_node_indices]
                    
                    # Save point clouds for this group
                    for i, (points, normals, transform, node) in enumerate(zip(group_point_clouds, group_normals, group_transforms, group_node_info)):
                        # Apply global transformation to point cloud
                        if transform is not None and not np.allclose(transform, 0.0):
                            transformed_points = dataset_utils.transform_points(points, transform)
                            if normals is not None:
                                transformed_normals = dataset_utils.transform_normals(normals, transform)
                            else:
                                transformed_normals = None
                        else:
                            transformed_points = points
                            transformed_normals = normals
                            logger.warning(f"No valid global transformation for node {node['name']}, using original coordinates")
                        
                        # Save transformed point cloud
                        submap_filename = f"submap_{i:02d}_{node['name'].replace('.ply', '')}.ply"
                        submap_path = os.path.join(sample_dir, submap_filename)
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(transformed_points)
                        if transformed_normals is not None:
                            pcd.normals = o3d.utility.Vector3dVector(transformed_normals)
                        o3d.io.write_point_cloud(submap_path, pcd, write_ascii=False)
                        
                        # Save transformation matrix for reference
                        transform_filename = f"submap_{i:02d}_{node['name'].replace('.ply', '')}_transform.txt"
                        transform_path = os.path.join(sample_dir, transform_filename)
                        np.savetxt(transform_path, transform, fmt='%.6f')
                    
                    # Save metadata for this group sample
                    group_edges = [edge for edge in edges if edge['source_id'] in [node['id'] for node in group_node_info] and edge['target_id'] in [node['id'] for node in group_node_info]]
                    
                    sample_metadata = {
                        'graph_name': f"{graph_name}_group_{group_idx}",
                        'original_graph_name': graph_name,
                        'original_graph_size': original_size,
                        'processed_graph_size': graph_size,
                        'group_size': len(group_node_indices),
                        'group_index': group_idx,
                        'num_submaps': len(group_point_clouds),
                        'node_info': group_node_info,
                        'edges': group_edges,
                        'buildings': list(set(node['building'] for node in group_node_info)),
                        'stages': list(set(node['stage'] for node in group_node_info)),
                        'coordinate_system': 'global',
                        'transformations_applied': True,
                        'processing_mode': 'connected_groups',
                        'stage_filtering': {
                            'same_stage_only': same_stage_only,
                            'preferred_stage': preferred_stage,
                            'group_stages': list(set(node['stage'] for node in group_node_info)) if group_node_info else None,
                            'filtering_method': 'group_level' if same_stage_only else 'none'
                        },
                        'group_generation': {
                            'sample_count_multiplier': sample_count_multiplier,
                            'min_group_size': min_submaps_per_sample,
                            'max_group_size': max_submaps_per_sample,
                            'overlap_constraints': {
                                'min_overlap_ratio': min_overlap_ratio,
                                'max_overlap_ratio': max_overlap_ratio
                            }
                        },
                        'downsampling': {
                            'method': 'none',
                            'voxel_size': None,
                            'num_points': None,
                            'note': 'NSS multi preserves original point cloud quality'
                        }
                    }
                    
                    metadata_path = os.path.join(sample_dir, "metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(sample_metadata, f, indent=2, default=str)
                    
                    num_samples_generated += 1
                
                logger.debug(f"Generated {len(groups)} group samples from graph {graph_name}")
                
            else:
                # Original behavior: create one sample from entire graph
                sample_dir = os.path.join(output_dir, f"sample_{num_samples_generated:06d}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save point clouds and transformations (no downsampling for NSS multi, similar to NSS)
                for i, (points, normals, transform, node) in enumerate(zip(point_clouds, normals_list, global_transforms, node_info)):
                    # Skip downsampling for NSS multi dataset to preserve original point cloud quality
                    # This is consistent with NSS dataset processing
                    
                    # Apply global transformation to point cloud
                    if transform is not None and not np.allclose(transform, 0.0):
                        # Transform points to global coordinate system
                        transformed_points = dataset_utils.transform_points(points, transform)
                        
                        # Transform normals if available
                        if normals is not None:
                            transformed_normals = dataset_utils.transform_normals(normals, transform)
                        else:
                            transformed_normals = None
                    else:
                        # No valid transformation, use original points
                        transformed_points = points
                        transformed_normals = normals
                        logger.warning(f"No valid global transformation for node {node['name']}, using original coordinates")
                    
                    # Save transformed point cloud
                    submap_filename = f"submap_{i:02d}_{node['name'].replace('.ply', '')}.ply"
                    submap_path = os.path.join(sample_dir, submap_filename)
                    
                    # Create Open3D point cloud and save
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(transformed_points)
                    if transformed_normals is not None:
                        pcd.normals = o3d.utility.Vector3dVector(transformed_normals)
                    o3d.io.write_point_cloud(submap_path, pcd, write_ascii=False)
                    
                    # Save transformation matrix for reference
                    transform_filename = f"submap_{i:02d}_{node['name'].replace('.ply', '')}_transform.txt"
                    transform_path = os.path.join(sample_dir, transform_filename)
                    np.savetxt(transform_path, transform, fmt='%.6f')
                
                # Save metadata for the sample
                sample_metadata = {
                    'graph_name': graph_name,
                    'original_graph_size': original_size,
                    'processed_graph_size': graph_size,
                    'num_submaps': len(point_clouds),
                    'node_info': node_info,
                    'edges': edges,
                    'buildings': list(set(node['building'] for node in node_info)),
                    'stages': list(set(node['stage'] for node in node_info)),
                    'coordinate_system': 'global',
                    'transformations_applied': True,
                    'processing_mode': 'entire_graph',
                    'stage_filtering': {
                        'same_stage_only': same_stage_only,
                        'preferred_stage': preferred_stage,
                        'graph_stages': list(set(node['stage'] for node in node_info)) if node_info else None,
                        'filtering_method': 'graph_level' if same_stage_only and not generate_groups else 'none'
                    },
                    'downsampling': {
                        'method': 'none',
                        'voxel_size': None,
                        'num_points': None,
                        'note': 'NSS multi preserves original point cloud quality'
                    }
                }
                
                metadata_path = os.path.join(sample_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(sample_metadata, f, indent=2, default=str)
                
                num_samples_generated += 1
            
        except Exception as e:
            logger.error(f"Error processing pose graph {graph_idx}: {e}")
            continue
    
    # Calculate final statistics
    stats = {
        'total_graphs_processed': num_samples_generated,
        'total_graphs_available': total_graphs,
        'graph_sizes': {
            'mean': np.mean(graph_sizes) if graph_sizes else 0,
            'std': np.std(graph_sizes) if graph_sizes else 0,
            'min': np.min(graph_sizes) if graph_sizes else 0,
            'max': np.max(graph_sizes) if graph_sizes else 0
        },
        'node_counts': {
            'mean': np.mean(node_counts) if node_counts else 0,
            'std': np.std(node_counts) if node_counts else 0,
            'min': np.min(node_counts) if node_counts else 0,
            'max': np.max(node_counts) if node_counts else 0
        },
        'building_distribution': building_counts,
        'stage_combinations': {f"{s[0]}->{s[1]}": count for s, count in stage_combinations.items()},
        'same_stage_edges': same_stage_edges,
        'cross_stage_edges': cross_stage_edges,
        'processing_method': 'connected_groups' if generate_groups else 'multiway_graphs',
        'coordinate_system': 'global',
        'transformations_applied': True,
        'group_generation': {
            'enabled': generate_groups,
            'sample_count_multiplier': sample_count_multiplier if generate_groups else None,
            'min_group_size': min_submaps_per_sample if generate_groups else None,
            'max_group_size': max_submaps_per_sample if generate_groups else None,
            'overlap_constraints': {
                'min_overlap_ratio': min_overlap_ratio,
                'max_overlap_ratio': max_overlap_ratio
            } if generate_groups else None
        },
        'stage_filtering': {
            'same_stage_only': same_stage_only,
            'preferred_stage': preferred_stage,
            'filtering_method': 'group_level' if (same_stage_only and generate_groups) else ('graph_level' if same_stage_only else 'none')
        },
        'downsampling_applied': False,  # NSS multi preserves original point clouds
        'downsample_method': 'none',
        'voxel_size': None,
        'num_points_downsample': None
    }
    
    if edge_overlap_ratios:
        stats['edge_overlaps'] = {
            'mean': np.mean(edge_overlap_ratios),
            'std': np.std(edge_overlap_ratios),
            'min': np.min(edge_overlap_ratios),
            'max': np.max(edge_overlap_ratios)
        }
    
    logger.info(f"NSS multiway processing complete: {num_samples_generated} samples generated")
    if generate_groups:
        logger.info(f"Processing mode: Connected groups generation (multiplier: {sample_count_multiplier})")
        logger.info(f"Group size constraints: {min_submaps_per_sample}-{max_submaps_per_sample} nodes per group")
    else:
        logger.info("Processing mode: Entire pose graphs as single samples")
    logger.info(f"Same-stage edges: {same_stage_edges}, Cross-stage edges: {cross_stage_edges}")
    logger.info(f"Average graph size: {stats['graph_sizes']['mean']:.1f} ± {stats['graph_sizes']['std']:.1f}")
    logger.info(f"Building distribution: {building_counts}")
    logger.info("All point clouds have been transformed to global coordinate system using pose graph transformations")
    
    return num_samples_generated, stats

def set_random_seeds(seed: int):
    """
    Set random seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set PyTorch to use deterministic algorithms when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set all random seeds to: {seed}")


def process_tls_dataset(data_loader,
                        output_dir: str,
                        sequence_name: str,
                        max_samples: Optional[int] = None,
                        sample_count_multiplier: float = 0.1,
                        min_frames_per_submap: int = 1,
                        max_frames_per_submap: int = 1, # Fixed to 1 for TLS as each PLY is a submap
                        min_spatial_threshold: float = 0.0, # Not used for TLS direct processing
                        max_spatial_threshold: float = 9999.0, # Not used for TLS direct processing
                        min_submaps_per_sample: int = 2,
                        max_submaps_per_sample: int = 10,
                        min_overlap_ratio: float = 0.001,
                        max_overlap_ratio: float = 0.8,
                        min_frame_interval: int = 0,
                        max_frame_interval: Optional[int] = None,
                        overlap_method: str = 'fast',
                        overlap_voxel_size: float = 2.0,
                        max_attempts: int = 50,
                        voxel_size: float = 0.25,
                        downsample_method: str = 'voxel',
                        num_points_downsample: Optional[int] = None,
                        enable_deskewing: bool = False, # Not applicable for TLS static scans
                        max_frames_per_sequence: Optional[int] = None # Not applicable for TLS direct processing
                        ) -> Tuple[int, Dict[str, Any]]:
    """
    Processes TLS dataset (ETH, WHU_TLS) by treating each AlignedPointCloud PLY file
    as a single "frame" and generating samples for training.
    
    Args:
        data_loader: An instance of TLSSequenceInterface.
        output_dir: The base output directory for the processed data.
        sequence_name: The name of the current sequence (e.g., 'ETH', 'WHU_TLS').
        max_samples: Maximum number of samples to generate for this sequence.
        sample_count_multiplier: Multiplier for automatically determining num_samples.
        min_frames_per_submap: Minimum frames to combine into each submap (should be 1 for TLS).
        max_frames_per_submap: Maximum frames to combine into each submap (should be 1 for TLS).
        min_spatial_threshold: Minimum spatial distance between submap centers.
        max_spatial_threshold: Maximum spatial distance between submap centers.
        min_submaps_per_sample: Minimum number of submaps per training sample.
        max_submaps_per_sample: Maximum number of submaps per training sample.
        min_overlap_ratio: Minimum overlap ratio between submaps.
        max_overlap_ratio: Maximum overlap ratio between submaps.
        min_frame_interval: Minimum interval between first frame IDs of submaps.
        overlap_method: Method to calculate overlap ratio.
        overlap_voxel_size: Voxel size for overlap calculation.
        max_attempts: Maximum number of attempts to find valid submap combinations.
        voxel_size: Voxel size for final submap downsampling.
        downsample_method: Downsampling method for submaps.
        num_points_downsample: Target number of points after downsampling.
        enable_deskewing: Whether to enable deskewing (ignored for TLS).
        max_frames_per_sequence: Maximum frames to load per sequence (ignored for TLS).

    Returns:
        A tuple of (num_generated_samples, statistics).
    """
    logger.info(f"Starting processing for TLS sequence '{sequence_name}'...")

    data_loader.set_sequence(sequence_name)
    total_frames = len(data_loader)

    if total_frames == 0:
        logger.warning(f"No frames found for TLS sequence '{sequence_name}'. Skipping.")
        return 0, {}

    # Adjust min/max submaps per sample to not exceed total_frames
    min_submaps_per_sample = min(min_submaps_per_sample, total_frames)
    max_submaps_per_sample = min(max_submaps_per_sample, total_frames)
    
    # Ensure min_submaps_per_sample is at least 1 if total_frames > 0
    if total_frames > 0:
        min_submaps_per_sample = max(1, min_submaps_per_sample)

    # Determine number of samples to generate
    
    num_samples_to_generate = max(1, int(total_frames * sample_count_multiplier))
    if max_samples is not None:
        num_samples_to_generate = min(num_samples_to_generate, max_samples)
    
    logger.info(f"Generating {num_samples_to_generate} samples for TLS sequence '{sequence_name}' (total frames: {total_frames})")

    generated_samples = 0
    generated_points_counts = []

    # For TLS, each item from the data_loader is already a "submap" (a single PLY).
    # We need to combine these "submaps" into multi-submap training samples.
    
    # 1. Load all frame data from the sequence once
    logger.info(f"Loading all {total_frames} frames for TLS sequence '{sequence_name}'...")
    all_frames_data = [data_loader[i] for i in tqdm(range(total_frames), desc="Loading TLS frames")]

    points_list = [frame['points'] for frame in all_frames_data]
    normals_list = [frame['normals'] for frame in all_frames_data]
    poses = [frame['pose'] for frame in all_frames_data] # Identity poses for TLS
    frame_ids = [frame['frame_id'] for frame in all_frames_data]

    # For TLS, each frame is a submap. So submap_boundaries are just (frame_id, frame_id)
    submap_boundaries = [(fid, fid) for fid in frame_ids]
    # Submap centers are derived from poses (which are identity, so all centers are origin)
    submap_centers = [dataset_utils.get_pose_center(p) for p in poses]

    # Statistics collection
    submap_counts = []
    submap_frame_counts = [] # Will always be 1 for TLS
    temporal_differences = []
    spatial_differences = []

    for sample_idx in tqdm(range(num_samples_to_generate), desc=f"Generating samples for {sequence_name}"):
        # Select spatially close submaps (frames) using the utility function
        selected_indices = select_spatially_close_submaps(
            submap_boundaries=submap_boundaries,
            submap_centers=submap_centers,
            poses=poses,
            points_list=points_list,
            frame_ids=frame_ids,
            min_spatial_threshold=min_spatial_threshold, # Effectively unused if poses are identity
            max_spatial_threshold=max_spatial_threshold, # Effectively unused if poses are identity
            min_submaps_per_sample=min_submaps_per_sample,
            max_submaps_per_sample=max_submaps_per_sample,
            min_overlap_ratio=min_overlap_ratio,
            max_overlap_ratio=max_overlap_ratio,
            overlap_method=overlap_method,
            min_frame_interval=min_frame_interval,
            max_frame_interval=max_frame_interval,
            overlap_voxel_size=overlap_voxel_size,
            max_attempts=max_attempts
        )

        if not selected_indices:
            logger.debug(f"Could not find spatially close submaps for sample {sample_idx} in sequence {sequence_name}. Skipping.")
            continue

        # Retrieve the selected submap data based on indices
        current_sample_submaps_data = [all_frames_data[idx] for idx in selected_indices]

        # Prepare sample data for HDF5 / PLY saving
        sample_data = {
            'submaps': [],
            'relative_poses': [],
            'overlaps': [],
            'pair_indices': []
        }

        # Add submaps
        for submap_data in current_sample_submaps_data:
            sample_data['submaps'].append({
                'points': submap_data['points'],
                'normals': submap_data['normals'],
                'frame_id': submap_data['frame_id'],
                'pose': submap_data['pose'] # This will be identity for TLS
            })
            generated_points_counts.append(len(submap_data['points']))
            submap_frame_counts.append(1) # Each TLS frame is 1 submap

        # Generate all pairwise relative poses and overlaps
        # For TLS, poses are identity, so relative_pose is also identity
        for i in range(len(current_sample_submaps_data)):
            for j in range(i + 1, len(current_sample_submaps_data)):
                submap_i_data = current_sample_submaps_data[i]
                submap_j_data = current_sample_submaps_data[j]

                relative_pose = np.eye(4, dtype=np.float32) 
                
                overlap = dataset_utils.calculate_point_cloud_overlap_ratio_fast(
                    submap_i_data['points'], submap_j_data['points'], overlap_voxel_size, 20000
                )
                
                sample_data['pair_indices'].append((i, j))
                sample_data['relative_poses'].append(relative_pose)
                sample_data['overlaps'].append(overlap)

                # Collect temporal and spatial differences for statistics
                # For TLS with identity poses, spatial difference is always 0.0
                # Temporal difference is also 0.0 as there's no inherent time order in static scans
                spatial_differences.append(0.0)
                temporal_differences.append(0.0) 
        
        # Save sample as PLY files
        output_sub_dir = os.path.join(output_dir, f"sample_{generated_samples:06d}")
        os.makedirs(output_sub_dir, exist_ok=True)
        
        # Save each submap
        submap_paths = []
        for i, submap in enumerate(sample_data['submaps']):
            ply_path = os.path.join(output_sub_dir, f"submap_{i:02d}.ply")
            dataset_utils.save_points_to_ply(submap['points'], ply_path, normals=submap['normals'])
            submap_paths.append(os.path.relpath(ply_path, output_dir))
            
        # Save metadata for the sample
        sample_metadata = {
            'num_submaps': len(sample_data['submaps']),
            'submap_frame_ids': [s['frame_id'] for s in sample_data['submaps']],
            'submap_paths': submap_paths,
            'pair_indices': sample_data['pair_indices'],
            'relative_poses': [p.tolist() for p in sample_data['relative_poses']],
            'overlaps': sample_data['overlaps'],
            'global_poses': [s['pose'].tolist() for s in sample_data['submaps']] # These are identity for TLS
        }
        with open(os.path.join(output_sub_dir, "metadata.json"), 'w') as f:
            json.dump(sample_metadata, f, indent=2)
            
        generated_samples += 1
        submap_counts.append(len(sample_data['submaps']))

    # Compile statistics
    stats = _calculate_statistics(submap_counts, submap_frame_counts, temporal_differences, spatial_differences)
    
    logger.info(f"Finished processing TLS sequence '{sequence_name}'. Generated {generated_samples} samples.")
    
    return generated_samples, stats 