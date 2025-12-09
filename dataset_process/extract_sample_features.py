#!/usr/bin/env python3
"""
Extract Sample Features using miniSpinNet

This script takes the output from generate_training_samples (either HDF5 file or folders)
and extracts features for each sample using miniSpinNet encoder.

For each sample:
1. Load all submaps (point clouds) in the sample
2. Combine all points from all submaps
3. Apply farthest point sampling to get K total points
4. Use miniSpinNet to extract features for each sampled point
5. Save sampled points + features as PLY files maintaining folder structure

Usage:
    python ./dataset_process/extract_sample_features.py --input /path/to/training_data --output /path/to/features

    # For an indoor dataset (for example, NSS)
    python ./dataset_process/extract_sample_features.py --input ./dataset/lidar_rpf_training_data/nss_pair_v1 --output ./dataset/lidar_rpf_training_data/nss_pair_v1_processed_db_05 --des_r 0.5 --voxel_size 0.1 -r 0.5 --log_level DEBUG

"""

import os
import sys
import numpy as np
import h5py
import open3d as o3d
import torch
import argparse
import logging
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Union
import json
import time
from datetime import datetime

# Add the current directory and parent directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import necessary modules
from utils import dataset_utils
from utils.spinnet.patch_embedder import MiniSpinNet
from rectified_point_flow.data.transform import center_pcd, rotate_pcd_yaw
from utils.processing_utils import set_random_seeds
from utils.io_utils import get_dataset_name, convert_to_hdf5, load_sample_from_folder, load_sample_from_hdf5, save_processed_sample
from utils.feature_extraction_metadata_utils import save_processing_metadata, print_detailed_statistics
from utils.validation_utils import _validate_and_setup_args
from utils.dataset_utils import save_num_points_to_folder
from utils.split_utils import copy_and_update_data_split
from utils.point_sampling_utils import calculate_adaptive_sample_count_per_part, allocate_fps_points, apply_batched_fps

# Setup logging
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Feature extractor using miniSpinNet."""
    
    def __init__(self, 
                 model_config: Dict = None,
                 des_r: float = 3.0,
                 is_aligned_to_global_z: bool = True,
                 checkpoint_path: str = None,
                 device: str = 'auto'):
        """
        Initialize the feature extractor.
        
        Args:
            model_config: Configuration for miniSpinNet model
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Default model configuration
        default_config = {
            'num_points_per_patch': 512,
            'rad_n': 3,
            'azi_n': 20,
            'ele_n': 7,
            'delta': 0.8,
            'voxel_sample': 10,
        }
        
        if model_config:
            default_config.update(model_config)
        
        self.model_config = default_config
        self.model = self._build_model()
        self.des_r = des_r
        self.is_aligned_to_global_z = is_aligned_to_global_z
        
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
            else:
                logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
                logger.info("Using randomly initialized miniSpinNet weights")
            
        logger.info(f"FeatureExtractor initialized on device: {self.device}")
        # logger.info(f"Model config: {self.model_config}")
        # logger.info(f"Model architecture: {self.model}")

        # Model Parameter Info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")
    
    def _build_model(self) -> MiniSpinNet:
        """Build miniSpinNet model."""
        model = MiniSpinNet(**self.model_config)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            # Load the checkpoint and accommodate different formats
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Filter to only include 'Desc.' parameters and remove the prefix
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('Desc.'):
                    # Remove 'Desc.' prefix to match model parameter names
                    new_key = key[5:]  # Remove first 5 characters ('Desc.')
                    filtered_state_dict[new_key] = value
            
            # Load the filtered state dict into the model
            self.model.load_state_dict(filtered_state_dict, strict=False)
            self.model.eval()
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")

            # Check parameters for specific layers
            for layer_name, layer in [('pnt_layer', self.model.pnt_layer), 
                                      ('pool_layer', self.model.pool_layer), 
                                      ('conv_net', self.model.conv_net)]:
                param_name, param_tensor = next(layer.named_parameters())
                logger.debug(f"Checking sample parameter for '{layer_name}.{param_name}':")
                logger.debug(f"  - Mean: {param_tensor.data.mean():.6f}, Std: {param_tensor.data.std():.6f}")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def extract_features(self, points: Union[np.ndarray, torch.Tensor], 
                        keypoints: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Extract features for a point cloud.
        
        Args:
            points: Input point cloud (N, 3) - numpy array or torch tensor
            keypoints: Keypoints for feature extraction (K, 3) - numpy array or torch tensor. If None, use all points.
        
        Returns:
            Dictionary containing extracted features and metadata
        """
        def _to_tensor_with_batch(data):
            """Convert data to tensor with batch dimension."""
            if isinstance(data, torch.Tensor):
                tensor = data.float().to(self.device)
                return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor
            else:
                return torch.from_numpy(data).float().unsqueeze(0).to(self.device)
        
        # Handle empty inputs
        points_is_tensor = isinstance(points, torch.Tensor)
        is_empty = points.numel() == 0 if points_is_tensor else len(points) == 0
        if is_empty:
            empty_result = torch.tensor([]) if points_is_tensor else np.array([])
            return {'features': empty_result, 'keypoints': empty_result}
        
        # Use all points as keypoints if not specified
        if keypoints is None:
            keypoints = points.clone() if points_is_tensor else points.copy()
        
        # Convert to tensors with batch dimension
        points_tensor = _to_tensor_with_batch(points)
        keypoints_tensor = _to_tensor_with_batch(keypoints)
        
        with torch.no_grad():
            try:
                # Extract features using miniSpinNet
                result = self.model(
                    pts=points_tensor,
                    kpts=keypoints_tensor,
                    des_r=self.des_r,
                    is_aligned_to_global_z=self.is_aligned_to_global_z
                )
                
                # Extract features (descriptors)
                features = result['desc']  # Keep as tensor: (1, K, feature_dim)
                features = features.squeeze(0) if features.dim() == 3 else features  # Remove batch dim
                
                # Return in same format as input
                return {
                    'features': features if points_is_tensor else features.cpu().numpy(),
                    'keypoints': keypoints,
                    }
                
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                # Return empty features on failure
                feature_dim = 32  # Default feature dimension for miniSpinNet
                keypoint_len = keypoints.shape[0] if keypoints.dim() > 1 else len(keypoints)
                
                if points_is_tensor:
                    empty_features = torch.zeros(keypoint_len, feature_dim, device=self.device)
                else:
                    empty_features = np.zeros((len(keypoints), feature_dim))
                
                return {
                'features': empty_features,
                    'keypoints': keypoints,
                }


class SampleProcessor:
    """Process training samples and extract features."""
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor,
                 num_points: int = 5000,
                 skip_point_sampling: bool = False,
                 remove_outliers: bool = True,
                 outlier_nb_neighbors: int = 20,
                 outlier_std_ratio: float = 2.0,
                 min_points_per_part: int = 100,
                 max_points_per_part: int = 10000,
                 global_seed: int = 42,
                 allocation_method: str = 'point_count',
                 voxel_size: float = 1.0,
                 voxel_ratio: float = 0.1):
        """
        Initialize sample processor.
        
        Args:
            feature_extractor: Feature extractor instance
            num_points: Number of points to sample using FPS (used for fixed allocation methods)
            remove_outliers: Whether to remove statistical outliers for FPS sampling (not from feature extraction context)
            outlier_nb_neighbors: Number of neighbors to consider for outlier removal
            outlier_std_ratio: Standard deviation ratio threshold for outlier removal
            min_points_per_part: Minimum number of points each part should have after FPS
            max_points_per_part: Maximum number of points each part should have after FPS
            global_seed: Global random seed for all random operations including FPS
            allocation_method: Method for allocating FPS points ('point_count', 'spatial_coverage', or 'voxel_adaptive')
            voxel_size: Voxel size in meters for spatial coverage calculation (default: 1.0)
            voxel_ratio: Ratio of occupied voxels to sample points for voxel_adaptive method (default: 0.1)
        """
        self.feature_extractor = feature_extractor
        self.num_points = num_points
        self.skip_point_sampling = skip_point_sampling
        self.remove_outliers = remove_outliers
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.min_points_per_part = min_points_per_part
        self.max_points_per_part = max_points_per_part
        self.global_seed = global_seed
        self.allocation_method = allocation_method
        self.voxel_size = voxel_size
        self.voxel_ratio = voxel_ratio

        # Validate allocation method
        if self.allocation_method not in ['point_count', 'spatial_coverage', 'voxel_adaptive']:
            raise ValueError(f"allocation_method must be 'point_count', 'spatial_coverage', or 'voxel_adaptive', got: {self.allocation_method}")
    
    def _set_sample_seed(self, sample_idx: int = 0):
        """
        Set random seed for processing a specific sample.
        Combines global_seed with sample index for deterministic but varied randomness.
        
        Args:
            sample_idx: Index of the current sample being processed
        """
        sample_seed = self.global_seed + sample_idx * 1000
        
        # Set seeds for this sample processing
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        logger.debug(f"Set sample seed to: {sample_seed} (global: {self.global_seed}, sample: {sample_idx})")
    
    def process_sample(self, parts_points: List[np.ndarray], 
                      parts_normals: List[Optional[np.ndarray]] = None) -> List[Dict[str, np.ndarray]]:
        """
        Process a single sample with processing pipeline:
        1. Prepare parts (no augmentation applied)
        2. Remove statistical outliers for FPS sampling (optional)
        3. Apply proportional batched FPS on filtered data (optimized tensor operations)
        4. Extract features using original data (with outliers) as context and sampled points as keypoints
        5. Return sampled coordinates and features for each part separately
        
        Optimizations:
        - Keeps data as tensors throughout FPS and feature extraction pipeline
        - Only converts to numpy at the very end for storage
        - Avoids redundant tensor/numpy conversions
        - Uses seeded random operations for reproducible results
        
        Args:
            parts_points: List of point clouds for each part
            parts_normals: List of optional normals for each part
            
        Returns:
            List of dictionaries, one for each part, containing processed data
        """
        if not parts_points or len(parts_points) == 0:
            return []
        
        n_parts = len(parts_points)
        parts_normals = parts_normals or [None] * n_parts
        
        # Step 1: Prepare parts (no centering or rotation needed)
        logger.debug(f"Processing {n_parts} parts")
        
        original_parts = []  # Store original coordinates
        original_normals = []
        
        for i, (points, normals) in enumerate(zip(parts_points, parts_normals)):
            if len(points) == 0:
                continue
                
            # Store original points and normals directly
            original_parts.append(points.copy())
            original_normals.append(normals.copy() if normals is not None else None)
        
        if not original_parts:
            return {
                'sampled_points': np.array([]),
                'sampled_normals': None,
                'features': np.array([])
            }

        # Optional fast path: skip FPS and use all points as keypoints
        if self.skip_point_sampling:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            part_results = []
            for i in range(len(original_parts)):
                points_tensor = torch.from_numpy(original_parts[i]).float().to(device)
                feat_res = self.feature_extractor.extract_features(
                    points=points_tensor,
                    keypoints=points_tensor,
                )
                features = feat_res['features']
                part_results.append({
                    'sampled_points': original_parts[i],
                    'sampled_normals': original_normals[i] if original_normals[i] is not None else None,
                    'features': features.cpu().numpy() if isinstance(features, torch.Tensor) else features,
                })
            return part_results
        
        # Step 2: Remove statistical outliers for FPS (optional)
        fps_parts = original_parts  # Use original parts for FPS sampling
        fps_normals = original_normals  # Use original normals
        
        if self.remove_outliers:
            logger.debug(f"Removing statistical outliers from {len(original_parts)} parts for FPS sampling")
            
            fps_parts = []
            fps_normals = []
            
            for i, (orig_part, orig_normals_i) in enumerate(zip(original_parts, original_normals)):
                if len(orig_part) < self.outlier_nb_neighbors:
                    # If part has fewer points than required neighbors, skip outlier removal
                    logger.debug(f"Part {i} has {len(orig_part)} points, less than required neighbors {self.outlier_nb_neighbors}, skipping outlier removal")
                    fps_parts.append(orig_part)
                    fps_normals.append(orig_normals_i)
                    continue
                
                try:
                    # Create point cloud for outlier removal
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(orig_part)
                    
                    # Remove statistical outliers
                    pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
                        nb_neighbors=self.outlier_nb_neighbors,
                        std_ratio=self.outlier_std_ratio
                    )
                    
                    inlier_indices = np.array(inlier_indices)
                    
                    if len(inlier_indices) == 0:
                        logger.warning(f"All points removed as outliers from part {i}, keeping original for FPS")
                        fps_parts.append(orig_part)
                        fps_normals.append(orig_normals_i)
                    else:
                        # Apply the same filtering to coordinates and normals for FPS
                        fps_parts.append(orig_part[inlier_indices])
                        
                        if orig_normals_i is not None:
                            fps_normals.append(orig_normals_i[inlier_indices])
                        else:
                            fps_normals.append(None)
                        
                        outliers_removed = len(orig_part) - len(inlier_indices)
                        outlier_ratio = outliers_removed / len(orig_part) * 100
                        logger.debug(f"Part {i}: removed {outliers_removed} outliers ({outlier_ratio:.1f}%) for FPS, kept {len(inlier_indices)} points")
                
                except Exception as e:
                    logger.warning(f"Outlier removal failed for part {i}: {e}, keeping original for FPS")
                    fps_parts.append(orig_part)
                    fps_normals.append(orig_normals_i)
        
        # Step 3: Apply proportional batched FPS with minimum points per part constraint
        # Calculate adaptive sample count for voxel_adaptive method
        if self.allocation_method == 'voxel_adaptive':
            # Calculate adaptive sample count based on occupied voxels after outlier removal
            adaptive_sample_counts_per_part = calculate_adaptive_sample_count_per_part(
                fps_parts, self.voxel_size, self.voxel_ratio, self.min_points_per_part, self.max_points_per_part
            )
            total_adaptive_points = sum(adaptive_sample_counts_per_part)
            logger.debug(f"Applying proportional batched FPS to get {total_adaptive_points} total points using {self.allocation_method} allocation")
            
            # For voxel adaptive, pass the actual point arrays and per-part sample counts
            target_per_part = allocate_fps_points(
                fps_parts, 
                self.allocation_method,
                self.num_points, # Not used for voxel_adaptive here, but kept for function signature consistency
                self.min_points_per_part,
                self.voxel_size,
                self.voxel_ratio,
                total_sample_points=adaptive_sample_counts_per_part
            )
        else:
            logger.debug(f"Applying proportional batched FPS to get {self.num_points} total points using {self.allocation_method} allocation")
            
            # Pass appropriate data based on allocation method
            if self.allocation_method == 'spatial_coverage':
                # For spatial coverage, pass the actual point arrays
                target_per_part = allocate_fps_points(
                    fps_parts, 
                    self.allocation_method,
                    self.num_points,
                    self.min_points_per_part,
                    self.voxel_size,
                    self.voxel_ratio,
                )
            else: # point_count method
                # For point count, pass the point counts
                pts_per_part = np.array([len(part) for part in fps_parts])
                target_per_part = allocate_fps_points(
                    pts_per_part, 
                    self.allocation_method,
                    self.num_points,
                    self.min_points_per_part,
                    self.voxel_size,
                    self.voxel_ratio,
                )
        
        # Log the final allocation
        pts_per_part = np.array([len(part) for part in fps_parts])
        for i, (original_count, target_count) in enumerate(zip(pts_per_part, target_per_part)):
            logger.debug(f"Part {i}: {original_count} -> {target_count} points")
        
        # Use appropriate target for logging
        if self.allocation_method == 'voxel_adaptive':
            target_total = total_adaptive_points
        else:
            target_total = self.num_points
        logger.debug(f"Total points allocated: {target_per_part.sum()} (target: {target_total})")
        
        # Apply batched FPS using PyTorch3D
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare batched data for PyTorch3D FPS
        batch_parts = []
        batch_normals = []
        batch_lengths = []
        batch_k = []
        
        for i, (part, part_normals, target_points) in enumerate(
            zip(fps_parts, fps_normals, target_per_part)
        ):
            if target_points == 0:
                continue
                
            batch_parts.append(torch.from_numpy(part).float())
            if part_normals is not None:
                batch_normals.append(torch.from_numpy(part_normals).float())
            else:
                batch_normals.append(None)
            
            batch_lengths.append(len(part))
            batch_k.append(min(target_points, len(part)))  # Don't exceed available points
        
        if not batch_parts:
            return []
        
        # Pad to same length for batching
        max_points = max(batch_lengths)
        padded_parts = []
        padded_normals = []
        
        for i, (part, norms) in enumerate(zip(batch_parts, batch_normals)):
            if len(part) < max_points:
                # we need to actually avoid this case (pad zero is dangerous)
                # Pad with zeros
                pad_size = max_points - len(part)
                part_padded = torch.cat([part, torch.zeros(pad_size, 3)], dim=0)
                if norms is not None:
                    norms_padded = torch.cat([norms, torch.zeros(pad_size, 3)], dim=0)
                else:
                    norms_padded = None
            else:
                part_padded = part
                norms_padded = norms
            
            padded_parts.append(part_padded)
            padded_normals.append(norms_padded)
        
        # Stack into batch tensors
        batch_parts_tensor = torch.stack(padded_parts).to(device)  # (N, max_points, 3)
        batch_lengths_tensor = torch.tensor(batch_lengths, dtype=torch.int64).to(device)  # (N,)
        batch_k_tensor = torch.tensor(batch_k, dtype=torch.int64).to(device)  # (N,)
        
        # Apply batched FPS using PyTorch3D (no fallback!)
        fps_start_time = time.perf_counter()
        sampled_parts, indices_tensor = apply_batched_fps(
            batch_parts_tensor, batch_lengths_tensor, batch_k_tensor, self.global_seed, device
        )
        fps_end_time = time.perf_counter()
        fps_duration = fps_end_time - fps_start_time
        logger.debug(f"FPS computation time: {fps_duration:.4f} seconds for {len(batch_parts_tensor)} parts")
        
        # Extract sampled points using indices (keep as tensors to avoid redundant conversions)
        sampled_points_list = []
        sampled_normals_list = []
        
        for i, (k_i, indices_i) in enumerate(zip(batch_k_tensor, indices_tensor)):
            # Get valid indices (not padded) - keep as tensor
            valid_indices = indices_i[:k_i]
            
            # Sample from point tensors (keep as tensors)
            sampled_points_list.append(batch_parts_tensor[i][valid_indices])
            
            if padded_normals[i] is not None:
                normals_tensor = padded_normals[i].to(device)
                sampled_normals_list.append(normals_tensor[valid_indices])
            else:
                sampled_normals_list.append(None)
        
        if not sampled_parts:
            return []
        
        # Process each part individually
        part_results = []
        
        for i, (sampled_pts, sampled_norms) in enumerate(zip(sampled_points_list, sampled_normals_list)):
            # Check if part is empty (now checking tensor)
            if sampled_pts.numel() == 0:
                continue
                
            logger.debug(f"Extracting features for part {i} with {sampled_pts.shape[0]} points")
            
            # Extract features for this part using original data (with outliers) as context
            # but sampled keypoints (without outliers) for feature extraction
            # Pass tensors directly to avoid redundant conversions
            feature_extraction_start_time = time.perf_counter()
            feature_result = self.feature_extractor.extract_features(
                points=torch.from_numpy(original_parts[i]).float().to(sampled_pts.device),  # Convert numpy to tensor once
                keypoints=sampled_pts,  # Already a tensor - no conversion needed
            )
            feature_extraction_end_time = time.perf_counter()
            feature_extraction_duration = feature_extraction_end_time - feature_extraction_start_time
            logger.debug(f"Feature extraction time for part {i}: {feature_extraction_duration:.4f} seconds ({sampled_pts.shape[0]} keypoints)")

            features = feature_result['features']
            # print(features[0:3])
            
            # Store result for this part (convert tensors back to numpy for storage)
            part_result = {
                'sampled_points': sampled_pts.cpu().numpy(),  # Convert tensor back to numpy for storage
                'sampled_normals': sampled_norms.cpu().numpy() if sampled_norms is not None else None,  # Convert tensor back to numpy
                'features': features.cpu().numpy() if isinstance(features, torch.Tensor) else features,  # Convert tensor to numpy for storage
            }
            part_results.append(part_result)
        
        return part_results
    

def process_from_folders(input_dir: str, 
                        output_dir: str, 
                        processor: SampleProcessor,
                        dataset_name: Optional[str] = None,
                        hdf5_only: bool = False,
                        dry_run: bool = False) -> Dict:
    """
    Process samples from folder structure.
    
    Args:
        input_dir: Input directory containing training data
        output_dir: Output directory for processed features
        processor: Sample processor instance
        dataset_name: Dataset name (auto-detected if None)
        hdf5_only: If True, skip processing and only convert existing PLY/NPY files to HDF5
        dry_run: If True, only show what would be processed without doing it
        
    Returns:
        Processing statistics
    """
    logger.info("Processing samples from folder structure...")
    
    # Auto-detect dataset name if not provided
    dataset_name = get_dataset_name(input_dir, dataset_name)

    # print('Dataset_name', dataset_name)
    
    # Look for the nested dataset directory structure
    dataset_dir = os.path.join(input_dir, dataset_name)
    # print('Dataset_dir', dataset_dir)
    if not os.path.exists(dataset_dir):
        # Fallback: use input_dir directly if nested structure doesn't exist
        dataset_dir = input_dir
    
    # Find all sample directories (supports nested structures like ThreeDMatch)
    sample_dirs = []
    if os.path.exists(dataset_dir):
        for sequence_name in os.listdir(dataset_dir):
            sequence_path = os.path.join(dataset_dir, sequence_name)
            if os.path.isdir(sequence_path):
                # Check for direct samples in this directory
                for sample_name in os.listdir(sequence_path):
                    if sample_name.startswith('sample_'):
                        sample_path = os.path.join(sequence_path, sample_name)
                        if os.path.isdir(sample_path):
                            relative_path = os.path.join(sequence_name, sample_name)
                            sample_dirs.append((relative_path, sample_path))
                
                # Also check for nested sequence directories (e.g., scene_name/seq-01/sample_XXXXXX)
                for subdir_name in os.listdir(sequence_path):
                    subdir_path = os.path.join(sequence_path, subdir_name)
                    # print('Subdir_path', subdir_path)
                    if os.path.isdir(subdir_path) and not (subdir_name.startswith('sample_')):
                        for sample_name in os.listdir(subdir_path):
                            if sample_name.startswith('sample_') or sample_name.startswith('fracture_'):
                                sample_path = os.path.join(subdir_path, sample_name)
                                if os.path.isdir(sample_path):
                                    relative_path = os.path.join(sequence_name, subdir_name, sample_name)
                                    # print(relative_path, sample_path)
                                    sample_dirs.append((relative_path, sample_path))
    
    if not sample_dirs:
        logger.error(f"No sample directories found in {input_dir}")
        return {'processed_samples': 0, 'failed_samples': 0}
    
    logger.info(f"Found {len(sample_dirs)} samples to process")
    
    # Skip processing if hdf5_only mode or dry run
    if hdf5_only or dry_run:
        if hdf5_only:
            logger.info("HDF5-only mode: skipping PLY/NPY processing")
        if dry_run:
            logger.info("Dry run mode: showing what would be processed")
            for relative_path, sample_path in sample_dirs[:5]:  # Show first 5 samples
                logger.info(f"  Would process: {relative_path}")
            if len(sample_dirs) > 5:
                logger.info(f"  ... and {len(sample_dirs) - 5} more samples")
        processed_count = len(sample_dirs)  # Assume all samples exist
        failed_count = 0
        sample_num_points = []  # Empty for dry run/hdf5_only mode
    else:
        # Process samples
        processed_count = 0
        failed_count = 0
        sample_num_points = []  # Track num_points for each sample
        sample_part_counts = []  # Track number of parts per sample
        sample_part_points = []  # Track points per part for each sample
        all_part_points = []  # Track all individual part point counts
        
        for relative_path, sample_path in tqdm(sample_dirs, desc="Processing samples"):
            try:
                logger.debug(f"Processing sample: {relative_path}")
                
                # Set sample-specific seed
                processor._set_sample_seed(processed_count)
                
                # Load sample
                parts_points, parts_normals, part_names = load_sample_from_folder(sample_path)
                
                if not parts_points:
                    logger.warning(f"No parts loaded for sample: {relative_path}")
                    failed_count += 1
                    sample_num_points.append(0)
                    sample_part_counts.append(0)
                    sample_part_points.append([])
                    continue
                
                # Process sample
                part_results = processor.process_sample(parts_points, parts_normals)
                
                # Calculate statistics for this sample
                part_point_counts = [len(part_result['sampled_points']) for part_result in part_results]
                total_sample_points = sum(part_point_counts)
                
                sample_num_points.append(total_sample_points)
                sample_part_counts.append(len(part_results))
                sample_part_points.append(part_point_counts)
                all_part_points.extend(part_point_counts)
                
                # Save processed sample parts
                sample_output_dir = os.path.join(output_dir, dataset_name, relative_path + '_processed')
                save_processed_sample(part_results, part_names, sample_output_dir, input_sample_dir=sample_path)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process sample {relative_path}: {e}")
                sample_num_points.append(0)  # Add 0 for failed samples
                sample_part_counts.append(0)
                sample_part_points.append([])
                failed_count += 1
    
    logger.info(f"Processing complete: {processed_count} processed, {failed_count} failed")
    
    # Copy and update data_split folder
    if not dry_run:
        copy_and_update_data_split(input_dir, output_dir, dataset_name)
        
        # Save num_points data to folder structure
        if sample_num_points:
            save_num_points_to_folder(output_dir, dataset_name, sample_num_points, sample_dirs)
    else:
        logger.info(f"Would copy and update data_split folder to: {output_dir}/data_split")
        if sample_num_points:
            logger.info(f"Would save num_points data for {len(sample_num_points)} samples")
    
    return {
        'processed_samples': processed_count,
        'failed_samples': failed_count,
        'total_samples': len(sample_dirs),
        'sample_num_points': sample_num_points,  # Include for HDF5 conversion
        'sample_part_counts': sample_part_counts if not (hdf5_only or dry_run) else [],
        'sample_part_points': sample_part_points if not (hdf5_only or dry_run) else [],
        'all_part_points': all_part_points if not (hdf5_only or dry_run) else []
    }


def process_from_hdf5(hdf5_path: str, 
                     output_dir: str, 
                     processor: SampleProcessor,
                     dataset_name: Optional[str] = None,
                     hdf5_only: bool = False,
                     dry_run: bool = False) -> Dict:
    """
    Process samples from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory for processed features
        processor: Sample processor instance
        dataset_name: Dataset name (auto-detected if None)
        hdf5_only: If True, skip processing and only convert existing PLY/NPY files to HDF5
        dry_run: If True, only show what would be processed without doing it
        
    Returns:
        Processing statistics
    """
    logger.info(f"Processing samples from HDF5: {hdf5_path}")
    
    # Auto-detect dataset name if not provided
    dataset_name = get_dataset_name(hdf5_path, dataset_name)
    
    processed_count = 0
    failed_count = 0
    
    try:
        with h5py.File(hdf5_path, 'r') as h5_file:
            # Find all sample paths
            sample_paths = []
            
            def collect_samples(name, obj):
                if isinstance(obj, h5py.Group):
                    # Check if this looks like a sample path (contains numeric submaps)
                    if any(key.isdigit() for key in obj.keys()):
                        sample_paths.append(name)
            
            h5_file.visititems(collect_samples)
            
            if not sample_paths:
                logger.error(f"No samples found in HDF5 file: {hdf5_path}")
                return {'processed_samples': 0, 'failed_samples': 0}
            
            logger.info(f"Found {len(sample_paths)} samples to process")
            
            # Skip processing if hdf5_only mode or dry run
            if hdf5_only or dry_run:
                if hdf5_only:
                    logger.info("HDF5-only mode: skipping PLY/NPY processing")
                if dry_run:
                    logger.info("Dry run mode: showing what would be processed")
                    for sample_path in sample_paths[:5]:  # Show first 5 samples
                        logger.info(f"  Would process: {sample_path}")
                    if len(sample_paths) > 5:
                        logger.info(f"  ... and {len(sample_paths) - 5} more samples")
                processed_count = len(sample_paths)  # Assume all samples exist
                failed_count = 0
                sample_num_points = []  # Empty for dry run/hdf5_only mode
            else:
                # Process samples
                sample_num_points = []  # Track num_points for each sample
                sample_part_counts = []  # Track number of parts per sample
                sample_part_points = []  # Track points per part for each sample
                all_part_points = []  # Track all individual part point counts
                
                for sample_path in tqdm(sample_paths, desc="Processing samples"):
                    try:
                        logger.debug(f"Processing sample: {sample_path}")
                        
                        # Set sample-specific seed
                        processor._set_sample_seed(processed_count)
                        
                        # Load sample
                        parts_points, parts_normals, part_names = load_sample_from_hdf5(h5_file, sample_path)
                        
                        if not parts_points:
                            logger.warning(f"No parts loaded for sample: {sample_path}")
                            failed_count += 1
                            sample_num_points.append(0)
                            sample_part_counts.append(0)
                            sample_part_points.append([])
                            continue
                        
                        # Process sample
                        part_results = processor.process_sample(parts_points, parts_normals)
                        
                        # Calculate statistics for this sample
                        part_point_counts = [len(part_result['sampled_points']) for part_result in part_results]
                        total_sample_points = sum(part_point_counts)
                        
                        sample_num_points.append(total_sample_points)
                        sample_part_counts.append(len(part_results))
                        sample_part_points.append(part_point_counts)
                        all_part_points.extend(part_point_counts)
                        
                        # Save processed sample parts
                        sample_output_dir = os.path.join(output_dir, dataset_name, sample_path + '_processed')
                        save_processed_sample(part_results, part_names, sample_output_dir)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process sample {sample_path}: {e}")
                        sample_num_points.append(0)  # Add 0 for failed samples
                        sample_part_counts.append(0)
                        sample_part_points.append([])
                        failed_count += 1
                    
    except Exception as e:
        logger.error(f"Failed to open HDF5 file {hdf5_path}: {e}")
        return {'processed_samples': 0, 'failed_samples': 0}
    
    logger.info(f"Processing complete: {processed_count} processed, {failed_count} failed")
    
    # Copy and update data_split folder
    hdf5_dir = os.path.dirname(hdf5_path)
    if not dry_run:
        copy_and_update_data_split(hdf5_dir, output_dir, dataset_name)
        
        # Save num_points data to folder structure
        if sample_num_points:
            # For HDF5 processing, we need to create sample_dirs format
            sample_dirs = [(sample_path, '') for sample_path in sample_paths]
            save_num_points_to_folder(output_dir, dataset_name, sample_num_points, sample_dirs)
    else:
        logger.info(f"Would copy and update data_split folder to: {output_dir}/data_split")
        if sample_num_points:
            logger.info(f"Would save num_points data for {len(sample_num_points)} samples")
    
    return {
        'processed_samples': processed_count,
        'failed_samples': failed_count,
        'total_samples': len(sample_paths),
        'sample_num_points': sample_num_points,  # Include for HDF5 conversion
        'sample_part_counts': sample_part_counts if not (hdf5_only or dry_run) else [],
        'sample_part_points': sample_part_points if not (hdf5_only or dry_run) else [],
        'all_part_points': all_part_points if not (hdf5_only or dry_run) else []
    }


def main():
    parser = argparse.ArgumentParser(description='Extract features from training samples using miniSpinNet')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input path (directory with PLY files or HDF5 file)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for processed samples with features')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (auto-detected if not provided)')
    
    # HDF5 output arguments
    parser.add_argument('--save_hdf5', action='store_true', default=True,
                        help='Convert processed PLY/NPY files to HDF5 format after processing')
    parser.add_argument('--hdf5_output', type=str, default=None,
                        help='Path for output HDF5 file (auto-generated if not provided when --save_hdf5 is used)')
    parser.add_argument('--hdf5_only', action='store_true', default=False,
                        help='Only convert existing PLY/NPY files to HDF5, skip feature extraction processing')
    
    # Processing arguments
    parser.add_argument('--num_points', '-k', type=int, default=5000,
                        help='Number of points to sample using FPS (default: 5000), now deprecated')
    parser.add_argument('--global_seed', type=int, default=42,
                        help='Global random seed for all random operations including FPS (default: 42)')
    parser.add_argument('--min_points_per_part', type=int, default=300,
                        help='Minimum number of points each part should have after FPS (default: 100)')
    parser.add_argument('--max_points_per_part', type=int, default=10000,
                        help='Maximum number of points each part should have after FPS (default: 10000)')
    parser.add_argument('--skip_point_sampling', action='store_true', default=False,
                        help='Skip farthest point sampling and use all points as keypoints for feature extraction')
    
    # Outlier removal arguments
    parser.add_argument('--remove_outliers', action='store_true', default=True,
                        help='Remove statistical outliers for FPS sampling, but keep all points for feature extraction context (default: True)')
    parser.add_argument('--no_remove_outliers', dest='remove_outliers', action='store_false',
                        help='Disable statistical outlier removal')
    parser.add_argument('--outlier_nb_neighbors', type=int, default=20,
                        help='Number of neighbors for outlier removal (default: 20)')
    parser.add_argument('--outlier_std_ratio', type=float, default=2.5,
                        help='Standard deviation ratio for outlier removal (default: 2.5)')
    
    # Allocation method and voxel size
    parser.add_argument('--allocation_method', type=str, default='voxel_adaptive',
                        choices=['point_count', 'spatial_coverage', 'voxel_adaptive'],
                        help='Method for allocating FPS points (default: voxel_adaptive)')
    parser.add_argument('--voxel_size', type=float, default=1.0,
                        help='Voxel size in meters for spatial coverage calculation (default: 1.0)')
    parser.add_argument('--voxel_ratio', '-r', type=float, default=0.05,
                        help='Ratio of occupied voxels to sample points for voxel_adaptive method (default: 0.05), should also be considered together with --voxel_size, \
                        the smaller the ratio, the fewer the sample points, we set a larger value (for example 0.2) for scan level data')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='./weights/weights/spinnet_3dmatch_bufferx.pth',
                        help='Path to miniSpinNet checkpoint (default: ./weights/weights/spinnet_3dmatch_bufferx.pth), select from ./weights/weights/spinnet_3dmatch_bufferx.pth, ./weights/weights/spinnet_kitti_bufferx.pth')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for feature extraction (default: auto)')
    parser.add_argument('--is_aligned_to_global_z', action='store_true', default=True,
                        help='Align point clouds to global Z axis before feature extraction (default: True)')
    parser.add_argument('--no_is_aligned_to_global_z', dest='is_aligned_to_global_z', action='store_false',
                        help='Do not align point clouds to global Z axis before feature extraction')
    
    # miniSpinNet configuration
    parser.add_argument('--des_r', type=float, default=5.0,
                        help='Description radius for miniSpinNet in meters (default: 5.0)')
    parser.add_argument('--num_points_per_patch', type=int, default=512,
                        help='Number of points per patch for miniSpinNet (default: 512)')
    
    # Utility arguments
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show what would be processed without actually doing it')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set random seeds for reproducibility
    set_random_seeds(args.global_seed)
    
    # Validate arguments
    if not _validate_and_setup_args(args):
        return
    
    # Show dry run mode if enabled
    if args.dry_run:
        logger.info("=" * 50)
        logger.info("DRY RUN MODE - No files will be created or modified")
        logger.info("=" * 50)
    
    # Create output directory
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)
    else:
        logger.info(f"Would create output directory: {args.output}")
    
    # Initialize feature extractor and processor (only needed for non-hdf5-only mode)
    if not args.hdf5_only:
        if args.dry_run:
            logger.info("Would initialize feature extractor and sample processor")
            processor = None  # Skip expensive initialization in dry run
        else:
            model_config = {
                'num_points_per_patch': args.num_points_per_patch,
                'is_aligned_to_global_z': args.is_aligned_to_global_z,
                'des_r': args.des_r,
            }
            
            feature_extractor = FeatureExtractor(
                model_config=model_config,
                des_r=args.des_r,
                is_aligned_to_global_z=args.is_aligned_to_global_z,
                checkpoint_path=args.checkpoint,
                device=args.device
            )
            
            processor = SampleProcessor(
                feature_extractor=feature_extractor,
                num_points=args.num_points,
                skip_point_sampling=args.skip_point_sampling,
                remove_outliers=args.remove_outliers,
                outlier_nb_neighbors=args.outlier_nb_neighbors,
                outlier_std_ratio=args.outlier_std_ratio,
                min_points_per_part=args.min_points_per_part,
                max_points_per_part=args.max_points_per_part,
                global_seed=args.global_seed,
                allocation_method=args.allocation_method,
                voxel_size=args.voxel_size,
                voxel_ratio=args.voxel_ratio,
            )
    else:
        processor = None  # Not needed for HDF5-only mode
    
    # Process samples
    if args.input.endswith('.hdf5') or args.input.endswith('.h5'):
        stats = process_from_hdf5(args.input, args.output, processor, args.dataset_name, args.hdf5_only, args.dry_run)
    else:
        stats = process_from_folders(args.input, args.output, processor, args.dataset_name, args.hdf5_only, args.dry_run)
    
    # Convert to HDF5 if requested
    if args.save_hdf5 or args.hdf5_only:
        # Use output directory name as dataset name instead of input directory name
        dataset_name = get_dataset_name(args.output, args.dataset_name)
        if args.dry_run:
            logger.info(f"Would convert to HDF5 format: {args.hdf5_output}")
        else:
            logger.info("Converting to HDF5 format...")
            convert_to_hdf5(args.output, dataset_name, args.hdf5_output, stats, args)
    
    # Save metadata
    if args.dry_run:
        logger.info(f"Would save processing metadata to: {args.output}/feature_extraction_metadata.json")
    else:
        save_processing_metadata(args.output, stats, args)
        
        # Also save a comprehensive metadata summary
        comprehensive_metadata = {
            'script_info': {
                'name': 'extract_sample_features.py',
                'description': 'Feature extraction from training samples using miniSpinNet',
                'version': '1.0',
                'timestamp': datetime.now().isoformat()
            },
            'processing_summary': {
                'input_path': args.input,
                'output_path': args.output,
                'dataset_name': args.dataset_name,
                'processed_samples': stats.get('processed_samples', 0),
                'failed_samples': stats.get('failed_samples', 0),
                'total_samples': stats.get('total_samples', 0),
                'success_rate': f"{stats.get('processed_samples', 0) / max(stats.get('total_samples', 1), 1) * 100:.1f}%"
            },
            'feature_extraction_config': {
                'feature_extractor': 'miniSpinNet',
                'num_points_fps': args.num_points,
                'min_points_per_part': args.min_points_per_part,
                'des_r': args.des_r,
                'num_points_per_patch': args.num_points_per_patch,
                'is_aligned_to_global_z': args.is_aligned_to_global_z,
                'checkpoint_path': args.checkpoint,
                'device': args.device
            },
            'processing_config': {
                'global_seed': args.global_seed,
                'skip_point_sampling': args.skip_point_sampling,
                'remove_outliers': args.remove_outliers,
                'outlier_nb_neighbors': args.outlier_nb_neighbors,
                'outlier_std_ratio': args.outlier_std_ratio,
                'allocation_method': args.allocation_method,
                'voxel_size': args.voxel_size,
                'voxel_ratio': args.voxel_ratio,
                # 'max_sample_points': args.max_sample_points,
                # 'min_sample_points': args.min_sample_points
            },
            'output_config': {
                'save_hdf5': args.save_hdf5,
                'hdf5_only': args.hdf5_only,
                'hdf5_output': args.hdf5_output
            }
        }
        
        comprehensive_metadata_path = os.path.join(args.output, 'comprehensive_metadata.json')
        with open(comprehensive_metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive metadata: {comprehensive_metadata_path}")
    
    # Print detailed statistics summary
    print_detailed_statistics(stats, args)
    
    logger.info("=" * 50)
    if args.dry_run:
        logger.info("DRY RUN COMPLETE!")
        logger.info("Summary of what would be processed:")
    elif args.hdf5_only:
        logger.info("HDF5 CONVERSION COMPLETE!")
    else:
        logger.info("FEATURE EXTRACTION COMPLETE!")
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    if args.save_hdf5 or args.hdf5_only:
        logger.info(f"HDF5 Output: {args.hdf5_output}")
    
    if not args.hdf5_only:
        logger.info(f"{'Would process' if args.dry_run else 'Processed'} samples: {stats['processed_samples']}")
        logger.info(f"{'Would fail' if args.dry_run else 'Failed'} samples: {stats['failed_samples']}")
        if stats['total_samples'] > 0:
            logger.info(f"Success rate: {stats['processed_samples'] / stats['total_samples'] * 100:.1f}%")
    
    if not args.dry_run:
        if args.save_hdf5 and not args.hdf5_only:
            logger.info("Note: Data saved in both PLY/NPY folder structure and HDF5 format")
        elif args.hdf5_only:
            logger.info("Note: Only HDF5 file generated from existing PLY/NPY files")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()