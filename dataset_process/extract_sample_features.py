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

# Add the current directory and parent directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from utils.spinnet.patch_embedder import MiniSpinNet
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
    def __init__(self, model_config: Dict = None, des_r: float = 3.0, is_aligned_to_global_z: bool = True,
                 checkpoint_path: str = None, device: str = 'auto'):
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
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        elif checkpoint_path:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def _build_model(self) -> MiniSpinNet:
        """Build miniSpinNet model."""
        model = MiniSpinNet(**self.model_config)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            filtered = {k[5:]: v for k, v in state_dict.items() if k.startswith('Desc.')}
            self.model.load_state_dict(filtered, strict=False)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    def extract_features(self, points: Union[np.ndarray, torch.Tensor],
                        keypoints: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict:
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
                
                return {'features': empty_features, 'keypoints': keypoints}


class SampleProcessor:
    def __init__(self, feature_extractor: FeatureExtractor, num_points: int = 5000, skip_point_sampling: bool = False,
                 remove_outliers: bool = True, outlier_nb_neighbors: int = 20, outlier_std_ratio: float = 2.0,
                 min_points_per_part: int = 100, max_points_per_part: int = 10000, global_seed: int = 42,
                 allocation_method: str = 'point_count', voxel_size: float = 1.0, voxel_ratio: float = 0.1):
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

        if self.allocation_method not in ['point_count', 'spatial_coverage', 'voxel_adaptive']:
            raise ValueError(f"allocation_method must be 'point_count', 'spatial_coverage', or 'voxel_adaptive'")

    def _set_sample_seed(self, sample_idx: int = 0):
        seed = self.global_seed + sample_idx * 1000
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def process_sample(self, parts_points: List[np.ndarray], parts_normals: List[Optional[np.ndarray]] = None) -> List[Dict]:
        if not parts_points or len(parts_points) == 0:
            return []
        
        n_parts = len(parts_points)
        parts_normals = parts_normals or [None] * n_parts
        
        original_parts = []
        original_normals = []
        
        for i, (points, normals) in enumerate(zip(parts_points, parts_normals)):
            if len(points) == 0:
                continue
                
            original_parts.append(points.copy())
            original_normals.append(normals.copy() if normals is not None else None)
        
        if not original_parts:
            return []

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
        
        fps_parts, fps_normals = original_parts, original_normals
        if self.remove_outliers:
            
            fps_parts = []
            fps_normals = []
            
            for i, (orig_part, orig_normals_i) in enumerate(zip(original_parts, original_normals)):
                if len(orig_part) < self.outlier_nb_neighbors:
                    fps_parts.append(orig_part)
                    fps_normals.append(orig_normals_i)
                    continue
                
                try:
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
                        fps_parts.append(orig_part[inlier_indices])
                        fps_normals.append(orig_normals_i[inlier_indices] if orig_normals_i is not None else None)
                
                except Exception as e:
                    logger.warning(f"Outlier removal failed for part {i}: {e}")
                    fps_parts.append(orig_part)
                    fps_normals.append(orig_normals_i)

        # Pre-FPS random downsampling for very large parts
        if self.max_points_per_part is not None:
            pre_fps_parts = []
            pre_fps_normals = []
            for i, (part, norms) in enumerate(zip(fps_parts, fps_normals)):
                pre_fps_cap = 20 * self.max_points_per_part
                if len(part) > pre_fps_cap:
                    indices = np.random.choice(len(part), pre_fps_cap, replace=False)
                    pre_fps_parts.append(part[indices].copy())
                    if norms is not None:
                        pre_fps_normals.append(norms[indices].copy())
                    else:
                        pre_fps_normals.append(None)
                else:
                    pre_fps_parts.append(part)
                    pre_fps_normals.append(norms)

            fps_parts = pre_fps_parts
            fps_normals = pre_fps_normals
        
        # FPS allocation
        if self.allocation_method == 'voxel_adaptive':
            # Calculate adaptive sample count based on occupied voxels after outlier removal
            adaptive_sample_counts_per_part = calculate_adaptive_sample_count_per_part(
                fps_parts, self.voxel_size, self.voxel_ratio, self.min_points_per_part, self.max_points_per_part
            )
            total_adaptive_points = sum(adaptive_sample_counts_per_part)
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
            if self.allocation_method == 'spatial_coverage':
                target_per_part = allocate_fps_points(
                    fps_parts, 
                    self.allocation_method,
                    self.num_points,
                    self.min_points_per_part,
                    self.voxel_size,
                    self.voxel_ratio,
                )
            else:
                pts_per_part = np.array([len(part) for part in fps_parts])
                target_per_part = allocate_fps_points(
                    pts_per_part, 
                    self.allocation_method,
                    self.num_points,
                    self.min_points_per_part,
                    self.voxel_size,
                    self.voxel_ratio,
                )
        
        # Apply batched FPS
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
        
        max_points = max(batch_lengths)
        padded_parts = []
        padded_normals = []
        for i, (part, norms) in enumerate(zip(batch_parts, batch_normals)):
            if len(part) < max_points:
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
        
        sampled_parts, indices_tensor = apply_batched_fps(
            batch_parts_tensor, batch_lengths_tensor, batch_k_tensor, self.global_seed, device
        )
        
        sampled_points_list = []
        sampled_normals_list = []
        
        for i, (k_i, indices_i) in enumerate(zip(batch_k_tensor, indices_tensor)):
            valid_indices = indices_i[:k_i]
            sampled_points_list.append(batch_parts_tensor[i][valid_indices])
            
            if padded_normals[i] is not None:
                normals_tensor = padded_normals[i].to(device)
                sampled_normals_list.append(normals_tensor[valid_indices])
            else:
                sampled_normals_list.append(None)
        
        if not sampled_parts:
            return []
        
        part_results = []
        for i, (sampled_pts, sampled_norms) in enumerate(zip(sampled_points_list, sampled_normals_list)):
            if sampled_pts.numel() == 0:
                continue
            feature_result = self.feature_extractor.extract_features(
                points=torch.from_numpy(original_parts[i]).float().to(sampled_pts.device),
                keypoints=sampled_pts,
            )
            features = feature_result['features']
            part_result = {
                'sampled_points': sampled_pts.cpu().numpy(),
                'sampled_normals': sampled_norms.cpu().numpy() if sampled_norms is not None else None,
                'features': features.cpu().numpy() if isinstance(features, torch.Tensor) else features,
            }
            part_results.append(part_result)
        return part_results
    

def process_from_folders(input_dir: str, output_dir: str, processor: SampleProcessor,
                        dataset_name: Optional[str] = None, hdf5_only: bool = False, dry_run: bool = False) -> Dict:
    dataset_name = get_dataset_name(input_dir, dataset_name)
    dataset_dir = os.path.join(input_dir, dataset_name)
    if not os.path.exists(dataset_dir):
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
                
                for subdir_name in os.listdir(sequence_path):
                    subdir_path = os.path.join(sequence_path, subdir_name)
                    if os.path.isdir(subdir_path) and not subdir_name.startswith('sample_'):
                        for sample_name in os.listdir(subdir_path):
                            if sample_name.startswith('sample_') or sample_name.startswith('fracture_'):
                                sample_path = os.path.join(subdir_path, sample_name)
                                if os.path.isdir(sample_path):
                                    sample_dirs.append((os.path.join(sequence_name, subdir_name, sample_name), sample_path))
    
    if not sample_dirs:
        logger.error(f"No sample directories found in {input_dir}")
        return {'processed_samples': 0, 'failed_samples': 0}
    
    logger.info(f"Found {len(sample_dirs)} samples to process")
    
    if hdf5_only or dry_run:
        processed_count = len(sample_dirs)
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

    return {
        'processed_samples': processed_count,
        'failed_samples': failed_count,
        'total_samples': len(sample_dirs),
        'sample_num_points': sample_num_points,  # Include for HDF5 conversion
        'sample_part_counts': sample_part_counts if not (hdf5_only or dry_run) else [],
        'sample_part_points': sample_part_points if not (hdf5_only or dry_run) else [],
        'all_part_points': all_part_points if not (hdf5_only or dry_run) else []
    }


def process_from_hdf5(hdf5_path: str, output_dir: str, processor: SampleProcessor,
                     dataset_name: Optional[str] = None, hdf5_only: bool = False, dry_run: bool = False) -> Dict:
    
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
            
            if hdf5_only or dry_run:
                processed_count = len(sample_paths)
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
            sample_dirs = [(p, '') for p in sample_paths]
            save_num_points_to_folder(output_dir, dataset_name, sample_num_points, sample_dirs)

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
    
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)
    
    if not args.hdf5_only:
        if args.dry_run:
            processor = None
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
        processor = None

    if args.input.endswith('.hdf5') or args.input.endswith('.h5'):
        stats = process_from_hdf5(args.input, args.output, processor, args.dataset_name, args.hdf5_only, args.dry_run)
    else:
        stats = process_from_folders(args.input, args.output, processor, args.dataset_name, args.hdf5_only, args.dry_run)
    
    if args.save_hdf5 or args.hdf5_only:
        dataset_name = get_dataset_name(args.output, args.dataset_name)
        if not args.dry_run:
            convert_to_hdf5(args.output, dataset_name, args.hdf5_output, stats, args)
    
    if not args.dry_run:
        save_processing_metadata(args.output, stats, args)
        
        comprehensive_metadata = {
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
            },
            'output_config': {
                'save_hdf5': args.save_hdf5,
                'hdf5_only': args.hdf5_only,
                'hdf5_output': args.hdf5_output
            }
        }
        
        with open(os.path.join(args.output, 'comprehensive_metadata.json'), 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)

    print_detailed_statistics(stats, args)


if __name__ == "__main__":
    main()