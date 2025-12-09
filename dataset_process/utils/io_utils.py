#!/usr/bin/env python3
"""
I/O Utilities for Training Sample Generation

This module contains functions for saving training samples, creating metadata,
and converting data to different formats.
"""

import os
import numpy as np
import open3d as o3d
import h5py
import json
import logging
import shutil
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any
from datetime import datetime
from tqdm import tqdm


# Default color map for point cloud visualization (64 distinct colors)
CMAP_DEFAULT = [
    [0.99, 0.55, 0.38],  # Coral/Orange
    [0.52, 0.75, 0.90],  # Sky Blue
    [0.65, 0.85, 0.33],  # Lime Green
    [0.91, 0.54, 0.76],  # Pink
    [0.79, 0.38, 0.69],  # Purple
    [1.00, 0.85, 0.18],  # Yellow
    [0.90, 0.77, 0.58],  # Tan/Beige
    [0.84, 0.00, 0.00],  # Red
    [0.00, 0.65, 0.93],  # Blue
    [0.55, 0.24, 1.00],  # Violet
    [0.00, 0.80, 0.40],  # Green
    [1.00, 0.50, 0.00],  # Orange
    [0.20, 0.60, 0.80],  # Cyan Blue
    [0.90, 0.20, 0.30],  # Rose Red
    [0.40, 0.70, 0.40],  # Forest Green
    [0.70, 0.30, 0.60],  # Magenta
    [0.30, 0.50, 0.70],  # Steel Blue
    [0.80, 0.60, 0.20],  # Brown/Gold
    [0.50, 0.80, 0.70],  # Mint Green
    [0.90, 0.40, 0.50],  # Salmon
    [0.20, 0.40, 0.60],  # Navy Blue
    [0.60, 0.80, 0.20],  # Chartreuse
    [0.80, 0.30, 0.40],  # Crimson
    [0.40, 0.60, 0.80],  # Light Blue
    [0.70, 0.50, 0.30],  # Sienna
    [0.30, 0.70, 0.50],  # Teal
    [0.90, 0.60, 0.30],  # Peach
    [0.50, 0.30, 0.70],  # Indigo
    [0.60, 0.40, 0.20],  # Dark Brown
    [0.20, 0.80, 0.60],  # Turquoise
    [0.95, 0.70, 0.85],  # Lavender Pink
    [0.35, 0.45, 0.55],  # Slate Gray
    [0.85, 0.45, 0.65],  # Hot Pink
    [0.25, 0.75, 0.85],  # Aqua Blue
    [0.70, 0.85, 0.50],  # Light Green
    [0.45, 0.25, 0.55],  # Dark Purple
    [0.95, 0.80, 0.40],  # Light Gold
    [0.15, 0.50, 0.35],  # Dark Green
    [0.80, 0.50, 0.70],  # Orchid
    [0.55, 0.65, 0.85],  # Periwinkle
    [0.90, 0.30, 0.60],  # Deep Pink
    [0.40, 0.80, 0.60],  # Emerald
    [0.65, 0.25, 0.45],  # Burgundy
    [0.30, 0.85, 0.75],  # Aquamarine
    [0.75, 0.35, 0.25],  # Rust Red
    [0.50, 0.50, 0.70],  # Blue Gray
    [0.85, 0.65, 0.40],  # Amber
    [0.20, 0.30, 0.50],  # Midnight Blue
    [0.95, 0.50, 0.30],  # Coral Red
    [0.60, 0.70, 0.30],  # Olive Green
    [0.70, 0.20, 0.50],  # Deep Rose
    [0.35, 0.60, 0.45],  # Sea Green
    [0.80, 0.40, 0.20],  # Burnt Orange
    [0.45, 0.55, 0.75],  # Powder Blue
    [0.90, 0.50, 0.70],  # Rose Pink
    [0.25, 0.65, 0.40],  # Jade Green
    [0.65, 0.45, 0.25],  # Coffee Brown
    [0.40, 0.30, 0.60],  # Deep Indigo
    [0.85, 0.75, 0.50],  # Khaki
    [0.50, 0.40, 0.30],  # Taupe
    [0.75, 0.60, 0.45],  # Caramel
    [0.30, 0.40, 0.55],  # Charcoal Blue
]


logger = logging.getLogger(__name__)

def get_dataset_name(input_path: str, provided_name: Optional[str] = None) -> str:
    """
    Get dataset name from input path or use provided name.
    
    Args:
        input_path: Input file or directory path
        provided_name: User-provided dataset name (optional)
        
    Returns:
        Dataset name string
    """
    if provided_name:
        return provided_name
    
    if input_path.endswith(('.hdf5', '.h5')):
        return os.path.splitext(os.path.basename(input_path))[0]
    else:
        return os.path.basename(input_path.rstrip('/')) 

def load_sample_from_folder(sample_path: str) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str]]:
    """
    Load a sample from folder containing PLY files, keeping parts separate.
    
    Args:
        sample_path: Path to sample folder
        
    Returns:
        Tuple of (parts_points, parts_normals, part_names) - lists of point clouds, normals, and part names
    """
    ply_files = sorted([f for f in os.listdir(sample_path) if f.endswith('.ply')])
    
    if not ply_files:
        logger.warning(f"No PLY files found in {sample_path}")
        return [], [], []
    
    parts_points = []
    parts_normals = []
    part_names = []
    
    for ply_file in ply_files:
        ply_path = os.path.join(sample_path, ply_file)
        
        try:
            # Load with Open3D
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            
            if len(points) > 0:
                parts_points.append(points)
                parts_normals.append(normals)
                # Store part name without extension
                part_names.append(os.path.splitext(ply_file)[0])
                    
        except Exception as e:
            logger.warning(f"Failed to load {ply_path}: {e}")
            continue
    
    return parts_points, parts_normals, part_names
    
def load_sample_from_hdf5(h5_file, fragment_path: str) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str]]:
    """
    Load a sample from HDF5 file, keeping parts separate.
    
    Args:
        h5_file: Open HDF5 file handle
        fragment_path: Path to fragment in HDF5
        
    Returns:
        Tuple of (parts_points, parts_normals, part_names) - lists of point clouds, normals, and part names
    """
    try:
        fragment_group = h5_file[fragment_path]
        parts_points = []
        parts_normals = []
        part_names = []
        
        # Iterate through submaps in sorted order
        submap_keys = sorted(fragment_group.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for submap_key in submap_keys:
            submap_group = fragment_group[submap_key]
            
            if 'vertices' in submap_group:
                points = submap_group['vertices'][:]
                normals = submap_group['normals'][:] if 'normals' in submap_group else None
                
                if len(points) > 0:
                    parts_points.append(points)
                    parts_normals.append(normals)
                    part_names.append(f"part_{submap_key}")
        
        return parts_points, parts_normals, part_names
        
    except Exception as e:
        logger.warning(f"Failed to load sample {fragment_path} from HDF5: {e}")
        return [], [], []

def save_processed_sample(part_results: List[Dict[str, np.ndarray]], 
                         part_names: List[str], 
                         output_sample_dir: str,
                         input_sample_dir: Optional[str] = None):
    """
    Save processed sample parts as separate PLY and NPY files in a sample folder.
    
    Args:
        part_results: List of processed data for each part
        part_names: List of part names
        output_sample_dir: Output directory for this sample
        input_sample_dir: Original input sample directory to copy pose files from (optional)
    """
    if not part_results:
        logger.warning(f"No parts to save for {output_sample_dir}")
        return
    
    # Create sample output directory
    os.makedirs(output_sample_dir, exist_ok=True)
    
    for part_result, part_name in zip(part_results, part_names):
        sampled_points = part_result['sampled_points']
        sampled_normals = part_result['sampled_normals']
        features = part_result['features']
        
        if len(sampled_points) == 0:
            logger.warning(f"No points to save for part {part_name}")
            continue
        
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sampled_points)
            
            if sampled_normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
            
            # Save PLY file for this part
            ply_path = os.path.join(output_sample_dir, f'{part_name}.ply')
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
            
            # Save features NPY file for this part
            feature_path = os.path.join(output_sample_dir, f'features_{part_name}.npy')
            np.save(feature_path, features)
            
            logger.debug(f"Saved part {part_name}: {ply_path}")
            logger.debug(f"Saved features: {feature_path}")

            # Copy pose txt file if available in input sample directory
            if input_sample_dir is not None:
                # Try both naming conventions for backward compatibility
                pose_src_old = os.path.join(input_sample_dir, f"{part_name}_pose.txt")
                pose_src_new = os.path.join(input_sample_dir, f"pose_{part_name}.txt")
                
                pose_src = None
                if os.path.exists(pose_src_new):
                    pose_src = pose_src_new
                    pose_dst = os.path.join(output_sample_dir, f"pose_{part_name}.txt")
                elif os.path.exists(pose_src_old):
                    pose_src = pose_src_old
                    pose_dst = os.path.join(output_sample_dir, f"pose_{part_name}.txt")
                
                if pose_src:
                    try:
                        shutil.copyfile(pose_src, pose_dst)
                        logger.debug(f"Copied pose file: {pose_src} -> {pose_dst}")
                    except Exception as e:
                        logger.warning(f"Failed to copy pose file for part {part_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save part {part_name}: {e}")

def create_metadata_json(output_dir: str,
                        sequence_stats: Dict,
                        split_info: Dict,
                        processing_args: Dict) -> str:
    """
    Create a metadata.json file containing information about all generated samples.
    
    Args:
        output_dir: Root output directory
        sequence_stats: Statistics from processing each sequence
        split_info: Information about data splits
        processing_args: Arguments used for processing
    
    Returns:
        Path to the created metadata.json file
    """
    logger.info("Creating metadata.json...")
    
    # Clean and prepare processing args for JSON serialization
    def clean_args_for_json(args_dict):
        """Clean arguments to ensure they can be serialized to JSON."""
        cleaned_args = {}
        for key, value in args_dict.items():
            # Skip private attributes
            if key.startswith('_'):
                continue
            
            # Handle different types
            if value is None:
                cleaned_args[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned_args[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert tuples to lists and ensure all elements are serializable
                cleaned_args[key] = [str(item) if not isinstance(item, (str, int, float, bool)) else item for item in value]
            elif isinstance(value, dict):
                cleaned_args[key] = clean_args_for_json(value)
            else:
                # Convert other types to string
                cleaned_args[key] = str(value)
        return cleaned_args
    
    cleaned_processing_args = clean_args_for_json(processing_args)
    
    # Reconstruct command line for reproducibility
    def reconstruct_command_line(args_dict):
        """Reconstruct the command line from arguments for reproducibility."""
        cmd_parts = ['python', 'dataset_process/generate_training_samples.py']
        
        # Add dataset and data root
        if args_dict.get('dataset'):
            cmd_parts.extend(['--dataset', str(args_dict['dataset'])])
        if args_dict.get('data_root'):
            cmd_parts.extend(['--data_root', str(args_dict['data_root'])])
        if args_dict.get('output_dir'):
            cmd_parts.extend(['--output_dir', str(args_dict['output_dir'])])
        
        # Add sequences if specified
        if args_dict.get('sequences'):
            cmd_parts.extend(['--sequences'] + [str(s) for s in args_dict['sequences']])
        
        # Add key parameters
        if args_dict.get('num_samples'):
            cmd_parts.extend(['--num_samples', str(args_dict['num_samples'])])
        if args_dict.get('sample_count_multiplier'):
            cmd_parts.extend(['--sample_count_multiplier', str(args_dict['sample_count_multiplier'])])
        
        # Add submap parameters
        if args_dict.get('min_frames_per_submap'):
            cmd_parts.extend(['--min_frames_per_submap', str(args_dict['min_frames_per_submap'])])
        if args_dict.get('max_frames_per_submap'):
            cmd_parts.extend(['--max_frames_per_submap', str(args_dict['max_frames_per_submap'])])
        if args_dict.get('min_spatial_threshold'):
            cmd_parts.extend(['--min_spatial_threshold', str(args_dict['min_spatial_threshold'])])
        if args_dict.get('max_spatial_threshold'):
            cmd_parts.extend(['--max_spatial_threshold', str(args_dict['max_spatial_threshold'])])
        if args_dict.get('min_submaps_per_sample'):
            cmd_parts.extend(['--min_submaps_per_sample', str(args_dict['min_submaps_per_sample'])])
        if args_dict.get('max_submaps_per_sample'):
            cmd_parts.extend(['--max_submaps_per_sample', str(args_dict['max_submaps_per_sample'])])
        
        # Add processing parameters
        if args_dict.get('voxel_size'):
            cmd_parts.extend(['--voxel_size', str(args_dict['voxel_size'])])
        if args_dict.get('downsample_method'):
            cmd_parts.extend(['--downsample_method', str(args_dict['downsample_method'])])
        if args_dict.get('num_points_downsample'):
            cmd_parts.extend(['--num_points_downsample', str(args_dict['num_points_downsample'])])
        
        # Add overlap parameters
        if args_dict.get('min_overlap_ratio'):
            cmd_parts.extend(['--min_overlap_ratio', str(args_dict['min_overlap_ratio'])])
        if args_dict.get('max_overlap_ratio'):
            cmd_parts.extend(['--max_overlap_ratio', str(args_dict['max_overlap_ratio'])])
        if args_dict.get('overlap_method'):
            cmd_parts.extend(['--overlap_method', str(args_dict['overlap_method'])])
        
        # Add frame parameters
        if args_dict.get('start_frame'):
            cmd_parts.extend(['--start_frame', str(args_dict['start_frame'])])
        if args_dict.get('end_frame'):
            cmd_parts.extend(['--end_frame', str(args_dict['end_frame'])])
        if args_dict.get('max_frames_per_sequence'):
            cmd_parts.extend(['--max_frames_per_sequence', str(args_dict['max_frames_per_sequence'])])
        
        # Add split parameters
        if args_dict.get('train_ratio'):
            cmd_parts.extend(['--train_ratio', str(args_dict['train_ratio'])])
        if args_dict.get('split_by_sequence'):
            cmd_parts.append('--split_by_sequence')
        if args_dict.get('guarantee_loop_closure'):
            cmd_parts.append('--guarantee_loop_closure')
        if args_dict.get('mixed_val_split'):
            cmd_parts.append('--mixed_val_split')
        if args_dict.get('val_sequences'):
            cmd_parts.extend(['--val_sequences'] + [str(s) for s in args_dict['val_sequences']])
        
        # Add other flags
        if args_dict.get('random_downsample'):
            cmd_parts.append('--random_downsample')
        if args_dict.get('estimate_normals'):
            cmd_parts.append('--estimate_normals')
        if args_dict.get('enable_deskewing'):
            cmd_parts.append('--enable_deskewing')
        if args_dict.get('create_hdf5'):
            cmd_parts.append('--create_hdf5')
        
        # Add seed
        if args_dict.get('seed'):
            cmd_parts.extend(['--seed', str(args_dict['seed'])])
        
        return ' '.join(cmd_parts)
    
    reconstructed_command = reconstruct_command_line(cleaned_processing_args)
    
    # Collect detailed information about each sample
    samples_info = []
    sequence_summary = {}
    
    for sequence, stats in sequence_stats.items():
        sequence_dir = os.path.join(output_dir, os.path.basename(output_dir), sequence)
        if os.path.exists(sequence_dir):
            sample_dirs = sorted([d for d in os.listdir(sequence_dir) if d.startswith('sample_')])
            
            # Collect sequence-specific information
            sequence_samples = []
            total_submaps_in_sequence = 0
            submap_counts_in_sequence = []
            
            for sample_dir in sample_dirs:
                sample_path = os.path.join(sequence_dir, sample_dir)
                
                # Count PLY files in this sample
                ply_files = [f for f in os.listdir(sample_path) if f.endswith('.ply')]
                num_submaps = len(ply_files)
                
                # Extract sample information from filename
                sample_info = {
                    'sample_id': sample_dir,
                    'sequence': sequence,
                    'fragment_path': f"{os.path.basename(output_dir)}/{sequence}/{sample_dir}",
                    'num_submaps': num_submaps,
                    'submap_files': ply_files,
                    'sample_path': sample_path
                }
                samples_info.append(sample_info)
                sequence_samples.append(sample_info)
                
                total_submaps_in_sequence += num_submaps
                submap_counts_in_sequence.append(num_submaps)
            
            # Create sequence summary
            sequence_summary[sequence] = {
                'num_samples': len(sequence_samples),
                'total_submaps': total_submaps_in_sequence,
                'avg_submaps_per_sample': total_submaps_in_sequence / len(sequence_samples) if sequence_samples else 0,
                'min_submaps_per_sample': min(submap_counts_in_sequence) if submap_counts_in_sequence else 0,
                'max_submaps_per_sample': max(submap_counts_in_sequence) if submap_counts_in_sequence else 0,
                'submap_count_distribution': {
                    str(count): submap_counts_in_sequence.count(count) for count in set(submap_counts_in_sequence)
                },
                'sample_ids': [s['sample_id'] for s in sequence_samples],
                'processing_statistics': stats.get('statistics', {})
            }
    
    # Create overall dataset summary
    total_samples = len(samples_info)
    total_submaps = sum(seq_sum['total_submaps'] for seq_sum in sequence_summary.values())
    all_submap_counts = [sample['num_submaps'] for sample in samples_info]
    
    dataset_summary = {
        'total_sequences': len(sequence_summary),
        'total_samples': total_samples,
        'total_submaps': total_submaps,
        'avg_samples_per_sequence': total_samples / len(sequence_summary) if sequence_summary else 0,
        'avg_submaps_per_sample': total_submaps / total_samples if total_samples > 0 else 0,
        'min_submaps_per_sample': min(all_submap_counts) if all_submap_counts else 0,
        'max_submaps_per_sample': max(all_submap_counts) if all_submap_counts else 0,
        'sequence_names': list(sequence_summary.keys()),
        'overall_submap_count_distribution': {
            str(count): all_submap_counts.count(count) for count in set(all_submap_counts)
        }
    }
    
    # Create metadata structure
    metadata = {
        'dataset_info': {
            'name': os.path.basename(output_dir),
            'description': 'Generated training samples from LiDAR sequences',
            'creation_date': datetime.now().isoformat(),
            'total_samples': total_samples,
            'sequences': list(sequence_stats.keys())
        },
        'dataset_summary': dataset_summary,
        'sequence_summary': sequence_summary,
        'processing_args': cleaned_processing_args,
        'command_line_info': {
            'script_name': 'generate_training_samples.py',
            'reconstructed_command': reconstructed_command,
            'args_summary': {
                'dataset': cleaned_processing_args.get('dataset', 'unknown'),
                'data_root': cleaned_processing_args.get('data_root', 'unknown'),
                'output_dir': cleaned_processing_args.get('output_dir', 'unknown'),
                'sequences': cleaned_processing_args.get('sequences', []),
                'num_samples': cleaned_processing_args.get('num_samples'),
                'sample_count_multiplier': cleaned_processing_args.get('sample_count_multiplier'),
                'voxel_size': cleaned_processing_args.get('voxel_size'),
                'downsample_method': cleaned_processing_args.get('downsample_method'),
                'train_ratio': cleaned_processing_args.get('train_ratio'),
                'split_by_sequence': cleaned_processing_args.get('split_by_sequence'),
                'mixed_val_split': cleaned_processing_args.get('mixed_val_split'),
                'val_sequences': cleaned_processing_args.get('val_sequences', []),
                'seed': cleaned_processing_args.get('seed')
            }
        },
        'data_splits': split_info,
        'sequence_statistics': sequence_stats,
        'samples': samples_info
    }
    
    # Write metadata file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Created metadata file: {metadata_file}")
    logger.info(f"Included {len(cleaned_processing_args)} processing arguments in metadata")
    logger.info(f"Reconstructed command line for reproducibility: {reconstructed_command}")
    return metadata_file

def save_training_sample(submaps: List[np.ndarray], 
                        submap_meta: List[Dict],
                        output_path: str, 
                        sample_idx: int,
                        sequence_name: str,
                        submap_normals: Optional[List[np.ndarray]] = None,
                        global_transform_matrix: Optional[np.ndarray] = None):
    """
    Save a training sample containing multiple submaps with meta information in filenames.
    
    Args:
        submaps: List of point cloud arrays for each submap
        submap_meta: List of metadata dictionaries for each submap
        output_path: Output directory path for this sample
        sample_idx: Sample index number
        sequence_name: Name of the sequence
        submap_normals: Optional list of normal arrays for each submap
        global_transform_matrix: Optional global transformation matrix
    """
    from . import dataset_utils
    
    os.makedirs(output_path, exist_ok=True)
    
    for submap_idx, (submap_points, meta) in enumerate(zip(submaps, submap_meta)):
        if len(submap_points) == 0:
            continue
        
        # Apply global transformation if provided
        transformed_points = submap_points
        transformed_normals = None
        
        if global_transform_matrix is not None:
            # Check if it's a 3x3 or 4x4 matrix
            if global_transform_matrix.shape == (3, 3):
                # Convert 3x3 to 4x4 by adding homogeneous row/column
                full_transform = np.eye(4, dtype=global_transform_matrix.dtype)
                full_transform[:3, :3] = global_transform_matrix
            elif global_transform_matrix.shape == (4, 4):
                full_transform = global_transform_matrix
            else:
                logger.warning(f"Invalid global transformation matrix shape: {global_transform_matrix.shape}, skipping transformation")
                full_transform = None
            
            if full_transform is not None:
                # Transform points
                transformed_points = dataset_utils.transform_points(submap_points, full_transform)
                
                # Transform normals if available
                if submap_normals and submap_idx < len(submap_normals) and submap_normals[submap_idx] is not None:
                    transformed_normals = dataset_utils.transform_normals(submap_normals[submap_idx], full_transform)
                    
        else:
            # No transformation, use original normals
            if submap_normals and submap_idx < len(submap_normals):
                transformed_normals = submap_normals[submap_idx]
            
        # Create filename and path
        filename = f"sample_{sample_idx:06d}_submap_{submap_idx:02d}_{sequence_name}_frame{meta['start_frame']}_frame{meta['end_frame']}.ply"
        filepath = os.path.join(output_path, filename)
        
        # Create and save point cloud with transformed points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_points)
        
        if transformed_normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(transformed_normals)
        
        o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)

        # Save the 4x4 pose matrix of a randomly selected frame from this submap, if available in meta
        if 'pose_matrix' in meta and isinstance(meta['pose_matrix'], np.ndarray) and meta['pose_matrix'].shape == (4, 4):
            # Create pose filename with "pose" at the beginning
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            pose_filename = f"pose_{base_filename}.txt"
            pose_filepath = os.path.join(os.path.dirname(filepath), pose_filename)
            try:
                # Apply global transformation to the pose matrix if provided
                transformed_pose = meta['pose_matrix']
                if global_transform_matrix is not None and full_transform is not None:
                    # Transform the pose: T_transformed = T_global * T_original
                    transformed_pose = full_transform @ meta['pose_matrix']
                
                np.savetxt(pose_filepath, transformed_pose, fmt='%.8f')
            except Exception as e:
                logger.warning(f"Failed to save pose for {filepath}: {e}")
    
    logger.debug(f"Saved training sample {sample_idx} with {len(submaps)} submaps (global transform applied: {global_transform_matrix is not None})")

def convert_to_hdf5(output_dir: str, 
                   dataset_name: str,
                   hdf5_output_path: str,
                   stats: Dict,
                   args: argparse.Namespace) -> str:
    """
    Convert processed PLY and NPY files to HDF5 format for efficient training.
    Based on the convert_to_hdf5 function in generate_training_samples.py but includes features.
    
    Args:
        output_dir: Root output directory containing processed PLY/NPY files
        dataset_name: Name of the dataset
        hdf5_output_path: Path for the output HDF5 file
        stats: Processing statistics
        args: Command line arguments
    
    Returns:
        Path to the created HDF5 file
    """
    logger.info("Converting processed PLY/NPY files to HDF5 format...")
    logger.debug(f"Output directory: '{output_dir}'")
    logger.debug(f"HDF5 output path: '{hdf5_output_path}'")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(hdf5_output_path), exist_ok=True)
    
    with h5py.File(hdf5_output_path, 'w') as h5_file:
        # Create data_split structure (following the reference format)
        data_split_group = h5_file.create_group("data_split")
        if not dataset_name:  # Fallback if still empty
            dataset_name = "dataset"
        logger.info(f"Dataset name for HDF5: '{dataset_name}'")
        dataset_split_group = data_split_group.create_group(dataset_name)
        
        # Find and read train/val splits
        data_split_dir = os.path.join(output_dir, 'data_split')
        if not os.path.exists(data_split_dir):
            logger.warning(f"No data_split folder found at {data_split_dir}")
            all_samples = []
            train_samples = []
            val_samples = []
        else:
            train_samples = []
            val_samples = []
            
            # Read all available split files
            split_files = {
                'train': 'train.txt',
                'val': 'val.txt',
                'train_random': 'train_random.txt',
                'val_random': 'val_random.txt',
                'train_sequence': 'train_sequence.txt',
                'val_sequence': 'val_sequence.txt'
            }
            
            split_data = {}
            
            for split_name, filename in split_files.items():
                split_file = os.path.join(data_split_dir, filename)
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        samples = [line.strip() for line in f if line.strip()]
                        # Remove '_processed' suffix for HDF5 keys
                        samples = [s.replace('_processed', '') for s in samples]
                        split_data[split_name] = samples
                        logger.info(f"Found {len(samples)} samples in {split_name} split")
                else:
                    logger.debug(f"Split file not found: {split_file}")
            
            # Store all available splits in HDF5
            for split_name, samples in split_data.items():
                if samples:  # Only create dataset if we have samples
                    samples_encoded = [name.encode('utf-8') for name in samples]
                    dataset_split_group.create_dataset(split_name, data=samples_encoded)
                    logger.info(f"Stored {split_name} split with {len(samples)} samples")
            
            # Use primary train/val for all_samples (fallback to any available)
            if 'train' in split_data and 'val' in split_data:
                train_samples = split_data['train']
                val_samples = split_data['val']
            elif 'train_random' in split_data and 'val_random' in split_data:
                train_samples = split_data['train_random']
                val_samples = split_data['val_random']
            elif 'train_sequence' in split_data and 'val_sequence' in split_data:
                train_samples = split_data['train_sequence']
                val_samples = split_data['val_sequence']
            else:
                # Fallback: use any available splits
                all_split_samples = []
                for samples in split_data.values():
                    all_split_samples.extend(samples)
                train_samples = all_split_samples
                val_samples = []
            
            all_samples = train_samples + val_samples
        
        # If we don't have split files, find all processed samples
        if not all_samples:
            logger.info("No split files found, scanning for all processed samples...")
            all_samples = []
            dataset_path = os.path.join(output_dir, dataset_name)
            if os.path.exists(dataset_path):
                for sequence_name in os.listdir(dataset_path):
                    sequence_path = os.path.join(dataset_path, sequence_name)
                    if os.path.isdir(sequence_path):
                        for sample_name in os.listdir(sequence_path):
                            if sample_name.endswith('_processed'):
                                # Remove '_processed' suffix for HDF5 key
                                original_name = sample_name.replace('_processed', '')
                                relative_path = os.path.join(sequence_name, original_name)
                                all_samples.append(relative_path)
        
        if not all_samples:
            logger.error(f"No processed samples found in {output_dir}")
            return hdf5_output_path
        
        logger.info(f"Converting {len(all_samples)} samples to HDF5...")
        
        # Process each sample and store in HDF5
        sample_num_points = []  # Store num_points for each sample
        
        for sample_idx, fragment_path in enumerate(tqdm(all_samples, desc="Converting to HDF5")):
            # Skip empty fragment paths
            if not fragment_path or not fragment_path.strip():
                logger.warning(f"Skipping empty fragment path at index {sample_idx}")
                sample_num_points.append(0)  # Add 0 for empty samples
                continue

            # print(f"Processing sample: {fragment_path}")
            
            # Path to the processed sample directory containing PLY and NPY files
            processed_sample_path = os.path.join(output_dir, fragment_path + '_processed')
            
            if os.path.exists(processed_sample_path):
                # Create group for this fragment (using original path without _processed)
                fragment_group = h5_file.create_group(fragment_path)
                
                # Get all PLY files in this sample
                ply_files = sorted([f for f in os.listdir(processed_sample_path) if f.endswith('.ply')])
                
                # Track total points in this sample
                sample_total_points = 0
                
                for submap_idx, ply_file in enumerate(ply_files):
                    ply_path = os.path.join(processed_sample_path, ply_file)
                    
                    # Extract part name from PLY filename
                    part_name = os.path.splitext(ply_file)[0]
                    
                    # Load point cloud from PLY
                    pcd = o3d.io.read_point_cloud(ply_path)
                    points = np.asarray(pcd.points)
                    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
                    
                    # Add to sample total points
                    sample_total_points += len(points)
                    
                    # Load corresponding features from NPY
                    feature_file = f'features_{part_name}.npy'
                    feature_path = os.path.join(processed_sample_path, feature_file)
                    features = None
                    if os.path.exists(feature_path):
                        try:
                            features = np.load(feature_path)
                        except Exception as e:
                            logger.warning(f"Failed to load features from {feature_path}: {e}")
                    
                    # Create submap group
                    submap_group = fragment_group.create_group(str(submap_idx))
                    
                    # Store vertices (points) - compress for efficiency
                    submap_group.create_dataset('vertices', data=points.astype(np.float32), compression='gzip')
                    
                    # Store normals if available
                    if normals is not None and len(normals) > 0:
                        submap_group.create_dataset('normals', data=normals.astype(np.float32), compression='gzip')
                    
                    # Store features if available
                    if features is not None and len(features) > 0:
                        submap_group.create_dataset('features', data=features.astype(np.float32), compression='gzip')
                    
                    # Store pose information if available
                    # Look for corresponding pose file (pose_*.txt)
                    pose_filename = f"pose_{part_name}.txt"
                    pose_path = os.path.join(processed_sample_path, pose_filename)
                    
                    if os.path.exists(pose_path):
                        try:
                            pose_matrix = np.loadtxt(pose_path, dtype=np.float64)
                            if pose_matrix.shape == (4, 4):
                                submap_group.create_dataset('pose', data=pose_matrix.astype(np.float32), compression='gzip')
                                logger.debug(f"Stored pose for part {part_name} in {fragment_path}")
                            else:
                                logger.warning(f"Invalid pose matrix shape {pose_matrix.shape} in {pose_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load pose from {pose_path}: {e}")
                    else:
                        logger.debug(f"No pose file found for part {part_name} in {fragment_path}")
                    
                    # Store metadata as attributes
                    submap_group.attrs['part_name'] = part_name.encode('utf-8')
                    submap_group.attrs['num_points'] = len(points)
                    if features is not None:
                        submap_group.attrs['feature_dim'] = features.shape[-1] if len(features.shape) > 1 else 1
                
                # Store the total points for this sample
                sample_num_points.append(sample_total_points)
            else:
                logger.warning(f"Processed sample directory not found: {processed_sample_path}")
                sample_num_points.append(0)  # Add 0 for missing samples
        
        # Store num_points data in HDF5 format (following the precompute_num_points.py structure)
        # Use sample_num_points from stats if available, otherwise compute from HDF5 data
        stats_num_points = stats.get('sample_num_points', [])
        if stats_num_points or sample_num_points:
            final_num_points = stats_num_points if stats_num_points else sample_num_points
            
            # Create num_points group structure: /num_points/dataset_name/split
            if "num_points" not in h5_file:
                num_points_group = h5_file.create_group("num_points")
            else:
                num_points_group = h5_file["num_points"]
            
            if not dataset_name:
                dataset_name = "dataset"  # fallback
            
            if dataset_name not in num_points_group:
                num_points_group_dataset = num_points_group.create_group(dataset_name)
            else:
                num_points_group_dataset = num_points_group[dataset_name]
            
            # Create normalized mapping for samples
            def normalize_path_hdf5(path: str) -> str:
                """Normalize path by removing dataset prefix and _processed suffix for matching."""
                if path.startswith(f"{dataset_name}/"):
                    path = path[len(f"{dataset_name}/"):]
                if path.endswith("_processed"):
                    path = path[:-len("_processed")]
                return path
            
            sample_to_num_points = {normalize_path_hdf5(sample): final_num_points[i] for i, sample in enumerate(all_samples)}
            
            # Create num_points datasets for each available split
            for split_name, samples in split_data.items():
                if samples:  # Only create dataset if we have samples
                    split_num_points = []
                    for sample in samples:
                        normalized_sample = normalize_path_hdf5(sample)
                        if normalized_sample in sample_to_num_points:
                            split_num_points.append(sample_to_num_points[normalized_sample])
                        else:
                            logger.warning(f"Sample {sample} (normalized: {normalized_sample}) not found in processed data for num_points")
                            split_num_points.append(0)
                    
                    if split_num_points:
                        # Remove existing dataset if it exists
                        if split_name in num_points_group_dataset:
                            del num_points_group_dataset[split_name]
                        num_points_group_dataset.create_dataset(split_name, data=split_num_points, dtype=np.int32)
                        logger.info(f"Stored num_points for {split_name} split: {len(split_num_points)} samples")
                        logger.debug(f"{split_name} num_points stats: min={min(split_num_points)}, max={max(split_num_points)}, mean={np.mean(split_num_points):.1f}")
            
            # Also store all samples under 'all' split for completeness
            if 'all' in num_points_group_dataset:
                del num_points_group_dataset['all']
            num_points_group_dataset.create_dataset('all', data=final_num_points, dtype=np.int32)
            logger.info(f"Stored num_points for 'all' split: {len(final_num_points)} samples")
            logger.debug(f"all num_points stats: min={min(final_num_points)}, max={max(final_num_points)}, mean={np.mean(final_num_points):.1f}")
        
        # Save processing metadata as root attributes
        h5_file.attrs['script'] = 'extract_sample_features.py'
        h5_file.attrs['timestamp'] = datetime.now().isoformat().encode('utf-8')
        h5_file.attrs['processed_samples'] = stats.get('processed_samples', 0)
        h5_file.attrs['failed_samples'] = stats.get('failed_samples', 0)
        h5_file.attrs['total_samples'] = stats.get('total_samples', 0)
        
        # Save processing parameters
        h5_file.attrs['num_points_fps'] = args.num_points
        h5_file.attrs['min_points_per_part'] = args.min_points_per_part
        h5_file.attrs['global_seed'] = args.global_seed
        h5_file.attrs['remove_outliers'] = args.remove_outliers
        h5_file.attrs['outlier_nb_neighbors'] = args.outlier_nb_neighbors
        h5_file.attrs['outlier_std_ratio'] = args.outlier_std_ratio
        h5_file.attrs['allocation_method'] = args.allocation_method
        h5_file.attrs['voxel_size'] = args.voxel_size
        h5_file.attrs['voxel_ratio'] = args.voxel_ratio
        #h5_file.attrs['max_sample_points'] = args.max_sample_points
        #h5_file.attrs['min_sample_points'] = args.min_sample_points
        
        # Save feature extractor parameters
        h5_file.attrs['feature_extractor'] = 'miniSpinNet'
        h5_file.attrs['des_r'] = args.des_r
        h5_file.attrs['checkpoint_path'] = (args.checkpoint or 'None').encode('utf-8')
        h5_file.attrs['device'] = args.device.encode('utf-8')
        h5_file.attrs['num_points_per_patch'] = args.num_points_per_patch
        
        # Save input/output paths for reproducibility
        h5_file.attrs['input_path'] = args.input.encode('utf-8')
        h5_file.attrs['output_path'] = args.output.encode('utf-8')
        if args.dataset_name:
            h5_file.attrs['dataset_name'] = args.dataset_name.encode('utf-8')
        
        # Save HDF5-specific parameters
        h5_file.attrs['save_hdf5'] = args.save_hdf5
        h5_file.attrs['hdf5_only'] = args.hdf5_only
        if args.hdf5_output:
            h5_file.attrs['hdf5_output_path'] = args.hdf5_output.encode('utf-8')
        
        # Save data structure information
        h5_file.attrs['includes_pose_data'] = True
        h5_file.attrs['includes_features'] = True
        h5_file.attrs['includes_normals'] = True
    
    logger.info(f"Created HDF5 file: {hdf5_output_path}")
    logger.info(f"Total samples converted: {len(all_samples)}")
    if train_samples or val_samples:
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Val samples: {len(val_samples)}")
    
    return hdf5_output_path 