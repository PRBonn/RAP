#!/usr/bin/env python3
"""
Visualize Registered Point Clouds using Estimated Poses

This script loads point clouds from an input directory and transforms them using
estimated poses from a result directory. Visualizes one sample at a time with
keyboard navigation (spacebar for next sample).

Usage:
    python visualize_registered_pointclouds.py \
        --input ./dataset/lidar_rpf_training_data/scannet_test_v1 \
        --results ./dataset/lidar_rpf_training_data/scannet_test_v1_result \
        --sequence scene0025_01
    
    # Visualize all sequences
    python visualize_registered_pointclouds.py \
        --input ./dataset/lidar_rpf_training_data/scannet_test_v1 \
        --results ./dataset/lidar_rpf_training_data/scannet_test_v1_result
"""

import argparse
import os
import glob
import re
import logging
import json
import numpy as np
import open3d as o3d
import natsort
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataset_process.utils.io_utils import CMAP_DEFAULT

# Logger will be configured in main() after parsing arguments
logger = logging.getLogger(__name__)

class RegisteredPointCloudVisualizer:
    """Visualizer for registered point clouds using estimated poses."""
    
    def __init__(self,
                 input_dir: str,
                 poses_dir: str,
                 point_size: float = 3.0,
                 background_color: List[float] = [1.0, 1.0, 1.0],
                 no_coordinate_frame: bool = True,
                 estimate_normals: bool = False,
                 normal_estimation_radius: float = 0.1,
                 generation: str = "generation00",
                 max_points_per_fragment: int = 10000,
                 remove_outliers: bool = False,
                 outlier_nb_neighbors: int = 20,
                 outlier_std_ratio: float = 2.0):
        """
        Initialize the registered point cloud visualizer.
        
        Args:
            input_dir: Root directory containing point cloud files
            poses_dir: Root directory containing pose files
            point_size: Size of points in the visualization
            background_color: Background color as [R, G, B] in range [0, 1]
            no_coordinate_frame: Whether to hide coordinate frame
            estimate_normals: If True, estimate normals for point clouds that don't have them
            normal_estimation_radius: Radius for normal estimation
            generation: Generation subdirectory name (default: "generation00")
            max_points_per_fragment: Maximum points to load per fragment
            remove_outliers: If True, apply stochastic outlier removal to input point clouds
            outlier_nb_neighbors: Number of neighbors for outlier removal (default: 20)
            outlier_std_ratio: Standard deviation ratio for outlier removal (default: 2.0)
        """
        self.input_dir = input_dir
        self.poses_dir = poses_dir
        self.point_size = point_size
        self.background_color = background_color
        self.no_coordinate_frame = no_coordinate_frame
        self.estimate_normals = estimate_normals
        self.normal_estimation_radius = normal_estimation_radius
        self.generation = generation
        self.max_points_per_fragment = max_points_per_fragment
        self.remove_outliers = remove_outliers
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        
        # Navigation state
        self.current_sequence = None
        self.current_sample_idx = None
        self.sequence_samples_cache = {}  # Cache for sequence sample lists
        self.vis = None
        self.colored_pcds = []
        self.original_pcds = []  # Store original (untransformed) point clouds
        self.registered_pcds = []  # Store registered (transformed) point clouds
        self.generated_pcds = []  # Store generated point clouds from processed directory
        self.generated_gt_pcd = None  # Store merged GT point cloud
        self.random_yaw_pcds = []  # Store input point clouds with random yaw rotations
        self.view_mode = "registered"  # "original", "registered", "generated", "gt", or "random_yaw"
        
        # Generation cycling state
        self.cycle_generations = False  # Whether to cycle through generations instead of samples
        self.current_generation_idx = 0  # Current generation index when cycling
        self.available_generations = []  # List of available generation names for current sample
        
        # Background toggle state
        self.use_white_background = True
        self.white_background = [1.0, 1.0, 1.0]
        self.black_background = [0.0, 0.0, 0.0]
        
        if np.mean(self.background_color) > 0.5:
            self.use_white_background = True
        else:
            self.use_white_background = False
    
    def compute_adaptive_normal_radius(self, points: np.ndarray) -> float:
        """
        Compute adaptive normal estimation radius based on point cloud scale.
        Uses 2% of average extent, with a minimum of 0.05.
        
        Args:
            points: Point cloud points as numpy array (N, 3)
        
        Returns:
            Adaptive radius value
        """
        if len(points) == 0:
            return self.normal_estimation_radius
        
        # Compute extent (bounding box size)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extent = max_coords - min_coords
        
        # Use adaptive radius based on point cloud scale (2% of average extent)
        adaptive_radius = max(np.mean(extent) * 0.02, 0.05)
        
        return adaptive_radius
    
    def apply_outlier_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply stochastic outlier removal to a point cloud.
        
        Args:
            pcd: Open3D point cloud
        
        Returns:
            Point cloud with outliers removed
        """
        if not self.remove_outliers:
            return pcd
        
        points = np.asarray(pcd.points)
        if len(points) < self.outlier_nb_neighbors:
            # Not enough points for outlier removal
            logger.debug(f"Skipping outlier removal: only {len(points)} points (need at least {self.outlier_nb_neighbors})")
            return pcd
        
        try:
            # Apply statistical outlier removal
            # This returns (cleaned_pcd, indices) where indices are the inlier indices
            pcd_cleaned, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio
            )
            
            num_removed = len(points) - len(np.asarray(pcd_cleaned.points))
            if num_removed > 0:
                logger.debug(f"Removed {num_removed} outliers ({num_removed/len(points)*100:.1f}%)")
            
            # Colors and normals are automatically preserved by remove_statistical_outlier
            # as it filters by indices, so we don't need to manually copy them
            
            return pcd_cleaned
        except Exception as e:
            logger.warning(f"Failed to apply outlier removal: {e}, returning original point cloud")
            return pcd
    
    def center_pcd(self, pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center point cloud at origin."""
        center = np.mean(pcd, axis=0)
        pcd = pcd - center
        return pcd, center
    
    def translate_point_cloud(self, pcd: o3d.geometry.PointCloud, translation: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Translate a point cloud by a given translation vector.
        
        Args:
            pcd: Open3D point cloud
            translation: Translation vector (3D)
        
        Returns:
            Translated point cloud
        """
        points = np.asarray(pcd.points)
        translated_points = points - translation  # Subtract to move towards origin
        
        translated_pcd = o3d.geometry.PointCloud()
        translated_pcd.points = o3d.utility.Vector3dVector(translated_points)
        
        # Copy normals if available
        if pcd.has_normals():
            translated_pcd.normals = pcd.normals
        
        # Copy colors if available
        if pcd.has_colors():
            translated_pcd.colors = pcd.colors
        
        return translated_pcd
    
    def apply_random_yaw_rotation(self, pcd: o3d.geometry.PointCloud, skip_rotation: bool = False) -> o3d.geometry.PointCloud:
        """
        Apply random yaw rotation (around z-axis) with small roll and pitch perturbations to a point cloud.
        First centers the point cloud at origin, then applies random rotations:
        - Yaw: random rotation around z-axis (0 to 2π)
        - Roll: random rotation around x-axis (±15 degrees)
        - Pitch: random rotation around y-axis (±15 degrees)
        
        Args:
            pcd: Open3D point cloud
            skip_rotation: If True, only center the point cloud without applying rotation
        
        Returns:
            Point cloud with random rotations applied (or just centered if skip_rotation=True)
        """
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return pcd
        
        # Center the point cloud at origin
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # If skip_rotation is True, just return centered point cloud
        if skip_rotation:
            centered_pcd = o3d.geometry.PointCloud()
            centered_pcd.points = o3d.utility.Vector3dVector(centered_points)
            
            # Copy normals if available
            if pcd.has_normals():
                centered_pcd.normals = pcd.normals
            
            # Copy colors if available
            if pcd.has_colors():
                centered_pcd.colors = pcd.colors
            
            return centered_pcd
        
        # Generate random rotation angles
        yaw_angle = np.random.uniform(0, 2 * np.pi)  # Full yaw rotation
        roll_angle = np.random.uniform(-15, 15) * np.pi / 180  # ±15 degrees roll
        pitch_angle = np.random.uniform(-15, 15) * np.pi / 180  # ±15 degrees pitch
        
        # Create rotation matrices
        # Roll rotation around x-axis
        cos_roll = np.cos(roll_angle)
        sin_roll = np.sin(roll_angle)
        R_roll = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        # Pitch rotation around y-axis
        cos_pitch = np.cos(pitch_angle)
        sin_pitch = np.sin(pitch_angle)
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Yaw rotation around z-axis
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        # Combine rotations: R = R_yaw * R_pitch * R_roll
        # This applies roll first, then pitch, then yaw
        rotation_matrix = R_yaw @ R_pitch @ R_roll
        
        # Apply rotation to centered points
        rotated_points = (rotation_matrix @ centered_points.T).T
        
        # Create new point cloud
        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
        
        # Transform normals if available
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            if len(normals) == len(points):
                # Only rotate normals (no translation for normals)
                rotated_normals = (rotation_matrix @ normals.T).T
                rotated_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)
        
        # Copy colors if available
        if pcd.has_colors():
            rotated_pcd.colors = pcd.colors
        
        return rotated_pcd
    
    def find_max_points_generated_part_index(self, generated_pcds: List[o3d.geometry.PointCloud]) -> Optional[int]:
        """
        Find the index of the generated part with the most points.
        
        Args:
            generated_pcds: List of generated point clouds
        
        Returns:
            Index of the part with the most points, or None if list is empty
        """
        if not generated_pcds:
            return None
        
        max_points = -1
        max_idx = None
        
        for idx, pcd in enumerate(generated_pcds):
            points = np.asarray(pcd.points)
            num_points = len(points)
            if num_points > max_points:
                max_points = num_points
                max_idx = idx
        
        return max_idx
    
    def create_random_yaw_point_clouds(self, input_pcds: List[o3d.geometry.PointCloud], 
                                      generated_pcds: Optional[List[o3d.geometry.PointCloud]] = None) -> List[o3d.geometry.PointCloud]:
        """
        Create random rotated versions of input point clouds.
        Each part is centered at origin and has random rotations applied:
        - Yaw: random rotation around z-axis (0 to 2π)
        - Roll: random rotation around x-axis (±15 degrees)
        - Pitch: random rotation around y-axis (±15 degrees)
        
        If generated_pcds are provided, the input part corresponding to the generated part
        with the most points will NOT be rotated (only centered).
        
        Args:
            input_pcds: List of original input point clouds
            generated_pcds: Optional list of generated point clouds to determine which part to skip rotation
        
        Returns:
            List of point clouds with random rotations applied
        """
        # Find which part should not be rotated (if generated_pcds are available)
        skip_rotation_idx = None
        if generated_pcds:
            max_points_idx = self.find_max_points_generated_part_index(generated_pcds)
            if max_points_idx is not None and max_points_idx < len(input_pcds):
                skip_rotation_idx = max_points_idx
                logger.info(f"Skipping rotation for input part {skip_rotation_idx} (corresponds to generated part with most points: {len(np.asarray(generated_pcds[max_points_idx].points))} points)")
        
        random_yaw_pcds = []
        for idx, pcd in enumerate(input_pcds):
            skip_rotation = (skip_rotation_idx == idx)
            rotated_pcd = self.apply_random_yaw_rotation(pcd, skip_rotation=skip_rotation)
            random_yaw_pcds.append(rotated_pcd)
        return random_yaw_pcds
    
    def merge_input_point_clouds(self, input_pcds: List[o3d.geometry.PointCloud]) -> Optional[np.ndarray]:
        """
        Merge input point clouds and return all points as a numpy array.
        Used as fallback for center of mass calculation when GT is not available.
        
        Args:
            input_pcds: List of input point clouds
        
        Returns:
            Merged points as numpy array, or None if no points available
        """
        if not input_pcds:
            return None
        
        all_points = []
        for pcd in input_pcds:
            points = np.asarray(pcd.points)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return None
        
        merged_points = np.concatenate(all_points, axis=0)
        return merged_points
    
    def find_sequence_directories(self, input_dir: str, sequence: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find input and result sequence directories.
        
        Returns:
            Tuple of (input_sequence_dir, result_sequence_dir) or (None, None) if not found
            If result directory doesn't exist separately, result_sequence_dir may be the same as input_sequence_dir
        """
        sequence_parts = sequence.split('/')
        search_sequence = sequence_parts[-1] if sequence_parts else sequence
        
        # Find input sequence directory
        input_seq_dir = None
        candidate = os.path.join(self.input_dir, sequence)
        if os.path.exists(candidate):
            input_seq_dir = candidate
        
        if input_seq_dir is None and len(sequence_parts) > 1:
            candidate = self.input_dir
            for part in sequence_parts:
                candidate = os.path.join(candidate, part)
                if not os.path.exists(candidate):
                    candidate = None
                    break
            if candidate:
                input_seq_dir = candidate
        
        if input_seq_dir is None:
            def find_seq_dir(root_dir: str, target: str, max_depth: int = 3) -> Optional[str]:
                if max_depth <= 0:
                    return None
                if os.path.basename(root_dir) == target or target in os.path.basename(root_dir):
                    return root_dir
                if os.path.isdir(root_dir):
                    try:
                        for item in os.listdir(root_dir):
                            item_path = os.path.join(root_dir, item)
                            if os.path.isdir(item_path):
                                result = find_seq_dir(item_path, target, max_depth - 1)
                                if result:
                                    return result
                    except PermissionError:
                        pass
                return None
            
            input_seq_dir = find_seq_dir(self.input_dir, search_sequence)
        
        # Find result sequence directory
        result_seq_dir = None
        
        # First, try to find a separate result directory
        candidate = os.path.join(self.poses_dir, sequence)
        if os.path.exists(candidate):
            result_seq_dir = candidate
        
        if result_seq_dir is None and len(sequence_parts) > 1:
            candidate = self.poses_dir
            for part in sequence_parts:
                candidate = os.path.join(candidate, part)
                if not os.path.exists(candidate):
                    candidate = None
                    break
            if candidate:
                result_seq_dir = candidate
        
        if result_seq_dir is None:
            def find_result_seq_dir(root_dir: str, target: str, max_depth: int = 3) -> Optional[str]:
                if max_depth <= 0:
                    return None
                if os.path.basename(root_dir) == target or target in os.path.basename(root_dir):
                    return root_dir
                if os.path.isdir(root_dir):
                    try:
                        for item in os.listdir(root_dir):
                            item_path = os.path.join(root_dir, item)
                            if os.path.isdir(item_path):
                                result = find_result_seq_dir(item_path, target, max_depth - 1)
                                if result:
                                    return result
                    except PermissionError:
                        pass
                return None
            
            result_seq_dir = find_result_seq_dir(self.poses_dir, search_sequence)
        
        # If no separate result directory found, use input directory (for cases where transforms are alongside PLY files)
        if result_seq_dir is None and input_seq_dir is not None:
            # Check if transform files exist in the input directory
            # This handles cases like nss_ss_v1 where transforms are in the same directory as PLY files
            result_seq_dir = input_seq_dir
        
        return input_seq_dir, result_seq_dir
    
    def find_sample_directories(self, sequence_dir: str) -> List[str]:
        """
        Find all sample directories in a sequence directory.
        Samples can be:
        1. Directories starting with 'sample_' (original format)
        2. Directories containing PLY files directly (modelnet format)
        """
        sample_dirs = []
        if not os.path.exists(sequence_dir):
            return sample_dirs
        
        # First, look for sample_* directories (original format)
        for item in os.listdir(sequence_dir):
            item_path = os.path.join(sequence_dir, item)
            if os.path.isdir(item_path) and item.startswith('sample_'):
                sample_dirs.append(item_path)
        
        # If no sample_* directories found, look for directories containing PLY files (modelnet format)
        if not sample_dirs:
            for item in os.listdir(sequence_dir):
                item_path = os.path.join(sequence_dir, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains PLY files
                    ply_files = glob.glob(os.path.join(item_path, "*.ply"))
                    if ply_files:
                        sample_dirs.append(item_path)
        
        return sorted(sample_dirs)
    
    def get_available_generations(self, result_sample_dir: str) -> List[str]:
        """
        Find all available generation names for a sample directory.
        Returns a list of generation names (e.g., ['generation_selected', 'generation00', 'generation01', ...])
        sorted by priority: generation_selected first, then generation00, generation01, etc.
        """
        if not os.path.exists(result_sample_dir):
            return []
        
        generations = set()
        
        # Find all transform files with generation patterns
        transform_patterns = [
            "*generation*_transform.txt",
            "*_generation*transform.txt",
        ]
        
        for pattern in transform_patterns:
            matches = glob.glob(os.path.join(result_sample_dir, pattern))
            for match in matches:
                basename = os.path.basename(match)
                # Extract generation name
                # Try to match patterns like: *generation_selected*transform.txt, *generation00*transform.txt
                gen_match = re.search(r'generation(_selected|\d+)', basename)
                if gen_match:
                    gen_name = gen_match.group(0)  # e.g., 'generation_selected' or 'generation00'
                    generations.add(gen_name)
        
        # Sort: generation_selected first, then generation00, generation01, etc.
        def sort_key(gen_name):
            if gen_name == "generation_selected":
                return (0, "")  # Highest priority
            else:
                # Extract number from generation00, generation01, etc.
                num_match = re.search(r'generation(\d+)', gen_name)
                if num_match:
                    return (1, int(num_match.group(1)))  # Sort by number
                return (2, gen_name)  # Other patterns last
        
        sorted_generations = sorted(generations, key=sort_key)
        return sorted_generations
    
    def find_matching_transform_file(self, ply_filename: str, result_sample_dir: str, generation: Optional[str] = None) -> Optional[str]:
        """
        Find matching transform file for a PLY file.
        If generation is specified, uses that generation. Otherwise, first tries to find 
        *generation_selected*.txt files, then falls back to generation00.
        """
        basename = os.path.splitext(os.path.basename(ply_filename))[0]
        part_match = None
        
        submap_match = re.search(r'submap_(\d+)', basename)
        if submap_match:
            part_num = int(submap_match.group(1))
            part_match = f"part{part_num:02d}"
        
        if part_match is None:
            part_match_obj = re.search(r'part_(\d+)', basename)
            if part_match_obj:
                part_num = int(part_match_obj.group(1))
                part_match = f"part{part_num:02d}"
        
        # Also handle simple "part(\d+)" pattern without underscore (e.g., "part1.ply")
        if part_match is None:
            part_match_obj = re.search(r'part(\d+)', basename)
            if part_match_obj:
                part_num = int(part_match_obj.group(1))
                part_match = f"part{part_num:02d}"
        
        # If generation is specified, use it directly
        if generation:
            if part_match:
                # Try multiple patterns for the specified generation
                patterns = [
                    os.path.join(result_sample_dir, f"*{generation}*{part_match}_transform.txt"),
                    os.path.join(result_sample_dir, f"*{generation}_{part_match}_transform.txt"),
                    os.path.join(result_sample_dir, f"*_{generation}_*{part_match}_transform.txt"),
                    os.path.join(result_sample_dir, f"*{part_match}_{generation}*transform.txt"),
                ]
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        return matches[0]
            
            sample_match = re.search(r'sample_(\d+)', basename)
            if sample_match:
                sample_id = sample_match.group(1)
                patterns = [
                    os.path.join(result_sample_dir, f"*sample{sample_id}*{generation}*transform.txt"),
                    os.path.join(result_sample_dir, f"*sample{sample_id}*_{generation}_*transform.txt"),
                ]
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        if part_match:
                            for match in matches:
                                if part_match in match:
                                    return match
                        if matches:
                            return matches[0]
            return None
        
        # If no generation specified, use default priority: generation_selected first, then generation00
        if part_match:
            # Try generation_selected pattern first (more flexible pattern)
            patterns_selected = [
                os.path.join(result_sample_dir, f"*generation_selected*{part_match}_transform.txt"),
                os.path.join(result_sample_dir, f"*_generation_selected_*{part_match}_transform.txt"),
                os.path.join(result_sample_dir, f"*{part_match}_generation_selected*transform.txt"),
            ]
            for pattern_selected in patterns_selected:
                matches_selected = glob.glob(pattern_selected)
                if matches_selected:
                    return matches_selected[0]
            
            # Fall back to generation00
            pattern = os.path.join(result_sample_dir, f"*generation00_{part_match}_transform.txt")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        
        sample_match = re.search(r'sample_(\d+)', basename)
        if sample_match:
            sample_id = sample_match.group(1)
            # Try generation_selected pattern first (more flexible pattern)
            patterns_selected = [
                os.path.join(result_sample_dir, f"*sample{sample_id}*generation_selected*transform.txt"),
                os.path.join(result_sample_dir, f"*sample{sample_id}*_generation_selected_*transform.txt"),
            ]
            for pattern_selected in patterns_selected:
                matches_selected = glob.glob(pattern_selected)
                if matches_selected:
                    if part_match:
                        for match in matches_selected:
                            if part_match in match:
                                return match
                    if matches_selected:
                        return matches_selected[0]
            
            # Fall back to generation00
            pattern = os.path.join(result_sample_dir, f"*sample{sample_id}*generation00*transform.txt")
            matches = glob.glob(pattern)
            if matches:
                if part_match:
                    for match in matches:
                        if part_match in match:
                            return match
                return matches[0]
        
        return None
    
    def find_matching_transform_file_by_index(self, part_index: int, result_sample_dir: str, generation: Optional[str] = None) -> Optional[str]:
        """
        Find matching transform file by part index (for PLY files without part numbers in their names).
        
        Args:
            part_index: Zero-based index of the part (0, 1, 2, ...)
            result_sample_dir: Directory containing transform files
            generation: Optional generation name to use (e.g., 'generation00', 'generation_selected')
        
        Returns:
            Path to matching transform file, or None if not found
        """
        part_match = f"part{part_index:02d}"
        
        # If generation is specified, use it directly
        if generation:
            patterns = [
                os.path.join(result_sample_dir, f"*{generation}*{part_match}_transform.txt"),
                os.path.join(result_sample_dir, f"*{generation}_{part_match}_transform.txt"),
                os.path.join(result_sample_dir, f"*_{generation}_*{part_match}_transform.txt"),
                os.path.join(result_sample_dir, f"*{part_match}_{generation}*transform.txt"),
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
            return None
        
        # If no generation specified, use default priority: generation_selected first, then generation00
        # Try generation_selected pattern first
        patterns_selected = [
            os.path.join(result_sample_dir, f"*generation_selected*{part_match}_transform.txt"),
            os.path.join(result_sample_dir, f"*_generation_selected_*{part_match}_transform.txt"),
            os.path.join(result_sample_dir, f"*{part_match}_generation_selected*transform.txt"),
        ]
        for pattern_selected in patterns_selected:
            matches_selected = glob.glob(pattern_selected)
            if matches_selected:
                return matches_selected[0]
        
        # Fall back to generation00
        pattern = os.path.join(result_sample_dir, f"*generation00*{part_match}_transform.txt")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        
        return None
    
    def load_pose_from_file(self, pose_file: str) -> Optional[np.ndarray]:
        """Load 4x4 pose matrix from text file."""
        try:
            pose = np.loadtxt(pose_file)
            if pose.shape != (4, 4):
                if pose.shape == (16,):
                    pose = pose.reshape(4, 4)
                else:
                    logger.warning(f"Invalid pose shape {pose.shape} in {pose_file}")
                    return None
            return pose
        except Exception as e:
            logger.warning(f"Failed to load pose from {pose_file}: {e}")
            return None
    
    def load_json_metadata(self, result_sample_dir: str) -> Optional[Dict]:
        """
        Load JSON metadata file from result sample directory.
        Tries multiple patterns to find metadata files, prioritizing generation_selected.
        """
        # Priority order: generation_selected > generation00 > other generation files > metadata.json > any *.json
        
        # 1. Try generation_selected JSON files first (matching transform file priority)
        pattern_selected = os.path.join(result_sample_dir, "*generation_selected.json")
        matches_selected = glob.glob(pattern_selected)
        if matches_selected:
            try:
                with open(matches_selected[0], 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from generation_selected file: {os.path.basename(matches_selected[0])}")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load generation_selected metadata file {matches_selected[0]}: {e}")
        
        # 2. Try generation00 JSON files
        pattern_generation00 = os.path.join(result_sample_dir, "*generation00.json")
        matches_generation00 = glob.glob(pattern_generation00)
        if matches_generation00:
            try:
                with open(matches_generation00[0], 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from generation00 file: {os.path.basename(matches_generation00[0])}")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load generation00 metadata file {matches_generation00[0]}: {e}")
        
        # 3. Try any generation*.json files
        pattern_generation = os.path.join(result_sample_dir, "*generation*.json")
        matches_generation = glob.glob(pattern_generation)
        if matches_generation:
            # Sort to get the highest generation number
            matches_generation_sorted = sorted(matches_generation, reverse=True)
            try:
                with open(matches_generation_sorted[0], 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from generation file: {os.path.basename(matches_generation_sorted[0])}")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load generation metadata file {matches_generation_sorted[0]}: {e}")
        
        # 4. Try exact metadata.json
        metadata_path = os.path.join(result_sample_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from metadata.json")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load metadata.json: {e}")
        
        # 5. Try metadata_sample_*.json pattern
        pattern_metadata_sample = os.path.join(result_sample_dir, "metadata_sample_*.json")
        matches_metadata_sample = glob.glob(pattern_metadata_sample)
        if matches_metadata_sample:
            try:
                with open(matches_metadata_sample[0], 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from metadata_sample file: {os.path.basename(matches_metadata_sample[0])}")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load metadata_sample file {matches_metadata_sample[0]}: {e}")
        
        # 6. Last resort: try any *.json file (but skip transform files)
        pattern_any_json = os.path.join(result_sample_dir, "*.json")
        matches_any_json = glob.glob(pattern_any_json)
        # Filter out transform-related files
        matches_any_json = [m for m in matches_any_json if "transform" not in os.path.basename(m).lower()]
        if matches_any_json:
            try:
                with open(matches_any_json[0], 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from JSON file: {os.path.basename(matches_any_json[0])}")
                    return metadata
            except Exception as e:
                logger.debug(f"Failed to load JSON file {matches_any_json[0]}: {e}")
        
        return None
    
    def print_metadata(self, metadata: Dict):
        """Print metadata in a formatted way."""
        if not metadata:
            return
        
        logger.info("=" * 60)
        logger.info("METADATA:")
        logger.info("=" * 60)
        
        def print_dict(d, indent=0):
            """Recursively print dictionary with indentation."""
            for key, value in d.items():
                if isinstance(value, dict):
                    logger.info("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                elif isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], dict):
                        logger.info("  " * indent + f"{key}: [")
                        for i, item in enumerate(value[:5]):  # Limit to first 5 items
                            logger.info("  " * (indent + 1) + f"Item {i}:")
                            print_dict(item, indent + 2)
                        if len(value) > 5:
                            logger.info("  " * (indent + 1) + f"... ({len(value) - 5} more items)")
                        logger.info("  " * indent + "]")
                    else:
                        logger.info("  " * indent + f"{key}: {value}")
                else:
                    logger.info("  " * indent + f"{key}: {value}")
        
        print_dict(metadata)
        logger.info("=" * 60)
    
    def load_sample_point_clouds(self, input_sample_dir: str, result_sample_dir: str, return_transforms: bool = False, generation: Optional[str] = None) -> Union[Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[str]], Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[str], List[np.ndarray]]]:
        """
        Load both original and transformed point clouds for a single sample.
        
        Args:
            input_sample_dir: Directory containing input PLY files
            result_sample_dir: Directory containing transform files
            return_transforms: Whether to return transform matrices
            generation: Optional generation name to use (e.g., 'generation00', 'generation_selected')
        
        Returns:
            If return_transforms=False: Tuple of (list of original point clouds, list of registered point clouds, list of part names)
            If return_transforms=True: Tuple of (list of original point clouds, list of registered point clouds, list of part names, list of transforms)
            Returns empty lists if result directory doesn't exist or no valid point clouds found
        """
        sample_name = os.path.basename(input_sample_dir)
        
        # Handle case where result_sample_dir might be the same as input_sample_dir
        # (for datasets like nss_ss_v1 where transforms are alongside PLY files)
        if result_sample_dir == input_sample_dir:
            # Transforms are in the same directory as PLY files
            result_sample_dir = input_sample_dir
        elif not os.path.exists(result_sample_dir):
            # If separate result directory doesn't exist, check if transforms are in input directory
            # This handles cases where transforms are alongside PLY files
            if os.path.exists(input_sample_dir):
                # Check if transform files exist in input directory
                transform_files = glob.glob(os.path.join(input_sample_dir, "*transform.txt"))
                if transform_files:
                    # Transforms are in the same directory as PLY files
                    result_sample_dir = input_sample_dir
                else:
                    logger.warning(f"Result sample directory not found: {result_sample_dir}")
                    logger.warning(f"  Skipping sample {sample_name} - no registered results available")
                    if return_transforms:
                        return [], [], [], []
                    return [], [], []
        else:
            # Separate result directory exists (normal case)
            pass
        
        if not os.path.exists(input_sample_dir):
            logger.warning(f"Input sample directory not found: {input_sample_dir}")
            if return_transforms:
                return [], [], [], []
            return [], [], []
        
        # Load and print metadata
        metadata = self.load_json_metadata(result_sample_dir)
        if metadata:
            self.print_metadata(metadata)
        else:
            logger.debug(f"No metadata JSON file found in {result_sample_dir}")
        
        ply_files = natsort.natsorted(glob.glob(os.path.join(input_sample_dir, "*.ply")))
        if not ply_files:
            logger.warning(f"No PLY files found in {input_sample_dir}")
            if return_transforms:
                return [], [], [], []
            return [], [], []
        
        original_point_clouds = []
        registered_point_clouds = []
        part_names = []
        transforms_list = []  # Store transforms for potential reapplication
        part_colors = self.generate_part_colors(len(ply_files))
        skipped_count = 0
        
        for idx, ply_file in enumerate(ply_files):
            ply_filename = os.path.basename(ply_file)
            part_name = os.path.splitext(ply_filename)[0]
            
            # Try to find transform file by filename pattern first
            transform_file = self.find_matching_transform_file(ply_filename, result_sample_dir, generation=generation)
            
            # If not found by filename pattern, try matching by index (for files without part numbers in name)
            if transform_file is None:
                transform_file = self.find_matching_transform_file_by_index(idx, result_sample_dir, generation=generation)
            
            if transform_file is None:
                logger.debug(f"No transform file found for {ply_filename} (index {idx}), skipping")
                skipped_count += 1
                continue
            
            # Log which type of transform file was found
            if "generation_selected" in transform_file:
                logger.debug(f"Using generation_selected transform file: {os.path.basename(transform_file)}")
            else:
                logger.debug(f"Using generation00 transform file: {os.path.basename(transform_file)}")
            
            transform = self.load_pose_from_file(transform_file)
            if transform is None:
                skipped_count += 1
                continue
            
            try:
                pcd = o3d.io.read_point_cloud(ply_file)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    logger.debug(f"Empty point cloud for {part_name}, skipping")
                    skipped_count += 1
                    continue
                
                # Create original point cloud (before transformation)
                pcd_original = o3d.geometry.PointCloud()
                pcd_original.points = o3d.utility.Vector3dVector(points.copy())
                
                # Copy colors if available (before outlier removal)
                if pcd.has_colors():
                    pcd_original.colors = pcd.colors
                
                # Apply outlier removal if enabled
                pcd_original = self.apply_outlier_removal(pcd_original)
                points = np.asarray(pcd_original.points)  # Update points after outlier removal
                
                if len(points) == 0:
                    logger.debug(f"Point cloud for {part_name} is empty after outlier removal, skipping")
                    skipped_count += 1
                    continue
                
                # Add normals to original if available (after outlier removal)
                normals_orig = None
                if pcd_original.has_normals():
                    normals_orig = np.asarray(pcd_original.normals)
                    if len(normals_orig) == len(points):
                        pcd_original.normals = o3d.utility.Vector3dVector(normals_orig.copy())
                        normals_orig = normals_orig.copy()  # Keep a copy for later use
                    else:
                        normals_orig = None
                
                # Estimate normals for original if requested
                if normals_orig is None and self.estimate_normals:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(points)
                    # Use adaptive radius based on point cloud scale
                    adaptive_radius = self.compute_adaptive_normal_radius(points)
                    pcd_temp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=adaptive_radius,
                            max_nn=30
                        )
                    )
                    pcd_temp.orient_normals_consistent_tangent_plane(k=15)
                    normals_orig = np.asarray(pcd_temp.normals)
                    pcd_original.normals = o3d.utility.Vector3dVector(normals_orig)
                
                # Transform points
                points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
                points_transformed = (transform @ points_homogeneous.T).T[:, :3]
                
                # Create registered point cloud (after transformation)
                pcd_transformed = o3d.geometry.PointCloud()
                pcd_transformed.points = o3d.utility.Vector3dVector(points_transformed)
                
                # Transform normals if available
                normals_transformed = None
                if normals_orig is not None:
                    # Transform normals (only rotation, no translation)
                    normals_transformed = (transform[:3, :3] @ normals_orig.T).T
                    pcd_transformed.normals = o3d.utility.Vector3dVector(normals_transformed)
                
                # Estimate normals for transformed if requested and not already available
                if normals_transformed is None and self.estimate_normals:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(points_transformed)
                    # Use adaptive radius based on point cloud scale
                    adaptive_radius = self.compute_adaptive_normal_radius(points_transformed)
                    pcd_temp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=adaptive_radius,
                            max_nn=30
                        )
                    )
                    pcd_temp.orient_normals_consistent_tangent_plane(k=15)
                    normals_transformed = np.asarray(pcd_temp.normals)
                    pcd_transformed.normals = o3d.utility.Vector3dVector(normals_transformed)
                
                # Set color for both
                color = part_colors[idx % len(part_colors)]
                pcd_original.paint_uniform_color(color)
                pcd_transformed.paint_uniform_color(color)
                
                original_point_clouds.append(pcd_original)
                registered_point_clouds.append(pcd_transformed)
                part_names.append(part_name)
                transforms_list.append(transform)  # Store transform for potential reapplication
            except Exception as e:
                logger.warning(f"Error processing {part_name}: {e}, skipping")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} parts (missing transforms or errors)")
        
        if return_transforms:
            return original_point_clouds, registered_point_clouds, part_names, transforms_list
        else:
            return original_point_clouds, registered_point_clouds, part_names
    
    def load_generated_point_clouds(self, result_sample_dir: str, generation: Optional[str] = None) -> List[o3d.geometry.PointCloud]:
        """
        Load generated point cloud PLY files from the processed directory.
        If generation is specified, uses that generation. Otherwise, prioritizes generation_selected files, 
        then falls back to generation00.
        
        NOTE: Generated point clouds are loaded as-is without any transformation or shift.
        They are already in the center of mass frame and should not be modified.
        
        Args:
            result_sample_dir: Directory containing generated PLY files
            generation: Optional generation name to use (e.g., 'generation00', 'generation_selected')
        
        Returns:
            List of generated point clouds (already in center of mass frame, no transformation applied)
        """
        if not os.path.exists(result_sample_dir):
            return []
        
        generated_pcds = []
        
        # If generation is specified, use it directly
        if generation:
            patterns = [
                f"*{generation}_part*.ply",
                f"*{generation}*part*.ply",
            ]
        else:
            # Priority: generation_selected > generation00 > other generation files
            patterns = [
                "*generation_selected_part*.ply",
                "*generation00_part*.ply",
                "*generation*_part*.ply",
            ]
        
        for pattern in patterns:
            ply_files = sorted(glob.glob(os.path.join(result_sample_dir, pattern)))
            if ply_files:
                logger.debug(f"Found {len(ply_files)} generated PLY files matching pattern: {pattern}")
                
                # Extract part numbers and sort by part number
                ply_files_with_parts = []
                for ply_file in ply_files:
                    basename = os.path.basename(ply_file)
                    # Extract part number from filename (e.g., part00, part01, etc.)
                    part_match = re.search(r'part(\d+)', basename)
                    if part_match:
                        part_num = int(part_match.group(1))
                        ply_files_with_parts.append((part_num, ply_file))
                    else:
                        # If no part number found, use index
                        ply_files_with_parts.append((len(ply_files_with_parts), ply_file))
                
                # Sort by part number
                ply_files_with_parts.sort(key=lambda x: x[0])
                
                # Generate colors using the same scheme as original/registered
                num_parts = len(ply_files_with_parts)
                part_colors = self.generate_part_colors(num_parts)
                
                for idx, (part_num, ply_file) in enumerate(ply_files_with_parts):
                    try:
                        pcd = o3d.io.read_point_cloud(ply_file)
                        points = np.asarray(pcd.points)
                        
                        if len(points) == 0:
                            logger.debug(f"Empty generated point cloud: {os.path.basename(ply_file)}")
                            continue
                        
                        # These are already transformed, so use them as-is
                        # Add normals if available
                        if pcd.has_normals():
                            normals = np.asarray(pcd.normals)
                            if len(normals) == len(points):
                                pcd.normals = o3d.utility.Vector3dVector(normals)
                        
                        # Estimate normals if requested and not available
                        if not pcd.has_normals() and self.estimate_normals:
                            # Use adaptive radius based on point cloud scale
                            adaptive_radius = self.compute_adaptive_normal_radius(points)
                            pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                    radius=adaptive_radius,
                                    max_nn=30
                                )
                            )
                            pcd.orient_normals_consistent_tangent_plane(k=15)
                        
                        # Use the same part-wise color scheme as original/registered
                        # Use the index in the sorted list to get the correct color
                        color = part_colors[idx % len(part_colors)]
                        pcd.paint_uniform_color(color)
                        
                        generated_pcds.append(pcd)
                    except Exception as e:
                        logger.warning(f"Error loading generated point cloud {ply_file}: {e}")
                        continue
                
                # Only use the first pattern that matches
                if generated_pcds:
                    break
        
        return generated_pcds
    
    def load_and_merge_gt_point_clouds(self, result_sample_dir: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Load and merge GT (ground truth) point cloud PLY files from the processed directory.
        
        NOTE: GT point clouds are loaded and merged as-is without any transformation or shift.
        They are already in the center of mass frame and should not be modified.
        
        Returns:
            Merged GT point cloud (already in center of mass frame, no transformation applied), or None if no GT files found
        """
        if not os.path.exists(result_sample_dir):
            return None
        
        # Look for GT files
        gt_pattern = "*gt_part*.ply"
        gt_files = sorted(glob.glob(os.path.join(result_sample_dir, gt_pattern)))
        
        if not gt_files:
            logger.debug(f"No GT PLY files found matching pattern: {gt_pattern}")
            return None
        
        logger.debug(f"Found {len(gt_files)} GT PLY files")
        
        # Extract part numbers and sort by part number
        gt_files_with_parts = []
        for gt_file in gt_files:
            basename = os.path.basename(gt_file)
            # Extract part number from filename (e.g., part00, part01, etc.)
            part_match = re.search(r'part(\d+)', basename)
            if part_match:
                part_num = int(part_match.group(1))
                gt_files_with_parts.append((part_num, gt_file))
            else:
                # If no part number found, use index
                gt_files_with_parts.append((len(gt_files_with_parts), gt_file))
        
        # Sort by part number
        gt_files_with_parts.sort(key=lambda x: x[0])
        
        # Load and merge all GT point clouds
        all_points = []
        all_normals = []
        all_colors = []
        
        # Generate colors using the same scheme as original/registered
        num_parts = len(gt_files_with_parts)
        part_colors = self.generate_part_colors(num_parts)
        
        for idx, (part_num, gt_file) in enumerate(gt_files_with_parts):
            try:
                pcd = o3d.io.read_point_cloud(gt_file)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    logger.debug(f"Empty GT point cloud: {os.path.basename(gt_file)}")
                    continue
                
                # Get normals if available
                normals = None
                if pcd.has_normals():
                    normals_array = np.asarray(pcd.normals)
                    if len(normals_array) == len(points):
                        normals = normals_array
                
                # Estimate normals if requested and not available
                if normals is None and self.estimate_normals:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(points)
                    # Use adaptive radius based on point cloud scale
                    adaptive_radius = self.compute_adaptive_normal_radius(points)
                    pcd_temp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=adaptive_radius,
                            max_nn=30
                        )
                    )
                    pcd_temp.orient_normals_consistent_tangent_plane(k=15)
                    normals = np.asarray(pcd_temp.normals)
                
                # Get colors if available, otherwise use part color
                if pcd.has_colors():
                    colors_array = np.asarray(pcd.colors)
                    if len(colors_array) == len(points):
                        colors = colors_array
                    else:
                        colors = np.tile(part_colors[idx % len(part_colors)], (len(points), 1))
                else:
                    colors = np.tile(part_colors[idx % len(part_colors)], (len(points), 1))
                
                all_points.append(points)
                if normals is not None:
                    all_normals.append(normals)
                all_colors.append(colors)
                
            except Exception as e:
                logger.warning(f"Error loading GT point cloud {gt_file}: {e}")
                continue
        
        if not all_points:
            return None
        
        # Merge all point clouds
        merged_pcd = o3d.geometry.PointCloud()
        merged_points = np.concatenate(all_points, axis=0)
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        if all_normals:
            merged_normals = np.concatenate(all_normals, axis=0)
            merged_pcd.normals = o3d.utility.Vector3dVector(merged_normals)
        
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        
        logger.info(f"Merged {len(gt_files_with_parts)} GT parts into single point cloud with {len(merged_points)} points")
        
        return merged_pcd
    
    def generate_part_colors(self, num_parts: int) -> List[List[float]]:
        """Generate distinct colors for each part."""
        colors = []
        for i in range(num_parts):
            color_idx = i % len(CMAP_DEFAULT)
            colors.append(CMAP_DEFAULT[color_idx])
        return colors
    
    def get_sequence_samples(self, sequence: str) -> List[str]:
        """Get list of sample directory names for a sequence."""
        if sequence in self.sequence_samples_cache:
            return self.sequence_samples_cache[sequence]
        
        input_seq_dir, _ = self.find_sequence_directories(self.input_dir, sequence)
        if input_seq_dir is None:
            return []
        
        sample_dirs = self.find_sample_directories(input_seq_dir)
        sample_names = [os.path.basename(d) for d in sample_dirs]
        
        self.sequence_samples_cache[sequence] = sample_names
        return sample_names
    
    def get_available_sequences(self) -> List[str]:
        """Get list of available sequences."""
        sequences = []
        
        def has_direct_samples(dir_path: str) -> bool:
            """Check if directory directly contains sample directories (not nested)."""
            sample_dirs = self.find_sample_directories(dir_path)
            return len(sample_dirs) > 0
        
        def find_sequences_recursive(root_dir: str, current_path: str = "", max_depth: int = 5) -> List[str]:
            """Recursively find sequences (directories containing samples)."""
            if max_depth <= 0:
                return []
            
            found_sequences = []
            
            try:
                if not os.path.isdir(root_dir):
                    return []
                
                # Check if current directory contains samples
                if has_direct_samples(root_dir):
                    if current_path:
                        found_sequences.append(current_path)
                    else:
                        # Use basename if no path built yet
                        found_sequences.append(os.path.basename(root_dir))
                
                # Recursively search subdirectories
                for item in os.listdir(root_dir):
                    item_path = os.path.join(root_dir, item)
                    if os.path.isdir(item_path):
                        # Skip common non-sequence directories
                        if item in ['.', '..', '__pycache__', '.git', 'data_split']:
                            continue
                        
                        # Build path for nested structure
                        if current_path:
                            new_path = f"{current_path}/{item}"
                        else:
                            new_path = item
                        
                        # Recursively search
                        sub_sequences = find_sequences_recursive(item_path, new_path, max_depth - 1)
                        found_sequences.extend(sub_sequences)
                        
            except PermissionError:
                pass
            except Exception as e:
                logger.debug(f"Error searching {root_dir}: {e}")
            
            return found_sequences
        
        # First, check direct subdirectories for sequences (directly contain samples)
        for item in os.listdir(self.input_dir):
            item_path = os.path.join(self.input_dir, item)
            if os.path.isdir(item_path) and has_direct_samples(item_path):
                sequences.append(item)
        
        # If no sequences found, check one level deeper (for nested structure like scannet_test_v1/scannet_test_v1/)
        if not sequences:
            for item in os.listdir(self.input_dir):
                item_path = os.path.join(self.input_dir, item)
                if os.path.isdir(item_path):
                    # Check subdirectories one level deeper
                    try:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path) and has_direct_samples(subitem_path):
                                # Use nested path as sequence name
                                sequence_name = f"{item}/{subitem}" if item != subitem else subitem
                                if sequence_name not in sequences:
                                    sequences.append(sequence_name)
                    except PermissionError:
                        pass
        
        # If still no sequences found, do recursive search (for deeply nested structures like modelnet/modelnet/category/object_id)
        if not sequences:
            sequences = find_sequences_recursive(self.input_dir)
            # Remove duplicates while preserving order
            seen = set()
            unique_sequences = []
            for seq in sequences:
                if seq not in seen:
                    seen.add(seq)
                    unique_sequences.append(seq)
            sequences = unique_sequences
        
        return sorted(sequences)
    
    def find_next_sample(self) -> Tuple[Optional[str], Optional[int]]:
        """Find the next sample to load."""
        if self.current_sequence is None or self.current_sample_idx is None:
            return None, None
        
        samples = self.get_sequence_samples(self.current_sequence)
        if not samples:
            return None, None
        
        if self.current_sample_idx + 1 < len(samples):
            return self.current_sequence, self.current_sample_idx + 1
        
        # Move to next sequence
        sequences = self.get_available_sequences()
        try:
            current_seq_idx = sequences.index(self.current_sequence)
            if current_seq_idx + 1 < len(sequences):
                next_sequence = sequences[current_seq_idx + 1]
                next_samples = self.get_sequence_samples(next_sequence)
                if next_samples:
                    return next_sequence, 0
            # Loop back to first
            first_sequence = sequences[0]
            first_samples = self.get_sequence_samples(first_sequence)
            if first_samples:
                return first_sequence, 0
        except ValueError:
            pass
        
        return None, None
    
    def find_next_sequence(self) -> Tuple[Optional[str], Optional[int]]:
        """Find the first sample of the next sequence to load."""
        if self.current_sequence is None:
            return None, None
        
        sequences = self.get_available_sequences()
        if not sequences:
            return None, None
        
        try:
            current_seq_idx = sequences.index(self.current_sequence)
            if current_seq_idx + 1 < len(sequences):
                next_sequence = sequences[current_seq_idx + 1]
            else:
                # If we're at the last sequence, loop back to first
                next_sequence = sequences[0]
            
            # Get first sample of next sequence
            next_samples = self.get_sequence_samples(next_sequence)
            if next_samples:
                return next_sequence, 0
        except ValueError:
            pass
        
        return None, None
    
    def reload_current_sample_with_generation(self, generation: str) -> bool:
        """
        Reload the current sample with a specific generation.
        
        Args:
            generation: Generation name to load (e.g., 'generation00', 'generation_selected')
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if self.current_sequence is None or self.current_sample_idx is None:
            return False
        
        try:
            samples = self.get_sequence_samples(self.current_sequence)
            if not samples or self.current_sample_idx >= len(samples):
                return False
            
            sample_name = samples[self.current_sample_idx]
            
            input_seq_dir, result_seq_dir = self.find_sequence_directories(self.input_dir, self.current_sequence)
            if input_seq_dir is None or result_seq_dir is None:
                return False
            
            input_sample_dir = os.path.join(input_seq_dir, sample_name)
            
            # Determine result sample directory name
            if sample_name.startswith('sample_'):
                result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                    result_sample_dir = input_sample_dir
            else:
                result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                if not os.path.exists(result_sample_dir):
                    result_sample_dir = os.path.join(result_seq_dir, sample_name)
                    if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                        result_sample_dir = input_sample_dir
            
            # Load point clouds with the specified generation
            original_pcds, registered_pcds, part_names, transforms_list = self.load_sample_point_clouds(
                input_sample_dir, result_sample_dir, return_transforms=True, generation=generation
            )
            
            if not registered_pcds:
                return False
            
            # Load generated point clouds if enabled
            if hasattr(self, 'show_generated') and self.show_generated:
                self.generated_gt_pcd = self.load_and_merge_gt_point_clouds(result_sample_dir)
                self.generated_pcds = self.load_generated_point_clouds(result_sample_dir, generation=generation)
            else:
                self.generated_pcds = []
                self.generated_gt_pcd = None
            
            # Create random yaw rotated versions
            original_pcds_before_translation = []
            for pcd in original_pcds:
                pcd_copy = o3d.geometry.PointCloud()
                pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
                if pcd.has_normals():
                    pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
                if pcd.has_colors():
                    pcd_copy.colors = pcd.colors
                original_pcds_before_translation.append(pcd_copy)
            
            self.random_yaw_pcds = self.create_random_yaw_point_clouds(
                original_pcds_before_translation,
                generated_pcds=self.generated_pcds if hasattr(self, 'show_generated') and self.show_generated else None
            )
            
            # Compute center of mass and translate
            center_of_mass = None
            merged_input_points = self.merge_input_point_clouds(original_pcds)
            if merged_input_points is not None:
                _, center_of_mass = self.center_pcd(merged_input_points)
            
            if center_of_mass is not None:
                self.original_pcds = [self.translate_point_cloud(pcd, center_of_mass) for pcd in original_pcds]
                
                self.registered_pcds = []
                for idx, (pcd_original, transform) in enumerate(zip(self.original_pcds, transforms_list)):
                    points = np.asarray(pcd_original.points)
                    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
                    points_transformed = (transform @ points_homogeneous.T).T[:, :3]
                    
                    pcd_registered = o3d.geometry.PointCloud()
                    pcd_registered.points = o3d.utility.Vector3dVector(points_transformed)
                    
                    if pcd_original.has_normals():
                        normals_orig = np.asarray(pcd_original.normals)
                        normals_transformed = (transform[:3, :3] @ normals_orig.T).T
                        pcd_registered.normals = o3d.utility.Vector3dVector(normals_transformed)
                    
                    if pcd_original.has_colors():
                        pcd_registered.colors = pcd_original.colors
                    else:
                        color = self.generate_part_colors(len(self.original_pcds))[idx % len(self.original_pcds)]
                        pcd_registered.paint_uniform_color(color)
                    
                    self.registered_pcds.append(pcd_registered)
            else:
                self.original_pcds = original_pcds
                self.registered_pcds = registered_pcds
            
            # Select point clouds based on view mode
            if self.view_mode == "generated" and self.generated_pcds:
                point_clouds = self.generated_pcds.copy()
            elif self.view_mode == "gt" and self.generated_gt_pcd is not None:
                point_clouds = [self.generated_gt_pcd]
            elif self.view_mode == "random_yaw" and self.random_yaw_pcds:
                point_clouds = self.random_yaw_pcds.copy()
            elif self.view_mode == "original":
                point_clouds = self.original_pcds
            else:
                point_clouds = self.registered_pcds
            
            # Update visualization
            if self.vis is not None:
                self.vis.clear_geometries()
                
                for pcd in point_clouds:
                    self.vis.add_geometry(pcd, False)
                    self.vis.update_geometry(pcd)
                
                if not self.no_coordinate_frame:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                    self.vis.add_geometry(coordinate_frame, False)
                    self.vis.update_geometry(coordinate_frame)
                
                self._center_and_fit_view(point_clouds)
                self.vis.update_renderer()
            
            self.colored_pcds = point_clouds
            
            logger.info(f"Reloaded sample with generation: {generation}")
            return True
            
        except Exception as e:
            logger.warning(f"Error reloading sample with generation {generation}: {e}")
            return False
    
    def cycle_to_next_generation(self) -> bool:
        """
        Cycle to the next available generation for the current sample.
        If we've cycled through all generations, moves to the next sample.
        
        Returns:
            True if successfully cycled, False otherwise
        """
        if not self.cycle_generations or not self.available_generations:
            return False
        
        # Check if we're at the last generation
        if self.current_generation_idx + 1 >= len(self.available_generations):
            # Move to next sample instead
            logger.info("Cycled through all generations, moving to next sample...")
            return self.load_next_sample()
        
        # Move to next generation
        self.current_generation_idx += 1
        generation = self.available_generations[self.current_generation_idx]
        
        logger.info(f"Cycling to generation {self.current_generation_idx + 1}/{len(self.available_generations)}: {generation}")
        
        return self.reload_current_sample_with_generation(generation)
    
    def load_next_sample(self, max_attempts: int = 10):
        """Load and visualize the next sample."""
        if max_attempts <= 0:
            logger.error("Maximum attempts reached. Could not find a valid sample to load.")
            return False
        
        next_sequence, next_sample_idx = self.find_next_sample()
        
        if next_sequence is None or next_sample_idx is None:
            logger.warning("No next sample available")
            return False
        
        try:
            self.current_sequence = next_sequence
            self.current_sample_idx = next_sample_idx
            
            samples = self.get_sequence_samples(next_sequence)
            sample_name = samples[next_sample_idx]
            
            input_seq_dir, result_seq_dir = self.find_sequence_directories(self.input_dir, next_sequence)
            if input_seq_dir is None or result_seq_dir is None:
                logger.warning(f"Could not find sequence directories for {next_sequence}")
                # Try next sample
                return self.load_next_sample(max_attempts - 1)
            
            input_sample_dir = os.path.join(input_seq_dir, sample_name)
            
            # Determine result sample directory name
            # For sample_* format: use sample_*_processed
            # For other formats (like fracture_*): try both with and without _processed suffix
            # Also handle case where transforms are in the same directory as PLY files
            if sample_name.startswith('sample_'):
                result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                # If _processed doesn't exist, check if transforms are in the same directory
                if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                    result_sample_dir = input_sample_dir
            else:
                # Try with _processed suffix first, then without
                result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                if not os.path.exists(result_sample_dir):
                    result_sample_dir = os.path.join(result_seq_dir, sample_name)
                    # If still doesn't exist and result_seq_dir == input_seq_dir, use input_sample_dir
                    if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                        result_sample_dir = input_sample_dir
            
            # Initialize available generations if cycling is enabled
            if self.cycle_generations:
                self.available_generations = self.get_available_generations(result_sample_dir)
                if self.available_generations:
                    self.current_generation_idx = 0
                    logger.info(f"Found {len(self.available_generations)} available generation(s): {self.available_generations}")
                else:
                    logger.warning(f"No generations found for sample {sample_name}, disabling generation cycling")
                    self.cycle_generations = False
            
            # Load point clouds (both original and registered) and get transforms
            # Use current generation if cycling is enabled
            current_generation = None
            if self.cycle_generations and self.available_generations:
                current_generation = self.available_generations[self.current_generation_idx]
            
            original_pcds, registered_pcds, part_names, transforms_list = self.load_sample_point_clouds(
                input_sample_dir, result_sample_dir, return_transforms=True, generation=current_generation
            )
            
            if not registered_pcds:
                logger.warning(f"No point clouds loaded for sample {sample_name}")
                # Try to load next sample automatically
                logger.info("Attempting to load next available sample...")
                return self.load_next_sample(max_attempts - 1)  # Recursively try next sample with limit
            
            # Load generated point clouds if enabled
            # NOTE: Generated and GT point clouds are loaded as-is without any transformation
            # They are already in center of mass frame and will be used directly for visualization
            if hasattr(self, 'show_generated') and self.show_generated:
                # Load GT point cloud (for visualization only, not for center of mass calculation)
                self.generated_gt_pcd = self.load_and_merge_gt_point_clouds(result_sample_dir)
                if self.generated_gt_pcd is not None:
                    logger.info(f"Loaded GT point cloud (for visualization only, NOT used for center of mass)")
                
                # Load generated point clouds (no transformation applied)
                self.generated_pcds = self.load_generated_point_clouds(result_sample_dir, generation=current_generation)
                if self.generated_pcds:
                    logger.info(f"Loaded {len(self.generated_pcds)} generated point cloud(s) (no transformation applied)")
            else:
                self.generated_pcds = []
                self.generated_gt_pcd = None
            
            # Create random yaw rotated versions of input point clouds
            # Clone original point clouds BEFORE translation for random yaw view
            # (These are the untranslated original point clouds from load_sample_point_clouds)
            original_pcds_before_translation = []
            for pcd in original_pcds:
                # Create a deep copy
                pcd_copy = o3d.geometry.PointCloud()
                pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
                if pcd.has_normals():
                    pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
                if pcd.has_colors():
                    pcd_copy.colors = pcd.colors
                original_pcds_before_translation.append(pcd_copy)
            
            # Create random yaw rotated point clouds (centered at origin, random yaw per part)
            # Skip rotation for the part corresponding to the generated part with most points
            self.random_yaw_pcds = self.create_random_yaw_point_clouds(
                original_pcds_before_translation, 
                generated_pcds=self.generated_pcds if hasattr(self, 'show_generated') and self.show_generated else None
            )
            
            # Always use merged input point clouds for center of mass computation
            center_of_mass = None
            merged_input_points = self.merge_input_point_clouds(original_pcds)
            if merged_input_points is not None:
                _, center_of_mass = self.center_pcd(merged_input_points)
                logger.info(f"Using merged input point clouds center of mass: {center_of_mass}")
            
            # Translate original point clouds to center of mass frame first
            # Then re-apply transforms to get registered point clouds in center of mass frame
            # Generated and GT are already in center of mass frame, so don't translate them
            if center_of_mass is not None:
                # Translate original point clouds to center of mass frame
                self.original_pcds = [self.translate_point_cloud(pcd, center_of_mass) for pcd in original_pcds]
                
                # Re-apply transforms to translated original point clouds to get registered in center of mass frame
                self.registered_pcds = []
                for idx, (pcd_original, transform) in enumerate(zip(self.original_pcds, transforms_list)):
                    # Apply transform to the translated original point cloud
                    points = np.asarray(pcd_original.points)
                    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
                    points_transformed = (transform @ points_homogeneous.T).T[:, :3]
                    
                    pcd_registered = o3d.geometry.PointCloud()
                    pcd_registered.points = o3d.utility.Vector3dVector(points_transformed)
                    
                    # Transform normals if available
                    if pcd_original.has_normals():
                        normals_orig = np.asarray(pcd_original.normals)
                        normals_transformed = (transform[:3, :3] @ normals_orig.T).T
                        pcd_registered.normals = o3d.utility.Vector3dVector(normals_transformed)
                    
                    # Copy colors
                    if pcd_original.has_colors():
                        pcd_registered.colors = pcd_original.colors
                    else:
                        # Use same color as original
                        color = self.generate_part_colors(len(self.original_pcds))[idx % len(self.original_pcds)]
                        pcd_registered.paint_uniform_color(color)
                    
                    self.registered_pcds.append(pcd_registered)
            else:
                # If no center available, keep original registered point clouds
                self.original_pcds = original_pcds
                self.registered_pcds = registered_pcds
            
            # Select point clouds based on view mode (use translated versions)
            if self.view_mode == "generated" and self.generated_pcds:
                point_clouds = self.generated_pcds.copy()  # Only generated, no GT
            elif self.view_mode == "gt" and self.generated_gt_pcd is not None:
                point_clouds = [self.generated_gt_pcd]  # Only GT
            elif self.view_mode == "random_yaw" and self.random_yaw_pcds:
                point_clouds = self.random_yaw_pcds.copy()  # Random yaw rotated input point clouds
            elif self.view_mode == "original":
                point_clouds = self.original_pcds  # Use translated original point clouds
            else:
                point_clouds = self.registered_pcds  # Use translated registered point clouds
            
            # Clear current visualization
            if self.vis is not None:
                self.vis.clear_geometries()
            
            # Add new geometries
            if self.vis is not None:
                for pcd in point_clouds:
                    self.vis.add_geometry(pcd, False)
                    self.vis.update_geometry(pcd)
                
                if not self.no_coordinate_frame:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                    self.vis.add_geometry(coordinate_frame, False)
                    self.vis.update_geometry(coordinate_frame)
                
                self._center_and_fit_view(point_clouds)
                self.vis.update_renderer()
            
            self.colored_pcds = point_clouds
            
            logger.info("=" * 60)
            logger.info(f"LOADED NEXT SAMPLE: Sequence {next_sequence}, Sample {sample_name}")
            logger.info(f"  Parts: {len(part_names)}")
            view_name_map = {
                "original": "Original (untransformed)",
                "registered": "Registered (transformed)",
                "generated": "Generated (from processed directory)",
                "gt": "GT (ground truth)",
                "random_yaw": "Input with Random Yaw Rotation"
            }
            logger.info(f"  View: {view_name_map.get(self.view_mode, 'Unknown')}")
            logger.info("=" * 60)
            return True
            
        except FileNotFoundError as e:
            logger.warning(f"File not found: {e}")
            # Try next sample
            return self.load_next_sample(max_attempts - 1)
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            # Try next sample
            return self.load_next_sample(max_attempts - 1)
        except Exception as e:
            logger.warning(f"Error loading sample: {e}")
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            # Try next sample
            return self.load_next_sample(max_attempts - 1)
    
    def load_next_sequence(self, max_attempts: int = 50):
        """
        Load and visualize the first sample of the next sequence.
        Tries multiple samples in the target sequence before moving to the next sequence.
        """
        if max_attempts <= 0:
            logger.error("Maximum attempts reached. Could not find a valid sequence to load.")
            return False
        
        next_sequence, next_sample_idx = self.find_next_sequence()
        
        if next_sequence is None or next_sample_idx is None:
            logger.warning("No next sequence available")
            return False
        
        # Try multiple samples in the target sequence before giving up
        samples = self.get_sequence_samples(next_sequence)
        if not samples:
            logger.warning(f"No samples found in sequence {next_sequence}, trying next sequence...")
            # Update current sequence to skip this empty sequence
            self.current_sequence = next_sequence
            self.current_sample_idx = 0
            return self.load_next_sequence(max_attempts - 1)
        
        # Try up to all samples in the target sequence (or max_attempts, whichever is smaller)
        max_samples_to_try = min(len(samples), max_attempts)
        
        for attempt in range(max_samples_to_try):
            sample_idx = (next_sample_idx + attempt) % len(samples)
            sample_name = samples[sample_idx]
            
            try:
                input_seq_dir, result_seq_dir = self.find_sequence_directories(self.input_dir, next_sequence)
                if input_seq_dir is None or result_seq_dir is None:
                    logger.warning(f"Could not find sequence directories for {next_sequence}")
                    # Try next sequence
                    self.current_sequence = next_sequence
                    self.current_sample_idx = sample_idx
                    return self.load_next_sequence(max_attempts - max_samples_to_try)
                
                input_sample_dir = os.path.join(input_seq_dir, sample_name)
                
                # Determine result sample directory name
                # For sample_* format: use sample_*_processed
                # For other formats (like fracture_*): try both with and without _processed suffix
                if sample_name.startswith('sample_'):
                    result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                else:
                    # Try with _processed suffix first, then without
                    result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
                    if not os.path.exists(result_sample_dir):
                        result_sample_dir = os.path.join(result_seq_dir, sample_name)
                
                # Check if result directory exists before trying to load
                if not os.path.exists(result_sample_dir):
                    logger.debug(f"Result sample directory not found: {result_sample_dir}, trying next sample...")
                    continue  # Try next sample in this sequence
                
                # Initialize available generations if cycling is enabled
                if self.cycle_generations:
                    self.available_generations = self.get_available_generations(result_sample_dir)
                    if self.available_generations:
                        self.current_generation_idx = 0
                        logger.info(f"Found {len(self.available_generations)} available generation(s): {self.available_generations}")
                    else:
                        logger.warning(f"No generations found for sample {sample_name}, disabling generation cycling")
                        self.cycle_generations = False
                
                # Load point clouds (both original and registered) and get transforms
                # Use current generation if cycling is enabled
                current_generation = None
                if self.cycle_generations and self.available_generations:
                    current_generation = self.available_generations[self.current_generation_idx]
                
                original_pcds, registered_pcds, part_names, transforms_list = self.load_sample_point_clouds(
                    input_sample_dir, result_sample_dir, return_transforms=True, generation=current_generation
                )
                
                if not registered_pcds:
                    logger.debug(f"No point clouds loaded for sample {sample_name}, trying next sample...")
                    continue  # Try next sample in this sequence
                
                # Successfully loaded point clouds - update state and proceed with visualization
                self.current_sequence = next_sequence
                self.current_sample_idx = sample_idx
                
                # Load generated point clouds if enabled
                # NOTE: Generated and GT point clouds are loaded as-is without any transformation
                # They are already in center of mass frame and will be used directly for visualization
                if hasattr(self, 'show_generated') and self.show_generated:
                    # Load GT point cloud (for visualization only, not for center of mass calculation)
                    self.generated_gt_pcd = self.load_and_merge_gt_point_clouds(result_sample_dir)
                    if self.generated_gt_pcd is not None:
                        logger.info(f"Loaded GT point cloud (for visualization only, NOT used for center of mass)")
                    
                    # Load generated point clouds (no transformation applied)
                    self.generated_pcds = self.load_generated_point_clouds(result_sample_dir, generation=current_generation)
                    if self.generated_pcds:
                        logger.info(f"Loaded {len(self.generated_pcds)} generated point cloud(s) (no transformation applied)")
                else:
                    self.generated_pcds = []
                    self.generated_gt_pcd = None
                
                # Always use merged input point clouds for center of mass computation
                center_of_mass = None
                merged_input_points = self.merge_input_point_clouds(original_pcds)
                if merged_input_points is not None:
                    _, center_of_mass = self.center_pcd(merged_input_points)
                    logger.info(f"Using merged input point clouds center of mass: {center_of_mass}")
                
                # Translate original point clouds to center of mass frame first
                # Then re-apply transforms to get registered point clouds in center of mass frame
                # Generated and GT are already in center of mass frame, so don't translate them
                if center_of_mass is not None:
                    # Translate original point clouds to center of mass frame
                    self.original_pcds = [self.translate_point_cloud(pcd, center_of_mass) for pcd in original_pcds]
                    
                    # Re-apply transforms to translated original point clouds to get registered in center of mass frame
                    self.registered_pcds = []
                    for idx, (pcd_original, transform) in enumerate(zip(self.original_pcds, transforms_list)):
                        # Apply transform to the translated original point cloud
                        points = np.asarray(pcd_original.points)
                        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
                        points_transformed = (transform @ points_homogeneous.T).T[:, :3]
                        
                        pcd_registered = o3d.geometry.PointCloud()
                        pcd_registered.points = o3d.utility.Vector3dVector(points_transformed)
                        
                        # Transform normals if available
                        if pcd_original.has_normals():
                            normals_orig = np.asarray(pcd_original.normals)
                            normals_transformed = (transform[:3, :3] @ normals_orig.T).T
                            pcd_registered.normals = o3d.utility.Vector3dVector(normals_transformed)
                        
                        # Copy colors
                        if pcd_original.has_colors():
                            pcd_registered.colors = pcd_original.colors
                        else:
                            # Use same color as original
                            color = self.generate_part_colors(len(self.original_pcds))[idx % len(self.original_pcds)]
                            pcd_registered.paint_uniform_color(color)
                        
                        self.registered_pcds.append(pcd_registered)
                else:
                    # If no center available, keep original registered point clouds
                    self.original_pcds = original_pcds
                    self.registered_pcds = registered_pcds
                
                # Create random yaw rotated versions of input point clouds
                # Clone original point clouds before translation for random yaw view
                original_pcds_before_translation = []
                for pcd in original_pcds:
                    # Create a deep copy
                    pcd_copy = o3d.geometry.PointCloud()
                    pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
                    if pcd.has_normals():
                        pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
                    if pcd.has_colors():
                        pcd_copy.colors = pcd.colors
                    original_pcds_before_translation.append(pcd_copy)
                
                # Create random yaw rotated point clouds (centered at origin, random yaw per part)
                # Skip rotation for the part corresponding to the generated part with most points
                self.random_yaw_pcds = self.create_random_yaw_point_clouds(
                    original_pcds_before_translation,
                    generated_pcds=self.generated_pcds if hasattr(self, 'show_generated') and self.show_generated else None
                )
                
                # Select point clouds based on view mode (use translated versions)
                if self.view_mode == "generated" and self.generated_pcds:
                    point_clouds = self.generated_pcds.copy()  # Only generated, no GT
                elif self.view_mode == "gt" and self.generated_gt_pcd is not None:
                    point_clouds = [self.generated_gt_pcd]  # Only GT
                elif self.view_mode == "random_yaw" and self.random_yaw_pcds:
                    point_clouds = self.random_yaw_pcds.copy()  # Random yaw rotated input point clouds
                elif self.view_mode == "original":
                    point_clouds = self.original_pcds  # Use translated original point clouds
                else:
                    point_clouds = self.registered_pcds  # Use translated registered point clouds
                
                # Clear current visualization
                if self.vis is not None:
                    self.vis.clear_geometries()
                
                # Add new geometries
                if self.vis is not None:
                    for pcd in point_clouds:
                        self.vis.add_geometry(pcd, False)
                        self.vis.update_geometry(pcd)
                    
                    if not self.no_coordinate_frame:
                        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                        self.vis.add_geometry(coordinate_frame, False)
                        self.vis.update_geometry(coordinate_frame)
                    
                    self._center_and_fit_view(point_clouds)
                    self.vis.update_renderer()
                
                self.colored_pcds = point_clouds
                
                logger.info("=" * 60)
                logger.info(f"JUMPED TO NEXT SEQUENCE: Sequence {next_sequence}, Sample {sample_name}")
                logger.info(f"  Parts: {len(part_names)}")
                view_name_map = {
                    "original": "Original (untransformed)",
                    "registered": "Registered (transformed)",
                    "generated": "Generated (from processed directory)",
                    "gt": "GT (ground truth)"
                }
                logger.info(f"  View: {view_name_map.get(self.view_mode, 'Unknown')}")
                logger.info("=" * 60)
                return True
                
            except Exception as e:
                logger.debug(f"Error loading sample {sample_name}: {e}")
                import traceback
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                continue  # Try next sample in this sequence
        
        # If we've exhausted all samples in this sequence, try the next sequence
        logger.info(f"Tried {max_samples_to_try} sample(s) in sequence {next_sequence} but none had valid results. Moving to next sequence...")
        self.current_sequence = next_sequence
        self.current_sample_idx = len(samples) - 1  # Mark as having tried all samples
        return self.load_next_sequence(max_attempts - max_samples_to_try)
    
    def toggle_transform_view(self):
        """Cycle between original, registered, generated, GT, and random_yaw point cloud views."""
        # Cycle through: original -> registered -> generated -> gt -> random_yaw -> original
        if self.view_mode == "original":
            if self.registered_pcds:
                self.view_mode = "registered"
                point_clouds = self.registered_pcds
                view_name = "Registered (transformed)"
            elif self.generated_pcds:
                self.view_mode = "generated"
                point_clouds = self.generated_pcds.copy()
                view_name = "Generated (from processed directory)"
            elif self.generated_gt_pcd is not None:
                self.view_mode = "gt"
                point_clouds = [self.generated_gt_pcd]
                view_name = "GT (ground truth)"
            elif self.random_yaw_pcds:
                self.view_mode = "random_yaw"
                point_clouds = self.random_yaw_pcds.copy()
                view_name = "Input with Random Yaw Rotation"
            else:
                logger.warning("No other views available")
                return
        elif self.view_mode == "registered":
            if self.generated_pcds:
                self.view_mode = "generated"
                point_clouds = self.generated_pcds.copy()
                view_name = "Generated (from processed directory)"
            elif self.generated_gt_pcd is not None:
                self.view_mode = "gt"
                point_clouds = [self.generated_gt_pcd]
                view_name = "GT (ground truth)"
            elif self.random_yaw_pcds:
                self.view_mode = "random_yaw"
                point_clouds = self.random_yaw_pcds.copy()
                view_name = "Input with Random Yaw Rotation"
            elif self.original_pcds:
                self.view_mode = "original"
                point_clouds = self.original_pcds
                view_name = "Original (untransformed)"
            else:
                logger.warning("No other views available")
                return
        elif self.view_mode == "generated":
            if self.generated_gt_pcd is not None:
                self.view_mode = "gt"
                point_clouds = [self.generated_gt_pcd]
                view_name = "GT (ground truth)"
            elif self.random_yaw_pcds:
                self.view_mode = "random_yaw"
                point_clouds = self.random_yaw_pcds.copy()
                view_name = "Input with Random Yaw Rotation"
            elif self.original_pcds:
                self.view_mode = "original"
                point_clouds = self.original_pcds
                view_name = "Original (untransformed)"
            elif self.registered_pcds:
                self.view_mode = "registered"
                point_clouds = self.registered_pcds
                view_name = "Registered (transformed)"
            else:
                logger.warning("No other views available")
                return
        elif self.view_mode == "gt":
            if self.random_yaw_pcds:
                self.view_mode = "random_yaw"
                point_clouds = self.random_yaw_pcds.copy()
                view_name = "Input with Random Yaw Rotation"
            elif self.original_pcds:
                self.view_mode = "original"
                point_clouds = self.original_pcds
                view_name = "Original (untransformed)"
            elif self.registered_pcds:
                self.view_mode = "registered"
                point_clouds = self.registered_pcds
                view_name = "Registered (transformed)"
            elif self.generated_pcds:
                self.view_mode = "generated"
                point_clouds = self.generated_pcds.copy()
                view_name = "Generated (from processed directory)"
            else:
                logger.warning("No other views available")
                return
        else:  # random_yaw
            if self.original_pcds:
                self.view_mode = "original"
                point_clouds = self.original_pcds
                view_name = "Original (untransformed)"
            elif self.registered_pcds:
                self.view_mode = "registered"
                point_clouds = self.registered_pcds
                view_name = "Registered (transformed)"
            elif self.generated_pcds:
                self.view_mode = "generated"
                point_clouds = self.generated_pcds.copy()
                view_name = "Generated (from processed directory)"
            elif self.generated_gt_pcd is not None:
                self.view_mode = "gt"
                point_clouds = [self.generated_gt_pcd]
                view_name = "GT (ground truth)"
            else:
                logger.warning("No other views available")
                return
        
        logger.info(f"Switched to {view_name} view")
        
        # Update visualization if visualizer exists
        if self.vis is not None:
            # Clear all geometries
            self.vis.clear_geometries()
            
            # Add new point clouds
            for pcd in point_clouds:
                self.vis.add_geometry(pcd, False)
                self.vis.update_geometry(pcd)
            
            # Add coordinate frame if enabled
            if not self.no_coordinate_frame:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                self.vis.add_geometry(coordinate_frame, False)
                self.vis.update_geometry(coordinate_frame)
            
            # Don't reset view - preserve current camera position
            self.vis.update_renderer()
        
        self.colored_pcds = point_clouds
    
    def toggle_background_color(self):
        """Toggle between white and black background colors."""
        self.use_white_background = not self.use_white_background
        
        if self.use_white_background:
            new_background = self.white_background
            logger.info("Switched to white background")
        else:
            new_background = self.black_background
            logger.info("Switched to black background")
        
        self.background_color = new_background
        
        if self.vis is not None:
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array(new_background)
    
    def _center_and_fit_view(self, point_clouds: List[o3d.geometry.PointCloud]):
        """Center and fit the view to show all point clouds properly."""
        if not point_clouds:
            return
        
        view_control = self.vis.get_view_control()
        
        all_points = []
        for pcd in point_clouds:
            points = np.asarray(pcd.points)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return
        
        all_points = np.concatenate(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        
        front = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
        lookat = center
        
        view_control.set_front(front)
        view_control.set_lookat(lookat)
        view_control.set_up(up)
        view_control.set_zoom(0.8)
    
    def visualize_sample(self, sequence: str, sample_idx: int = 0):
        """Visualize a specific sample."""
        self.current_sequence = sequence
        self.current_sample_idx = sample_idx
        
        samples = self.get_sequence_samples(sequence)
        if not samples:
            raise ValueError(f"No samples found for sequence {sequence}")
        
        if sample_idx >= len(samples):
            raise ValueError(f"Sample index {sample_idx} out of range for sequence {sequence}")
        
        sample_name = samples[sample_idx]
        
        input_seq_dir, result_seq_dir = self.find_sequence_directories(self.input_dir, sequence)
        if input_seq_dir is None or result_seq_dir is None:
            raise ValueError(f"Could not find sequence directories for {sequence}")
        
        input_sample_dir = os.path.join(input_seq_dir, sample_name)
        
        # Determine result sample directory name
        # For sample_* format: use sample_*_processed
        # For other formats (like fracture_*): try both with and without _processed suffix
        # Also handle case where transforms are in the same directory as PLY files
        if sample_name.startswith('sample_'):
            result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
            # If _processed doesn't exist, check if transforms are in the same directory
            if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                result_sample_dir = input_sample_dir
        else:
            # Try with _processed suffix first, then without
            result_sample_dir = os.path.join(result_seq_dir, f"{sample_name}_processed")
            if not os.path.exists(result_sample_dir):
                result_sample_dir = os.path.join(result_seq_dir, sample_name)
                # If still doesn't exist and result_seq_dir == input_seq_dir, use input_sample_dir
                if not os.path.exists(result_sample_dir) and result_seq_dir == input_seq_dir:
                    result_sample_dir = input_sample_dir
        
        # Initialize available generations if cycling is enabled
        if self.cycle_generations:
            self.available_generations = self.get_available_generations(result_sample_dir)
            if self.available_generations:
                self.current_generation_idx = 0
                logger.info(f"Found {len(self.available_generations)} available generation(s): {self.available_generations}")
            else:
                logger.warning(f"No generations found for sample {sample_name}, disabling generation cycling")
                self.cycle_generations = False
        
        # Load point clouds (both original and registered) and get transforms
        # Use current generation if cycling is enabled
        current_generation = None
        if self.cycle_generations and self.available_generations:
            current_generation = self.available_generations[self.current_generation_idx]
        
        original_pcds, registered_pcds, part_names, transforms_list = self.load_sample_point_clouds(
            input_sample_dir, result_sample_dir, return_transforms=True, generation=current_generation
        )
        
        if not registered_pcds:
            logger.warning(f"No point clouds loaded for sample {sample_name}")
            logger.warning("This may be because:")
            logger.warning("  - Result sample directory doesn't exist")
            logger.warning("  - No transform files found")
            logger.warning("  - All PLY files failed to load")
            logger.info("Attempting to find next available sample...")
            # Try to find and load next sample
            next_sequence, next_sample_idx = self.find_next_sample()
            if next_sequence and next_sample_idx is not None:
                logger.info(f"Found next sample: {next_sequence}, index {next_sample_idx}")
                return self.visualize_sample(next_sequence, next_sample_idx)
            else:
                raise ValueError(f"No valid samples found. Could not load sample {sample_name} and no next sample available.")
        
        # Load generated point clouds if enabled
        # NOTE: Generated and GT point clouds are loaded as-is without any transformation
        # They are already in center of mass frame and will be used directly for visualization
        if hasattr(self, 'show_generated') and self.show_generated:
            # Load GT point cloud (for visualization only, not for center of mass calculation)
            self.generated_gt_pcd = self.load_and_merge_gt_point_clouds(result_sample_dir)
            if self.generated_gt_pcd is not None:
                logger.info(f"Loaded GT point cloud (for visualization only, NOT used for center of mass)")
            
            # Load generated point clouds (no transformation applied)
            self.generated_pcds = self.load_generated_point_clouds(result_sample_dir, generation=current_generation)
            if self.generated_pcds:
                logger.info(f"Loaded {len(self.generated_pcds)} generated point cloud(s) (no transformation applied)")
        else:
            self.generated_pcds = []
            self.generated_gt_pcd = None
        
        # Create random yaw rotated versions of input point clouds
        # Clone original point clouds BEFORE translation for random yaw view
        # (These are the untranslated original point clouds from load_sample_point_clouds)
        original_pcds_before_translation = []
        for pcd in original_pcds:
            # Create a deep copy
            pcd_copy = o3d.geometry.PointCloud()
            pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
            if pcd.has_normals():
                pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
            if pcd.has_colors():
                pcd_copy.colors = pcd.colors
            original_pcds_before_translation.append(pcd_copy)
        
        # Create random yaw rotated point clouds (centered at origin, random yaw per part)
        # Skip rotation for the part corresponding to the generated part with most points
        self.random_yaw_pcds = self.create_random_yaw_point_clouds(
            original_pcds_before_translation,
            generated_pcds=self.generated_pcds if hasattr(self, 'show_generated') and self.show_generated else None
        )
        
        # Always use merged input point clouds for center of mass computation
        center_of_mass = None
        merged_input_points = self.merge_input_point_clouds(original_pcds)
        if merged_input_points is not None:
            _, center_of_mass = self.center_pcd(merged_input_points)
            logger.info(f"Using merged input point clouds center of mass: {center_of_mass}")
        
        # Translate original point clouds to center of mass frame first
        # Then re-apply transforms to get registered point clouds in center of mass frame
        # Generated and GT are already in center of mass frame, so don't translate them
        if center_of_mass is not None:
            # Translate original point clouds to center of mass frame
            self.original_pcds = [self.translate_point_cloud(pcd, center_of_mass) for pcd in original_pcds]
            
            # Re-apply transforms to translated original point clouds to get registered in center of mass frame
            self.registered_pcds = []
            for idx, (pcd_original, transform) in enumerate(zip(self.original_pcds, transforms_list)):
                # Apply transform to the translated original point cloud
                points = np.asarray(pcd_original.points)
                points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
                points_transformed = (transform @ points_homogeneous.T).T[:, :3]
                
                pcd_registered = o3d.geometry.PointCloud()
                pcd_registered.points = o3d.utility.Vector3dVector(points_transformed)
                
                # Transform normals if available
                if pcd_original.has_normals():
                    normals_orig = np.asarray(pcd_original.normals)
                    normals_transformed = (transform[:3, :3] @ normals_orig.T).T
                    pcd_registered.normals = o3d.utility.Vector3dVector(normals_transformed)
                
                # Copy colors
                if pcd_original.has_colors():
                    pcd_registered.colors = pcd_original.colors
                else:
                    # Use same color as original
                    color = self.generate_part_colors(len(self.original_pcds))[idx % len(self.original_pcds)]
                    pcd_registered.paint_uniform_color(color)
                
                self.registered_pcds.append(pcd_registered)
        else:
            # If no center available, keep original registered point clouds
            self.original_pcds = original_pcds
            self.registered_pcds = registered_pcds
        
        # Create random yaw rotated versions of input point clouds
        # Clone original point clouds before translation for random yaw view
        original_pcds_before_translation = []
        for pcd in original_pcds:
            # Create a deep copy
            pcd_copy = o3d.geometry.PointCloud()
            pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
            if pcd.has_normals():
                pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
            if pcd.has_colors():
                pcd_copy.colors = pcd.colors
            original_pcds_before_translation.append(pcd_copy)
        
        # Create random yaw rotated point clouds (centered at origin, random yaw per part)
        # Skip rotation for the part corresponding to the generated part with most points
        self.random_yaw_pcds = self.create_random_yaw_point_clouds(
            original_pcds_before_translation,
            generated_pcds=self.generated_pcds if hasattr(self, 'show_generated') and self.show_generated else None
        )
        
        # Select point clouds based on view mode (use translated versions)
        if self.view_mode == "generated" and self.generated_pcds:
            point_clouds = self.generated_pcds.copy()  # Only generated, no GT
        elif self.view_mode == "gt" and self.generated_gt_pcd is not None:
            point_clouds = [self.generated_gt_pcd]  # Only GT
        elif self.view_mode == "random_yaw" and self.random_yaw_pcds:
            point_clouds = self.random_yaw_pcds.copy()  # Random yaw rotated input point clouds
        elif self.view_mode == "original":
            point_clouds = self.original_pcds  # Use translated original point clouds
        else:
            point_clouds = self.registered_pcds  # Use translated registered point clouds
        
        logger.info("=" * 60)
        logger.info(f"SAMPLE POINT COUNT SUMMARY:")
        logger.info(f"  Sample: {sample_name}")
        logger.info(f"  Number of Parts: {len(point_clouds)}")
        total_points = sum(len(np.asarray(pcd.points)) for pcd in point_clouds)
        logger.info(f"  TOTAL POINTS: {total_points:6d}")
        logger.info("=" * 60)
        
        # Create visualizer
        try:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            has_key_callback = True
        except AttributeError:
            self.vis = o3d.visualization.Visualizer()
            has_key_callback = False
        
        self.vis.create_window(window_name="Registered Point Cloud Visualizer")
        
        # Add point clouds
        for pcd in point_clouds:
            self.vis.add_geometry(pcd)
        
        # Add coordinate frame
        if not self.no_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.vis.add_geometry(coordinate_frame)
        
        # Set render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.point_size = self.point_size
        
        # Center and fit view
        self._center_and_fit_view(point_clouds)
        
        self.colored_pcds = point_clouds
        
        # Register keyboard callbacks
        if has_key_callback:
            try:
                def background_callback(vis):
                    self.toggle_background_color()
                    return False
                
                def next_sample_callback(vis):
                    # If generation cycling is enabled, cycle through generations instead of samples
                    if self.cycle_generations and self.available_generations:
                        success = self.cycle_to_next_generation()
                        if success:
                            logger.info(f"Spacebar pressed: Cycled to generation {self.available_generations[self.current_generation_idx]}")
                        else:
                            logger.warning("Spacebar pressed: Failed to cycle to next generation")
                    else:
                        success = self.load_next_sample()
                        if success:
                            logger.info("Spacebar pressed: Loaded next sample")
                        else:
                            logger.warning("Spacebar pressed: Failed to load next sample")
                    return False
                
                def toggle_transform_callback(vis):
                    self.toggle_transform_view()
                    return False
                
                def next_sequence_callback(vis):
                    success = self.load_next_sequence()
                    if success:
                        logger.info("N pressed: Jumped to next sequence")
                    else:
                        logger.warning("N pressed: Failed to jump to next sequence")
                    return False
                
                self.vis.register_key_callback(ord('B'), background_callback)
                self.vis.register_key_callback(ord(' '), next_sample_callback)
                self.vis.register_key_callback(ord('T'), toggle_transform_callback)
                self.vis.register_key_callback(ord('N'), next_sequence_callback)
                logger.info("Keyboard callbacks registered successfully")
            except Exception as e:
                logger.warning(f"Could not register keyboard callback: {e}")
                has_key_callback = False
        
        logger.info("Visualization ready. Press 'Q' to quit.")
        logger.info("Controls:")
        logger.info("  - Mouse: Rotate view")
        logger.info("  - Scroll: Zoom")
        logger.info("  - Shift+Mouse: Pan")
        logger.info("  - R: Reset view")
        logger.info("  - H: Print help")
        
        if has_key_callback:
            logger.info("  - B: Toggle background color")
            logger.info("  - T: Cycle between original, registered, generated, GT, and random_yaw views")
            if self.cycle_generations:
                logger.info("  - Spacebar: Cycle through generations for current sample")
            else:
                logger.info("  - Spacebar: Load next sample")
            logger.info("  - N: Jump to next sequence (first sample of next sequence)")
        
        view_name_map = {
            "original": "Original (untransformed)",
            "registered": "Registered (transformed)",
            "generated": "Generated (from processed directory)",
            "gt": "GT (ground truth)"
        }
        logger.info(f"  Current view: {view_name_map.get(self.view_mode, 'Unknown')}")
        
        logger.info("=" * 60)
        logger.info(f"CURRENT SAMPLE: Sequence {sequence}, Sample {sample_name}")
        logger.info("=" * 60)
        
        # Run visualizer
        self.vis.run()
        
        self.vis.destroy_window()
        self.vis = None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize registered point clouds using estimated poses"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input directory containing point cloud files")
    parser.add_argument("--results", "-r", type=str, required=True,
                       help="Directory containing result files (poses and generated point clouds)")
    parser.add_argument("--sequence", "-s", type=str, default=None,
                       help="Specific sequence to visualize (default: all sequences)")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index to start with (default: 0)")
    parser.add_argument("--max_points_per_fragment", type=int, default=10000,
                       help="Maximum points to load per fragment (default: 10000)")
    parser.add_argument("--generation", type=str, default="generation00",
                       help="Generation subdirectory name (default: generation00)")
    parser.add_argument("--point_size", type=float, default=3.0,
                       help="Point size in visualization (default: 3.0)")
    parser.add_argument("--background_color", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Background color as R G B values in [0,1] (default: 1.0 1.0 1.0 - white)")
    parser.add_argument("--show_coordinate_frame", action='store_true',
                       help="Show coordinate frame (hidden by default)")
    parser.add_argument("--estimate_normals", "-n", action='store_true', default=False,
                       help="Estimate normals for point clouds that don't have them")
    parser.add_argument("--normal_estimation_radius", type=float, default=1.0,
                       help="Radius for normal estimation (default: 1.0)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set the logging level (default: INFO)")
    parser.add_argument("--show_generated", action='store_true', default=False,
                       help="Load and show generated point clouds from processed directory (generation_selected PLY files)")
    parser.add_argument("--remove_outliers", action='store_true', default=False,
                       help="Apply stochastic outlier removal to input point clouds")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=20,
                       help="Number of neighbors for outlier removal (default: 20)")
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0,
                       help="Standard deviation ratio for outlier removal (default: 2.0)")
    parser.add_argument("--cycle_generations", action='store_true', default=False,
                       help="Cycle through different generations for each sample (press spacebar to cycle)")
    
    args = parser.parse_args()
    
    # Configure logging based on argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Expand user paths
    input_dir = os.path.expanduser(args.input)
    results_dir = os.path.expanduser(args.results)
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Create visualizer
    visualizer = RegisteredPointCloudVisualizer(
        input_dir=input_dir,
        poses_dir=results_dir,
        point_size=args.point_size,
        background_color=args.background_color,
        no_coordinate_frame=not args.show_coordinate_frame,
        estimate_normals=args.estimate_normals,
        normal_estimation_radius=args.normal_estimation_radius,
        generation=args.generation,
        max_points_per_fragment=args.max_points_per_fragment,
        remove_outliers=args.remove_outliers,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio
    )
    
    # Set flag for showing generated point clouds
    visualizer.show_generated = args.show_generated
    
    # Set flag for cycling through generations
    visualizer.cycle_generations = args.cycle_generations
    
    # Get list of all available sequences from input directory
    available_sequences = visualizer.get_available_sequences()
    
    if not available_sequences:
        logger.error(f"No sequences found in input directory: {input_dir}")
        return
    
    logger.info(f"Found {len(available_sequences)} sequence(s) in input directory:")
    for seq in available_sequences:
        logger.info(f"  - {seq}")
    
    # If sequence not specified, randomly select one
    if args.sequence is None:
        import random
        args.sequence = random.choice(available_sequences)
        logger.info(f"No sequence specified, randomly selected: {args.sequence}")
    elif args.sequence not in available_sequences:
        logger.warning(f"Specified sequence '{args.sequence}' not found in available sequences")
        logger.warning(f"Available sequences: {available_sequences}")
        import random
        args.sequence = random.choice(available_sequences)
        logger.info(f"Randomly selected instead: {args.sequence}")
    
    # Visualize first sample
    try:
        visualizer.visualize_sample(args.sequence, args.sample_idx)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
