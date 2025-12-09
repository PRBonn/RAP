#!/usr/bin/env python3
"""
Visualize Sample Features using Open3D

This script visualizes processed training samples with local features.
Features are colorized using PCA (Principal Component Analysis) where the first 3 
principal components are mapped to RGB colors, or using distinct colors for each part.

Features:
- White background as default (better for presentations/papers)
- Bird's Eye View (BEV) as default viewing angle
- Automatic random sequence selection if none specified
- Toggle between PCA-based colors and part index colors using 'C' key
- Toggle between white and black background using 'B' key
- Load next sample using Spacebar (next in sequence, or first of next sequence)
- Estimate normals for point clouds that don't have them
- Interactive 3D visualization with mouse controls

Usage:
    # Basic usage with PCA colors (default) - sequence will be randomly selected
    python ./dataset_process/visualize_sample_features.py --input /path/to/processed --sample_id 123
    
    # Specify a particular sequence
    python ./dataset_process/visualize_sample_features.py --input /path/to/processed --sequence 00 --sample_id 123
    
    # Start with part index colors
    python ./dataset_process/visualize_sample_features.py --input /path/to/processed --sequence 00 --sample_id 123 --color_mode part
    
    # With custom visualization options
    python ./dataset_process/visualize_sample_features.py --input /path/to/processed --sequence 00 --sample_id 123 --point_size 5.0 --color_mode pca
    
    # Visualize raw samples (before feature extraction) with part-based coloring
    python ./dataset_process/visualize_sample_features.py --input /path/to/raw_samples_output --sequence 00 --sample_id 123 --raw_samples
    
    # Estimate normals for point clouds that don't have them
    python ./dataset_process/visualize_sample_features.py --input /path/to/processed --sequence 00 --sample_id 123 --estimate_normals --normal_estimation_radius 0.2
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import glob
import threading
from dataset_process.utils.io_utils import CMAP_DEFAULT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SampleVisualizer:
    """Visualizer for processed training samples with feature colorization."""
    
    def __init__(self, 
                 pca_components: int = 3,
                 point_size: float = 3.0,
                 background_color: List[float] = [1.0, 1.0, 1.0],
                 no_coordinate_frame: bool = False,
                 raw_samples: bool = False,
                 estimate_normals: bool = False,
                 normal_estimation_radius: float = 0.1):
        """
        Initialize the sample visualizer.
        
        Args:
            pca_components: Number of PCA components to use (should be 3 for RGB)
            point_size: Size of points in the visualization
            background_color: Background color as [R, G, B] in range [0, 1] (default: white)
            raw_samples: If True, visualize raw samples (before feature extraction),
                         only showing parts colored by index.
            estimate_normals: If True, estimate normals for point clouds that don't have them
            normal_estimation_radius: Radius for normal estimation (default: 0.1)
        """
        self.pca_components = pca_components
        self.point_size = point_size
        self.background_color = background_color
        self.no_coordinate_frame = no_coordinate_frame
        self.raw_samples = raw_samples
        self.estimate_normals = estimate_normals
        self.normal_estimation_radius = normal_estimation_radius
        self.pca = PCA(n_components=pca_components)
        self.scaler = MinMaxScaler()
        
        # Color mode state
        self.use_pca_colors = True
        self.colored_pcds = []
        self.pca_colors = []
        self.part_colors = []
        self.vis = None
        
        # Background toggle state
        self.use_white_background = True
        self.white_background = [1.0, 1.0, 1.0]
        self.black_background = [0.0, 0.0, 0.0]
        
        # Check if initial background is closer to white or black
        if np.mean(self.background_color) > 0.5:
            self.use_white_background = True
        else:
            self.use_white_background = False
        
        # Sample navigation state
        self.base_dir = None
        self.current_sequence = None
        self.current_sample_id = None
        self.padding_length = 6
        self.sequence_samples_cache = {}  # Cache for sequence sample lists
        
        # Global PCA state for consistent coloring across samples
        self.global_pca = None
        self.global_scaler = None
        self.use_global_pca = False
        
        # Supported directory name prefixes
        self.directory_prefixes = ['sample', 'fracture', 'part']
    
    def _detect_dataset_name(self, base_dir: str, sequence: str) -> Optional[str]:
        """
        Detect dataset name from directory structure.
        
        Args:
            base_dir: Base directory containing processed samples
            sequence: Sequence name
            
        Returns:
            Dataset name if found, None otherwise
        """
        if not base_dir or not os.path.exists(base_dir):
            return None
        
        # Check if base_dir itself is a dataset directory (e.g., named "modelnet")
        base_dir_name = os.path.basename(base_dir.rstrip('/'))
        if base_dir_name.lower().startswith('modelnet'):
            # Check if sequences exist directly under base_dir
            seq_path = os.path.join(base_dir, sequence)
            if os.path.exists(seq_path):
                sample_dirs = self._find_sample_directories(seq_path)
                if sample_dirs:
                    return base_dir_name
        
        # Try direct structure first: base_dir/sequence/
        seq_path = os.path.join(base_dir, sequence)
        if os.path.exists(seq_path):
            # Check if this is a direct sequence (no dataset parent)
            sample_dirs = self._find_sample_directories(seq_path)
            if sample_dirs:
                # Check if base_dir itself is a dataset directory
                # by checking if it looks like a dataset name
                if any(base_dir_name.lower().startswith(prefix) for prefix in ['modelnet', 'partnet', 'ikea']):
                    return base_dir_name
                return None
        
        # Try dataset structure: base_dir/dataset_name/sequence/
        for dataset_dir in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                seq_path = os.path.join(dataset_path, sequence)
                if os.path.exists(seq_path):
                    sample_dirs = self._find_sample_directories(seq_path)
                    if sample_dirs:
                        return dataset_dir
                
                # Also check nested structure: base_dir/dataset_name/scene/seq/
                for subseq_dir in os.listdir(dataset_path):
                    subseq_path = os.path.join(dataset_path, subseq_dir)
                    if os.path.isdir(subseq_path):
                        nested_seq_path = os.path.join(subseq_path, sequence.split('/')[-1] if '/' in sequence else sequence)
                        if os.path.exists(nested_seq_path):
                            sample_dirs = self._find_sample_directories(nested_seq_path)
                            if sample_dirs:
                                return dataset_dir
        
        return None
    
    def _determine_padding_length(self, base_dir: str, sequence: str) -> int:
        """
        Determine appropriate padding length by examining actual directory names.
        
        Args:
            base_dir: Base directory containing processed samples
            sequence: Sequence name
            
        Returns:
            Padding length detected from actual directories (default: 6)
        """
        dataset_name = self._detect_dataset_name(base_dir, sequence)
        if dataset_name and dataset_name.lower().startswith('modelnet'):
            return 1
        
        # Try to detect actual padding length from directory names
        # Find the sequence path
        seq_path = None
        
        # Try direct structure: base_dir/sequence/
        direct_path = os.path.join(base_dir, sequence)
        if os.path.exists(direct_path):
            seq_path = direct_path
        
        # Try dataset structure: base_dir/dataset_name/sequence/
        if seq_path is None:
            for dataset_dir in os.listdir(base_dir):
                dataset_path = os.path.join(base_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    candidate_path = os.path.join(dataset_path, sequence)
                    if os.path.exists(candidate_path):
                        seq_path = candidate_path
                        break
        
        # If we found the sequence path, check actual directory names
        if seq_path:
            sample_dirs = self._find_sample_directories(seq_path)
            if sample_dirs:
                # Extract sample ID from first directory and check its length
                sample_name = os.path.basename(sample_dirs[0])
                sample_id = self._extract_sample_id(sample_name)
                if sample_id and sample_id.isdigit():
                    # Count leading zeros + digits to determine padding
                    # If sample_id is "00000", padding_length should be 5
                    # If sample_id is "0", padding_length should be 1
                    # But we want the total length including leading zeros
                    padding_length = len(sample_id)
                    logger.debug(f"Detected padding_length={padding_length} from directory '{sample_name}' (sample_id='{sample_id}')")
                    return padding_length
        
        # Default fallback
        logger.debug(f"Could not detect padding length from directories, using default 6")
        return 6
    
    def toggle_background_color(self):
        """Toggle between white and black background colors."""
        self.use_white_background = not self.use_white_background
        
        if self.use_white_background:
            new_background = self.white_background
            logger.info("Switched to white background")
        else:
            new_background = self.black_background
            logger.info("Switched to black background")
        
        # Update background color
        self.background_color = new_background
        
        # Update visualization if visualizer exists
        if self.vis is not None:
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array(new_background)
    
    def _find_sample_directories(self, directory: str) -> List[str]:
        """
        Find all sample directories matching any supported prefix pattern.
        
        Args:
            directory: Directory to search in
            
        Returns:
            List of matching directory paths
        """
        sample_dirs = []
        for prefix in self.directory_prefixes:
            if self.raw_samples:
                pattern = os.path.join(directory, f"{prefix}_*")
            else:
                pattern = os.path.join(directory, f"{prefix}_*_processed")
            matches = glob.glob(pattern)
            sample_dirs.extend(matches)
        return sample_dirs
    
    def _extract_sample_id(self, directory_name: str) -> Optional[str]:
        """
        Extract sample ID from directory name, supporting multiple prefixes.
        
        Args:
            directory_name: Name of the directory (e.g., "sample_123", "fracture_456_processed")
            
        Returns:
            Sample ID string, or None if not matched
        """
        for prefix in self.directory_prefixes:
            if self.raw_samples:
                if directory_name.startswith(f"{prefix}_"):
                    return directory_name[len(f"{prefix}_"):]
            else:
                if directory_name.startswith(f"{prefix}_") and directory_name.endswith("_processed"):
                    return directory_name[len(f"{prefix}_"):-10]  # Remove prefix_ and _processed
        return None
    
    def generate_part_colors(self, num_parts: int) -> List[np.ndarray]:
        """
        Generate distinct colors for each part using the default color map from render.py.
        
        Args:
            num_parts: Number of parts to generate colors for
            
        Returns:
            List of color arrays, one per part
        """
        colors = []
        
        for i in range(num_parts):
            # Use default color map, cycling through colors if we have more parts than colors
            color_idx = i % len(CMAP_DEFAULT)
            color = CMAP_DEFAULT[color_idx]
            colors.append(color)
        
        logger.info(f"Generated {num_parts} distinct part colors using default color map")
        if num_parts > len(CMAP_DEFAULT):
            logger.info(f"Note: Cycling through colors since {num_parts} parts > {len(CMAP_DEFAULT)} available colors")
        
        return colors
    
    def load_sample_data(self, sample_dir: str, center_pcds: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[Optional[np.ndarray]]]:
        """
        Load point clouds and features from a processed sample directory.
        
        Args:
            sample_dir: Path to processed sample directory
            center_pcds: Whether to center all point clouds at origin
            
        Returns:
            Tuple of (point_clouds, features, part_names, normals) or (point_clouds, part_ids, part_names, normals) if raw_samples is True.
            normals is a list of normal arrays (one per part), or None if normals are not available.
        """
        if not os.path.exists(sample_dir):
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        
        # Find all PLY files
        ply_files = sorted(glob.glob(os.path.join(sample_dir, "*.ply")))
        
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {sample_dir}")
        
        point_clouds = []
        features = []
        part_names = []
        part_ids = []
        normals = []

        for i, ply_file in enumerate(ply_files):
            # Extract part name from PLY filename
            part_name = os.path.splitext(os.path.basename(ply_file))[0]
            
            try:
                # Load point cloud
                pcd = o3d.io.read_point_cloud(ply_file)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    logger.warning(f"Empty point cloud in {ply_file}")
                    continue
                
                # Check for normals and estimate if needed
                part_normals = None
                if pcd.has_normals():
                    part_normals = np.asarray(pcd.normals)
                    if len(part_normals) != len(points):
                        logger.warning(f"Mismatch between points ({len(points)}) and normals ({len(part_normals)}) for {part_name}")
                        part_normals = None
                    else:
                        logger.debug(f"Loaded normals for part '{part_name}': {len(part_normals)} normals")
                
                # Estimate normals if they don't exist and estimation is requested
                if part_normals is None and self.estimate_normals:
                    logger.info(f"Estimating normals for part '{part_name}' (radius={self.normal_estimation_radius})")
                    # Estimate normals using Open3D
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=self.normal_estimation_radius,
                            max_nn=30
                        )
                    )
                    # Orient normals consistently (optional, but helps with visualization)
                    pcd.orient_normals_consistent_tangent_plane(k=15)
                    part_normals = np.asarray(pcd.normals)
                    logger.info(f"Estimated {len(part_normals)} normals for part '{part_name}'")
                elif part_normals is None:
                    logger.debug(f"No normals found for part '{part_name}'")
                
                if self.raw_samples:
                    point_clouds.append(points)
                    part_names.append(part_name)
                    part_ids.append(np.full(len(points), i, dtype=int)) # Assign unique part ID
                    normals.append(part_normals)
                    logger.debug(f"Loaded raw part '{part_name}': {len(points)} points")
                else:
                    # Find corresponding feature file
                    feature_file = os.path.join(sample_dir, f"features_{part_name}.npy")
                    
                    if not os.path.exists(feature_file):
                        logger.warning(f"Feature file not found for {part_name}: {feature_file}")
                        continue

                    # Load features
                    part_features = np.load(feature_file)
                    
                    if len(points) != len(part_features):
                        logger.warning(f"Mismatch between points ({len(points)}) and features ({len(part_features)}) for {part_name}")
                        continue
                    
                    point_clouds.append(points)
                    features.append(part_features)
                    part_names.append(part_name)
                    normals.append(part_normals)
                    
                    logger.debug(f"Loaded part '{part_name}': {len(points)} points, {part_features.shape[1]} feature dims")
                
            except Exception as e:
                logger.error(f"Failed to load {part_name}: {e}")
                continue
        
        if not point_clouds:
            raise RuntimeError(f"No valid parts loaded from {sample_dir}")
        
        # Log detailed point count information
        total_points = sum(len(points) for points in point_clouds)
        
        logger.info("=" * 60)
        logger.info(f"SAMPLE POINT COUNT SUMMARY:")
        logger.info(f"  Sample Directory: {os.path.basename(sample_dir)}")
        logger.info(f"  Number of Parts: {len(point_clouds)}")

        if self.raw_samples:
            logger.info("  Mode: Raw Samples (no features)")
            for i, (points, part_name) in enumerate(zip(point_clouds, part_names)):
                percentage = (len(points) / total_points) * 100
                logger.info(f"  Part {i+1:2d} ({part_name:<12}): {len(points):6d} points ({percentage:5.1f}%) [Part ID: {i}]")
            
        else:
            feature_dims = features[0].shape[1] if features else 0
            logger.info(f"  Feature Dimensions: {feature_dims}")
            for i, (points, part_name) in enumerate(zip(point_clouds, part_names)):
                percentage = (len(points) / total_points) * 100
                logger.info(f"  Part {i+1:2d} ({part_name:<12}): {len(points):6d} points ({percentage:5.1f}%) ")
        logger.info(f"  TOTAL POINTS: {total_points:6d}")
        
        # Log normal availability
        normals_count = sum(1 for n in normals if n is not None)
        if normals_count > 0:
            logger.info(f"  Normals available: {normals_count}/{len(normals)} parts")
        else:
            logger.info(f"  Normals available: None")
        logger.info("=" * 60)
        
        # Center point clouds if requested
        if center_pcds:
            point_clouds, center_offset = self.center_point_clouds(point_clouds)
            logger.info(f"Point clouds centered at origin with offset: {center_offset}")
            # Note: Normals don't need to be modified when centering (they're direction vectors)
        
        if self.raw_samples:
            all_part_ids = np.concatenate(part_ids, axis=0)
            return point_clouds, all_part_ids, part_names, normals
        else:
            return point_clouds, features, part_names, normals
    
    def compute_pca_colors(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute PCA-based colors for features.
        
        Args:
            features: List of feature arrays, one per part
            
        Returns:
            List of RGB color arrays, one per part
        """
        if self.raw_samples:
            logger.warning("Attempted to compute PCA colors in raw samples mode. Returning empty list.")
            return []
        
        # Concatenate all features for global PCA
        all_features = np.concatenate(features, axis=0)
        logger.info(f"Computing PCA on {all_features.shape[0]} points with {all_features.shape[1]} features")
        
        # Fit PCA on all features
        pca_features = self.pca.fit_transform(all_features)
        
        # Normalize PCA components to [0, 1] range
        pca_normalized = self.scaler.fit_transform(pca_features)
        
        # Split back into parts
        colors = []
        start_idx = 0
        
        for feature_array in features:
            end_idx = start_idx + len(feature_array)
            part_colors = pca_normalized[start_idx:end_idx]
            colors.append(part_colors)
            start_idx = end_idx
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return colors
    
    def create_colored_point_clouds(self, 
                                  point_clouds: List[np.ndarray], 
                                  colors_data: Union[List[np.ndarray], np.ndarray],
                                  part_names: List[str],
                                  normals: Optional[List[Optional[np.ndarray]]] = None) -> List[o3d.geometry.PointCloud]:
        """
        Create Open3D point clouds with both PCA and part index colors.
        
        Args:
            point_clouds: List of point arrays
            colors_data: List of PCA-based color arrays (RGB in [0, 1]) OR a single np.ndarray of part_ids
            part_names: List of part names
            normals: Optional list of normal arrays (one per part), or None if not available
            
        Returns:
            List of colored Open3d point clouds
        """
        colored_pcds = []
        
        # Generate part index colors
        part_base_colors = self.generate_part_colors(len(point_clouds))
        
        # Store colors for toggling
        if not self.raw_samples:
            self.pca_colors = colors_data # In non-raw mode, colors_data is pca_colors
        self.part_colors = []
        
        for i, (points, part_name) in enumerate(zip(point_clouds, part_names)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Add normals if available
            if normals and i < len(normals) and normals[i] is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals[i])
            
            if self.raw_samples:
                # For raw samples, merge all into one PCD and color by part ID
                # This loop will run for each part, but we only create one merged PCD
                # We need to collect all points and colors first
                pass # Actual merging and coloring will happen after the loop
            else:
                # Generate part index colors (same color for all points in this part)
                part_color = np.tile(part_base_colors[i], (len(points), 1))
                self.part_colors.append(part_color)
                
                # Set initial colors based on current mode
                if self.use_pca_colors:
                    pcd.colors = o3d.utility.Vector3dVector(colors_data[i]) # In non-raw mode, colors_data[i] is pca_part_colors
                else:
                    pcd.colors = o3d.utility.Vector3dVector(part_color)
                
            colored_pcds.append(pcd)
            logger.info(f"Created colored point cloud for '{part_name}' (Part {i})")
        
        if self.raw_samples:
            # Merge all point clouds into a single one for raw samples and color by part ID
            merged_pcd = o3d.geometry.PointCloud()
            all_points = np.concatenate(point_clouds, axis=0)
            all_part_ids_flat = colors_data # In raw mode, colors_data is the concatenated part_ids

            # Generate colors based on part IDs
            unique_part_ids = np.unique(all_part_ids_flat)
            num_unique_parts = len(unique_part_ids)
            part_id_colors = self.generate_part_colors(num_unique_parts)

            # Map part IDs to colors
            colors_array = np.zeros((len(all_points), 3))
            for i, part_id in enumerate(unique_part_ids):
                colors_array[all_part_ids_flat == part_id] = part_id_colors[i]

            merged_pcd.points = o3d.utility.Vector3dVector(all_points)
            merged_pcd.colors = o3d.utility.Vector3dVector(colors_array)
            
            # Merge normals if available
            if normals:
                all_normals = []
                for part_normals in normals:
                    if part_normals is not None:
                        all_normals.append(part_normals)
                if all_normals:
                    merged_normals = np.concatenate(all_normals, axis=0)
                    merged_pcd.normals = o3d.utility.Vector3dVector(merged_normals)
            
            colored_pcds = [merged_pcd] # Only one PCD for raw samples
            logger.info(f"Created single merged point cloud for raw samples, colored by {num_unique_parts} part indices.")

        self.colored_pcds = colored_pcds
        return colored_pcds
    
    def toggle_color_mode(self):
        """Toggle between PCA colors and part index colors."""
        self.use_pca_colors = not self.use_pca_colors
        
        if self.raw_samples:
            logger.info("Color mode toggle ignored: Raw samples mode only supports part index coloring.")
            return
        
        if self.use_pca_colors:
            logger.info("Switched to PCA-based colors")
            colors_to_use = self.pca_colors
        else:
            logger.info("Switched to part index-based colors")
            colors_to_use = self.part_colors
        
        # Update colors for all point clouds
        for pcd, colors in zip(self.colored_pcds, colors_to_use):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Update visualization if visualizer exists
        if self.vis is not None:
            for pcd in self.colored_pcds:
                self.vis.update_geometry(pcd)
    
    def visualize_sample(self, 
                        sample_dir: str,
                        window_name: Optional[str] = None,
                        show_coordinate_frame: bool = True,
                        save_screenshot: Optional[str] = None,
                        center_pcds: bool = True,
                        raw_samples: bool = False) -> None:
        """
        Visualize a processed sample with PCA-colorized features.
        
        Args:
            sample_dir: Path to processed sample directory
            window_name: Custom window name
            show_coordinate_frame: Whether to show coordinate frame
            save_screenshot: Path to save screenshot (optional)
            center_pcds: Whether to center all point clouds at origin (default: True)
            raw_samples: If True, visualize raw samples (before feature extraction),
                         only showing parts colored by index.
        """
        logger.info(f"Visualizing sample from: {sample_dir}")
        
        # Load sample data
        point_clouds, features, part_names, normals = self.load_sample_data(sample_dir, center_pcds=center_pcds)
        
        # Compute PCA colors (use global PCA if available)
        if self.use_global_pca:
            pca_colors = self.apply_global_pca_colors(features)
        else:
            pca_colors = self.compute_pca_colors(features)
        
        # Create colored point clouds (both PCA and part colors)
        if raw_samples:
            # In raw samples mode, features are actually part_ids
            # And there is no pca_colors needed, so we pass the part_ids as colors_data
            colored_pcds = self.create_colored_point_clouds(point_clouds, features, part_names, normals)
        else:
            colored_pcds = self.create_colored_point_clouds(point_clouds, pca_colors, part_names, normals)
        
        # Try to use VisualizerWithKeyCallback, fallback to regular Visualizer
        try:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            has_key_callback = True
        except AttributeError:
            self.vis = o3d.visualization.Visualizer()
            has_key_callback = False
            logger.info("Key callback not supported in this Open3D version, using manual toggle")
        
        self.vis.create_window(window_name="Sample 3D Visualizer")
        
        # Add point clouds to visualizer
        for pcd in colored_pcds:
            self.vis.add_geometry(pcd)
        
        # Add coordinate frame if requested
        if show_coordinate_frame and not self.no_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.vis.add_geometry(coordinate_frame)
        
        # Set render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.point_size = self.point_size
        
        # Center and fit view to point cloud
        self._center_and_fit_view()
        
        # Register keyboard callback for color toggling if supported
        if has_key_callback:
            try:
                # Create callback function for color toggling
                def color_callback(vis):
                    self.toggle_color_mode()
                    return False
                
                # Create callback function for background toggling
                def background_callback(vis):
                    self.toggle_background_color()
                    return False
                
                # Create callback function for loading next sample
                def next_sample_callback(vis):
                    success = self.load_next_sample()
                    if success:
                        logger.info("Spacebar pressed: Loaded next sample")
                    else:
                        logger.warning("Spacebar pressed: Failed to load next sample")
                    return False
                
                # Create callback function for loading next sequence
                def next_sequence_callback(vis):
                    success = self.load_next_sequence()
                    if success:
                        logger.info("N pressed: Jumped to next sequence")
                    else:
                        logger.warning("N pressed: Failed to jump to next sequence")
                    return False
                
                self.vis.register_key_callback(ord('C'), color_callback)
                self.vis.register_key_callback(ord('B'), background_callback)
                self.vis.register_key_callback(ord(' '), next_sample_callback)  # Spacebar
                self.vis.register_key_callback(ord('N'), next_sequence_callback)  # N key
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
            logger.info("  - C: Toggle between PCA colors and part index colors")
            logger.info("  - B: Toggle background color")
            logger.info("  - Spacebar: Load next sample (next in sequence, or first of next sequence)")
            logger.info("  - N: Jump to next sequence (first sample of next sequence)")
        
        if self.raw_samples:
            logger.info("  Note: In raw samples mode, 'C' toggle is disabled as only part index colors are shown.")
        else:
            logger.info(f"  - Color mode: {self.get_color_mode_description()}")
            logger.info("    (To change color mode, close and restart with different initial setting)")
        
        logger.info(f"Current color mode: {'PCA-based' if self.use_pca_colors else 'Part index-based'}")
        logger.info("=" * 60)
        # Display padded sample ID for consistency
        padded_sample_id = self.current_sample_id.zfill(self.padding_length) if self.current_sample_id else "N/A"
        logger.info(f"CURRENT SAMPLE: Sequence {self.current_sequence}, Sample {padded_sample_id}")
        logger.info("=" * 60)
        
        # Run visualizer
        if has_key_callback:
            # Use the callback-based approach
            self.vis.run()
        else:
            # Use manual polling approach for compatibility
            self._run_with_manual_controls()
        
        # Save screenshot if requested
        if save_screenshot:
            self.vis.capture_screen_image(save_screenshot)
            logger.info(f"Screenshot saved to: {save_screenshot}")
        
        self.vis.destroy_window()
        self.vis = None  # Clean up reference
    
    def _run_with_manual_controls(self):
        """Run visualizer with manual control polling for older Open3D versions."""
        logger.info("Manual control mode - press ESC in console and type commands + Enter:")
        logger.info("  'c' = toggle colors, 'b' = toggle background, 'n' = next sample, 's' = next sequence, 'q' = quit")
        logger.info(f"Current: Sequence {self.current_sequence}, Sample {self.current_sample_id}")
        
        def input_thread():
            while True:
                try:
                    command = input().strip().lower()
                    if command == 'c':
                        self.toggle_color_mode()
                    elif command == 'b':
                        self.toggle_background_color()
                    elif command == 'n':
                        success = self.load_next_sample()
                        if success:
                            logger.info("Loaded next sample")
                        else:
                            logger.warning("Failed to load next sample")
                    elif command == 's':
                        success = self.load_next_sequence()
                        if success:
                            logger.info("Jumped to next sequence")
                        else:
                            logger.warning("Failed to jump to next sequence")
                    elif command in ['q', 'quit', 'exit']:
                        break
                except (EOFError, KeyboardInterrupt):
                    break
        
        # Start input thread
        input_thread_obj = threading.Thread(target=input_thread, daemon=True)
        input_thread_obj.start()
        
        # Run the visualizer
        self.vis.run()
    
    def get_color_mode_description(self) -> str:
        """Get description of current color mode."""
        if self.use_pca_colors:
            return "PCA-based colors (features)"
        else:
            return "Part index-based colors (distinct per part)"
    
    def set_navigation_state(self, base_dir: str, sequence: str, sample_id: str, padding_length: Optional[int] = None):
        """
        Set the navigation state for sample browsing.
        
        Args:
            base_dir: Base directory containing processed samples
            sequence: Current sequence name
            sample_id: Current sample ID
            padding_length: Length to pad sample ID to (auto-detected if None)
        """
        self.base_dir = base_dir
        self.current_sequence = sequence
        self.current_sample_id = sample_id
        
        # Auto-detect padding_length if not provided
        if padding_length is None:
            padding_length = self._determine_padding_length(base_dir, sequence)
            dataset_name = self._detect_dataset_name(base_dir, sequence)
            if dataset_name:
                logger.info(f"Auto-detected padding_length={padding_length} for dataset '{dataset_name}'")
        
        self.padding_length = padding_length
        logger.info(f"Navigation state set: {base_dir}, sequence {sequence}, sample {sample_id}, padding_length={padding_length}")
    
    def compute_global_pca(self, max_samples: int = 1000):
        """
        Compute global PCA on a subset of all available samples for consistent coloring.
        
        Args:
            max_samples: Maximum number of samples to use for PCA computation (for performance)
        """
        if not self.base_dir or not os.path.exists(self.base_dir):
            logger.warning("Base directory not set, cannot compute global PCA")
            return False
        
        logger.info("Computing global PCA for consistent coloring across samples...")
        
        # Get all available sequences
        sequences = self.get_available_sequences()
        if not sequences:
            logger.warning("No sequences found for global PCA computation")
            return False
        
        all_features = []
        sample_count = 0
        
        # Collect features from samples across all sequences
        for sequence in sequences:
            if sample_count >= max_samples:
                break
                
            samples = self.get_sequence_samples(sequence)
            for sample_id in samples:
                if sample_count >= max_samples:
                    break
                    
                try:
                    # Find sample directory
                    sample_dir = find_sample_directory(self.base_dir, sequence, sample_id, self.padding_length, raw_samples=self.raw_samples, directory_prefixes=self.directory_prefixes)
                    
                    # Load sample data
                    point_clouds, features, part_names, normals = self.load_sample_data(sample_dir, center_pcds=False)
                    
                    # Concatenate all features from this sample
                    if features:
                        sample_features = np.concatenate(features, axis=0)
                        all_features.append(sample_features)
                        sample_count += 1
                        
                        if sample_count % 100 == 0:
                            logger.info(f"Processed {sample_count} samples for global PCA...")
                    
                except Exception as e:
                    logger.warning(f"Failed to load sample {sequence}/{sample_id} for global PCA: {e}")
                    continue
        
        if not all_features:
            logger.warning("No features collected for global PCA")
            return False
        
        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"Computing global PCA on {all_features.shape[0]} points with {all_features.shape[1]} features from {sample_count} samples")
        
        # Compute global PCA
        self.global_pca = PCA(n_components=self.pca_components)
        self.global_scaler = MinMaxScaler()
        
        # Fit PCA and scaler
        pca_features = self.global_pca.fit_transform(all_features)
        self.global_scaler.fit(pca_features)
        
        logger.info(f"Global PCA computed successfully. Explained variance ratio: {self.global_pca.explained_variance_ratio_}")
        self.use_global_pca = True
        return True
    
    def apply_global_pca_colors(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply global PCA transformation to features for consistent coloring.
        
        Args:
            features: List of feature arrays, one per part
            
        Returns:
            List of RGB color arrays, one per part
        """
        if not self.use_global_pca or self.global_pca is None:
            logger.warning("Global PCA not available, falling back to local PCA")
            return self.compute_pca_colors(features)
        
        # Concatenate all features
        all_features = np.concatenate(features, axis=0)
        
        # Apply global PCA transformation
        pca_features = self.global_pca.transform(all_features)
        
        # Apply global scaling
        pca_normalized = self.global_scaler.transform(pca_features)
        
        # Split back into parts
        colors = []
        start_idx = 0
        
        for feature_array in features:
            end_idx = start_idx + len(feature_array)
            part_colors = pca_normalized[start_idx:end_idx]
            colors.append(part_colors)
            start_idx = end_idx
        
        return colors
    
    def get_available_sequences(self) -> List[str]:
        """Get list of available sequences in the base directory."""
        if not self.base_dir or not os.path.exists(self.base_dir):
            return []
        
        sequences = []
        
        # Check direct sequence directories
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory contains processed samples
                sample_dirs = self._find_sample_directories(item_path)
                if sample_dirs:
                    sequences.append(item)
        
        # Also check for dataset structure: base_dir/dataset_name/sequence/
        for dataset_dir in os.listdir(self.base_dir):
            dataset_path = os.path.join(self.base_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                for seq_dir in os.listdir(dataset_path):
                    seq_path = os.path.join(dataset_path, seq_dir)
                    if os.path.isdir(seq_path):
                        sample_dirs = self._find_sample_directories(seq_path)
                        if sample_dirs and seq_dir not in sequences:
                            sequences.append(seq_dir)
                        
                        # Also check for nested structure: base_dir/dataset_name/scene/seq/
                        # (e.g., for ThreeDMatch: base_dir/dataset_name/bundlefusion-office0/seq-01/)
                        for subseq_dir in os.listdir(seq_path):
                            subseq_path = os.path.join(seq_path, subseq_dir)
                            if os.path.isdir(subseq_path):
                                # Check if subseq_dir itself matches a prefix pattern
                                if any(subseq_dir.startswith(f"{prefix}_") for prefix in self.directory_prefixes):
                                    continue
                                sample_dirs = self._find_sample_directories(subseq_path)
                                if sample_dirs:
                                    nested_seq_name = f"{seq_dir}/{subseq_dir}"
                                    if nested_seq_name not in sequences:
                                        sequences.append(nested_seq_name)
        
        return sorted(sequences)
    
    def get_sequence_samples(self, sequence: str) -> List[str]:
        """
        Get list of available sample IDs for a given sequence.
        
        Args:
            sequence: Sequence name (can be nested like "scene/seq-01")
            
        Returns:
            List of sample IDs (as strings)
        """
        if sequence in self.sequence_samples_cache:
            return self.sequence_samples_cache[sequence]
        
        if not self.base_dir or not os.path.exists(self.base_dir):
            return []
        
        sample_ids = []
        
        # Try direct structure: base_dir/sequence/{prefix}_XXXXXX{_processed}
        seq_path = os.path.join(self.base_dir, sequence)
        if os.path.exists(seq_path):
            sample_dirs = self._find_sample_directories(seq_path)
            for sample_dir in sample_dirs:
                # Extract sample ID from directory name
                sample_name = os.path.basename(sample_dir)
                sample_id = self._extract_sample_id(sample_name)
                if sample_id and sample_id not in sample_ids: # Ensure uniqueness
                    sample_ids.append(sample_id)
        
        # Try dataset structure: base_dir/*/sequence/{prefix}_XXXXXX{_processed}
        if not sample_ids:
            for dataset_dir in os.listdir(self.base_dir):
                dataset_path = os.path.join(self.base_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    seq_path = os.path.join(dataset_path, sequence)
                    if os.path.exists(seq_path):
                        sample_dirs = self._find_sample_directories(seq_path)
                        for sample_dir in sample_dirs:
                            sample_name = os.path.basename(sample_dir)
                            sample_id = self._extract_sample_id(sample_name)
                            if sample_id and sample_id not in sample_ids: # Ensure uniqueness
                                sample_ids.append(sample_id)
        
        # Sort sample IDs numerically
        sample_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
        
        # Cache the result
        self.sequence_samples_cache[sequence] = sample_ids
        
        return sample_ids
    
    def find_next_sample(self) -> Tuple[str, str]:
        """
        Find the next sample to load.
        
        Returns:
            Tuple of (next_sequence, next_sample_id)
        """
        if not self.current_sequence or not self.current_sample_id:
            logger.warning("Navigation state not set, cannot find next sample")
            return None, None
        
        # Get available sequences
        sequences = self.get_available_sequences()
        if not sequences:
            logger.warning("No sequences found")
            return None, None
        
        # Get samples for current sequence
        current_samples = self.get_sequence_samples(self.current_sequence)
        if not current_samples:
            logger.warning(f"No samples found for sequence {self.current_sequence}")
            return None, None
        
        # Pad current sample ID to 6 digits for proper indexing
        # In raw samples mode, sample IDs are usually not padded, so use the original ID for lookup
        sample_id_for_lookup = self.current_sample_id if self.raw_samples else self.current_sample_id.zfill(self.padding_length)
        
        # Find current sample index
        try:
            current_idx = current_samples.index(sample_id_for_lookup)
        except ValueError:
            # Try with padded version if it's a digit and not raw_samples (for robustness, though should be covered by sample_id_for_lookup)
            if sample_id_for_lookup.isdigit() and not self.raw_samples and self.current_sample_id.zfill(self.padding_length) in current_samples:
                current_idx = current_samples.index(self.current_sample_id.zfill(self.padding_length))
                logger.warning(f"Current sample '{sample_id_for_lookup}' not found, but found padded '{self.current_sample_id.zfill(self.padding_length)}' in sequence '{self.current_sequence}'. Using padded.")
            else:
                logger.warning(f"Current sample '{sample_id_for_lookup}' not found in sequence '{self.current_sequence}")
                return None, None
        
        # Check if there's a next sample in current sequence
        if current_idx + 1 < len(current_samples):
            next_sample_id = current_samples[current_idx + 1]
            logger.info(f"Next sample in sequence {self.current_sequence}: {next_sample_id}")
            return self.current_sequence, next_sample_id
        
        # If we're at the last sample, move to next sequence
        current_seq_idx = sequences.index(self.current_sequence)
        if current_seq_idx + 1 < len(sequences):
            next_sequence = sequences[current_seq_idx + 1]
            next_samples = self.get_sequence_samples(next_sequence)
            if next_samples:
                next_sample_id = next_samples[0]  # First sample of next sequence
                logger.info(f"Moving to next sequence {next_sequence}, first sample: {next_sample_id}")
                return next_sequence, next_sample_id
        
        # If we're at the last sample of the last sequence, loop back to first
        first_sequence = sequences[0]
        first_samples = self.get_sequence_samples(first_sequence)
        if first_samples:
            first_sample_id = first_samples[0]
            logger.info(f"Looping back to first sequence {first_sequence}, first sample: {first_sample_id}")
            return first_sequence, first_sample_id
        
        logger.warning("No next sample found")
        return None, None
    
    def find_next_sequence(self) -> Tuple[str, str]:
        """
        Find the first sample of the next sequence to load.
        
        Returns:
            Tuple of (next_sequence, first_sample_id)
        """
        if not self.current_sequence:
            logger.warning("Navigation state not set, cannot find next sequence")
            return None, None
        
        # Get available sequences
        sequences = self.get_available_sequences()
        if not sequences:
            logger.warning("No sequences found")
            return None, None
        
        # Find current sequence index
        try:
            current_seq_idx = sequences.index(self.current_sequence)
        except ValueError:
            logger.warning(f"Current sequence {self.current_sequence} not found in available sequences")
            return None, None
        
        # Check if there's a next sequence
        if current_seq_idx + 1 < len(sequences):
            next_sequence = sequences[current_seq_idx + 1]
        else:
            # If we're at the last sequence, loop back to first
            next_sequence = sequences[0]
        
        # Get first sample of next sequence
        next_samples = self.get_sequence_samples(next_sequence)
        if next_samples:
            first_sample_id = next_samples[0]
            logger.info(f"Jumping to next sequence {next_sequence}, first sample: {first_sample_id}")
            return next_sequence, first_sample_id
        else:
            logger.warning(f"No samples found in next sequence {next_sequence}")
            return None, None
    
    def load_next_sequence(self):
        """Load and visualize the first sample of the next sequence."""
        next_sequence, first_sample_id = self.find_next_sequence()
        
        if next_sequence is None or first_sample_id is None:
            logger.warning("No next sequence available")
            return False
        
        try:
            # Find the sample directory
            sample_dir = find_sample_directory(self.base_dir, next_sequence, first_sample_id, self.padding_length, raw_samples=self.raw_samples, directory_prefixes=self.directory_prefixes)
            
            # Update navigation state (store unpadded version for consistency)
            self.current_sequence = next_sequence
            self.current_sample_id = first_sample_id
            
            # Clear current visualization
            if self.vis is not None:
                # Clear all existing geometries (including point clouds and coordinate frame)
                self.vis.clear_geometries()
            
            # Load new sample data
            point_clouds, data_for_colors, part_names, normals = self.load_sample_data(sample_dir, center_pcds=True)
            
            # Compute PCA colors or use part_ids for coloring
            if self.raw_samples:
                pca_colors = [] # Not used in raw mode, but argument required for type consistency if we keep the signature
                colored_pcds = self.create_colored_point_clouds(point_clouds, data_for_colors, part_names, normals) # data_for_colors is part_ids here
            else:
                pca_colors = self.apply_global_pca_colors(data_for_colors) if self.use_global_pca else self.compute_pca_colors(data_for_colors)
                colored_pcds = self.create_colored_point_clouds(point_clouds, pca_colors, part_names, normals)
            
            # Add new geometries to visualizer
            if self.vis is not None:
                for pcd in colored_pcds:
                    self.vis.add_geometry(pcd, False) # Keep False for efficiency in loop
                    self.vis.update_geometry(pcd) # Explicitly update geometry
                
                # Add coordinate frame if enabled (it was cleared by clear_geometries)
                if not self.no_coordinate_frame:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                    self.vis.add_geometry(coordinate_frame, False)
                    self.vis.update_geometry(coordinate_frame) # Explicitly update geometry
                
                # Update view
                self._center_and_fit_view()
                self.vis.update_renderer()
            
            logger.info("=" * 60)
            # Display padded sample ID for consistency
            if self.raw_samples:
                logged_sample_id = first_sample_id
            else:
                logged_sample_id = first_sample_id.zfill(self.padding_length) if first_sample_id else "N/A"
            logger.info(f"JUMPED TO NEXT SEQUENCE: Sequence {next_sequence}, Sample {logged_sample_id}")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load next sequence: {e}")
            return False
    
    def load_next_sample(self):
        """Load and visualize the next sample."""
        next_sequence, next_sample_id = self.find_next_sample()
        
        if next_sequence is None or next_sample_id is None:
            logger.warning("No next sample available")
            return False
        
        try:
            # Find the sample directory
            sample_dir = find_sample_directory(self.base_dir, next_sequence, next_sample_id, self.padding_length, raw_samples=self.raw_samples, directory_prefixes=self.directory_prefixes)
            
            # Update navigation state (store unpadded version for consistency)
            self.current_sequence = next_sequence
            self.current_sample_id = next_sample_id
            
            # Clear current visualization
            if self.vis is not None:
                # Clear all existing geometries (including point clouds and coordinate frame)
                self.vis.clear_geometries()
            
            # Load new sample data
            point_clouds, data_for_colors, part_names, normals = self.load_sample_data(sample_dir, center_pcds=True)
            
            # Compute PCA colors or use part_ids for coloring
            if self.raw_samples:
                pca_colors = [] # Not used in raw mode, but argument required for type consistency if we keep the signature
                colored_pcds = self.create_colored_point_clouds(point_clouds, data_for_colors, part_names, normals) # data_for_colors is part_ids here
            else:
                pca_colors = self.apply_global_pca_colors(data_for_colors) if self.use_global_pca else self.compute_pca_colors(data_for_colors)
                colored_pcds = self.create_colored_point_clouds(point_clouds, pca_colors, part_names, normals)
            
            # Add new geometries to visualizer
            if self.vis is not None:
                for pcd in colored_pcds:
                    self.vis.add_geometry(pcd, False) # Keep False for efficiency in loop
                    self.vis.update_geometry(pcd) # Explicitly update geometry
                
                # Add coordinate frame if enabled (it was cleared by clear_geometries)
                if not self.no_coordinate_frame:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                    self.vis.add_geometry(coordinate_frame, False)
                    self.vis.update_geometry(coordinate_frame) # Explicitly update geometry
                
                # Update view
                self._center_and_fit_view()
                self.vis.update_renderer()
                
            logger.info("=" * 60)
            # Display padded sample ID for consistency
            if self.raw_samples:
                logged_sample_id = next_sample_id
            else:
                logged_sample_id = next_sample_id.zfill(self.padding_length) if next_sample_id else "N/A"
            logger.info(f"LOADED NEXT SAMPLE: Sequence {next_sequence}, Sample {logged_sample_id}")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load next sample: {e}")
            return False
    
    def center_point_clouds(self, point_clouds: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Center all point clouds at the origin by subtracting their mean.
        
        Args:
            point_clouds: List of point cloud arrays
            
        Returns:
            Tuple of (centered_point_clouds, center_offset)
        """
        if not point_clouds:
            return point_clouds, np.zeros(3)
        
        # Concatenate all point clouds to find the global center
        all_points = np.concatenate(point_clouds, axis=0)
        center = np.mean(all_points, axis=0)
        
        # Center each point cloud
        centered_point_clouds = []
        for points in point_clouds:
            centered_points = points - center
            centered_point_clouds.append(centered_points)
        
        logger.info(f"Centered all point clouds at origin. Global center was: {center}")
        return centered_point_clouds, center
    
    def _center_and_fit_view(self):
        """Center and fit the view to show all point clouds properly."""
        if not self.colored_pcds:
            return
        
        # Get view control
        view_control = self.vis.get_view_control()
        
        # Calculate bounding box of all point clouds
        all_points = []
        for pcd in self.colored_pcds:
            points = np.asarray(pcd.points)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return
        
        # Concatenate all points to get overall bounding box
        all_points = np.concatenate(all_points, axis=0)
        
        # Calculate center and extent
        center = np.mean(all_points, axis=0)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        
        # Set camera parameters for a good view
        # Position camera at a distance proportional to the point cloud size
        camera_distance = max_extent * (2.0 if self.raw_samples else 1.0)  # Adjust this multiplier as needed
        
        # Set up a bird's eye view (top-down) by default
        # You can modify these angles for different viewing angles
        front = np.array([0.0, 0.0, 1.0])  # Looking down
        up = np.array([0.0, 1.0, 0.0])      # Y-axis up
        lookat = center                      # Look at the center of the point cloud
        
       
        # Set the view
        view_control.set_front(front)
        view_control.set_lookat(lookat)
        view_control.set_up(up)
        view_control.set_zoom(0.8)  # Adjust zoom level
        
        logger.info(f"Centered view on point cloud center: {center}")
        logger.info(f"Point cloud extent: {extent}, max extent: {max_extent}")


def find_sample_directory(base_dir: str, sequence: str, sample_id: str, padding_length: int = 6, raw_samples: bool = False, directory_prefixes: List[str] = None) -> str:
    """
    Find the processed sample directory.
    
    Args:
        base_dir: Base directory containing processed samples
        sequence: Sequence name (e.g., "00" or "bundlefusion-office0/seq-01")
        sample_id: Sample ID (e.g., "123" or "000123")
        padding_length: Length to pad sample ID to (default: 6)
        raw_samples: Whether to look for raw samples (without _processed suffix)
        directory_prefixes: List of directory name prefixes to try (default: ['sample', 'fracture', 'part'])
        
    Returns:
        Path to the sample directory
    """
    if directory_prefixes is None:
        directory_prefixes = ['sample', 'fracture', 'part']
    
    # Ensure sample_id is properly padded
    if sample_id.isdigit() and not raw_samples:
        # Only pad if the sample_id has fewer digits than padding_length
        # If it already has the correct number of digits (or more), use it as-is
        if len(sample_id) < padding_length:
            padded_sample_id = sample_id.zfill(padding_length)
        else:
            # Sample ID already has correct padding, use as-is
            padded_sample_id = sample_id
    else:
        padded_sample_id = sample_id
    
    suffix = '_processed' if not raw_samples else ''
    
    # Try different possible directory structures with all prefixes
    possible_paths = []
    for prefix in directory_prefixes:
        # Direct structure: base_dir/sequence/{prefix}_XXXXXX{_processed}
        possible_paths.append(
            os.path.join(base_dir, sequence, f"{prefix}_{padded_sample_id}{suffix}")
        )
        # Dataset structure: base_dir/dataset_name/sequence/{prefix}_XXXXXX{_processed}
        possible_paths.append(
            os.path.join(base_dir, "*", sequence, f"{prefix}_{padded_sample_id}{suffix}")
        )
    
    # Log attempted paths for debugging
    logger.debug(f"Searching for sample directory with sample_id='{sample_id}' -> padded_sample_id='{padded_sample_id}' (padding_length={padding_length})")
    
    for pattern in possible_paths:
        if "*" in pattern:
            # Use glob for wildcard patterns
            matches = glob.glob(pattern)
            if matches:
                logger.debug(f"Found sample directory via glob pattern: {matches[0]}")
                return matches[0]  # Return first match
        else:
            if os.path.exists(pattern):
                logger.debug(f"Found sample directory: {pattern}")
                return pattern
    
    # Log all attempted paths for better error messages
    logger.error(f"Sample directory not found. Attempted paths:")
    for path in possible_paths:
        logger.error(f"  - {path}")
    raise FileNotFoundError(f"Sample directory not found for sequence {sequence}, sample {padded_sample_id} (original: {sample_id}) in {base_dir}. Tried prefixes: {directory_prefixes}")


def main():
    parser = argparse.ArgumentParser(description='Visualize processed training samples with PCA-colorized features')
    
    # Input arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing processed samples')
    parser.add_argument('--sequence', '-s', type=str, default=None,
                        help='Sequence name (e.g., "00"). If not specified, a random sequence will be selected.')
    parser.add_argument('--sample_id', '--sample', type=str, default="0",
                        help='Sample ID (e.g., "123" or "000123") - will be padded automatically')
    
    # Formatting options
    parser.add_argument('--padding_length', type=int, default=None,
                        help='Length to pad sample ID to (default: auto-detect based on dataset name, 1 for modelnet, 6 for others)')
    
    # Visualization options
    parser.add_argument('--point_size', '-p', type=float, default=4.0,
                        help='Point size in visualization (default: 4.0)')
    parser.add_argument('--background_color', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Background color as R G B values in [0,1] (default: 1.0 1.0 1.0 - white)')
    parser.add_argument('--no_coordinate_frame', action='store_true', default=True, 
                        help='Hide coordinate frame')
    parser.add_argument('--save_screenshot', type=str, default=None,
                        help='Save screenshot to specified path')
    parser.add_argument('--window_name', type=str, default=None,
                        help='Custom window name')
    parser.add_argument('--color_mode', type=str, default='pca', choices=['pca', 'part'],
                        help='Initial color mode: pca (feature-based) or part (index-based) (default: pca)')
    parser.add_argument('--no_center', action='store_true',
                        help='Do not center point clouds at origin (keep original positions)')
    parser.add_argument('--raw_samples', '-r', action='store_true', default=False,
                        help='Visualize raw samples (before feature extraction), only showing parts colored by index')
    parser.add_argument('--estimate_normals', '-n', action='store_true', default=False,
                        help='Estimate normals for point clouds that don\'t have them')
    parser.add_argument('--normal_estimation_radius', type=float, default=0.5,
                        help='Radius for normal estimation (default: 0.1)')
    
    # PCA options
    parser.add_argument('--pca_components', type=int, default=3,
                        help='Number of PCA components (default: 3 for RGB)')
    parser.add_argument('--global_pca', action='store_true', default=True,
                        help='Compute global PCA on all samples for consistent coloring across samples')
    parser.add_argument('--max_pca_samples', type=int, default=50,
                        help='Maximum number of samples to use for global PCA computation (default: 50)')
    
    # Utility arguments
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return
    
    try:
        # Create visualizer first to access sequence discovery methods
        visualizer = SampleVisualizer(
            pca_components=args.pca_components,
            point_size=args.point_size,
            background_color=args.background_color,
            no_coordinate_frame=args.no_coordinate_frame,
            raw_samples=args.raw_samples,
            estimate_normals=args.estimate_normals,
            normal_estimation_radius=args.normal_estimation_radius
        )
        
        # Set navigation state temporarily to access sequence discovery
        # Use provided padding_length or None for auto-detection
        visualizer.set_navigation_state(args.input, "temp", "0", args.padding_length if args.padding_length is not None else 6)
        
        # If sequence not specified, select a random one
        if args.sequence is None:
            available_sequences = visualizer.get_available_sequences()
            if not available_sequences:
                logger.error(f"No sequences found in input directory: {args.input}")
                return
            
            import random
            args.sequence = random.choice(available_sequences)
            logger.info(f"No sequence specified, randomly selected: {args.sequence}")
        
        # Auto-detect padding_length if not explicitly provided (after sequence is determined)
        if args.padding_length is None:
            args.padding_length = visualizer._determine_padding_length(args.input, args.sequence)
            dataset_name = visualizer._detect_dataset_name(args.input, args.sequence)
            if dataset_name:
                logger.info(f"Auto-detected padding_length={args.padding_length} for dataset '{dataset_name}'")
            else:
                logger.info(f"Auto-detected padding_length={args.padding_length} (default)")
        
        # Get available samples for the selected sequence
        available_samples = visualizer.get_sequence_samples(args.sequence)
        if not available_samples:
            logger.error(f"No samples found in sequence {args.sequence} in directory: {args.input}")
            return

        # If sample_id is default "0" or not found, use the first available sample
        if args.sample_id == "0" or args.sample_id not in available_samples:
            if args.sample_id == "0":
                logger.info(f"No sample ID specified (default '0'), using first sample in sequence: {available_samples[0]}")
            else:
                logger.warning(f"Specified sample ID '{args.sample_id}' not found in sequence '{args.sequence}', using first sample: {available_samples[0]}")
            args.sample_id = available_samples[0]

        # Find sample directory (with automatic padding)
        sample_dir = find_sample_directory(args.input, args.sequence, args.sample_id, args.padding_length, raw_samples=args.raw_samples, directory_prefixes=visualizer.directory_prefixes)
        logger.info(f"Found sample directory: {sample_dir}")
        
        # Log the padded sample ID for user reference
        if args.sample_id.isdigit() and not args.raw_samples:
            padded_id = args.sample_id.zfill(args.padding_length)
            logger.info(f"Using padded sample ID: {args.sample_id} -> {padded_id}")
        elif args.raw_samples:
            logger.info(f"Using raw sample ID: {args.sample_id} (not padded in raw mode)")
        
        # Log initial sample information
        logger.info("=" * 60)
        logger.info(f"STARTING VISUALIZATION:")
        logger.info(f"  Input Directory: {args.input}")
        logger.info(f"  Sequence: {args.sequence}")
        logger.info(f"  Sample ID: {args.sample_id} (padded: {args.sample_id.zfill(args.padding_length)})")
        logger.info(f"  Sample Directory: {os.path.basename(sample_dir)}")
        logger.info("=" * 60)
        
        # Set initial color mode based on command line argument
        if args.color_mode == 'pca':
            visualizer.use_pca_colors = True
            logger.info("Initial color mode: PCA-based (feature-based)")
        else:
            visualizer.use_pca_colors = False
            logger.info("Initial color mode: Part index-based (distinct per part)")
        
        # If raw samples, force part index color mode
        if args.raw_samples:
            visualizer.use_pca_colors = False
            logger.info("Raw samples mode: Forcing initial color mode to part index-based.")

        # Set navigation state
        visualizer.set_navigation_state(args.input, args.sequence, args.sample_id, args.padding_length)
        
        # Compute global PCA if requested
        if args.global_pca and not args.raw_samples:
            logger.info("Global PCA requested - computing on all available samples...")
            success = visualizer.compute_global_pca(max_samples=args.max_pca_samples)
            if success:
                logger.info("Global PCA computed successfully - colors will be consistent across samples")
            else:
                logger.warning("Failed to compute global PCA - falling back to local PCA per sample")
        elif args.global_pca and args.raw_samples:
            logger.info("Global PCA skipped: Raw samples mode does not require feature-based coloring.")

        # Visualize sample
        visualizer.visualize_sample(
            sample_dir=sample_dir,
            window_name=args.window_name,
            show_coordinate_frame=not args.no_coordinate_frame,
            save_screenshot=args.save_screenshot,
            center_pcds=not args.no_center,
            raw_samples=args.raw_samples
        )
        
        logger.info("Visualization complete!")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 