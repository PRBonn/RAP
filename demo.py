#!/usr/bin/env python3
"""
Demo script for RAP inference on a folder of PLY point clouds.

This script:
1. Loads PLY point clouds from a folder
2. Applies voxel downsampling to each point cloud
3. Performs keypoint sampling and feature extraction (with outlier removal)
4. Runs RAP inference using the processed data
"""

import os
import sys
import argparse
import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import glob
import copy
from tqdm import tqdm
import urllib.request
import zipfile
import time
from natsort import natsorted
import torch

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'dataset_process'))

# Import required modules
from dataset_process.utils import dataset_utils
from dataset_process.utils.io_utils import CMAP_DEFAULT
from dataset_process.extract_sample_features import FeatureExtractor, SampleProcessor
from dataset_process.utils.processing_utils import set_random_seeds
from dataset_process.utils.io_utils import save_processed_sample
import hydra

logger = logging.getLogger(__name__)


def get_time():
    """
    :return: get timing statistics with GPU synchronization
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:  # issue #10
        torch.cuda.synchronize()
    return time.time()

# Coordinate frame transformation matrix (for 7-scenes, bundlefusion, rgbd-scenes)
COORDINATE_TRANSFORM = np.array([[0, 0, 1],
                                 [-1, 0, 0],
                                 [0, -1, 0]], dtype=np.float32)


def download_and_extract_weights(weights_url: str = "https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip",
                                 extract_to: Optional[str] = None) -> bool:
    """
    Download and extract weights zip file if checkpoint files are missing.
    
    Args:
        weights_url: URL to download weights zip file
        extract_to: Directory to extract to (default: current folder)
        
    Returns:
        True if successful, False otherwise
    """
    if extract_to is None:
        extract_to = current_dir
    
    weights_zip_path = os.path.join(extract_to, "weights.zip")
    
    try:
        logger.info(f"Downloading weights from {weights_url}...")
        logger.info("This may take a few minutes depending on your internet connection.")
        
        # Download with progress bar
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(weights_url, weights_zip_path, show_progress)
        sys.stdout.write("\n")
        logger.info("Download completed.")
        
        # Extract zip file
        logger.info(f"Extracting weights.zip to {extract_to}...")
        with zipfile.ZipFile(weights_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.info("Extraction completed.")
        
        # Clean up zip file
        if os.path.exists(weights_zip_path):
            os.remove(weights_zip_path)
            logger.info("Removed temporary weights.zip file.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download/extract weights: {e}")
        # Clean up partial download if exists
        if os.path.exists(weights_zip_path):
            try:
                os.remove(weights_zip_path)
            except:
                pass
        return False


def load_ply_file(ply_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a PLY file and return points and normals.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Tuple of (points, normals) where normals can be None
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    return points, normals

def process_point_clouds(loaded_point_clouds: List[Tuple[str, np.ndarray, Optional[np.ndarray]]],
                        output_folder: str,
                        voxel_size: float = 0.25,
                        feature_extractor: Optional[FeatureExtractor] = None,
                        sample_processor: Optional[SampleProcessor] = None,
                        checkpoint_path: str = './weights/spinnet_3dmatch_bufferx.pth',
                        des_r: float = 5.0,
                        is_aligned_to_global_z: bool = True,
                        remove_outliers: bool = True,
                        outlier_nb_neighbors: int = 20,
                        outlier_std_ratio: float = 2.5,
                        allocation_method: str = 'voxel_adaptive',
                        voxel_ratio: float = 0.05,
                        min_points_per_part: int = 200,
                        max_points_per_part: int = 20000,
                        global_seed: int = 42,
                        use_torch_downsampling: bool = True) -> str:
    """
    Process loaded point clouds: downsample, extract features, and save.
    
    Args:
        loaded_point_clouds: List of tuples (part_name, points, normals) where:
            - part_name: Name of the part/point cloud
            - points: numpy array of shape (N, 3)
            - normals: numpy array of shape (N, 3) or None
        output_folder: Output folder for processed data
        voxel_size: Voxel size for downsampling (default: 0.25m)
        feature_extractor: FeatureExtractor instance (created if None)
        sample_processor: SampleProcessor instance (created if None)
        checkpoint_path: Path to miniSpinNet checkpoint
        des_r: Description radius for miniSpinNet
        is_aligned_to_global_z: Whether to align to global Z axis
        remove_outliers: Whether to remove outliers
        outlier_nb_neighbors: Number of neighbors for outlier removal
        outlier_std_ratio: Standard deviation ratio for outlier removal
        allocation_method: Method for allocating FPS points
        voxel_ratio: Ratio for voxel_adaptive allocation
        min_points_per_part: Minimum points per part
        max_points_per_part: Maximum points per part
        global_seed: Random seed
        use_torch_downsampling: Use torch-based voxel downsampling for speedup
        
    Returns:
        Path to the created sample folder
    """
    if not loaded_point_clouds:
        raise ValueError("No point clouds provided")
    
    logger.info(f"Processing {len(loaded_point_clouds)} point clouds...")
    
    # Initialize feature extractor if not provided
    if feature_extractor is None:
        model_config = {
            'num_points_per_patch': 512,
            'is_aligned_to_global_z': is_aligned_to_global_z,
            'des_r': des_r,
        }
        feature_extractor = FeatureExtractor(
            model_config=model_config,
            des_r=des_r,
            is_aligned_to_global_z=is_aligned_to_global_z,
            checkpoint_path=checkpoint_path,
            device='auto'
        )
    
    allocated_voxel_size = 4 * voxel_size

    # Initialize sample processor if not provided
    if sample_processor is None:
        sample_processor = SampleProcessor(
            feature_extractor=feature_extractor,
            num_points=5000,  # Default, will be overridden by allocation_method
            skip_point_sampling=False,
            remove_outliers=remove_outliers,
            outlier_nb_neighbors=outlier_nb_neighbors,
            outlier_std_ratio=outlier_std_ratio,
            min_points_per_part=min_points_per_part,
            max_points_per_part=max_points_per_part,
            global_seed=global_seed,
            allocation_method=allocation_method,
            voxel_size=allocated_voxel_size,
            voxel_ratio=voxel_ratio,
        )
    
    # Create output sample folder
    sample_name = "demo_sample"
    sample_output_dir = os.path.join(output_folder, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Process each loaded point cloud
    parts_points = []
    parts_normals = []
    part_names = []
    
    logger.info("Downsampling point clouds...")
    start_time_downsampling = get_time()
    
    for part_name, points, normals in loaded_point_clouds:
        if len(points) == 0:
            logger.warning(f"Skipping empty point cloud: {part_name}")
            continue
        
        # Apply voxel downsampling
        downsampled_points, downsampled_normals = dataset_utils.downsample_points(
            points=points,
            normals=normals,
            method="voxel",
            voxel_size=voxel_size,
            use_torch=use_torch_downsampling
        )
        
        logger.info(f"  {part_name}: {len(points)} -> {len(downsampled_points)} points (voxel_size={voxel_size}m)")
        
        parts_points.append(downsampled_points)
        parts_normals.append(downsampled_normals)
        part_names.append(part_name)
    
    elapsed_time_downsampling = get_time() - start_time_downsampling
    logger.info(f"Voxel downsampling time: {elapsed_time_downsampling:.2f} seconds")
    
    if not parts_points:
        raise ValueError("No valid point clouds found after processing")
    
    logger.info(f"Processing {len(parts_points)} parts for feature extraction...")
    
    # Process sample using SampleProcessor (includes FPS and feature extraction)
    start_time_sample_processor = get_time()
    
    part_results = sample_processor.process_sample(parts_points, parts_normals)
    
    elapsed_time_sample_processor = get_time() - start_time_sample_processor
    logger.info(f"Sample processor time (FPS + feature extraction): {elapsed_time_sample_processor:.2f} seconds")
    
    if not part_results:
        raise ValueError("No parts returned from processing")
    
    # Log point counts for each sampled point cloud
    logger.info("Sampled point cloud statistics:")
    total_sampled_points = 0
    for i, (part_result, part_name) in enumerate(zip(part_results, part_names)):
        num_points = len(part_result['sampled_points'])
        total_sampled_points += num_points
        logger.info(f"  Part {i+1} ({part_name}): {num_points} points")
    logger.info(f"Total sampled points across all parts: {total_sampled_points}")
    
    # Save processed sample
    save_processed_sample(part_results, part_names, sample_output_dir)
    
    logger.info(f"Processed sample saved to: {sample_output_dir}")
    
    return sample_output_dir


def visualize_with_toggle(original_pcds: List[o3d.geometry.PointCloud],
                          registered_pcds: List[o3d.geometry.PointCloud],
                          point_size: float = 3.0,
                          background_color: List[float] = [1.0, 1.0, 1.0],
                          show_coordinate_frame: bool = False,
                          show_normals: bool = False):
    """
    Visualize point clouds with toggle between original and registered views.
    
    Press 'T' or 't' to toggle between original and registered point clouds.
    Press 'Q' or ESC to quit.
    
    Note: Normals should be computed on original_pcds before transformation.
          They will be automatically transformed when creating registered_pcds.
          Normals are saved with point clouds but not displayed as lines by default.
    
    Args:
        original_pcds: List of original point clouds (with normals if show_normals=True)
        registered_pcds: List of registered point clouds (normals inherited from original)
        point_size: Size of points in visualization
        background_color: Background color as [R, G, B] in [0, 1]
        show_coordinate_frame: Whether to show coordinate frame
        show_normals: Whether normals are available (they will be saved but not displayed as lines)
    """
    if not original_pcds or not registered_pcds:
        logger.warning("No point clouds to visualize")
        return
    
    # Check if normals are available when requested
    if show_normals:
        has_normals = all(pcd.has_normals() for pcd in original_pcds)
        if not has_normals:
            logger.warning("Normals requested but not all point clouds have normals computed.")
            show_normals = False
        else:
            logger.info("Normals are computed and saved with point clouds (not displayed as lines in visualization)")
    
    # State for toggle
    class VisualizationState:
        def __init__(self):
            self.show_registered = False
            self.vis = None
    
    state = VisualizationState()
    
    def toggle_view(vis):
        """Toggle between original and registered point clouds"""
        state.show_registered = not state.show_registered
        
        # Clear all geometries
        vis.clear_geometries()
        
        # Add coordinate frame if requested
        if show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            vis.add_geometry(coord_frame, reset_bounding_box=False)
        
        # Add appropriate point clouds
        if state.show_registered:
            logger.info("Showing REGISTERED point clouds")
            for pcd in registered_pcds:
                vis.add_geometry(pcd, reset_bounding_box=False)
        else:
            logger.info("Showing ORIGINAL point clouds")
            for pcd in original_pcds:
                vis.add_geometry(pcd, reset_bounding_box=False)
        
        # Update rendering
        vis.update_renderer()
        return False
    
    def key_callback(vis):
        """Handle key press events"""
        # Toggle on 'T' key
        toggle_view(vis)
        return False
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    state.vis = vis
    vis.create_window(window_name="Point Cloud Visualization - Press 'T' to toggle", 
                      width=1920, height=1080)
    
    # Register key callback for 'T' key
    vis.register_key_callback(ord('T'), key_callback)
    vis.register_key_callback(ord('t'), key_callback)  # Also handle lowercase
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray(background_color)
    
    # Note: Normals are computed when show_normals=True, but not displayed as lines by default
    # Uncomment the following lines to visualize normals as lines:
    # if show_normals:
    #     render_option.point_show_normal = True
    #     logger.info("Normal visualization enabled")
    
    # Add coordinate frame if requested
    if show_coordinate_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)
    
    # Initially show original point clouds
    logger.info("Showing ORIGINAL point clouds")
    logger.info("Press 'T' to toggle between original and registered views")
    logger.info("Press 'Q' or ESC to quit")
    
    for pcd in original_pcds:
        vis.add_geometry(pcd)
    
    # Set view control
    view_control = vis.get_view_control()
    
    # Run visualization
    vis.run()
    vis.destroy_window()
    
    logger.info("Visualization closed")


def main():
    parser = argparse.ArgumentParser(description='Demo script for RAP inference on PLY point clouds')
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder containing PLY point cloud files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output folder for processed data (default: temporary directory)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory for inference results (default: output/logs)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (default: None, will use the final directory name of input path)')
    parser.add_argument('--point_cloud_count', '-k', type=int, default=None,
                        help='Limit the number of point clouds to process (default: None, process all)')

    parser.add_argument('--apply_coordinate_transform', action='store_true', default=False,
                        help='Apply coordinate frame transformation (rotation matrix for certain rgb-d data, when the point cloud is in the camera frame, for example 3dmatch test data)')

    parser.add_argument('--adaptive_parameters', '-a', action='store_true', default=False,
                        help='Use adaptive parameters for pre-processing')
    
    # Voxel downsampling
    parser.add_argument('--voxel_size', type=float, default=0.25,
                        help='Voxel size for downsampling in meters (default: 0.25)')
    
    # Feature extraction
    parser.add_argument('--feature_extraction_checkpoint', type=str, default='./weights/mini_spinnet_t.pth',
                        help='Path to miniSpinNet checkpoint')
    parser.add_argument('--des_r', type=float, default=5.0,
                        help='Description radius for miniSpinNet in meters (default: 5.0)')
    parser.add_argument('--is_aligned_to_global_z', action='store_true', default=True,
                        help='Align point clouds to global Z axis (default: True)')
    parser.add_argument('--no_is_aligned_to_global_z', dest='is_aligned_to_global_z', action='store_false',
                        help='Do not align point clouds to global Z axis')
    
    # Outlier removal
    parser.add_argument('--remove_outliers', action='store_true', default=True,
                        help='Remove statistical outliers (default: True)')
    parser.add_argument('--no_remove_outliers', dest='remove_outliers', action='store_false',
                        help='Disable outlier removal')
    parser.add_argument('--outlier_nb_neighbors', type=int, default=20,
                        help='Number of neighbors for outlier removal (default: 20)')
    parser.add_argument('--outlier_std_ratio', type=float, default=2.5,
                        help='Standard deviation ratio for outlier removal (default: 2.5)')
    
    # Sampling parameters
    parser.add_argument('--allocation_method', type=str, default='voxel_adaptive',
                        choices=['point_count', 'spatial_coverage', 'voxel_adaptive'],
                        help='Method for allocating FPS points (default: voxel_adaptive)')
    parser.add_argument('--voxel_ratio', '-r', type=float, default=0.05,
                        help='Ratio for voxel_adaptive allocation (default: 0.1)')
    parser.add_argument('--min_points_per_part', type=int, default=200,
                        help='Minimum points per part (default: 200)')
    parser.add_argument('--max_points_per_part', type=int, default=20000,
                        help='Maximum points per part (default: 20000)')
    parser.add_argument('--global_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Inference
    parser.add_argument('--flow_model_checkpoint', type=str, default='./weights/rap_model.ckpt',
                        help='Path to PRFM checkpoint')
    parser.add_argument('--config', type=str, default='RAP_inference',
                        help='Config name for inference (default: RAP_inference)')
    parser.add_argument('--rigidity_forcing', action='store_true', default=True,
                        help='Enable rigidity forcing in flow model (default: True)')
    parser.add_argument('--no_rigidity_forcing', dest='rigidity_forcing', action='store_false',
                        help='Disable rigidity forcing in flow model')
    parser.add_argument('--n_generations', type=int, default=1,
                        help='Number of generations for flow matching (default: 1)')
    parser.add_argument('--inference_sampling_steps', type=int, default=10,
                        help='Number of inference sampling steps for flow matching (default: 10)')
    parser.add_argument('--skip_inference', action='store_true', default=False,
                        help='Skip inference and only process point clouds')
    parser.add_argument('--visualize', '-v', action='store_true', default=False,
                        help='Visualize registered point clouds after inference (requires inference to complete)')
    parser.add_argument('--point_size', type=float, default=3.0,
                        help='Point size for visualization (default: 3.0)')
    parser.add_argument('--background_color', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Background color as R G B values in [0,1] (default: 1.0 1.0 1.0 - white)')
    parser.add_argument('--show_coordinate_frame', action='store_true', default=False,
                        help='Show coordinate frame in visualization')
    parser.add_argument('-n', '--show_normals', action='store_true', default=False,
                        help='Compute normals for point clouds (normals will be saved but not displayed as lines in visualization)')
    parser.add_argument('--generation', type=str, default="generation_selected",
                        help='Generation to visualize (e.g., generation00, generation_selected). Default: generation_selected')
    parser.add_argument('--eval_on', action='store_true', default=False,
                        help='Evaluate on the registered point clouds (default: False)')
    parser.add_argument('--no_cleanup', dest='cleanup', action='store_false', default=True,
                        help='Do not clean up temporary dataset folder after processing')
    parser.add_argument('--save_trajectory', action='store_true', default=False,
                        help='Save trajectory as GIF animation (default: False)')

    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if checkpoint files exist, download weights if missing
    flow_checkpoint_exists = os.path.exists(args.flow_model_checkpoint)
    feature_checkpoint_exists = os.path.exists(args.feature_extraction_checkpoint)
    
    if not flow_checkpoint_exists or not feature_checkpoint_exists:
        logger.info("=" * 60)
        logger.info("Checkpoint files missing - downloading weights")
        logger.info("=" * 60)
        
        missing_files = []
        if not flow_checkpoint_exists:
            missing_files.append(f"Flow model checkpoint: {args.flow_model_checkpoint}")
        if not feature_checkpoint_exists:
            missing_files.append(f"Feature extraction checkpoint: {args.feature_extraction_checkpoint}")
        
        logger.info("Missing checkpoint files:")
        for missing_file in missing_files:
            logger.info(f"  - {missing_file}")
        
        # Ensure weights directory exists
        # Determine weights directory from checkpoint paths
        flow_checkpoint_dir = os.path.dirname(os.path.abspath(args.flow_model_checkpoint))
        feature_checkpoint_dir = os.path.dirname(os.path.abspath(args.feature_extraction_checkpoint))
        
        # Use the common directory or default to ./weights
        if flow_checkpoint_dir == feature_checkpoint_dir:
            weights_dir = flow_checkpoint_dir
        else:
            # If different directories, use the first one or default
            weights_dir = flow_checkpoint_dir if os.path.basename(flow_checkpoint_dir) == 'weights' else './weights'
        
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            logger.info(f"Created weights directory: {weights_dir}")
        
        # Download and extract weights
        if not download_and_extract_weights():
            logger.error("Failed to download weights. Please download manually from:")
            logger.error("  https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip")
            logger.error("  and extract to the current folder.")
            return 1
        
        # Verify checkpoint files exist after extraction
        flow_checkpoint_exists = os.path.exists(args.flow_model_checkpoint)
        feature_checkpoint_exists = os.path.exists(args.feature_extraction_checkpoint)
        
        if not flow_checkpoint_exists or not feature_checkpoint_exists:
            logger.warning("Some checkpoint files are still missing after extraction.")
            if not flow_checkpoint_exists:
                logger.warning(f"  Flow model checkpoint not found: {args.flow_model_checkpoint}")
            if not feature_checkpoint_exists:
                logger.warning(f"  Feature extraction checkpoint not found: {args.feature_extraction_checkpoint}")
            logger.warning("Please check the weights directory and ensure files are correctly extracted.")
        else:
            logger.info("=" * 60)
            logger.info("All checkpoint files are now available")
            logger.info("=" * 60)
    
    # Set random seeds
    set_random_seeds(args.global_seed)
    
    # Validate input folder
    if not os.path.isdir(args.input):
        logger.error(f"Input folder does not exist: {args.input}")
        return 1
    
    # Set dataset name: use provided name or derive from input path
    if args.dataset_name is None:
        # Get the final directory name from input path
        # Normalize path to handle trailing slashes
        input_path = os.path.normpath(args.input)
        dataset_name = os.path.basename(input_path)
        if not dataset_name:  # Handle case where input is root directory
            dataset_name = "demo_dataset"
        logger.info(f"Using dataset name from input path: {dataset_name}")
    else:
        dataset_name = args.dataset_name
        logger.info(f"Using provided dataset name: {dataset_name}")
    
    # Load all point clouds once
    logger.info("=" * 60)
    logger.info("Loading point clouds")
    logger.info("=" * 60)
    
    # Get all PLY files
    ply_files = natsorted([f for f in os.listdir(args.input) if f.endswith('.ply')])
    
    if not ply_files:
        logger.error(f"No PLY files found in {args.input}")
        return 1
    
    # Limit to first K point clouds if specified
    if args.point_cloud_count is not None:
        if args.point_cloud_count <= 0:
            logger.error(f"point_cloud_count must be positive, got {args.point_cloud_count}")
            return 1
        original_count = len(ply_files)
        ply_files = ply_files[:args.point_cloud_count]
        logger.info(f"Found {original_count} PLY files, limiting to first {len(ply_files)} files")
    else:
        logger.info(f"Found {len(ply_files)} PLY files")
    
    if args.apply_coordinate_transform:
        logger.info("Coordinate frame transformation will be applied to all point clouds")
    
    # Load point clouds and prepare for processing
    logger.info("Starting point cloud loading...")
    start_time_loading = get_time()
    
    loaded_point_clouds = []  # For processing: List[(part_name, points, normals)]
    visualization_pcds = []  # For visualization: List[o3d.geometry.PointCloud]
    bbox_dimensions = []
    
    for ply_file in ply_files:
        ply_path = os.path.join(args.input, ply_file)
        part_name = os.path.splitext(ply_file)[0]
        
        # Load point cloud
        points, normals = load_ply_file(ply_path)
        
        if len(points) == 0:
            logger.warning(f"Skipping empty point cloud: {ply_file}")
            continue
        
        # Apply coordinate frame transformation if requested
        if args.apply_coordinate_transform:
            # Apply rotation matrix to points: points @ R.T
            points = points @ COORDINATE_TRANSFORM.T
            # Apply same rotation to normals if they exist
            if normals is not None:
                normals = normals @ COORDINATE_TRANSFORM.T
        
        # Store loaded point cloud for processing
        loaded_point_clouds.append((part_name, points, normals))
        
        # Create Open3D point cloud for visualization (deep copy)
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points.copy())
        if normals is not None:
            pcd_vis.normals = o3d.utility.Vector3dVector(normals.copy())
        
        # Add colors for consistent visualization
        idx = len(visualization_pcds)
        rgb = CMAP_DEFAULT[idx % len(CMAP_DEFAULT)]
        pcd_vis.paint_uniform_color(rgb)
        
        visualization_pcds.append(pcd_vis)
        
        # Calculate bounding box for adaptive parameters if needed
        if args.adaptive_parameters:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()  # Returns [x_size, y_size, z_size]
            bbox_dimensions.append(extent)
            logger.debug(f"  {ply_file}: {len(points)} points, bbox extent = [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
        else:
            logger.debug(f"  {ply_file}: {len(points)} points")
    
    if not loaded_point_clouds:
        logger.error("No valid point clouds found")
        return 1
    
    elapsed_time_loading = get_time() - start_time_loading
    logger.info(f"Successfully loaded {len(loaded_point_clouds)} point clouds")
    logger.info(f"Point cloud loading time: {elapsed_time_loading:.2f} seconds")
    
    # Adaptive parameter setting based on point cloud size
    if args.adaptive_parameters:
        logger.info("=" * 60)
        logger.info("Analyzing point clouds for adaptive parameter setting")
        logger.info("=" * 60)
        
        if not bbox_dimensions:
            logger.error("No valid point clouds found for adaptive parameter analysis")
            return 1
        
        # Calculate median of x, y, z dimensions over all point clouds
        bbox_dimensions = np.array(bbox_dimensions)  # Shape: (n_clouds, 3)
        median_x = np.median(bbox_dimensions[:, 0])
        median_y = np.median(bbox_dimensions[:, 1])
        median_z = np.median(bbox_dimensions[:, 2])
        median_size = np.median([median_x, median_y, median_z])
        
        logger.info(f"Median bounding box dimensions: x={median_x:.3f}, y={median_y:.3f}, z={median_z:.3f}")
        logger.info(f"Median size (across x,y,z): {median_size:.3f}")
        
        # Set adaptive voxel_size: median / 500, bounded by [0.01, 0.6]
        adaptive_voxel_size = median_size / 500.0
        adaptive_voxel_size = max(0.01, min(0.6, adaptive_voxel_size))
        
        # Set adaptive des_r: 20 * voxel_size
        adaptive_des_r = 20.0 * adaptive_voxel_size
        
        # Override parameters
        args.voxel_size = adaptive_voxel_size
        args.des_r = adaptive_des_r
        
        logger.info("=" * 60)
        logger.info("Adaptively set parameters:")
        logger.info(f"  voxel_size = {adaptive_voxel_size:.6f}")
        logger.info(f"  des_r = {adaptive_des_r:.6f}")
        logger.info("=" * 60)
    
    # Setup output folder (default to current directory)
    if args.output is None:
        output_folder = os.path.join(os.getcwd(), 'demo_output')
        logger.info(f"Using default output directory: {output_folder}")
    else:
        output_folder = args.output
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup log directory
    if args.log_dir is None:
        log_dir = os.path.join(output_folder, 'logs')
    else:
        log_dir = args.log_dir
    
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Process point clouds
        logger.info("=" * 60)
        logger.info("STEP 1: Processing point clouds")
        logger.info("=" * 60)
        
        # Create data_root structure expected by RAP
        # The data module expects: data_root/dataset_name/sample_name/
        data_root = output_folder
        dataset_folder = os.path.join(data_root, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        logger.info("Starting preprocessing (point sampling and feature extraction)...")
        start_time_preprocessing = get_time()
        
        sample_output_dir = process_point_clouds(
            loaded_point_clouds=loaded_point_clouds,
            output_folder=dataset_folder,  # Save directly to dataset folder
            voxel_size=args.voxel_size,
            checkpoint_path=args.feature_extraction_checkpoint,
            des_r=args.des_r,
            is_aligned_to_global_z=args.is_aligned_to_global_z,
            remove_outliers=args.remove_outliers,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            allocation_method=args.allocation_method,
            voxel_ratio=args.voxel_ratio,
            min_points_per_part=args.min_points_per_part,
            max_points_per_part=args.max_points_per_part,
            global_seed=args.global_seed,
        )
        
        elapsed_time_preprocessing = get_time() - start_time_preprocessing
        logger.info(f"Sample saved to: {sample_output_dir}")
        logger.info(f"Preprocessing time (sampling + feature extraction): {elapsed_time_preprocessing:.2f} seconds")
        
        # Create data_split file so RAP can find the sample
        # The dataset expects: data_root/dataset_name/data_split/val.txt
        # (Note: even for test stage, the datamodule uses split="val")
        # containing fragment names (one per line), where each fragment is a folder
        # containing PLY files
        sample_name = os.path.basename(sample_output_dir)
        data_split_dir = os.path.join(dataset_folder, "data_split")
        os.makedirs(data_split_dir, exist_ok=True)
        
        # Create val.txt split file with the sample name
        val_split_file = os.path.join(data_split_dir, "val.txt")
        with open(val_split_file, 'w') as f:
            f.write(f"{sample_name}\n")
        
        logger.info(f"Created split file: {val_split_file} (contains: {sample_name})")
        
        # Initialize inference timing variable
        elapsed_time_inference = 0.0
        
        if not args.skip_inference:
            # Run inference
            logger.info("=" * 60)
            logger.info("STEP 2: Running flow matching inference")
            logger.info("=" * 60)
            
            # Run inference with hydra
            with hydra.initialize(config_path="./config", version_base="1.3"):
                overrides = [
                    f'data_root={data_root}',
                    f'log_dir={log_dir}',
                    f'data.dataset_names=[{dataset_name}]',
                ]
                if args.flow_model_checkpoint:
                    overrides.append(f'ckpt_path={args.flow_model_checkpoint}')
                # Set rigidity_forcing (default is True)
                overrides.append(f'model.rigidity_forcing={args.rigidity_forcing}')
                # Set n_generations (default is 1)
                overrides.append(f'model.n_generations={args.n_generations}')
                # Set inference_sampling_steps (default is 10)
                overrides.append(f'model.inference_sampling_steps={args.inference_sampling_steps}')
                # Set save_trajectory (default is False)
                overrides.append(f'visualizer.save_trajectory={args.save_trajectory}')
                # Set max_samples_per_batch to 1 when save_trajectory is enabled
                if args.save_trajectory:
                    overrides.append(f'visualizer.max_samples_per_batch=1')
                
                logger.info(f"Flow matching parameters: n_generations={args.n_generations}, inference_steps={args.inference_sampling_steps}")
                
                cfg = hydra.compose(
                    config_name=args.config,
                    overrides=overrides
                )
                
                # Import and run sample.py's main logic
                from sample import setup
                from rectified_point_flow.utils import print_eval_table
                
                model, datamodule, trainer = setup(cfg)
                
                logger.info("Running RAP inference (flow matching generation)...")
                start_time_inference = get_time()
                
                eval_results = trainer.test(
                    model=model,
                    datamodule=datamodule,
                    verbose=False,
                )
                
                elapsed_time_inference = get_time() - start_time_inference
                logger.info(f"Flow matching generation time: {elapsed_time_inference:.2f} seconds")
                
                if args.eval_on:
                    # Print results
                    sample_counts = []
                    part_count_ranges = []
                    for dataset_name in datamodule.dataset_names:
                        count = model.last_sample_counts.get(dataset_name, 0)
                        sample_counts.append(count)
                        part_range = model.last_part_count_ranges.get(dataset_name, (0, 0))
                        part_count_ranges.append(part_range)
                    
                    print_eval_table(eval_results, datamodule.dataset_names,
                                    sample_counts=sample_counts,
                                    part_count_ranges=part_count_ranges)
                
                logger.info(f"Visualizations saved to: {Path(cfg.get('log_dir')) / 'visualizations'}")
                logger.info(f"Evaluation results saved to: {Path(cfg.get('log_dir')) / 'results'}")
                
                # Apply transformations and save registered point clouds
                logger.info("=" * 60)
                logger.info("STEP 3: Applying transformations and saving registered point clouds")
                logger.info("=" * 60)
                
                import glob as glob_module
                import re
                
                # Set up paths
                input_vis_dir = args.input
                results_vis_dir = os.path.join(log_dir, 'results')
                
                # Find the actual results directory for this dataset
                if os.path.exists(results_vis_dir):
                    dataset_results_dir = os.path.join(results_vis_dir, dataset_name)
                    if os.path.exists(dataset_results_dir):
                        # Find the sample directory (may have nested structure)
                        sample_name = os.path.basename(sample_output_dir)
                        result_sample_dirs = []
                        
                        # Search for sample directories
                        for root, dirs, files in os.walk(dataset_results_dir):
                            if sample_name in root or any(f.endswith('_transform.txt') for f in files):
                                result_sample_dirs.append(root)
                        
                        if result_sample_dirs:
                            # Use the first matching directory
                            results_vis_dir = result_sample_dirs[0]
                            logger.info(f"Found results directory: {results_vis_dir}")
                        else:
                            # Fallback: use dataset_results_dir
                            results_vis_dir = dataset_results_dir
                            logger.info(f"Using dataset results directory: {results_vis_dir}")
                
                # Use pre-loaded point clouds for visualization (deep copies)
                original_pcds = [copy.deepcopy(pcd) for pcd in visualization_pcds]
                part_names = [part_name for part_name, _, _ in loaded_point_clouds]
                
                logger.info(f"Using {len(original_pcds)} pre-loaded point clouds for visualization")
                
                # Compute normals for original point clouds if needed for visualization
                if args.visualize and args.show_normals and original_pcds:
                    first_pcd = original_pcds[0]
                    bbox = first_pcd.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()
                    normal_radius = max(np.mean(extent) * 0.005, 0.05)
                    logger.info(f"Computing normals for original point clouds with radius: {normal_radius:.4f}")
                    
                    for pcd in tqdm(original_pcds, desc="Computing normals"):
                        if not pcd.has_normals():
                            pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                    radius=normal_radius, max_nn=15
                                )
                            )
                            pcd.orient_normals_consistent_tangent_plane(k=15)
                    
                    logger.info(f"Normals computed for {len(original_pcds)} point clouds")
                
                if original_pcds and os.path.exists(results_vis_dir):
                    # Load global transformations
                    generation_str = args.generation or "generation_selected"
                    
                    # Check if the specified generation exists, fallback to generation00 if not
                    generation_exists = False
                    for file in glob_module.glob(os.path.join(results_vis_dir, f"*{generation_str}_*transform.txt")):
                        generation_exists = True
                        break
                    
                    if not generation_exists:
                        logger.info(f"Generation '{generation_str}' not found, falling back to 'generation00'")
                        generation_str = "generation00"
                    
                    # Find global transform files
                    global_transform_file = None

                    # Try to find files with the generation suffix
                    for file in glob_module.glob(os.path.join(results_vis_dir, f"*{generation_str}_global_transform.txt")):
                        global_transform_file = file
                        break
                    
                    # If not found with generation suffix, try without (for backward compatibility)
                    if global_transform_file is None:
                        for file in glob_module.glob(os.path.join(results_vis_dir, "*_global_transform.txt")):
                            global_transform_file = file
                            break
                    
                    # Load part-specific transformations and apply to point clouds
                    registered_pcds = []

                    T_part_reference = np.eye(4)
                    
                    for idx, (pcd_original, part_name) in enumerate(zip(original_pcds, part_names)):
                        # Find part-specific transform file
                        part_transform_file = None
                        
                        # Try to find by part number in filename
                        part_match = re.search(r'part(\d+)', part_name, re.IGNORECASE)
                        if part_match:
                            part_num = part_match.group(1)
                            pattern = f"*{generation_str}_part{part_num:0>2}_transform.txt"
                            for file in glob_module.glob(os.path.join(results_vis_dir, pattern)):
                                part_transform_file = file
                                break
                        
                        # If not found, try by index
                        if part_transform_file is None:
                            pattern = f"*{generation_str}_part{idx:02d}_transform.txt"
                            for file in glob_module.glob(os.path.join(results_vis_dir, pattern)):
                                part_transform_file = file
                                break
                        
                        # If still not found, try without generation suffix
                        if part_transform_file is None:
                            if part_match:
                                part_num = part_match.group(1)
                                pattern = f"*_part{part_num:0>2}_transform.txt"
                                for file in glob_module.glob(os.path.join(results_vis_dir, pattern)):
                                    part_transform_file = file
                                    break
                        
                        if part_transform_file is None:
                            logger.warning(f"No transform file found for part {part_name} (index {idx}), skipping")
                            continue
                        
                        # Load part-specific transformation
                        T_part = np.loadtxt(part_transform_file) # (4, 4)

                        if idx == 0:
                            T_part_reference = T_part
                        
                        # transform relative the first frame
                        T_part = np.linalg.inv(T_part_reference) @ T_part # (4, 4)

                        pcd_registered = copy.deepcopy(pcd_original)

                        pcd_registered = pcd_registered.transform(T_part)

                        registered_pcds.append(pcd_registered)
                        
                        # Save registered point cloud to file
                        registered_output_dir = os.path.join(results_vis_dir, "registered")
                        os.makedirs(registered_output_dir, exist_ok=True)
                        
                        # Create output filename based on part name and generation
                        part_basename = os.path.splitext(part_name)[0]
                        registered_filename = f"{part_basename}_{generation_str}_registered.ply"
                        registered_filepath = os.path.join(registered_output_dir, registered_filename)
                        
                        # Save point cloud
                        o3d.io.write_point_cloud(registered_filepath, pcd_registered, write_ascii=False)
                        logger.info(f"Saved registered point cloud: {registered_filename}")
                    
                    logger.info(f"Applied transformations to {len(registered_pcds)} point clouds")
                    logger.info(f"Saved {len(registered_pcds)} registered point clouds to {os.path.join(results_vis_dir, 'registered')}")
                    
                    # Visualize if requested
                    if args.visualize:
                        logger.info("=" * 60)
                        logger.info("STEP 4: Visualizing point clouds")
                        logger.info("=" * 60)

                        # Implement visualization with toggle functionality
                        visualize_with_toggle(
                            original_pcds=original_pcds,
                            registered_pcds=registered_pcds,
                            point_size=args.point_size,
                            background_color=args.background_color,
                            show_coordinate_frame=args.show_coordinate_frame,
                            show_normals=args.show_normals
                        )
                        
                else:
                    if not original_pcds:
                        logger.warning("No point clouds found in input directory")
                    if not os.path.exists(results_vis_dir):
                        logger.warning(f"Results directory not found: {results_vis_dir}")
                
                # Clean up temporary dataset folder after processing
                if args.cleanup and os.path.exists(dataset_folder):
                    logger.info(f"Cleaning up temporary dataset folder: {dataset_folder}")
                    shutil.rmtree(dataset_folder)
                    logger.info("Cleanup completed")
                
        else:
            logger.info("Skipping inference (--skip_inference flag set)")
            logger.info(f"Processed data saved to: {output_folder}")
            logger.info(f"To run inference manually, use:")
            logger.info(f"  python sample.py --config {args.config} data_root={data_root} log_dir={log_dir} data.dataset_names=[{dataset_name}]")
            
            if args.visualize:
                logger.warning("Visualization requested but inference was skipped. Visualization requires inference results.")
        
        # Print timing summary
        logger.info("=" * 60)
        logger.info("Timing Summary:")
        logger.info(f"  Point cloud loading: {elapsed_time_loading:.2f} seconds")
        logger.info(f"  Preprocessing (sampling + feature extraction): {elapsed_time_preprocessing:.2f} seconds")
        if not args.skip_inference:
            logger.info(f"  Flow matching generation: {elapsed_time_inference:.2f} seconds")
            total_time = elapsed_time_loading + elapsed_time_preprocessing + elapsed_time_inference
        else:
            total_time = elapsed_time_loading + elapsed_time_preprocessing
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        
        # Clean up temporary dataset folder on error if it exists
        try:
            if 'dataset_folder' in locals() and os.path.exists(dataset_folder):
                logger.info(f"Cleaning up temporary dataset folder after error: {dataset_folder}")
                shutil.rmtree(dataset_folder)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up dataset folder: {cleanup_error}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())