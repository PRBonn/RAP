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
from dataset_process.utils.point_sampling_utils import (
    calculate_voxel_coverage,
    allocate_fps_points,
    calculate_adaptive_sample_count_per_part
)
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
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(weights_url, weights_zip_path, show_progress)
        sys.stdout.write("\n")
        with zipfile.ZipFile(weights_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(weights_zip_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download/extract weights: {e}")
        if os.path.exists(weights_zip_path):
            try:
                os.remove(weights_zip_path)
            except OSError:
                pass
        return False


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
                        use_torch_downsampling: bool = True,
                        feature_extraction_on: bool = True,
                        use_random_downsample: bool = False,
                        target_points_per_scan: Optional[int] = None) -> str:
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
        use_random_downsample: If True, use random downsampling instead of FPS (default: False)
        target_points_per_scan: Target number of points per scan for random downsampling.
                                If None, uses allocation_method to determine target points (default: None)
        
    Returns:
        Path to the created sample folder
    """
    if not loaded_point_clouds:
        raise ValueError("No point clouds provided")
    
    logger.info(f"Processing {len(loaded_point_clouds)} point clouds...")
    
    # Initialize feature extractor if not provided
    # If feature_extraction_on is False, skip feature extraction (features will be set to zero vectors)
    if feature_extractor is None:
        if not feature_extraction_on:
            logger.info("Feature extraction disabled: features will be set to zero vectors")
            # Create a dummy feature extractor that returns zero features
            class ZeroFeatureExtractor:
                def __init__(self, feature_dim=32):
                    self.feature_dim = feature_dim
                    self.device = 'cpu'
                
                def extract_features(self, points, keypoints=None):
                    if keypoints is None:
                        keypoints = points
                    points_is_tensor = isinstance(keypoints, torch.Tensor)
                    keypoint_len = keypoints.shape[0] if keypoints.dim() > 1 else len(keypoints)
                    
                    if points_is_tensor:
                        empty_features = torch.zeros(keypoint_len, self.feature_dim, device=keypoints.device)
                    else:
                        empty_features = np.zeros((keypoint_len, self.feature_dim))
                    
                    return {
                        'features': empty_features,
                        'keypoints': keypoints,
                    }
            
            feature_extractor = ZeroFeatureExtractor(feature_dim=32)
        else:
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
    
    allocated_voxel_size = 4.0 * voxel_size # adaptively set

    # Initialize sample processor if not provided
    # If using random downsampling, skip point sampling in SampleProcessor
    skip_point_sampling = use_random_downsample
    if sample_processor is None:
        sample_processor = SampleProcessor(
            feature_extractor=feature_extractor,
            num_points=5000,  # Default, will be overridden by allocation_method
            skip_point_sampling=skip_point_sampling,
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
        
        logger.debug(f"  {part_name}: {len(points)} -> {len(downsampled_points)} points (voxel_size={voxel_size}m)")
        
        parts_points.append(downsampled_points)
        parts_normals.append(downsampled_normals)
        part_names.append(part_name)
    
    elapsed_time_downsampling = get_time() - start_time_downsampling
    logger.info(f"Voxel downsampling time: {elapsed_time_downsampling:.2f} seconds")
    
    if not parts_points:
        raise ValueError("No valid point clouds found after processing")
    
    # Apply random downsampling if enabled
    if use_random_downsample:
        logger.info("Applying random downsampling to each scan...")
        start_time_random_downsample = get_time()
        
        # Calculate target points per scan
        if target_points_per_scan is not None:
            # Use fixed target for all scans
            target_per_scan = [target_points_per_scan] * len(parts_points)
        else:
            # Use allocation method to determine target points per scan
            if allocation_method == 'voxel_adaptive':
                target_per_scan = calculate_adaptive_sample_count_per_part(
                    parts_points, allocated_voxel_size, voxel_ratio, 
                    min_points_per_part, max_points_per_part
                )
            elif allocation_method == 'point_count':
                # Use point_count allocation with default num_points
                pts_per_part = np.array([len(part) for part in parts_points])
                target_per_scan = allocate_fps_points(
                    pts_per_part, allocation_method, 5000, 
                    min_points_per_part, allocated_voxel_size, voxel_ratio
                )
            elif allocation_method == 'spatial_coverage':
                target_per_scan = allocate_fps_points(
                    parts_points, allocation_method, 5000,
                    min_points_per_part, allocated_voxel_size, voxel_ratio
                )
            else:
                raise ValueError(f"Unknown allocation method: {allocation_method}")
        
        # Apply random downsampling to each scan
        np.random.seed(global_seed)
        randomly_downsampled_points = []
        randomly_downsampled_normals = []
        
        for i, (points, normals, target_points) in enumerate(zip(parts_points, parts_normals, target_per_scan)):
            num_points = len(points)
            target = min(int(target_points), num_points)  # Don't exceed available points
            
            if target < num_points:
                # Randomly sample target points
                indices = np.random.choice(num_points, size=target, replace=False)
                indices = np.sort(indices)  # Keep original order for consistency
                randomly_downsampled_points.append(points[indices])
                if normals is not None:
                    randomly_downsampled_normals.append(normals[indices])
                else:
                    randomly_downsampled_normals.append(None)
                logger.info(f"  Scan {i+1} ({part_names[i]}): {num_points} -> {target} points (random downsampling)")
            else:
                # Keep all points if target >= available points
                randomly_downsampled_points.append(points)
                randomly_downsampled_normals.append(normals)
                logger.info(f"  Scan {i+1} ({part_names[i]}): {num_points} points (no downsampling needed)")
        
        parts_points = randomly_downsampled_points
        parts_normals = randomly_downsampled_normals
        
        elapsed_time_random_downsample = get_time() - start_time_random_downsample
        logger.info(f"Random downsampling time: {elapsed_time_random_downsample:.2f} seconds")
    
    logger.info(f"Processing {len(parts_points)} parts for feature extraction...")
    
    # Process sample using SampleProcessor (includes FPS and feature extraction, or just feature extraction if random downsampling is used)
    start_time_sample_processor = get_time()
    
    part_results = sample_processor.process_sample(parts_points, parts_normals)
    
    elapsed_time_sample_processor = get_time() - start_time_sample_processor
    if use_random_downsample:
        logger.info(f"Sample processor time (feature extraction only): {elapsed_time_sample_processor:.2f} seconds")
    else:
        # Separate FPS and feature extraction times are logged by SampleProcessor at INFO level
        logger.info(f"Total sample processor time: {elapsed_time_sample_processor:.2f} seconds")
    
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
    
    if show_normals and not all(pcd.has_normals() for pcd in original_pcds):
        show_normals = False
    
    state = {'show_registered': False}

    def toggle_view(vis):
        state['show_registered'] = not state['show_registered']
        
        # Clear all geometries
        vis.clear_geometries()
        
        # Add coordinate frame if requested
        if show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            vis.add_geometry(coord_frame, reset_bounding_box=False)
        
        # Add appropriate point clouds
        if state['show_registered']:
            for pcd in registered_pcds:
                vis.add_geometry(pcd, reset_bounding_box=False)
        else:
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
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
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
    
    logger.info("Press 'T' to toggle, 'Q' to quit")
    
    for pcd in original_pcds:
        vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


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
    parser.add_argument('--use_random_downsample', action='store_true', default=False,
                        help='Use random downsampling instead of FPS (default: False)')
    parser.add_argument('--target_points_per_scan', type=int, default=None,
                        help='Target number of points per scan for random downsampling. '
                             'If None, uses allocation_method to determine target points (default: None)')
    
    # Inference
    parser.add_argument('--flow_model_checkpoint', type=str, default='./weights/rap_model.ckpt',
                        help='Path to PRFM checkpoint')
    parser.add_argument('--config', type=str, default='RAP_inference',
                        help='Config name for inference (default: RAP_inference)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model configuration to use (default: None, uses config default)')
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
    parser.add_argument('--use_original_colors', action='store_true', default=True,
                        help='Use original RGB colors from PLY files instead of scan index colors for visualization')
    parser.add_argument('--point_size', type=float, default=4.0,
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
    parser.add_argument('--output_generated', action='store_true', default=False,
                        help='Output the scaled and transformed generated point clouds instead of applying transformations to original point clouds (default: False)')
    parser.add_argument('--save_merged_pointcloud_steps', action='store_true', default=None,
                        help='Save merged point cloud at each generation step (default: None, uses config default)')
    parser.add_argument('--no_save_merged_pointcloud_steps', dest='save_merged_pointcloud_steps', action='store_false',
                        help='Disable saving merged point cloud at each generation step')

    
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
        logger.info("Checkpoint files missing - downloading weights...")
        if not download_and_extract_weights():
            logger.error("Failed to download weights. Get them from https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip")
            return 1
        flow_checkpoint_exists = os.path.exists(args.flow_model_checkpoint)
        feature_checkpoint_exists = os.path.exists(args.feature_extraction_checkpoint)
        if not flow_checkpoint_exists or not feature_checkpoint_exists:
            logger.error("Some checkpoint files still missing after extraction")
            return 1
    
    # Set random seeds
    set_random_seeds(args.global_seed)
    
    # Validate input folder
    if not os.path.isdir(args.input):
        logger.error(f"Input folder does not exist: {args.input}")
        return 1
    
    dataset_name = args.dataset_name or os.path.basename(os.path.normpath(args.input)) or "demo_dataset"
    
    # Get all PLY files
    ply_files = natsorted([f for f in os.listdir(args.input) if f.endswith('.ply')])
    
    if not ply_files:
        logger.error(f"No PLY files found in {args.input}")
        return 1
    
    if args.point_cloud_count is not None:
        if args.point_cloud_count <= 0:
            logger.error("point_cloud_count must be positive")
            return 1
        ply_files = ply_files[:args.point_cloud_count]
    
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
        
        # Load point cloud once (extract points, normals, and colors)
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        has_colors = pcd.has_colors() and len(np.asarray(pcd.colors)) > 0
        original_colors = pcd.colors if has_colors else None
        
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
        
        # Add colors for visualization
        if args.use_original_colors and has_colors:
            # Use original colors from PLY file
            pcd_vis.colors = original_colors
        else:
            # Use scan index colors for consistent visualization
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
    
    if args.adaptive_parameters:
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
        
        # Set adaptive voxel_size: median / 600, bounded by [min_voxel_size, max_voxel_size]
        if median_size < 5.0: # indoor scene or object 
            divide_factor = 200.0
        elif median_size < 30.0: # indoor scene
            divide_factor = 400.0
        elif median_size < 100.0: # outdoor scene
            divide_factor = 600.0
        elif median_size < 250.0: # outdoor large scene
            divide_factor = 800.0
        elif median_size < 500.0: # outdoor large scene
            divide_factor = 1000.0
        else: # outdoor very large scene
            divide_factor = 1200.0
        adaptive_voxel_size = median_size / divide_factor
        min_voxel_size = 0.0001
        max_voxel_size = 0.4
        adaptive_voxel_size = max(min_voxel_size, min(max_voxel_size, adaptive_voxel_size))
        
        # Set adaptive des_r: 20 * voxel_size
        adaptive_des_r = 20.0 * adaptive_voxel_size
        
        # Calculate allocated_voxel_size (used for voxel coverage calculation)
        allocated_voxel_size = 4.0 * adaptive_voxel_size
        
        voxel_coverages = []
        for part_name, points, normals in loaded_point_clouds:
            if len(points) == 0:
                continue
            voxel_coverage = calculate_voxel_coverage(points, allocated_voxel_size)
            voxel_coverages.append(voxel_coverage)
            logger.debug(f"  {part_name}: {voxel_coverage} occupied voxels")
        
        if not voxel_coverages:
            logger.error("No valid voxel coverages calculated for adaptive voxel_ratio")
            return 1
        
        # Calculate median voxel coverage
        median_voxel_coverage = np.median(voxel_coverages)
        current_median_point_count = median_voxel_coverage * args.voxel_ratio
        
        voxel_ratio_adjusted = False
        adaptive_voxel_ratio = args.voxel_ratio
        target_median_point_count_max = args.max_points_per_part
        if current_median_point_count > target_median_point_count_max:
            # Calculate adaptive voxel_ratio to achieve target median point count
            # voxel_ratio = target_point_count / voxel_coverage
            adaptive_voxel_ratio = target_median_point_count_max / median_voxel_coverage
            args.voxel_ratio = adaptive_voxel_ratio
            voxel_ratio_adjusted = True
            current_median_point_count = median_voxel_coverage * adaptive_voxel_ratio
        
        target_median_point_count_min = 500
        if current_median_point_count < target_median_point_count_min:
            adaptive_voxel_ratio = target_median_point_count_min / median_voxel_coverage
            args.voxel_ratio = adaptive_voxel_ratio
            voxel_ratio_adjusted = True
        elif not voxel_ratio_adjusted:
            adaptive_voxel_ratio = args.voxel_ratio
        
        # Override parameters
        args.voxel_size = adaptive_voxel_size
        args.des_r = adaptive_des_r
    
    output_folder = args.output or os.path.join(os.getcwd(), 'demo_output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    log_dir = args.log_dir or os.path.join(output_folder, 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Process point clouds
        
        # Create data_root structure expected by RAP
        # The data module expects: data_root/dataset_name/sample_name/
        data_root = output_folder
        dataset_folder = os.path.join(data_root, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        logger.info("Starting preprocessing (point sampling and feature extraction)...")
        start_time_preprocessing = get_time()
        
        # Determine if feature extraction should be enabled
        # For rap_12_po model, disable feature extraction
        feature_extraction_on = args.model != 'rap_12_po' if args.model else True
        
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
            feature_extraction_on=feature_extraction_on,
            use_random_downsample=args.use_random_downsample,
            target_points_per_scan=args.target_points_per_scan,
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
                if args.model:
                    overrides.append(f'model={args.model}')
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
                # Enable save_pointcloud_parts when output_generated is enabled
                if args.output_generated:
                    overrides.append(f'model.save_pointcloud_parts=true')
                    logger.info("Enabled save_pointcloud_parts to save generated point cloud parts")
                # Set save_merged_pointcloud_steps if specified
                if args.save_merged_pointcloud_steps is not None:
                    overrides.append(f'model.save_merged_pointcloud_steps={str(args.save_merged_pointcloud_steps).lower()}')
                
                model_name = args.model if args.model else "default (from config)"
                logger.info(f"Flow matching parameters: model={model_name}, n_generations={args.n_generations}, inference_steps={args.inference_sampling_steps}")
                
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
                if args.output_generated:
                    logger.info("STEP 3: Loading and saving generated point clouds")
                else:
                    logger.info("STEP 3: Applying transformations and saving registered point clouds")
                logger.info("=" * 60)
                
                import re
                results_vis_dir = os.path.join(log_dir, 'results', dataset_name)
                sample_name = os.path.basename(sample_output_dir)
                sample_results_dir = os.path.join(results_vis_dir, sample_name)
                if os.path.exists(sample_results_dir):
                    results_vis_dir = sample_results_dir
                
                original_pcds = [copy.deepcopy(pcd) for pcd in visualization_pcds]
                part_names = [p[0] for p in loaded_point_clouds]
                
                if args.visualize and args.show_normals and original_pcds:
                    extent = original_pcds[0].get_axis_aligned_bounding_box().get_extent()
                    normal_radius = max(np.mean(extent) * 0.005, 0.05)
                    for pcd in tqdm(original_pcds, desc="Computing normals"):
                        if not pcd.has_normals():
                            pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                    radius=normal_radius, max_nn=15
                                )
                            )
                            pcd.orient_normals_consistent_tangent_plane(k=15)
                
                if original_pcds and os.path.exists(results_vis_dir):
                    generation_str = args.generation
                    if not glob.glob(os.path.join(results_vis_dir, f"*{generation_str}_*transform.txt")):
                        generation_str = "generation00"
                    registered_pcds = []
                    
                    if args.output_generated:
                        generated_ply_files = natsorted(glob.glob(os.path.join(results_vis_dir, f"*{generation_str}*part*.ply")))
                        
                        if not generated_ply_files:
                            logger.warning(f"No generated PLY files found in {results_vis_dir}")
                            logger.warning("Make sure --output_generated was specified before inference, or enable model.save_pointcloud_parts in config")
                        else:
                            # Extract part numbers and sort by part number
                            ply_files_with_parts = []
                            for ply_file in generated_ply_files:
                                basename = os.path.basename(ply_file)
                                # Extract part number from filename (e.g., part00, part01, etc.)
                                part_match = re.search(r'part(\d+)', basename)
                                part_num = int(part_match.group(1)) if part_match else len(ply_files_with_parts)
                                ply_files_with_parts.append((part_num, ply_file))
                            
                            # Sort by part number
                            ply_files_with_parts.sort(key=lambda x: x[0])
                            
                            # Load generated point clouds
                            for idx, (part_num, ply_file) in enumerate(ply_files_with_parts):
                                try:
                                    pcd_generated = o3d.io.read_point_cloud(ply_file)
                                    points = np.asarray(pcd_generated.points)
                                    
                                    if len(points) == 0:
                                        logger.warning(f"Empty generated point cloud: {os.path.basename(ply_file)}")
                                        continue
                                    
                                    # Use the same color scheme as original point clouds
                                    if idx < len(CMAP_DEFAULT):
                                        rgb = CMAP_DEFAULT[idx % len(CMAP_DEFAULT)]
                                        pcd_generated.paint_uniform_color(rgb)
                                    
                                    # Compute normals for generated point clouds if needed for visualization
                                    if args.visualize and args.show_normals and not pcd_generated.has_normals():
                                        bbox = pcd_generated.get_axis_aligned_bounding_box()
                                        extent = bbox.get_extent()
                                        normal_radius = max(np.mean(extent) * 0.005, 0.05)
                                        pcd_generated.estimate_normals(
                                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                radius=normal_radius, max_nn=15
                                            )
                                        )
                                        pcd_generated.orient_normals_consistent_tangent_plane(k=15)
                                    
                                    registered_pcds.append(pcd_generated)
                                    
                                except Exception as e:
                                    logger.warning(f"Error loading generated point cloud {ply_file}: {e}")
                                    continue
                            
                            
                            # Save generated point clouds to registered folder (parallel to generation subfolder)
                            registered_output_dir = os.path.join(results_vis_dir, "registered")
                            os.makedirs(registered_output_dir, exist_ok=True)
                            
                            for idx, (pcd_generated, part_name) in enumerate(zip(registered_pcds, part_names[:len(registered_pcds)])):
                                # Create output filename based on part name and generation
                                part_basename = os.path.splitext(part_name)[0]
                                registered_filename = f"{part_basename}_{generation_str}_registered.ply"
                                registered_filepath = os.path.join(registered_output_dir, registered_filename)
                                
                                # Save point cloud
                                o3d.io.write_point_cloud(registered_filepath, pcd_generated, write_ascii=False)
                            
                            # Clean up temporary PLY files saved by save_pointcloud_parts
                            for pattern in ["*generation*_part*.ply", "*_input_part*.ply", "*_gt_part*.ply"]:
                                for f in glob.glob(os.path.join(results_vis_dir, pattern)):
                                    try:
                                        os.remove(f)
                                    except OSError:
                                        pass
                    else:
                        # Original behavior: Load part-specific transformations and apply to point clouds
                        T_part_reference = np.eye(4)
                        
                        for idx, (pcd_original, part_name) in enumerate(zip(original_pcds, part_names)):
                            # Find part-specific transform file (evaluator uses input filename in transform file name)
                            part_name_safe = part_name.replace("/", "_").replace("\\", "_")
                            matches = natsorted(glob.glob(os.path.join(results_vis_dir, f"*{generation_str}_{part_name_safe}_transform.txt")))
                            if not matches:
                                matches = natsorted(glob.glob(os.path.join(results_vis_dir, f"*_{part_name_safe}_transform.txt")))
                            part_transform_file = matches[0] if matches else None
                            
                            if part_transform_file is None:
                                logger.warning(f"No transform file found for part {part_name} (index {idx}), skipping")
                                continue
                            
                            # Load part-specific transformation
                            T_part = np.loadtxt(part_transform_file) # (4, 4)

                            if idx == 0:
                                T_part_reference = T_part
                            
                            # transform relative to the first frame
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
                    
                    if args.visualize:
                        visualize_with_toggle(
                            original_pcds=original_pcds,
                            registered_pcds=registered_pcds,
                            point_size=args.point_size,
                            background_color=args.background_color,
                            show_coordinate_frame=args.show_coordinate_frame,
                            show_normals=args.show_normals
                        )
                        
                else:
                    logger.warning("No point clouds or results directory found")
                
                if args.cleanup and os.path.exists(dataset_folder):
                    shutil.rmtree(dataset_folder)
                
        else:
            logger.info("Skipping inference (--skip_inference flag set)")
            logger.info(f"Processed data saved to: {output_folder}")
            logger.info(f"To run inference manually, use:")
            logger.info(f"  python sample.py --config {args.config} data_root={data_root} log_dir={log_dir} data.dataset_names=[{dataset_name}]")
            
            if args.visualize:
                logger.warning("Visualization requested but inference was skipped. Visualization requires inference results.")
        
        total_time = elapsed_time_loading + elapsed_time_preprocessing + (elapsed_time_inference if not args.skip_inference else 0)
        logger.info(f"Done. Total: {total_time:.1f}s (load: {elapsed_time_loading:.1f}s, preprocess: {elapsed_time_preprocessing:.1f}s" +
                    (f", inference: {elapsed_time_inference:.1f}s" if not args.skip_inference else "") + ")")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        
        # Clean up temporary dataset folder on error if it exists
        try:
            if 'dataset_folder' in locals() and os.path.exists(dataset_folder):
                shutil.rmtree(dataset_folder)
        except OSError:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())