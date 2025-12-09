import os
import json
import numpy as np
import logging
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter

logger = logging.getLogger(__name__)

def clean_args_for_json(args_dict: Dict) -> Dict:
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

def reconstruct_command_line(args_dict: Dict) -> str:
    """Reconstruct the command line from arguments for reproducibility."""
    cmd_parts = ['python', 'dataset_process/extract_sample_features.py']
    
    # Add input/output arguments
    if args_dict.get('input'):
        cmd_parts.extend(['--input', str(args_dict['input'])])
    if args_dict.get('output'):
        cmd_parts.extend(['--output', str(args_dict['output'])])
    if args_dict.get('dataset_name'):
        cmd_parts.extend(['--dataset_name', str(args_dict['dataset_name'])])
    
    # Add HDF5 arguments
    if args_dict.get('save_hdf5'):
        cmd_parts.append('--save_hdf5')
    if args_dict.get('hdf5_output'):
        cmd_parts.extend(['--hdf5_output', str(args_dict['hdf5_output'])])
    if args_dict.get('hdf5_only'):
        cmd_parts.append('--hdf5_only')
    
    # Add processing arguments
    if args_dict.get('num_points'):
        cmd_parts.extend(['--num_points', str(args_dict['num_points'])])
    if args_dict.get('global_seed'):
        cmd_parts.extend(['--global_seed', str(args_dict['global_seed'])])
    if args_dict.get('min_points_per_part'):
        cmd_parts.extend(['--min_points_per_part', str(args_dict['min_points_per_part'])])
    
    # Add outlier removal arguments
    if args_dict.get('remove_outliers'):
        cmd_parts.append('--remove_outliers')
    else:
        cmd_parts.append('--no_remove_outliers')
    if args_dict.get('outlier_nb_neighbors'):
        cmd_parts.extend(['--outlier_nb_neighbors', str(args_dict['outlier_nb_neighbors'])])
    if args_dict.get('outlier_std_ratio'):
        cmd_parts.extend(['--outlier_std_ratio', str(args_dict['outlier_std_ratio'])])
    
    # Add allocation method and voxel size
    if args_dict.get('allocation_method'):
        cmd_parts.extend(['--allocation_method', str(args_dict['allocation_method'])])
    if args_dict.get('voxel_size'):
        cmd_parts.extend(['--voxel_size', str(args_dict['voxel_size'])])
    
    # Add voxel-adaptive parameters
    if args_dict.get('voxel_ratio'):
        cmd_parts.extend(['--voxel_ratio', str(args_dict['voxel_ratio'])])
    # if args_dict.get('max_sample_points'):
    #     cmd_parts.extend(['--max_sample_points', str(args_dict['max_sample_points'])])
    # if args_dict.get('min_sample_points'):
    #     cmd_parts.extend(['--min_sample_points', str(args_dict['min_sample_points'])])
    
    # Add model arguments
    if args_dict.get('checkpoint'):
        cmd_parts.extend(['--checkpoint', str(args_dict['checkpoint'])])
    if args_dict.get('device'):
        cmd_parts.extend(['--device', str(args_dict['device'])])
    
    # Add miniSpinNet configuration
    if args_dict.get('des_r'):
        cmd_parts.extend(['--des_r', str(args_dict['des_r'])])
    if args_dict.get('num_points_per_patch'):
        cmd_parts.extend(['--num_points_per_patch', str(args_dict['num_points_per_patch'])])
    
    # Add utility arguments
    if args_dict.get('log_level'):
        cmd_parts.extend(['--log_level', str(args_dict['log_level'])])
    if args_dict.get('dry_run'):
        cmd_parts.append('--dry_run')
    
    return ' '.join(cmd_parts)

def save_processing_metadata(output_dir: str, 
                           stats: Dict, 
                           args: argparse.Namespace):
    """Save processing metadata and statistics."""
    
    cleaned_processing_args = clean_args_for_json(vars(args))
    reconstructed_command = reconstruct_command_line(cleaned_processing_args)
    
    metadata = {
        'processing_info': {
            'script': 'extract_sample_features.py',
            'timestamp': datetime.now().isoformat(),
            'description': 'Feature extraction from training samples using miniSpinNet',
        },
        'command_line_info': {
            'script_name': 'extract_sample_features.py',
            'reconstructed_command': reconstructed_command,
            'args_summary': {
                'input': cleaned_processing_args.get('input', 'unknown'),
                'output': cleaned_processing_args.get('output', 'unknown'),
                'dataset_name': cleaned_processing_args.get('dataset_name'),
                'num_points': cleaned_processing_args.get('num_points'),
                'global_seed': cleaned_processing_args.get('global_seed'),
                'min_points_per_part': cleaned_processing_args.get('min_points_per_part'),
                'remove_outliers': cleaned_processing_args.get('remove_outliers'),
                'outlier_nb_neighbors': cleaned_processing_args.get('outlier_nb_neighbors'),
                'outlier_std_ratio': cleaned_processing_args.get('outlier_std_ratio'),
                'allocation_method': cleaned_processing_args.get('allocation_method'),
                'voxel_size': cleaned_processing_args.get('voxel_size'),
                'voxel_ratio': cleaned_processing_args.get('voxel_ratio'),
                # 'max_sample_points': cleaned_processing_args.get('max_sample_points'),
                # 'min_sample_points': cleaned_processing_args.get('min_sample_points'),
                'des_r': cleaned_processing_args.get('des_r'),
                'device': cleaned_processing_args.get('device'),
                'save_hdf5': cleaned_processing_args.get('save_hdf5'),
                'hdf5_only': cleaned_processing_args.get('hdf5_only')
            }
        },
        'processing_args': cleaned_processing_args,
        'statistics': stats,
        'model_config': {
            'feature_extractor': 'miniSpinNet',
            'num_points_fps': args.num_points,
            'global_seed': args.global_seed,
            'min_points_per_part': args.min_points_per_part,
            'remove_outliers': args.remove_outliers,
            'outlier_nb_neighbors': args.outlier_nb_neighbors,
            'outlier_std_ratio': args.outlier_std_ratio,
            'allocation_method': args.allocation_method,
            'voxel_size': args.voxel_size,
            'voxel_ratio': args.voxel_ratio,
            # 'max_sample_points': args.max_sample_points,
            # 'min_sample_points': args.min_sample_points,
            'des_r': args.des_r,
            'num_points_per_patch': args.num_points_per_patch,
            'checkpoint_path': args.checkpoint,
            'device': args.device
        }
    }
    
    metadata_path = os.path.join(output_dir, 'feature_extraction_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved processing metadata: {metadata_path}")
    logger.info(f"Included {len(cleaned_processing_args)} processing arguments in metadata")
    logger.info(f"Reconstructed command line for reproducibility: {reconstructed_command}")


def print_detailed_statistics(stats: Dict, args: argparse.Namespace):
    """
    Print a comprehensive summary of processing statistics.
    
    Args:
        stats: Dictionary containing processing statistics
        args: Command line arguments
    """
    logger.info("=" * 60)
    logger.info("DETAILED PROCESSING STATISTICS")
    logger.info("=" * 60)
    
    # Basic sample statistics
    processed_samples = stats.get('processed_samples', 0)
    failed_samples = stats.get('failed_samples', 0)
    total_samples = stats.get('total_samples', 0)
    
    logger.info(f"📊 SAMPLE STATISTICS:")
    logger.info(f"   Total samples found: {total_samples}")
    logger.info(f"   Successfully processed: {processed_samples}")
    logger.info(f"   Failed to process: {failed_samples}")
    if total_samples > 0:
        success_rate = processed_samples / total_samples * 100
        logger.info(f"   Success rate: {success_rate:.1f}%")
    logger.info("")
    
    # Detailed statistics (only available if not dry_run or hdf5_only)
    sample_num_points = stats.get('sample_num_points', [])
    sample_part_counts = stats.get('sample_part_counts', [])
    sample_part_points = stats.get('sample_part_points', [])
    all_part_points = stats.get('all_part_points', [])
    
    if sample_num_points and len(sample_num_points) > 0:
        # Filter out zero values (failed samples) for meaningful statistics
        valid_sample_points = [points for points in sample_num_points if points > 0]
        
        if valid_sample_points:
            logger.info(f"🔢 POINT COUNT STATISTICS (Per Sample):")
            logger.info(f"   Total points across all samples: {sum(valid_sample_points):,}")
            logger.info(f"   Average points per sample: {np.mean(valid_sample_points):.1f}")
            logger.info(f"   Median points per sample: {np.median(valid_sample_points):.1f}")
            logger.info(f"   Min points per sample: {min(valid_sample_points):,}")
            logger.info(f"   Max points per sample: {max(valid_sample_points):,}")
            logger.info(f"   Std deviation: {np.std(valid_sample_points):.1f}")
            logger.info("")
    
    if sample_part_counts and len(sample_part_counts) > 0:
        # Filter out zero values (failed samples)
        valid_part_counts = [count for count in sample_part_counts if count > 0]
        
        if valid_part_counts:
            logger.info(f"🧩 PART COUNT STATISTICS (Per Sample):")
            logger.info(f"   Total parts across all samples: {sum(valid_part_counts):,}")
            logger.info(f"   Average parts per sample: {np.mean(valid_part_counts):.1f}")
            logger.info(f"   Median parts per sample: {np.median(valid_part_counts):.1f}")
            logger.info(f"   Min parts per sample: {min(valid_part_counts)}")
            logger.info(f"   Max parts per sample: {max(valid_part_counts)}")
            logger.info(f"   Std deviation: {np.std(valid_part_counts):.1f}")
            logger.info("")
    
    if all_part_points and len(all_part_points) > 0:
        # Filter out zero values
        valid_part_points = [points for points in all_part_points if points > 0]
        
        if valid_part_points:
            logger.info(f"⚫ POINT COUNT STATISTICS (Per Part):")
            logger.info(f"   Total individual parts: {len(valid_part_points):,}")
            logger.info(f"   Average points per part: {np.mean(valid_part_points):.1f}")
            logger.info(f"   Median points per part: {np.median(valid_part_points):.1f}")
            logger.info(f"   Min points per part: {min(valid_part_points):,}")
            logger.info(f"   Max points per part: {max(valid_part_points):,}")
            logger.info(f"   Std deviation: {np.std(valid_part_points):.1f}")
            logger.info("")
    
    # Processing configuration summary
    logger.info(f"⚙️ PROCESSING CONFIGURATION:")
    if not args.hdf5_only:
        logger.info(f"   Target points per sample (FPS): {args.num_points:,}")
        logger.info(f"   Min points per part: {args.min_points_per_part}")
        logger.info(f"   Allocation method: {args.allocation_method}")
        if args.allocation_method == 'voxel_adaptive':
            logger.info(f"   Voxel size: {args.voxel_size}m")
            logger.info(f"   Voxel ratio: {args.voxel_ratio}")
            # logger.info(f"   Max sample points: {args.max_sample_points:,}")
            # logger.info(f"   Min sample points: {args.min_sample_points}")
        elif args.allocation_method == 'spatial_coverage':
            logger.info(f"   Voxel size: {args.voxel_size}m")
        logger.info(f"   Remove outliers: {args.remove_outliers}")
        if args.remove_outliers:
            logger.info(f"   Outlier neighbors: {args.outlier_nb_neighbors}")
            logger.info(f"   Outlier std ratio: {args.outlier_std_ratio}")
        logger.info(f"   Feature extractor: miniSpinNet")
        logger.info(f"   Description radius: {args.des_r}m")
        logger.info(f"   Points per patch: {args.num_points_per_patch}")
        logger.info(f"   Global seed: {args.global_seed}")
        logger.info(f"   Device: {args.device}")
    
    if args.save_hdf5 or args.hdf5_only:
        logger.info(f"   HDF5 output: {args.hdf5_output}")
    
    logger.info("")
    
    # Show sample distribution if we have detailed data
    if sample_part_counts and len(sample_part_counts) > 0:
        valid_part_counts = [count for count in sample_part_counts if count > 0]
        if valid_part_counts:
            # Show distribution of parts per sample
            part_count_dist = Counter(valid_part_counts)
            logger.info(f"📈 PART COUNT DISTRIBUTION:")
            for parts, count in sorted(part_count_dist.items()):
                percentage = count / len(valid_part_counts) * 100
                logger.info(f"   {parts} parts: {count} samples ({percentage:.1f}%)")
            logger.info("")
    
    # Show point count percentiles if we have data
    if sample_num_points and len(sample_num_points) > 0:
        valid_sample_points = [points for points in sample_num_points if points > 0]
        if valid_sample_points and len(valid_sample_points) >= 5:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            logger.info(f"📊 SAMPLE POINT COUNT PERCENTILES:")
            for p in percentiles:
                value = np.percentile(valid_sample_points, p)
                logger.info(f"   {p}th percentile: {value:.0f} points")
            logger.info("")
    
    if all_part_points and len(all_part_points) > 0:
        valid_part_points = [points for points in all_part_points if points > 0]
        if valid_part_points and len(valid_part_points) >= 5:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            logger.info(f"📊 PART POINT COUNT PERCENTILES:")
            for p in percentiles:
                value = np.percentile(valid_part_points, p)
                logger.info(f"   {p}th percentile: {value:.0f} points")
            logger.info("")
    
    logger.info("=" * 60) 