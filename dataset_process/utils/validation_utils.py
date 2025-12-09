#!/usr/bin/env python3
"""
Validation Utilities for Training Sample Generation

This module contains functions for dry run validation, argument checking,
and other validation utilities.
"""

import os
import logging
import argparse
from typing import Dict, List, Tuple, Any

# Import necessary modules from other utils files
from .io_utils import get_dataset_name
from .preview_utils import preview_data_splits
from .submap_utils import get_default_num_samples

logger = logging.getLogger(__name__)

def _validate_and_setup_args(args: argparse.Namespace) -> bool:
    """
    Validate and setup command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if validation passed, False otherwise
    """
    # Validate input path
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return False
    
    # Validate and set HDF5 arguments
    if args.save_hdf5 or args.hdf5_only:
        if args.hdf5_output is None:
            # Auto-generate HDF5 output path based on output directory name
            # Use output directory name as dataset name instead of input directory name
            output_dataset_name = get_dataset_name(args.output, args.dataset_name)
            args.hdf5_output = os.path.join(args.output, f"{output_dataset_name}.hdf5")
            logger.info(f"Auto-generated HDF5 output path: {args.hdf5_output}")
        
        if not args.hdf5_output.endswith(('.hdf5', '.h5')):
            logger.error("HDF5 output file must have .hdf5 or .h5 extension")
            return False
    
    # Validate hdf5_only mode
    if args.hdf5_only:
        if not (args.save_hdf5 or args.hdf5_output):
            logger.error("--hdf5_only requires --save_hdf5 or --hdf5_output to be specified")
            return False
        logger.info("HDF5-only mode: will only convert existing PLY/NPY files to HDF5")
    
    return True

def perform_dry_run(args, data_loader, sequences_to_process):
    """Perform dry run to check configuration and data paths."""
    logger.info("=" * 50)
    logger.info("DRY RUN - Checking configuration and data paths")
    logger.info("=" * 50)
    
    # Check data root exists
    if not os.path.exists(args.data_root):
        logger.error(f"Data root path does not exist: {args.data_root}")
        return
    logger.info(f"✓ Data root path exists: {args.data_root}")
    
    # Check output directory can be created
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"✓ Output directory can be created: {args.output_dir}")
    except Exception as e:
        logger.error(f"✗ Cannot create output directory: {e}")
        return
    
    # Check each sequence and calculate total samples
    total_samples = 0
    sequence_samples = {}
    
    for sequence in sequences_to_process:
        logger.info(f"Checking sequence {sequence}...")
        try:
            sequence_info = data_loader.get_sequence_info(sequence)
            frame_count = sequence_info['frame_count']
            
            # Use effective frame count if keyframe filtering is available
            if hasattr(data_loader, 'get_effective_frame_count'):
                # Set sequence temporarily to get effective count
                try:
                    data_loader.set_sequence(sequence)
                    effective_count = data_loader.get_effective_frame_count()
                    if effective_count != frame_count:
                        logger.info(f"  ✓ Sequence {sequence}: {frame_count} frames -> {effective_count} keyframes")
                        frame_count = effective_count
                    else:
                        logger.info(f"  ✓ Sequence {sequence}: {frame_count} frames")
                except Exception as e:
                    logger.warning(f"  ! Could not get effective frame count for {sequence}: {e}")
                    logger.info(f"  ✓ Sequence {sequence}: {frame_count} frames (using original count)")
            else:
                logger.info(f"  ✓ Sequence {sequence}: {frame_count} frames")
            
            # Show frame limiting info
            if args.max_frames_per_sequence and frame_count > args.max_frames_per_sequence:
                logger.info(f"  ✓ Would limit to {args.max_frames_per_sequence} frames (random sampling)")
            elif args.max_frames_per_sequence:
                logger.info(f"  ✓ Would use all {frame_count} frames (within limit)")
            else:
                logger.info(f"  ✓ Would use all {frame_count} frames (no limit)")
            
            # Determine number of samples
            num_samples = args.num_samples or get_default_num_samples(sequence, frame_count, data_loader, args.sample_count_multiplier)
            
            # Apply max_samples_per_sequence limit
            if num_samples > args.max_samples_per_sequence:
                logger.info(f"  ✓ Would generate {num_samples} samples (limited from calculated value)")
                logger.info(f"  ✓ Applying max_samples_per_sequence limit: {args.max_samples_per_sequence}")
                num_samples = args.max_samples_per_sequence
            else:
                logger.info(f"  ✓ Would generate {num_samples} samples")
            
            # Store for total calculation
            sequence_samples[sequence] = num_samples
            total_samples += num_samples
            
        except Exception as e:
            logger.error(f"  ✗ Error checking sequence {sequence}: {e}")
    
    # Check HDF5 path if specified
    if args.create_hdf5 and args.hdf5_output_path:
        hdf5_dir = os.path.dirname(args.hdf5_output_path)
        if hdf5_dir and not os.path.exists(hdf5_dir):
            try:
                os.makedirs(hdf5_dir, exist_ok=True)
                logger.info(f"✓ HDF5 output directory can be created: {hdf5_dir}")
            except Exception as e:
                logger.error(f"✗ Cannot create HDF5 output directory: {e}")
    
    logger.info("=" * 50)
    logger.info("DRY RUN COMPLETE - All checks passed!")
    logger.info("=" * 50)
    logger.info("SAMPLE COUNT SUMMARY:")
    logger.info(f"  Total sequences to process: {len(sequences_to_process)}")
    logger.info(f"  Total samples to generate: {total_samples}")
    logger.info(f"  Max samples per sequence limit: {args.max_samples_per_sequence}")
    logger.info("  Per-sequence breakdown:")
    for sequence, num_samples in sequence_samples.items():
        logger.info(f"    {sequence}: {num_samples} samples")
    
    # Show data splitting preview
    if args.val_sequences:
        logger.info("=" * 50)
        logger.info("MANUAL VALIDATION SEQUENCES SPECIFIED:")
        logger.info(f"  Validation sequences: {args.val_sequences}")
        logger.info(f"  This will override automatic sequence-based splitting")
        if args.mixed_val_split:
            logger.info(f"  Mixed validation enabled: Additional random samples from training sequences will be added to validation")
        logger.info("=" * 50)
    else:
        if args.mixed_val_split:
            logger.info("=" * 50)
            logger.info("MIXED VALIDATION SPLIT ENABLED:")
            logger.info(f"  Validation will include sequence-based samples PLUS random samples from training sequences")
            logger.info(f"  Target validation ratio: {1 - args.train_ratio:.3f}")
            logger.info("=" * 50)
        preview_data_splits(sequence_samples, args.train_ratio, args.split_by_sequence, args.seed, data_loader, args.val_sequences)
    
    logger.info("=" * 50)

def process_sequences(args, data_loader, sequences_to_process, sequences_dir):
    """Process all sequences and return total samples and statistics."""
    from .processing_utils import process_sequence_with_loader
    
    total_samples = 0
    sequence_stats = {}
    
    for sequence in sequences_to_process:
        logger.info(f"Processing sequence {sequence}")
        
        # Get sequence info and determine samples
        sequence_info = data_loader.get_sequence_info(sequence)
        frame_count = sequence_info['frame_count']
        
        # Use estimated effective frame count to avoid expensive pre-loading for sample calculation
        if hasattr(data_loader, 'estimate_effective_frame_count'):
            try:
                estimated_frame_count = data_loader.estimate_effective_frame_count(sequence)
                if estimated_frame_count != frame_count:
                    logger.info(f"Using estimated effective frame count for {sequence}: {frame_count} -> ~{estimated_frame_count} frames (for sample calculation)")
                    frame_count = estimated_frame_count
            except Exception as e:
                logger.warning(f"Could not estimate effective frame count for {sequence}: {e}")
        
        num_samples = args.num_samples or get_default_num_samples(sequence, frame_count, data_loader, args.sample_count_multiplier)
        
        # Apply max_samples_per_sequence limit
        if num_samples > args.max_samples_per_sequence:
            logger.info(f"Limiting samples for sequence {sequence} from {num_samples} to {args.max_samples_per_sequence}")
            num_samples = args.max_samples_per_sequence
        
        # Log processing info
        if args.max_frames_per_sequence and frame_count > args.max_frames_per_sequence:
            logger.info(f"Sequence {sequence}: {frame_count} frames, limiting to {args.max_frames_per_sequence} frames, generating {num_samples} samples")
        else:
            logger.info(f"Sequence {sequence}: {frame_count} frames, generating {num_samples} samples")
        
        # Process sequence
        sequence_output_dir = os.path.join(sequences_dir, sequence)
        
        # Log actual effective frame count if available (after set_sequence is called in processing)
        logger.info(f"Starting processing for sequence {sequence}...")
        
        num_generated, stats = process_sequence_with_loader(
            data_loader=data_loader,
            sequence=sequence,
            output_dir=sequence_output_dir,
            num_samples_to_generate=num_samples,
            min_frames_per_submap=args.min_frames_per_submap,
            max_frames_per_submap=args.max_frames_per_submap,
            min_spatial_threshold=args.min_spatial_threshold,
            max_spatial_threshold=args.max_spatial_threshold,
            min_submaps_per_sample=args.min_submaps_per_sample,
            max_submaps_per_sample=args.max_submaps_per_sample,
            voxel_size=args.voxel_size,
            downsample_method=args.downsample_method,
            num_points_downsample=args.num_points_downsample,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_frames_per_sequence=args.max_frames_per_sequence,
            min_overlap_ratio=args.min_overlap_ratio,
            max_overlap_ratio=args.max_overlap_ratio,
            overlap_method=args.overlap_method,
            min_frame_interval=args.min_frame_interval,
            max_frame_interval=args.max_frame_interval,
            overlap_voxel_size=args.overlap_voxel_size,
            max_attempts=args.max_attempts,
            enable_deskewing=args.enable_deskewing,
            random_drop_to_single_frame=args.random_drop_to_single_frame
        )
        
        total_samples += num_generated
        sequence_stats[sequence] = {
            'training_samples_generated': num_generated,
            'statistics': stats
        }
        
        logger.info(f"Sequence {sequence}: {num_generated} training samples")
    
    return total_samples, sequence_stats 