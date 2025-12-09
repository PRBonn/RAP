#!/usr/bin/env python3
"""
Utilities for Training Sample Generation

This package contains utility modules for generating training samples and doing preprocessing for RAP.
"""

# Import commonly used functions for easier access
from .submap_utils import (
    get_default_num_samples,
    create_submap_from_frames,
    generate_submap_boundaries_for_sample,
    select_spatially_close_submaps,
    validate_no_overlap
)

from .io_utils import (
    create_metadata_json,
    convert_to_hdf5,
    get_dataset_name,
    load_sample_from_folder,
    load_sample_from_hdf5,
    save_processed_sample,
    save_training_sample
)

from .processing_utils import (
    process_nss_dataset,
    process_nss_multi_dataset,
    process_sequence_with_loader,
    set_random_seeds,
    process_tls_dataset
)

from .validation_utils import (
    perform_dry_run,
    process_sequences,
    _validate_and_setup_args
)

from .split_utils import (
    create_data_splits,
    create_nss_data_splits,
    create_random_data_splits_only,
    create_sequence_data_splits_only,
    copy_and_update_data_split
)

from .preview_utils import (
    preview_data_splits
)

from .feature_extraction_metadata_utils import (
    save_processing_metadata,
    print_detailed_statistics
)

from .point_sampling_utils import (
    calculate_voxel_coverage,
    calculate_adaptive_sample_count_per_part,
    allocate_fps_points,
    apply_batched_fps
)

# Import dataset utilities for easier access
from . import dataset_utils

__all__ = [
    # Submap utilities
    'get_default_num_samples',
    'create_submap_from_frames', 
    'generate_submap_boundaries_for_sample',
    'select_spatially_close_submaps',
    'validate_no_overlap',
    
    # I/O utilities
    'create_metadata_json',
    'convert_to_hdf5',
    'get_dataset_name',
    'load_sample_from_folder',
    'load_sample_from_hdf5',
    'save_processed_sample',
    'save_training_sample',
    
    # Processing utilities
    'process_nss_dataset',
    'process_nss_multi_dataset',
    'process_sequence_with_loader',
    'set_random_seeds',
    'process_tls_dataset',
    
    # Validation utilities
    'perform_dry_run',
    'process_sequences',
    '_validate_and_setup_args',
    
    # Split utilities
    'create_data_splits',
    'create_nss_data_splits', 
    'create_random_data_splits_only',
    'create_sequence_data_splits_only',
    'copy_and_update_data_split',
    
    # Preview utilities
    'preview_data_splits',
    
    # Feature Extraction Metadata Utilities
    'save_processing_metadata',
    'print_detailed_statistics',

    # Point Sampling Utilities
    'calculate_voxel_coverage',
    'calculate_adaptive_sample_count_per_part',
    'allocate_fps_points',
    'apply_batched_fps',
    
    # Dataset utilities
    'dataset_utils',
    'save_num_points_to_folder',
    'save_points_to_ply' # Explicitly expose save_points_to_ply
] 