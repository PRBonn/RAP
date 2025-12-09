"""
Data splitting utilities for training sample generation.

This module contains functions for creating train/val splits for different datasets.
"""

import os
import random
import logging
from typing import List, Tuple, Optional, Dict, Set
import json
try:
    import h5py
except ImportError:
    h5py = None  # Optional dependency

import glob
import shutil

try:
    from .preview_utils import preview_data_splits
except ImportError:
    preview_data_splits = None  # Optional dependency

try:
    from .io_utils import get_dataset_name  # For convert_to_hdf5 to get dataset name
except ImportError:
    get_dataset_name = None  # Optional dependency

logger = logging.getLogger(__name__)


def split_by_sequence(sequence_samples: Dict[str, List[str]], 
                     train_ratio: float, 
                     random_seed: int,
                     loop_closure_sequences: Optional[Set[str]] = None,
                     guarantee_loop_closure: bool = False,
                     val_sequences: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Split samples by sequence, keeping entire sequences together in train or val.
    
    Args:
        sequence_samples: Dict mapping sequence names to lists of sample paths
        train_ratio: Desired ratio of training samples (ignored if val_sequences is provided)
        random_seed: Random seed for reproducibility
        loop_closure_sequences: Set of sequence names that have loop closures
        guarantee_loop_closure: If True, guarantee at least one loop closure sequence in train
        val_sequences: List of sequence names to use for validation. If provided, overrides automatic splitting.
    
    Returns:
        Tuple of (train_samples, val_samples)
    """
    random.seed(random_seed)
    
    # Calculate sequence statistics
    sequence_info = []
    total_samples = 0
    
    for sequence, samples in sequence_samples.items():
        num_samples = len(samples)
        has_loop_closure = loop_closure_sequences is not None and sequence in loop_closure_sequences
        sequence_info.append({
            'sequence': sequence,
            'samples': samples,
            'num_samples': num_samples,
            'has_loop_closure': has_loop_closure
        })
        total_samples += num_samples
    
    if total_samples == 0:
        logger.warning("No samples found for splitting")
        return [], []
    
    # If val_sequences is provided, use manual splitting
    if val_sequences is not None:
        logger.info(f"Using manual validation sequences: {val_sequences}")
        
        # Validate that all specified validation sequences exist
        available_sequences = set(sequence_samples.keys())
        invalid_val_sequences = [seq for seq in val_sequences if seq not in available_sequences]
        if invalid_val_sequences:
            logger.warning(f"Invalid validation sequences specified: {invalid_val_sequences}")
            logger.warning(f"Available sequences: {sorted(available_sequences)}")
            # Remove invalid sequences
            val_sequences = [seq for seq in val_sequences if seq in available_sequences]
        
        # Split based on manual specification
        train_samples = []
        val_samples = []
        train_sequences = []
        val_sequences_set = set(val_sequences)
        
        for seq_info in sequence_info:
            if seq_info['sequence'] in val_sequences_set:
                val_samples.extend(seq_info['samples'])
                logger.debug(f"Sequence {seq_info['sequence']}: {seq_info['num_samples']} samples -> val (manual)")
            else:
                train_samples.extend(seq_info['samples'])
                train_sequences.append(seq_info['sequence'])
                logger.debug(f"Sequence {seq_info['sequence']}: {seq_info['num_samples']} samples -> train (manual)")
        
        actual_train_ratio = len(train_samples) / total_samples if total_samples > 0 else 0
        logger.info(f"Manual sequence-based split: {len(train_samples)} train, {len(val_samples)} val")
        logger.info(f"Actual train ratio: {actual_train_ratio:.3f}")
        logger.info(f"Train sequences: {sorted(train_sequences)}")
        logger.info(f"Val sequences: {sorted(val_sequences)}")
        
        return train_samples, val_samples
    
    # Original automatic splitting logic
    # Target number of training samples
    target_train_samples = int(total_samples * train_ratio)
    
    # Sort sequences by number of samples (descending) for better greedy allocation
    sequence_info.sort(key=lambda x: x['num_samples'], reverse=True)
    
    # Add some randomness while keeping deterministic behavior
    random.shuffle(sequence_info)
    sequence_info.sort(key=lambda x: x['num_samples'], reverse=True)
    
    # If guaranteeing loop closure, prioritize loop closure sequences
    if guarantee_loop_closure and loop_closure_sequences:
        loop_closure_info = [seq for seq in sequence_info if seq['has_loop_closure']]
        non_loop_info = [seq for seq in sequence_info if not seq['has_loop_closure']]
        
        if loop_closure_info:
            # Sort loop closure sequences by size (descending)
            loop_closure_info.sort(key=lambda x: x['num_samples'], reverse=True)
            random.shuffle(loop_closure_info)
            loop_closure_info.sort(key=lambda x: x['num_samples'], reverse=True)
            
            # Reconstruct sequence_info with loop closure sequences first
            sequence_info = loop_closure_info + non_loop_info
    
    # Greedy assignment: try to get as close as possible to target ratio
    train_samples = []
    val_samples = []
    current_train_count = 0
    has_loop_closure_in_train = False
    
    for seq_info in sequence_info:
        # If guaranteeing loop closure and we don't have one yet, prioritize loop closure sequences
        if guarantee_loop_closure and not has_loop_closure_in_train and seq_info['has_loop_closure']:
            # Force this loop closure sequence into train
            train_samples.extend(seq_info['samples'])
            current_train_count += seq_info['num_samples']
            has_loop_closure_in_train = True
            assignment = "train (loop closure guarantee)"
        else:
            # Normal greedy assignment logic
            if current_train_count + seq_info['num_samples'] <= target_train_samples:
                # Add to train if it doesn't exceed target
                train_samples.extend(seq_info['samples'])
                current_train_count += seq_info['num_samples']
                assignment = "train"
            else:
                # Check if adding to train or val gives us a ratio closer to target
                train_ratio_if_added = (current_train_count + seq_info['num_samples']) / total_samples
                train_ratio_if_not_added = current_train_count / total_samples
                
                diff_if_added = abs(train_ratio_if_added - train_ratio)
                diff_if_not_added = abs(train_ratio_if_not_added - train_ratio)
                
                if diff_if_added < diff_if_not_added:
                    # Adding to train gives better ratio
                    train_samples.extend(seq_info['samples'])
                    current_train_count += seq_info['num_samples']
                    assignment = "train"
                else:
                    # Adding to val gives better ratio
                    val_samples.extend(seq_info['samples'])
                    assignment = "val"
        
        logger.debug(f"Sequence {seq_info['sequence']}: {seq_info['num_samples']} samples -> {assignment}")
    
    actual_train_ratio = current_train_count / total_samples if total_samples > 0 else 0
    logger.info(f"Sequence-based split: {len(train_samples)} train, {len(val_samples)} val")
    logger.info(f"Actual train ratio: {actual_train_ratio:.3f} (target: {train_ratio:.3f})")
    
    if guarantee_loop_closure and loop_closure_sequences:
        train_loop_sequences = [seq['sequence'] for seq in sequence_info 
                              if seq['has_loop_closure'] and any(sample in train_samples for sample in seq['samples'])]
        logger.info(f"Loop closure sequences in train: {train_loop_sequences}")
    
    # Log sequence assignments
    train_sequences = []
    val_sequences = []
    for seq_info in sequence_info:
        if any(sample in train_samples for sample in seq_info['samples']):
            train_sequences.append(seq_info['sequence'])
        else:
            val_sequences.append(seq_info['sequence'])
    
    logger.info(f"Train sequences: {sorted(train_sequences)}")
    logger.info(f"Val sequences: {sorted(val_sequences)}")
    
    return train_samples, val_samples


def split_by_sequence_mixed_val(sequence_samples: Dict[str, List[str]], 
                               train_ratio: float, 
                               random_seed: int,
                               loop_closure_sequences: Optional[Set[str]] = None,
                               guarantee_loop_closure: bool = False,
                               val_sequences: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Split samples by sequence with mixed validation set.
    
    This creates a validation set that includes:
    1. All samples from validation sequences (sequence-based)
    2. Additional randomly selected (1-train_ratio) samples from training sequences
    """
    random.seed(random_seed)
    
    # First, do the regular sequence-based split to get base train/val sequences
    base_train_samples, base_val_samples = split_by_sequence(
        sequence_samples, train_ratio, random_seed, 
        loop_closure_sequences, guarantee_loop_closure, val_sequences
    )
    
    # Identify which sequences are in train vs val
    train_sequences = set()
    val_sequences = set()
    
    for sequence, samples in sequence_samples.items():
        if any(sample in base_train_samples for sample in samples):
            train_sequences.add(sequence)
        else:
            val_sequences.add(sequence)
    
    logger.info(f"Base split - Train sequences: {sorted(train_sequences)}, Val sequences: {sorted(val_sequences)}")
    
    # Calculate how many additional samples to add to validation from training sequences
    total_samples = len(base_train_samples) + len(base_val_samples)
    target_val_samples = int(total_samples * (1 - train_ratio))
    current_val_samples = len(base_val_samples)
    additional_val_needed = max(0, target_val_samples - current_val_samples)
    
    logger.info(f"Mixed validation split: Current val samples: {current_val_samples}, Target: {target_val_samples}, Additional needed: {additional_val_needed}")
    
    if additional_val_needed > 0 and base_train_samples:
        # Randomly select additional samples from training sequences
        random.seed(random_seed + 1)  # Use different seed to avoid affecting base split
        additional_val_samples = random.sample(base_train_samples, min(additional_val_needed, len(base_train_samples)))
        
        # Create final splits
        final_train_samples = [sample for sample in base_train_samples if sample not in additional_val_samples]
        final_val_samples = base_val_samples + additional_val_samples
        
        logger.info(f"Added {len(additional_val_samples)} random samples from training sequences to validation")
        logger.info(f"Final mixed split: {len(final_train_samples)} train, {len(final_val_samples)} val")
        
        return final_train_samples, final_val_samples
    else:
        logger.info("No additional validation samples needed, using base sequence split")
        return base_train_samples, base_val_samples


def create_data_splits(output_dir: str, 
                      sequence_stats: Dict,
                      train_ratio: float = 0.8,
                      random_seed: int = 42,
                      split_by_sequence_on: bool = False,
                      data_loader = None,
                      guarantee_loop_closure: bool = False,
                      val_sequences: Optional[List[str]] = None,
                      mixed_val_split: bool = False) -> Dict:
    """Create both sequence-based and sample-based train/val splits for the generated samples."""
    logger.info("Creating both sequence-based and sample-based data splits...")
    
    random.seed(random_seed)
    
    # Collect all sample paths organized by sequence
    sequence_samples = {}
    all_samples = []
    
    for sequence, stats in sequence_stats.items():
        sequence_dir = os.path.join(output_dir, os.path.basename(output_dir), sequence)
        if os.path.exists(sequence_dir):
            sample_dirs = [d for d in os.listdir(sequence_dir) if d.startswith('sample_')]
            sequence_sample_paths = [f"{os.path.basename(output_dir)}/{sequence}/{sample_dir}" for sample_dir in sample_dirs]
            sequence_samples[sequence] = sequence_sample_paths
            all_samples.extend(sequence_sample_paths)
    
    # Create splits directory
    data_split_dir = os.path.join(output_dir, "data_split")
    os.makedirs(data_split_dir, exist_ok=True)
    
    # Get loop closure sequences if data_loader is provided
    loop_closure_sequences = None
    if data_loader is not None:
        loop_closure_sequences = data_loader.get_loop_closure_sequences()
        logger.info(f"Loop closure sequences: {sorted(loop_closure_sequences)}")
    
    # 1. Create sequence-based splits (prevent data leakage)
    if mixed_val_split:
        logger.info("Creating mixed validation splits (train_mixed.txt, val_mixed.txt)...")
        seq_train_samples, seq_val_samples = split_by_sequence_mixed_val(
            sequence_samples, train_ratio, random_seed, 
            loop_closure_sequences, guarantee_loop_closure, val_sequences
        )
        
        seq_train_file = os.path.join(data_split_dir, "train_mixed.txt")
        seq_val_file = os.path.join(data_split_dir, "val_mixed.txt")
    else:
        logger.info("Creating sequence-based splits (train.txt, val.txt)...")
        seq_train_samples, seq_val_samples = split_by_sequence(
            sequence_samples, train_ratio, random_seed, 
            loop_closure_sequences, guarantee_loop_closure, val_sequences
        )
        
        seq_train_file = os.path.join(data_split_dir, "train.txt")
        seq_val_file = os.path.join(data_split_dir, "val.txt")
    
    for file_path, samples in [(seq_train_file, seq_train_samples), (seq_val_file, seq_val_samples)]:
        with open(file_path, 'w') as f:
            f.write('\n'.join(samples) + '\n')
    
    seq_actual_train_ratio = len(seq_train_samples) / len(all_samples) if len(all_samples) > 0 else 0
    split_type = "Mixed validation" if mixed_val_split else "Sequence-based"
    logger.info(f"{split_type} splits: {len(seq_train_samples)} train, {len(seq_val_samples)} val")
    logger.info(f"{split_type} actual train ratio: {seq_actual_train_ratio:.3f} (target: {train_ratio:.3f})")
    
    # 2. Create sample-based splits (random)
    logger.info("Creating sample-based splits (train_random.txt, val_random.txt)...")
    random.seed(random_seed)  # Reset seed for consistent randomization
    random_all_samples = all_samples.copy()  # Create copy to avoid modifying original
    random.shuffle(random_all_samples)
    num_train = int(len(random_all_samples) * train_ratio)
    rand_train_samples, rand_val_samples = random_all_samples[:num_train], random_all_samples[num_train:]
    
    rand_train_file = os.path.join(data_split_dir, "train_random.txt")
    rand_val_file = os.path.join(data_split_dir, "val_random.txt")
    
    for file_path, samples in [(rand_train_file, rand_train_samples), (rand_val_file, rand_val_samples)]:
        with open(file_path, 'w') as f:
            f.write('\n'.join(samples) + '\n')
    
    rand_actual_train_ratio = len(rand_train_samples) / len(all_samples) if len(all_samples) > 0 else 0
    logger.info(f"Sample-based splits: {len(rand_train_samples)} train, {len(rand_val_samples)} val")
    logger.info(f"Sample-based actual train ratio: {rand_actual_train_ratio:.3f} (target: {train_ratio:.3f})")
    
    # Determine which split to use as primary (based on original split_by_sequence preference)
    if split_by_sequence_on:
        primary_train_samples, primary_val_samples = seq_train_samples, seq_val_samples
        primary_train_file, primary_val_file = seq_train_file, seq_val_file
        primary_actual_ratio = seq_actual_train_ratio
        primary_method = "mixed-validation" if mixed_val_split else "sequence-based"
    else:
        primary_train_samples, primary_val_samples = rand_train_samples, rand_val_samples
        primary_train_file, primary_val_file = rand_train_file, rand_val_file
        primary_actual_ratio = rand_actual_train_ratio
        primary_method = "sample-based"
    
    split_info = {
        'total_samples': len(all_samples),
        'train_samples': len(primary_train_samples),
        'val_samples': len(primary_val_samples),
        'target_train_ratio': train_ratio,
        'actual_train_ratio': primary_actual_ratio,
        'split_method': primary_method,
        'split_by_sequence_on': split_by_sequence_on,
        'mixed_val_split': mixed_val_split,
        'train_file': primary_train_file,
        'val_file': primary_val_file,
        # Add information about both splits
        'sequence_based': {
            'train_samples': len(seq_train_samples),
            'val_samples': len(seq_val_samples),
            'actual_train_ratio': seq_actual_train_ratio,
            'train_file': seq_train_file,
            'val_file': seq_val_file,
            'mixed_val_split': mixed_val_split
        },
        'sample_based': {
            'train_samples': len(rand_train_samples),
            'val_samples': len(rand_val_samples),
            'actual_train_ratio': rand_actual_train_ratio,
            'train_file': rand_train_file,
            'val_file': rand_val_file
        }
    }
    
    logger.info("=" * 50)
    logger.info("CREATED BOTH TYPES OF DATA SPLITS:")
    split_desc = f"  {'Mixed validation' if mixed_val_split else 'Sequence-based'} ({'train_mixed.txt, val_mixed.txt' if mixed_val_split else 'train.txt, val.txt'}): {len(seq_train_samples)} train, {len(seq_val_samples)} val"
    logger.info(split_desc)
    logger.info(f"  Sample-based (train_random.txt, val_random.txt): {len(rand_train_samples)} train, {len(rand_val_samples)} val")
    logger.info(f"  Primary split method: {primary_method}")
    logger.info("=" * 50)
    
    return split_info


def create_nss_data_splits(output_dir: str, sequence_stats: Dict) -> Dict:
    """
    Create data splits for NSS and other non-sequential datasets.
    For NSS, directly use the sequence_stats which contains train/val splits
    without needing sequence folder structure.
    """
    logger.info("Creating NSS data splits (no random splitting - using predefined train/val)")
    
    # Get dataset name from output directory
    dataset_name = os.path.basename(output_dir)
    
    # For NSS, check the actual directory structure
    train_samples = []
    val_samples = []
    all_samples = []
    
    # NSS samples are saved in subdirectories: output_dir/dataset_name/train/ and output_dir/dataset_name/val/
    sequences_dir = os.path.join(output_dir, dataset_name)
    
    if os.path.exists(sequences_dir):
        # Check for train and val subdirectories
        for split_name in ['train', 'val']:
            split_dir = os.path.join(sequences_dir, split_name)
            if os.path.exists(split_dir):
                sample_dirs = [d for d in os.listdir(split_dir) if d.startswith('sample_') and os.path.isdir(os.path.join(split_dir, d))]
                if sample_dirs:
                    split_samples = [f"{dataset_name}/{split_name}/{sample_dir}" for sample_dir in sorted(sample_dirs)]
                    all_samples.extend(split_samples)
                    
                    if split_name == 'train':
                        train_samples = split_samples
                    elif split_name == 'val':
                        val_samples = split_samples
                    
                    logger.info(f"Found {len(split_samples)} samples in {split_name} split")
        
        if not all_samples:
            # Fallback: check if samples are directly in sequences_dir (single split case)
            sample_dirs = [d for d in os.listdir(sequences_dir) if d.startswith('sample_') and os.path.isdir(os.path.join(sequences_dir, d))]
            if sample_dirs:
                all_samples = [f"{dataset_name}/{sample_dir}" for sample_dir in sorted(sample_dirs)]
                
                # Determine split based on sequence_stats
                if 'train' in sequence_stats and 'val' not in sequence_stats:
                    train_samples = all_samples
                    val_samples = []
                elif 'val' in sequence_stats and 'train' not in sequence_stats:
                    train_samples = []
                    val_samples = all_samples
                else:
                    # Default: treat all as train
                    train_samples = all_samples
                    val_samples = []
                
                logger.info(f"Found {len(all_samples)} samples in sequences directory")
            else:
                raise ValueError(f"No sample directories found in sequences directory: {sequences_dir}")
    else:
        raise ValueError(f"Sequences directory does not exist: {sequences_dir}")
    
    logger.info(f"Found {len(all_samples)} total samples: {len(train_samples)} train, {len(val_samples)} val")
    
    # Create data_split directory
    data_split_dir = os.path.join(output_dir, "data_split")
    os.makedirs(data_split_dir, exist_ok=True)
    
    # Write split files
    train_file = os.path.join(data_split_dir, "train.txt")
    val_file = os.path.join(data_split_dir, "val.txt")
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_samples) + '\n')
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_samples) + '\n')
    
    actual_train_ratio = len(train_samples) / len(all_samples) if len(all_samples) > 0 else 0
    
    split_info = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'actual_train_ratio': actual_train_ratio,
        'split_method': 'predefined',
        'split_by_sequence_on': False,  # NSS doesn't use sequence folder structure
        'train_file': train_file,
        'val_file': val_file,
        'dataset_name': dataset_name,
        'sequences': list(sequence_stats.keys()),
        'random_splits_created': False  # No random splits for NSS
    }
    
    logger.info(f"NSS splits created: {len(train_samples)} train, {len(val_samples)} val")
    logger.info(f"Actual train ratio: {actual_train_ratio:.3f}")
    logger.info(f"Available splits: {list(sequence_stats.keys())}")
    
    return split_info


def create_random_data_splits_only(output_dir: str, train_ratio: float, random_seed: int) -> Dict:
    """
    Create random data split files for an already processed dataset without processing any data.
    
    Args:
        output_dir: Directory containing existing samples
        train_ratio: Ratio of samples to use for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split information
    """
    # Find all existing sample directories that contain sample_*.ply files
    sample_dirs_raw = []
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if dir_name.startswith('sample_'):
                sample_path = os.path.join(root, dir_name)
                # Check if this sample directory contains any .ply files
                if glob.glob(os.path.join(sample_path, "*.ply")):
                    sample_dirs_raw.append(sample_path)

    if not sample_dirs_raw:
        raise ValueError(f"No sample directories containing .ply files found in {output_dir}")

    # Convert to relative paths, ensuring uniqueness
    # The relative path should be from output_dir to the sample_ directory
    relative_sample_dirs = sorted(list(set(os.path.relpath(d, output_dir) for d in sample_dirs_raw)))

    if not relative_sample_dirs:
        raise ValueError(f"No unique relative sample directories found in {output_dir}")

    # Create random split
    random.seed(random_seed)
    random.shuffle(relative_sample_dirs)

    num_train = int(len(relative_sample_dirs) * train_ratio)
    train_samples = relative_sample_dirs[:num_train]
    val_samples = relative_sample_dirs[num_train:]

    # Create splits directory
    data_split_dir = os.path.join(output_dir, "data_split")
    os.makedirs(data_split_dir, exist_ok=True)

    # Write split files
    train_file = os.path.join(data_split_dir, "train_random.txt")
    val_file = os.path.join(data_split_dir, "val_random.txt")

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_samples) + '\n') # Add newline at end of file

    with open(val_file, 'w') as f:
        f.write('\n'.join(val_samples) + '\n') # Add newline at end of file

    logger.info(f"Created random split files: {train_file}, {val_file}")

    return {
        'total_samples': len(relative_sample_dirs),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'split_by_sequence_on': False,
        'train_file': train_file, # Add to return info
        'val_file': val_file     # Add to return info
    }


def create_sequence_data_splits_only(output_dir: str, 
                                   train_ratio: float, 
                                   random_seed: int,
                                   val_sequences: Optional[List[str]] = None,
                                   data_loader = None,
                                   guarantee_loop_closure: bool = False,
                                   mixed_val_split: bool = False) -> Dict:
    """
    Create sequence-based data split files for an already processed dataset without processing any data.
    
    Args:
        output_dir: Directory containing existing samples
        train_ratio: Ratio of samples to use for training
        random_seed: Random seed for reproducibility
        val_sequences: List of sequence names to use for validation
        data_loader: Data loader instance for loop closure information
        guarantee_loop_closure: Guarantee at least one loop closure sequence in training
        mixed_val_split: Create mixed validation split
        
    Returns:
        Dictionary with split information
    """
    # Find all existing sample directories (sequences)
    sequence_samples = {}
    
    # Look for sample files in subdirectories
    # First, check the main output directory structure
    main_dataset_dir = None
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item != 'data_split':
            # Check if this looks like the main dataset directory
            # (contains subdirectories that might be scenes/sequences)
            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            if subdirs:
                main_dataset_dir = item_path
                break
    
    # If we found a main dataset directory, look inside it
    search_dirs = [main_dataset_dir] if main_dataset_dir else [output_dir]
    
    for search_dir in search_dirs:
        for subdir in os.listdir(search_dir):
            subdir_path = os.path.join(search_dir, subdir)
            if os.path.isdir(subdir_path) and subdir != 'data_split':
                # Look for sample_ directories within this subdir_path
                found_sample_dirs_in_sequence = []
                for root, dirs, files in os.walk(subdir_path):
                    for dir_name in dirs:
                        if dir_name.startswith('sample_'):
                            sample_full_path = os.path.join(root, dir_name)
                            # Only add if it contains .ply files
                            if glob.glob(os.path.join(sample_full_path, "*.ply")):
                                found_sample_dirs_in_sequence.append(sample_full_path)
                
                if found_sample_dirs_in_sequence:
                    # Use the subdir as the sequence name
                    sequence_name = os.path.relpath(subdir_path, output_dir)
                    # Convert sample full paths to relative paths from output_dir
                    # and ensure uniqueness
                    relative_samples_for_sequence = sorted(list(set(os.path.relpath(s, output_dir) for s in found_sample_dirs_in_sequence)))
                    sequence_samples[sequence_name] = relative_samples_for_sequence
    
    if not sequence_samples:
        raise ValueError(f"No sample directories found in sequence subdirectories of {output_dir}")
    
    # Get loop closure information if available
    loop_closure_sequences = None
    if data_loader and hasattr(data_loader, 'get_loop_closure_sequences'):
        try:
            loop_closure_sequences = set(data_loader.get_loop_closure_sequences())
            logger.info(f"Found loop closure sequences: {loop_closure_sequences}")
        except Exception as e:
            logger.warning(f"Could not get loop closure information: {e}")
    
    # Create sequence-based split
    if mixed_val_split:
        train_samples, val_samples = split_by_sequence_mixed_val(
            sequence_samples=sequence_samples,
            train_ratio=train_ratio,
            random_seed=random_seed,
            loop_closure_sequences=loop_closure_sequences,
            guarantee_loop_closure=guarantee_loop_closure,
            val_sequences=val_sequences
        )
    else:
        train_samples, val_samples = split_by_sequence(
            sequence_samples=sequence_samples,
            train_ratio=train_ratio,
            random_seed=random_seed,
            loop_closure_sequences=loop_closure_sequences,
            guarantee_loop_closure=guarantee_loop_closure,
            val_sequences=val_sequences
        )
    
    # Create data_split directory
    data_split_dir = os.path.join(output_dir, "data_split")
    os.makedirs(data_split_dir, exist_ok=True)

    # Write split files
    train_file = os.path.join(data_split_dir, "train.txt")
    val_file = os.path.join(data_split_dir, "val.txt")

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_samples) + '\n') # Add newline at end of file

    with open(val_file, 'w') as f:
        f.write('\n'.join(val_samples) + '\n') # Add newline at end of file

    logger.info(f"Created sequence-based split files: {train_file}, {val_file}")

    total_samples_count = sum(len(samples) for samples in sequence_samples.values())

    return {
        'total_samples': total_samples_count,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'split_by_sequence_on': True,
        'sequence_samples': sequence_samples,
        'train_file': train_file, # Add to return info
        'val_file': val_file     # Add to return info
    }


def copy_and_update_data_split(input_dir: str, 
                               output_dir: str, 
                               dataset_name: str):
    """Copy and update data_split folder with new paths."""
    # Try multiple possible locations for data_split
    possible_locations = [
        os.path.join(input_dir, 'data_split'),  # At input_dir level
        os.path.join(input_dir, dataset_name, 'data_split'),  # At dataset level
    ]
    
    input_data_split = None
    for location in possible_locations:
        if os.path.exists(location):
            input_data_split = location
            break
    
    if input_data_split is None:
        logger.info(f"No data_split folder found in any expected location")
        return
    
    output_data_split = os.path.join(output_dir, 'data_split')
    
    # Create output data_split directory
    os.makedirs(output_data_split, exist_ok=True)
    
    # Process each txt file in data_split
    for filename in os.listdir(input_data_split):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_data_split, filename)
            output_file = os.path.join(output_data_split, filename)
            
            try:
                with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            # Update path to include _processed suffix
                            updated_line = line + '_processed'
                            f_out.write(updated_line + '\n')
                
                logger.info(f"Updated and copied: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to update {filename}: {e}")
    
    logger.info(f"Data split files copied and updated from {input_data_split} to {output_data_split}")


def create_threedmatch_test_data_splits(output_dir: str, sequence_stats: Dict) -> Dict:
    """
    Create data splits for ThreeDMatch test dataset.
    All samples go to validation split (no training samples).
    
    Args:
        output_dir: Output directory containing the processed samples
        sequence_stats: Dictionary containing statistics for each sequence
        
    Returns:
        Dictionary containing split information
    """
    logger.info("Creating ThreeDMatch test data splits (all samples go to validation)")
    
    dataset_name = os.path.basename(os.path.normpath(output_dir))
    sequences_dir = os.path.join(output_dir, dataset_name)
    
    all_samples = []
    train_samples = []  # Always empty for test dataset
    val_samples = []
    
    if os.path.exists(sequences_dir):
        # Collect all samples from all sequences
        for sequence_name in sequence_stats.keys():
            sequence_dir = os.path.join(sequences_dir, sequence_name)
            if os.path.exists(sequence_dir):
                sample_dirs = [d for d in os.listdir(sequence_dir) 
                              if d.startswith('sample_') and os.path.isdir(os.path.join(sequence_dir, d))]
                if sample_dirs:
                    sequence_samples = [f"{dataset_name}/{sequence_name}/{sample_dir}" 
                                      for sample_dir in sorted(sample_dirs)]
                    all_samples.extend(sequence_samples)
                    val_samples.extend(sequence_samples)  # All samples go to validation
                    logger.info(f"Found {len(sequence_samples)} samples in sequence {sequence_name}")
        
        if not all_samples:
            raise ValueError(f"No sample directories found in sequences directory: {sequences_dir}")
    else:
        raise ValueError(f"Sequences directory does not exist: {sequences_dir}")
    
    logger.info(f"Found {len(all_samples)} total samples: {len(train_samples)} train, {len(val_samples)} val")
    
    # Create data_split directory
    data_split_dir = os.path.join(output_dir, "data_split")
    os.makedirs(data_split_dir, exist_ok=True)
    
    # Write train.txt (empty for test dataset)
    train_file = os.path.join(data_split_dir, "train.txt")
    with open(train_file, 'w') as f:
        pass  # Write empty file
    logger.info(f"Created empty train split file: {train_file}")
    
    # Write val.txt (all samples)
    val_file = os.path.join(data_split_dir, "val.txt")
    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    logger.info(f"Created val split file with {len(val_samples)} samples: {val_file}")
    
    # Return split information
    split_info = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_ratio': 0.0,  # No training samples
        'val_ratio': 1.0,    # All samples are validation
        'split_by_sequence': True,
        'split_method': 'threedmatch_test_all_val',
        'train_sequences': [],
        'val_sequences': list(sequence_stats.keys()),
        'dataset_type': 'test_only'
    }
    
    logger.info("ThreeDMatch test data splits created successfully")
    logger.info(f"Split summary: {len(train_samples)} train, {len(val_samples)} val samples")
    
    return split_info 