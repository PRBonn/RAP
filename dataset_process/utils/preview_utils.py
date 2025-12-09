"""
Preview utilities for data splitting in training sample generation.

This module contains functions for previewing what data splits would look like
before actually processing the data.
"""

import random
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


def preview_data_splits(sequence_samples: Dict[str, int], 
                       train_ratio: float, 
                       split_by_sequence: bool, 
                       random_seed: int,
                       data_loader=None,
                       val_sequences: Optional[List[str]] = None):
    """Preview what the data splits would look like without actually creating them."""
    
    if not sequence_samples:
        return
    
    # Check for predefined splits
    using_predefined_splits = False
    predefined_val_sequences = []
    
    if val_sequences is None and data_loader is not None and hasattr(data_loader, 'predefined_split_available') and data_loader.predefined_split_available:
        predefined_val_sequences = data_loader.get_predefined_val_sequences()
        # Only use predefined sequences that would be processed
        predefined_val_sequences = [seq for seq in predefined_val_sequences if seq in sequence_samples]
        if predefined_val_sequences:
            val_sequences = predefined_val_sequences
            using_predefined_splits = True
    
    if using_predefined_splits:
        _preview_predefined_splits(sequence_samples, val_sequences)
    elif val_sequences is not None:
        _preview_manual_splits(sequence_samples, val_sequences)
    else:
        _preview_automatic_splits(sequence_samples, train_ratio, split_by_sequence, random_seed)


def _preview_predefined_splits(sequence_samples: Dict[str, int], val_sequences: List[str]):
    """Preview predefined splits."""
    logger.info("DATA SPLITTING PREVIEW (predefined):")
    logger.info("  Using predefined validation sequences from dataset")
    
    # Calculate splits based on predefined validation sequences
    train_sequences = [seq for seq in sequence_samples.keys() if seq not in val_sequences]
    val_sequences_actual = [seq for seq in sequence_samples.keys() if seq in val_sequences]
    
    train_samples_count = sum(sequence_samples[seq] for seq in train_sequences)
    val_samples_count = sum(sequence_samples[seq] for seq in val_sequences_actual)
    total_samples = train_samples_count + val_samples_count
    actual_train_ratio = train_samples_count / total_samples if total_samples > 0 else 0
    
    logger.info(f"  Split method: predefined")
    logger.info(f"  Predefined train samples: {train_samples_count}")
    logger.info(f"  Predefined val samples: {val_samples_count}")
    logger.info(f"  Predefined actual ratio: {actual_train_ratio:.3f}")
    logger.info(f"  Train sequences: {sorted(train_sequences)}")
    logger.info(f"  Val sequences: {sorted(val_sequences_actual)}")
    logger.info(f"  Data leakage: None (sequences kept separate)")
    
    # Show detailed sequence assignments
    logger.info("  Sequence assignments:")
    for sequence in sorted(sequence_samples.keys()):
        num_samples = sequence_samples[sequence]
        assignment = "VAL" if sequence in val_sequences_actual else "TRAIN"
        logger.info(f"    {sequence}: {num_samples:3d} samples → {assignment}")


def _preview_manual_splits(sequence_samples: Dict[str, int], val_sequences: List[str]):
    """Preview manual splits."""
    logger.info("DATA SPLITTING PREVIEW (manual):")
    logger.info(f"  Using manually specified validation sequences: {val_sequences}")
    
    # Calculate splits based on manual validation sequences
    train_sequences = [seq for seq in sequence_samples.keys() if seq not in val_sequences]
    val_sequences_actual = [seq for seq in sequence_samples.keys() if seq in val_sequences]
    
    train_samples_count = sum(sequence_samples[seq] for seq in train_sequences)
    val_samples_count = sum(sequence_samples[seq] for seq in val_sequences_actual)
    total_samples = train_samples_count + val_samples_count
    actual_train_ratio = train_samples_count / total_samples if total_samples > 0 else 0
    
    logger.info(f"  Split method: manual")
    logger.info(f"  Manual train samples: {train_samples_count}")
    logger.info(f"  Manual val samples: {val_samples_count}")
    logger.info(f"  Manual actual ratio: {actual_train_ratio:.3f}")
    logger.info(f"  Train sequences: {sorted(train_sequences)}")
    logger.info(f"  Val sequences: {sorted(val_sequences_actual)}")
    logger.info(f"  Data leakage: None (sequences kept separate)")
    
    # Show detailed sequence assignments
    logger.info("  Sequence assignments:")
    for sequence in sorted(sequence_samples.keys()):
        num_samples = sequence_samples[sequence]
        assignment = "VAL" if sequence in val_sequences_actual else "TRAIN"
        logger.info(f"    {sequence}: {num_samples:3d} samples → {assignment}")


def _preview_automatic_splits(sequence_samples: Dict[str, int], train_ratio: float, split_by_sequence: bool, random_seed: int):
    """Preview automatic splits."""
    split_method = "sequence-based" if split_by_sequence else "sample-based"
    logger.info(f"DATA SPLITTING PREVIEW ({split_method}):")
    
    # Convert sequence_samples (counts) to mock sample lists for splitting simulation
    mock_sequence_samples = {}
    for sequence, num_samples in sequence_samples.items():
        mock_sequence_samples[sequence] = [f"mock_sample_{i}" for i in range(num_samples)]
    
    if split_by_sequence:
        # Use the actual sequence-based splitting logic
        train_samples, val_samples, train_sequences, val_sequences = _preview_sequence_split(
            mock_sequence_samples, train_ratio, random_seed
        )
    else:
        # Simulate sample-based splitting
        train_samples, val_samples = _preview_sample_split(
            mock_sequence_samples, train_ratio, random_seed
        )
        train_sequences = list(sequence_samples.keys())  # All sequences appear in both
        val_sequences = list(sequence_samples.keys())
    
    total_samples = sum(sequence_samples.values())
    actual_train_ratio = len(train_samples) / total_samples if total_samples > 0 else 0
    
    logger.info(f"  Split method: {split_method}")
    logger.info(f"  Target train ratio: {train_ratio:.3f}")
    logger.info(f"  Predicted train samples: {len(train_samples)}")
    logger.info(f"  Predicted val samples: {len(val_samples)}")
    logger.info(f"  Predicted actual ratio: {actual_train_ratio:.3f}")
    logger.info(f"  Ratio deviation: ±{abs(actual_train_ratio - train_ratio):.3f}")
    
    if split_by_sequence:
        logger.info(f"  Train sequences: {sorted(train_sequences)}")
        logger.info(f"  Val sequences: {sorted(val_sequences)}")
        logger.info(f"  Data leakage: None (sequences kept separate)")
        
        # Show detailed sequence assignments
        logger.info("  Sequence assignments:")
        for sequence in sorted(sequence_samples.keys()):
            num_samples = sequence_samples[sequence]
            assignment = "TRAIN" if sequence in train_sequences else "VAL"
            logger.info(f"    {sequence}: {num_samples:3d} samples → {assignment}")
    else:
        # Count potential data leakage for sample-based splitting
        logger.info(f"  Potential data leakage: All {len(sequence_samples)} sequences")
        logger.info(f"  (samples from each sequence likely in both train/val)")


def _preview_sequence_split(sequence_samples: Dict[str, List[str]], 
                           train_ratio: float, 
                           random_seed: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Preview sequence-based splitting (simplified version of the actual logic)."""
    import random
    random.seed(random_seed)
    
    # Calculate total samples and target
    total_samples = sum(len(samples) for samples in sequence_samples.values())
    target_train_samples = int(total_samples * train_ratio)
    
    # Create sequence info and sort by size
    sequences = list(sequence_samples.items())
    sequences.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Add some randomness while keeping deterministic behavior
    random.shuffle(sequences)
    sequences.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Greedy assignment
    train_samples = []
    val_samples = []
    train_sequences = []
    val_sequences = []
    current_train_count = 0
    
    for seq_name, samples in sequences:
        num_samples = len(samples)
        
        if current_train_count + num_samples <= target_train_samples:
            train_samples.extend(samples)
            train_sequences.append(seq_name)
            current_train_count += num_samples
        else:
            # Check which assignment gives better ratio
            train_ratio_if_added = (current_train_count + num_samples) / total_samples
            train_ratio_if_not = current_train_count / total_samples
            
            diff_if_added = abs(train_ratio_if_added - train_ratio)
            diff_if_not = abs(train_ratio_if_not - train_ratio)
            
            if diff_if_added < diff_if_not:
                train_samples.extend(samples)
                train_sequences.append(seq_name)
                current_train_count += num_samples
            else:
                val_samples.extend(samples)
                val_sequences.append(seq_name)
    
    return train_samples, val_samples, train_sequences, val_sequences


def _preview_sample_split(sequence_samples: Dict[str, List[str]], 
                         train_ratio: float, 
                         random_seed: int) -> Tuple[List[str], List[str]]:
    """Preview sample-based splitting."""
    import random
    random.seed(random_seed)
    
    # Collect all samples
    all_samples = []
    for samples in sequence_samples.values():
        all_samples.extend(samples)
    
    # Calculate split
    num_train = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:]
    
    return train_samples, val_samples 