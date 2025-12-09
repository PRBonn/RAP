import logging
import random
import time
import bisect
from typing import List, Optional, Dict
import os

import h5py
import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from .dataset import PointCloudDataset

logger = logging.getLogger("Data")


def worker_init_fn(worker_id):
    """Worker function for initializing the h5 file."""
    worker_info = torch.utils.data.get_worker_info()
    concat_dataset: ConcatDataset = worker_info.dataset
    for dataset in concat_dataset.datasets:
        # Handle both direct PointCloudDataset and RandomSampledDataset wrapper
        if hasattr(dataset, '_h5_file') and dataset._h5_file is None and not dataset.use_folder:
            dataset._h5_file = h5py.File(
                dataset.data_path, "r", 
                libver='latest', 
                swmr=True, 
                rdcc_nbytes=256*1024*1024,  # increase the data chunk size to 256MB
                rdcc_nslots=1024*1024,      # increase the number of slots to 1M
            )


class ConcatPointCloudDataset(ConcatDataset):
    def __init__(self, datasets: list[PointCloudDataset]):
        super().__init__(datasets)
        self.sample_sizes = np.array([
            self.estimate_num_points(i) for i in range(len(self))
        ], dtype=np.int32)

    def estimate_num_points(self, idx: int) -> int:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        # Use bisect to find the dataset index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].estimate_num_points(sample_idx)


class DynamicBatchSampler(Sampler[list[int]]):
    """A batch sampler that packs samples until max_points_per_batch, shards the full index set across ranks.

    Args:
        dataset: any object with .estimate_num_points(idx: int) -> int, e.g. ConcatPointCloudDataset
        max_points_per_batch: maximum total points in a batch
        shuffle: whether to shuffle samples each epoch
        drop_last: if True, drop the final batch if its total < max_points_per_batch
        seed: base RNG seed for shuffle reproducibility
    """

    def __init__(
        self,
        dataset: ConcatPointCloudDataset,
        max_points_per_batch: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.max_points = max_points_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
    
    # GPU id and count
    def _get_rank_and_size(self):
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def set_epoch(self, epoch: int):
        """To be called at start of each epoch for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self):
        """Yields batches sequentially."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            rnd = random.Random(self.seed + self.epoch)
            rnd.shuffle(indices)

        # shard across DDP ranks
        rank, world_size = self._get_rank_and_size()
        if world_size > 1:
            indices = indices[rank :: world_size]

        # pack into dynamic batches
        batch, acc_pts = [], 0
        batches = []
        for idx in indices:
            pts = self.dataset.estimate_num_points(idx)
            if batch and (acc_pts + pts > self.max_points):
                # this one is the real batching information
                # print(f"Yielding batch - acc_pts: {acc_pts}, batch_size: {len(batch)}")
                batches.append(batch)
                batch, acc_pts = [], 0
            batch.append(idx)
            acc_pts += pts

        # final batch
        if batch and not self.drop_last: # final batch (when drop last is False)
            # print(f"Final batch - acc_pts: {acc_pts}, batch_size: {len(batch)}")
            batches.append(batch)
        
        # Pad with repeated batches to ensure all GPUs have the same number of batches
        if world_size > 1:
            max_batches = self.__len__()
            while len(batches) < max_batches:
                # Repeat the last batch to maintain synchronization
                if batches:
                    batches.append(batches[-1])
                else:
                    # If no batches at all, create a minimal batch with the first sample
                    if indices:
                        batches.append([indices[0]])
                    else:
                        break
        
        # Yield all batches
        for batch in batches:
            yield batch

    def _compute_num_batches(self, rank: int, world_size: int) -> int:
        indices = list(range(len(self.dataset)))
        indices = indices[rank :: world_size]
        count, acc_pts = 0, 0
        for idx in indices:
            pts = self.dataset.estimate_num_points(idx)
            if acc_pts + pts > self.max_points:
                count += 1
                acc_pts = 0
            acc_pts += pts
        if acc_pts and not self.drop_last:
            count += 1
        return count

    def __len__(self) -> int:
        """Length computed by the maximum number of batches over all ranks to ensure all GPUs complete."""
        _, world_size = self._get_rank_and_size()
        max_count = 0
        for rank in range(world_size):
            count = self._compute_num_batches(rank, world_size)
            max_count = max(max_count, count)
        return max_count


# collate that handles fixed-size vs variable-size
def variable_collate_fn(batch: list[dict]):
    """Collate function that handles fixed point count vs variable point count."""
    lengths = [b["pointclouds"].shape[0] for b in batch]
    total_points = sum(lengths)
    # print(f"Batch collated: {len(batch)} samples, {total_points} total points, points per sample: {lengths}")
    
    cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int64)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    out = {"cu_seqlens": cu_seqlens}

    # Per-point keys (concatenate)
    for k in [
        "pointclouds", "pointclouds_gt", "pointclouds_normals", "pointclouds_normals_gt", "part_indices", "anchor_indices", "features"
    ]:
        arrs = [b[k] for b in batch]
        out[k] = torch.from_numpy(np.concatenate(arrs, axis=0))

    # Per-sample and per-part keys (stack along new batch dim)
    for k in [
        "rotations", "translations", "points_per_part", "anchor_parts", "init_rotation", "scales", 
        "global_rotation", "global_translation"
    ]:
        arrs = [torch.from_numpy(b[k]) for b in batch]
        out[k] = torch.stack(arrs, dim=0)

    # Rest of the keys (list)
    for k in ["index", "name", "dataset_name", "num_parts", "overlap_threshold"]:
        out[k] = [b[k] for b in batch]
    # print(f"{time.time()}: Variable collate fn done")
    return out


class RandomSampledDataset(Dataset):
    """A dataset wrapper that randomly samples a subset of indices from the original dataset each epoch."""
    
    def __init__(self, dataset: Dataset, num_samples: int, seed: Optional[int] = None):
        """
        Args:
            dataset: The original dataset to sample from
            num_samples: Number of samples to use per epoch (if None, use all samples)
            seed: Random seed for reproducibility (if None, use random seed)
        """
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self._current_indices = None
        self._update_indices()
        
        # Forward HDF5-related attributes for worker initialization
        if hasattr(dataset, '_h5_file'):
            self._h5_file = dataset._h5_file
        if hasattr(dataset, 'use_folder'):
            self.use_folder = dataset.use_folder
        if hasattr(dataset, 'data_path'):
            self.data_path = dataset.data_path
    
    def estimate_num_points(self, idx: int) -> int:
        """Forward estimate_num_points to the underlying dataset."""
        return self.dataset.estimate_num_points(self._current_indices[idx])
    
    def _update_indices(self):
        """Update the random indices for the current epoch."""
        if self.num_samples is None or self.num_samples >= len(self.dataset):
            # Use all samples
            if self.num_samples is not None and self.num_samples > len(self.dataset):
                logger.warning(f"Sample limit ({self.num_samples}) exceeds dataset size ({len(self.dataset)}). Using all available samples.")
            self._current_indices = list(range(len(self.dataset)))
        else:
            # Randomly sample indices
            if self.seed is not None:
                random.seed(self.seed)
            self._current_indices = random.sample(range(len(self.dataset)), self.num_samples)
    
    def set_epoch(self, epoch: int):
        """Set the epoch to update random sampling."""
        if self.seed is not None:
            # Use epoch to create different random states
            random.seed(self.seed + epoch)
        self._update_indices()
    
    def __len__(self):
        return len(self._current_indices)
    
    def __getitem__(self, idx):
        return self.dataset[self._current_indices[idx]]


class PointCloudDataModule(L.LightningDataModule):
    """Lightning data module for point cloud data."""
    
    def __init__(
        self,
        data_root: str = "",
        dataset_names: List[str] = [],
        up_axis: dict[str, str] = {},
        min_parts: int = 2,
        max_parts: int = 64,
        min_points_per_part: int = 20,
        max_points_per_part: int = 500,
        poisson_sampling_radius: float = 0,
        min_dataset_size: int = 500,
        limit_val_samples: int = 0,
        random_scale_range: tuple[float, float] = (0.95, 1.05), # original: (0.75, 1.25)
        batch_size: int = 40,
        max_points_per_batch: int = 80000,
        num_workers: int = 16,
        multi_anchor: bool = False,
        multi_anchor_random_rate: float = 2.0,
        dataset_sample_limits: Optional[Dict[str, int]] = None,
        dataset_configs: Optional[Dict[str, Dict]] = None,
        overlap_threshold: float = 2.0,
        overlap_threshold_in_meters: bool = True,
        seed: int = 42,
        use_random_split: bool = False,
        force_use_ply: bool = False,
    ):
        """Data module for point cloud data.

        Args:
            data_root: Root directory of the dataset.
            dataset_names: List of dataset names to use.
            up_axis: Dictionary of dataset names to up axis, e.g. {"ikea": "y", "everyday": "z"}.
                     If not provided, the up axis is assumed to be 'y'. This only affects the visualization.
            min_parts: Minimum number of parts in a point cloud.
            max_parts: Maximum number of parts in a point cloud.
            max_points_per_part: Maximum number of points per part.
            poisson_sampling_radius: Minimum distance between points when using Poisson disk sampling. If 0, skips Poisson disk sampling.
            min_points_per_part: Minimum number of points per part.
            min_dataset_size: Minimum number of point clouds in a dataset.
            limit_val_samples: Number of point clouds to sample from the validation set.
            random_scale_range: Range of random scale to apply to the point cloud.
            max_points_per_batch: Maximum points per batch (for dynamic batching).
            num_workers: Number of workers to use for loading the data.
            multi_anchor: Whether to use multiple anchors for the point cloud.
            dataset_sample_limits: Dictionary mapping dataset names to number of samples per epoch.
                                 If None, use all samples. If a dataset is not in this dict, use all samples.
                                 Example: {"ikea": 1000, "partnet": 500} will use 1000 samples from ikea
                                 and 500 from partnet per epoch, while other datasets use all samples.
            dataset_configs: Dictionary mapping dataset names to keyword arguments for PointCloudDataset.
            overlap_threshold: The global overlap threshold.
            overlap_threshold_in_meters: Whether the overlap threshold is in meters.
            seed: Random seed for reproducibility.
            use_random_split: Whether to prefer train_random.txt/val_random.txt over train.txt/val.txt (default: False).
                             When True, tries random splits first and falls back to standard splits if unavailable.
                             When False, tries standard splits first and falls back to random splits if unavailable.
            force_use_ply: Whether to force loading PLY files directly instead of looking for HDF5 files (default: False).
                          When True, skips HDF5 detection and uses PLY files even if HDF5 files are present.
        """
        super().__init__()
        
        self.use_dynamic_batching = True # default
        
        self.data_root = data_root
        self.up_axis = up_axis
        self.min_parts = min_parts
        self.max_parts = max_parts
        # self.num_points_to_sample = num_points_to_sample
        self.max_points_per_part = max_points_per_part
        self.poisson_sampling_radius = poisson_sampling_radius
        self.min_points_per_part = min_points_per_part
        self.max_points_per_batch = max_points_per_batch
        self.num_workers = num_workers
        self.limit_val_samples = limit_val_samples
        self.min_dataset_size = min_dataset_size
        self.random_scale_range = random_scale_range
        self.multi_anchor = multi_anchor
        self.multi_anchor_random_rate = multi_anchor_random_rate
        self.dataset_sample_limits = dataset_sample_limits or {}
        self.dataset_configs = dataset_configs or {}
        self.overlap_threshold = overlap_threshold
        self.overlap_threshold_in_meters = overlap_threshold_in_meters
        self.seed = seed
        self.use_random_split = use_random_split
        self.force_use_ply = force_use_ply

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.train_datasets: Optional[List[Dataset]] = None  # Store individual datasets for epoch updates

        # Initialize dataset paths
        self.dataset_paths = {}
        self.dataset_names = []
        self._initialize_dataset_paths(dataset_names)
    
    def _initialize_dataset_paths(self, dataset_names: List[str]):
        """Initializes dataset paths by prioritizing HDF5 files inside dataset folders."""
        use_all_datasets = len(dataset_names) == 0
        
        # Discover all potential datasets from directories and .hdf5 files in the data root
        found_names = set()
        if not os.path.exists(self.data_root):
            logger.error(f"Data root not found: {self.data_root}")
            return
            
        for file in os.listdir(self.data_root):
            path = os.path.join(self.data_root, file)
            if os.path.isdir(path):
                found_names.add(file)
            elif file.endswith(".hdf5"):
                found_names.add(file.split(".")[0])
        
        # Filter datasets if specific names are provided
        if not use_all_datasets:
            found_names = found_names.intersection(set(dataset_names))
            
        # Determine the final path for each dataset
        for name in sorted(list(found_names)):
            dir_path = os.path.join(self.data_root, name)
            h5_in_dir_path = os.path.join(dir_path, f"{name}.hdf5")
            
            final_path = None
            log_msg = ""
            
            # Priority 1: HDF5 file inside the folder
            if os.path.isdir(dir_path) and os.path.exists(h5_in_dir_path):
                final_path = h5_in_dir_path
                log_msg = f"Found dataset '{name}' as HDF5 inside folder: {final_path}"
            # Priority 2: Folder itself (for PLY files)
            elif os.path.isdir(dir_path):
                final_path = dir_path
                log_msg = f"Found dataset '{name}' in folder format: {final_path}"
            # Priority 3: HDF5 file at the root level
            else:
                h5_root_path = os.path.join(self.data_root, f"{name}.hdf5")
                if os.path.exists(h5_root_path):
                    final_path = h5_root_path
                    log_msg = f"Found dataset '{name}' as HDF5 file: {final_path}"
            
            if final_path:
                self.dataset_names.append(name)
                self.dataset_paths[name] = final_path
                logger.info(log_msg)

        logger.info(f"Using {len(self.dataset_paths)} datasets: {list(self.dataset_paths.keys())}")
        
        # Determine split types for each dataset to ensure consistency
        self.dataset_split_types = {}
        for dataset_name in self.dataset_names:
            split_type = PointCloudDataset._determine_split_type(
                self.dataset_paths[dataset_name], 
                dataset_name, 
                self.use_random_split
            )
            self.dataset_split_types[dataset_name] = split_type
            logger.info(f"Dataset '{dataset_name}' will use {split_type} splits for consistency")
        
        # Log dataset sample limits
        if self.dataset_sample_limits:
            logger.info("Dataset sample limits per epoch:")
            for dataset_name, limit in self.dataset_sample_limits.items():
                logger.info(f"  {dataset_name}: {limit} samples")
            for dataset_name in self.dataset_names:
                if dataset_name not in self.dataset_sample_limits:
                    logger.info(f"  {dataset_name}: all samples")

    def setup(self, stage: str):
        """Set up datasets for training/validation/testing."""
        make_line = lambda: f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--"
        logger.info(make_line())
        logger.info(f"| {'Dataset':<16} | {'Split':<8} | {'Length':<8} | {'Parts':<8} |")
        logger.info(make_line())

        if stage == "fit":
            # Create individual datasets
            self.train_datasets = []
            for dataset_name in self.dataset_names:
                dataset_config = self.dataset_configs.get(dataset_name, {}).copy()
                dataset = PointCloudDataset(
                    split="train",
                    data_path=self.dataset_paths[dataset_name],
                    up_axis=self.up_axis.get(dataset_name, "y"),
                    dataset_name=dataset_name,
                    min_parts=self.min_parts,
                    max_parts=self.max_parts,
                    max_points_per_part=self.max_points_per_part,
                    poisson_sampling_radius=self.poisson_sampling_radius,
                    min_points_per_part=self.min_points_per_part,
                    min_dataset_size=self.min_dataset_size,
                    random_scale_range=self.random_scale_range,
                    multi_anchor=self.multi_anchor,
                    multi_anchor_random_rate=self.multi_anchor_random_rate,
                    overlap_threshold=dataset_config.pop("overlap_threshold", self.overlap_threshold),
                    overlap_threshold_in_meters=dataset_config.pop("overlap_threshold_in_meters", self.overlap_threshold_in_meters),
                    use_random_split=(self.dataset_split_types[dataset_name] == 'random'),
                    force_use_ply=self.force_use_ply,
                    **dataset_config,
                )
                
                # Apply random sampling if specified
                if dataset_name in self.dataset_sample_limits:
                    dataset = RandomSampledDataset(
                        dataset, 
                        self.dataset_sample_limits[dataset_name],
                        seed=self.seed
                    )
                    logger.info(f"Applied random sampling to {dataset_name}: {len(dataset)} samples per epoch")
                
                self.train_datasets.append(dataset)
            
            # Use ConcatPointCloudDataset for dynamic batching, regular ConcatDataset for fixed batching
            if self.use_dynamic_batching:
                self.train_dataset = ConcatPointCloudDataset(self.train_datasets)
            else:
                self.train_dataset = ConcatDataset(self.train_datasets)
            logger.info(make_line())
            
            val_datasets = [
                (
                    dataset_config := self.dataset_configs.get(dataset_name, {}).copy(),
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        max_points_per_part=self.max_points_per_part,
                        poisson_sampling_radius=self.poisson_sampling_radius,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                        overlap_threshold=dataset_config.pop("overlap_threshold", self.overlap_threshold),
                        overlap_threshold_in_meters=dataset_config.pop("overlap_threshold_in_meters", self.overlap_threshold_in_meters),
                        use_random_split=(self.dataset_split_types[dataset_name] == 'random'),
                        force_use_ply=self.force_use_ply,
                        **dataset_config,
                    )
                )[1] for dataset_name in self.dataset_names
            ]
            
            if self.use_dynamic_batching:
                self.val_dataset = ConcatPointCloudDataset(val_datasets)
            else:
                self.val_dataset = ConcatDataset(val_datasets)
            logger.info(make_line())
            logger.info("Total Train Samples Per Epoch: " + str(self.train_dataset.cumulative_sizes[-1]))
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))

        elif stage == "validate":
            val_datasets = [
                (
                    dataset_config := self.dataset_configs.get(dataset_name, {}).copy(),
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        max_points_per_part=self.max_points_per_part,
                        poisson_sampling_radius=self.poisson_sampling_radius,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                        overlap_threshold=dataset_config.pop("overlap_threshold", self.overlap_threshold),
                        overlap_threshold_in_meters=dataset_config.pop("overlap_threshold_in_meters", self.overlap_threshold_in_meters),
                        use_random_split=(self.dataset_split_types[dataset_name] == 'random'),
                        force_use_ply=self.force_use_ply,
                        **dataset_config,
                    )
                )[1] for dataset_name in self.dataset_names
            ]
            
            if self.use_dynamic_batching:
                self.val_dataset = ConcatPointCloudDataset(val_datasets)
            else:
                self.val_dataset = ConcatDataset(val_datasets)
            logger.info(make_line())
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))

        elif stage in ["test", "predict"]:
            test_datasets = [ 
                (
                    dataset_config := self.dataset_configs.get(dataset_name, {}).copy(),
                    PointCloudDataset(
                        split="val", # change split here
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        max_points_per_part=self.max_points_per_part,
                        poisson_sampling_radius=self.poisson_sampling_radius,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                        overlap_threshold=dataset_config.pop("overlap_threshold", self.overlap_threshold),
                        overlap_threshold_in_meters=dataset_config.pop("overlap_threshold_in_meters", self.overlap_threshold_in_meters),
                        use_random_split=(self.dataset_split_types[dataset_name] == 'random'),
                        force_use_ply=self.force_use_ply,
                        **dataset_config,
                    )
                )[1] for dataset_name in self.dataset_names
            ]

            self.test_dataset = test_datasets
            logger.info(make_line())
            logger.info("Total Test Samples: " + str(sum(len(dataset) for dataset in self.test_dataset)))

    def on_train_epoch_start(self):
        """Update random sampling for each dataset and sampler at the start of each epoch."""
        if self.train_datasets is not None:
            current_epoch = self.trainer.current_epoch
            
            # Update RandomSampledDataset epochs
            for dataset in self.train_datasets:
                if isinstance(dataset, RandomSampledDataset):
                    dataset.set_epoch(current_epoch)
            
            # Update the sampler epoch for proper shuffling
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(current_epoch)

    def train_dataloader(self):
        """Get training dataloader."""
        
        sampler = DynamicBatchSampler(self.train_dataset, self.max_points_per_batch, shuffle=True)
        collate_fn = variable_collate_fn

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            batch_sampler=sampler,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=collate_fn,
            # prefetch_factor=1,
        )


    def val_dataloader(self):
        """Get validation dataloader."""
        sampler = DynamicBatchSampler(self.val_dataset, self.max_points_per_batch, shuffle=False)
        collate_fn = variable_collate_fn

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            batch_sampler=sampler,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=collate_fn,
            # prefetch_factor=1,
        )


    def test_dataloader(self):
        """Get test dataloader."""
        collate_fn = variable_collate_fn
        
        # Return a list of DataLoaders, one for each test dataset
        dataloaders = []
        for dataset in self.test_dataset:
            # Each dataset gets its own sampler
            sampler = DynamicBatchSampler(dataset, self.max_points_per_batch, shuffle=False, drop_last=False)
            dataloaders.append(DataLoader(
                dataset,
                num_workers=self.num_workers,
                persistent_workers=False,
                batch_sampler=sampler,
                pin_memory=False,
                collate_fn=collate_fn,
            ))
        return dataloaders