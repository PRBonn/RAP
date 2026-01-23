import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import trimesh
from trimesh.exchange.ply import load_ply
from torch.utils.data import Dataset

from .transform import sample_points_uniform, sample_points_poisson, center_pcd, rotate_pcd, rotate_pcd_yaw, pad_data

logger = logging.getLogger("Data")


def _load_point_cloud_from_h5(group, part_name):
    """Load one mesh part from an HDF5 group, including features if available."""
    sub_grp = group[part_name]
    verts = np.array(sub_grp["vertices"][:])
    # norms = np.array(sub_grp["normals"][:]) if "normals" in sub_grp else None
    norms = None
    # faces = np.array(sub_grp["faces"][:]) if "faces" in sub_grp else np.array([])
    faces = None
    
    # Load features if available
    features = np.array(sub_grp["features"][:]) if "features" in sub_grp else None
    
    point_cloud = trimesh.Trimesh(vertices=verts, vertex_normals=norms, faces=faces, process=False)
    
    # Store features as a custom attribute
    if features is not None:
        point_cloud.features = features
    else:
        point_cloud.features = np.zeros((verts.shape[0], 32), dtype=np.float32)
        
    return point_cloud

def _load_point_cloud_from_ply(ply_path):
    """Load a mesh from a PLY file and corresponding features from NPY file if available."""
    with open(ply_path, "rb") as f:
        ply_data = load_ply(f)
    verts = ply_data["vertices"]
    norms = ply_data["vertex_normals"] if "vertex_normals" in ply_data else None
    faces = ply_data["faces"] if "faces" in ply_data else None
    
    point_cloud = trimesh.Trimesh(vertices=verts, vertex_normals=norms, faces=faces, process=False)
    
    # Try to load corresponding features file
    # Expected naming: if PLY is "part_name.ply", features should be "features_part_name.npy"
    folder = os.path.dirname(ply_path)
    ply_name = os.path.splitext(os.path.basename(ply_path))[0]  # part_name without extension
    feature_path = os.path.join(folder, f"features_{ply_name}.npy")
    
    if os.path.exists(feature_path):
        try:
            features = np.load(feature_path)
            point_cloud.features = features
            logger.debug(f"Loaded features from {feature_path}, shape: {features.shape}")
        except Exception as e:
            logger.warning(f"Failed to load features from {feature_path}: {e}")
            point_cloud.features = np.zeros((verts.shape[0], 32), dtype=np.float32)
    else:
        point_cloud.features = np.zeros((verts.shape[0], 32), dtype=np.float32)
        logger.debug(f"No features file found at {feature_path}")
    
    return point_cloud

    
class PointCloudDataset(Dataset):
    """Dataset class for multi-part point clouds and apply part-level augmentation.
    
    This dataset can load data from:
    1. A direct HDF5 file path (e.g., "data.hdf5")
    2. A directory containing PLY files organized by fragments
    3. A directory containing HDF5 files (automatically detected and used)
    
    When a directory is provided, the dataset will:
    - First look for HDF5 files (*.hdf5) in the directory
    - If HDF5 files are found and have valid structure, use the first one (or preferred one if specified)
    - If no HDF5 files are found or they have invalid structure, fall back to PLY files
    
    The dataset now supports loading pointwise features:
    - From HDF5: features are stored in 'features' dataset within each part group
    - From folders: features are loaded from 'features_{part_name}.npy' files
    - If input data is pre-downsampled (total points == num_points_to_sample), sampling is skipped
    
    Split file handling:
    - When use_random_split=True: tries train_random.txt/val_random.txt first, falls back to 
      train.txt/val.txt if random split files are not available or empty
    - When use_random_split=False: tries train.txt/val.txt first, falls back to 
      train_random.txt/val_random.txt if standard split files are not available or empty
    
    Args:
        split: Dataset split ("train", "val", "test")
        data_path: Path to HDF5 file or directory containing data
        dataset_name: Name of the dataset (used for HDF5 structure)
        up_axis: Up axis for coordinate system ("x", "y", "z")
        min_parts: Minimum number of parts per sample
        max_parts: Maximum number of parts per sample
        num_points_to_sample: Total number of points to sample across all parts
        min_points_per_part: Minimum points per part
        random_scale_range: Range for random scaling (min, max)
        multi_anchor: Whether to use multiple anchor parts
        limit_val_samples: Limit number of validation samples
        min_dataset_size: Minimum dataset size (will repeat if needed)
        num_threads: Number of threads for data loading
        overlap_threshold: Threshold for overlap detection
        overlap_threshold_in_meters: Whether the overlap threshold is in meters
        preferred_h5_file: Preferred HDF5 filename when multiple files exist in directory
        load_features: Whether to load and return pointwise features (default: True)
        use_random_split: Whether to prefer train_random.txt/val_random.txt over train.txt/val.txt (default: False).
                         When True, tries random splits first and falls back to standard splits if unavailable.
                         When False, tries standard splits first and falls back to random splits if unavailable.
        force_use_ply: Whether to force loading PLY files directly instead of looking for HDF5 files (default: False).
                      When True, skips HDF5 detection and uses PLY files even if HDF5 files are present.
        yaw_augmentation: Whether to use yaw augmentation (default: True)
        roll_pitch_range: Range for roll/pitch perturbations in degrees when using yaw augmentation (default: 5.0)
    """

    @classmethod
    def _determine_split_type(cls, data_path: str, dataset_name: str = "", use_random_split: bool = False) -> str:
        """Determine whether to use standard or random splits for consistency across all splits.
        
        This method checks the availability of split files and returns either 'standard' or 'random'
        to ensure all splits (train, val, test) use the same split type for consistency.
        
        Fallback logic:
        - If use_random_split=True: tries random splits first, falls back to standard if unavailable
        - If use_random_split=False: tries standard splits first, falls back to random if unavailable
        
        Args:
            data_path: Path to HDF5 file or directory containing data
            dataset_name: Name of the dataset (used for HDF5 structure)
            use_random_split: Whether random splits are explicitly requested
            
        Returns:
            'random' if random splits should be used, 'standard' otherwise
        """
        # Check folder format
        if os.path.isdir(data_path):
            if use_random_split:
                # Try random splits first, fall back to standard if unavailable
                random_splits_available = True
                for split in ['train', 'val']:
                    random_split_file = os.path.join(data_path, "data_split", f"{split}_random.txt")
                    if not os.path.exists(random_split_file) or os.path.getsize(random_split_file) == 0:
                        random_splits_available = False
                        break
                
                if random_splits_available:
                    return 'random'
                else:
                    # Fall back to standard splits
                    standard_splits_available = True
                    for split in ['train', 'val']:
                        split_file = os.path.join(data_path, "data_split", f"{split}.txt")
                        if not os.path.exists(split_file) or os.path.getsize(split_file) == 0:
                            standard_splits_available = False
                            break
                    
                    if standard_splits_available:
                        logger.info(f"Random splits not available for {data_path}, will use standard splits for consistency")
                        return 'standard'
                    else:
                        logger.warning(f"Neither random nor standard splits are available for {data_path}")
                        return 'standard'  # Default fallback
            else:
                # Try standard splits first, fall back to random if unavailable
                standard_splits_available = True
                for split in ['train', 'val']:
                    split_file = os.path.join(data_path, "data_split", f"{split}.txt")
                    if not os.path.exists(split_file) or os.path.getsize(split_file) == 0:
                        standard_splits_available = False
                        break
                
                if standard_splits_available:
                    return 'standard'
                else:
                    # Fall back to random splits
                    random_splits_available = True
                    for split in ['train', 'val']:
                        random_split_file = os.path.join(data_path, "data_split", f"{split}_random.txt")
                        if not os.path.exists(random_split_file) or os.path.getsize(random_split_file) == 0:
                            random_splits_available = False
                            break
                    
                    if random_splits_available:
                        logger.info(f"Standard splits not available for {data_path}, will use random splits for consistency")
                        return 'random'
                    else:
                        logger.warning(f"Neither standard nor random splits are available for {data_path}")
                        return 'standard'  # Default fallback
                    
        # Check HDF5 format
        elif data_path.endswith('.hdf5'):
            try:
                with h5py.File(data_path, "r") as h5:
                    if "data_split" in h5 and dataset_name in h5["data_split"]:
                        if use_random_split:
                            # Try random splits first, fall back to standard if unavailable
                            random_splits_available = True
                            for split in ['train', 'val']:
                                if f"{split}_random" not in h5["data_split"][dataset_name]:
                                    random_splits_available = False
                                    break
                            
                            if random_splits_available:
                                return 'random'
                            else:
                                # Fall back to standard splits
                                standard_splits_available = True
                                for split in ['train', 'val']:
                                    if split not in h5["data_split"][dataset_name]:
                                        standard_splits_available = False
                                        break
                                
                                if standard_splits_available:
                                    logger.info(f"Random splits not available for {dataset_name} in {data_path}, will use standard splits for consistency")
                                    return 'standard'
                                else:
                                    logger.warning(f"Neither random nor standard splits are available for {dataset_name} in {data_path}")
                                    return 'standard'  # Default fallback
                        else:
                            # Try standard splits first, fall back to random if unavailable
                            standard_splits_available = True
                            for split in ['train', 'val']:
                                if split not in h5["data_split"][dataset_name]:
                                    standard_splits_available = False
                                    break
                            
                            if standard_splits_available:
                                return 'standard'
                            else:
                                # Fall back to random splits
                                random_splits_available = True
                                for split in ['train', 'val']:
                                    if f"{split}_random" not in h5["data_split"][dataset_name]:
                                        random_splits_available = False
                                        break
                                
                                if random_splits_available:
                                    logger.info(f"Standard splits not available for {dataset_name} in {data_path}, will use random splits for consistency")
                                    return 'random'
                                else:
                                    logger.warning(f"Neither standard nor random splits are available for {dataset_name} in {data_path}")
                                    return 'standard'  # Default fallback
            except Exception as e:
                logger.warning(f"Error checking split availability in HDF5 file {data_path}: {e}")
        
        return 'standard'

    def _detect_h5_file_in_folder(self, folder_path: str) -> str | None:
        """Detect HDF5 files in the given folder and return the first one found."""
        h5_files = glob.glob(os.path.join(folder_path, "*.hdf5"))
        if h5_files:
            # Sort to ensure consistent selection
            h5_files.sort()
            
            # If a preferred file is specified, try to use it
            if self.preferred_h5_file:
                preferred_path = os.path.join(folder_path, self.preferred_h5_file)
                if preferred_path in h5_files:
                    selected_file = preferred_path
                    logger.info(f"Using preferred HDF5 file: {self.preferred_h5_file}")
                else:
                    logger.warning(f"Preferred HDF5 file '{self.preferred_h5_file}' not found, using first available")
                    selected_file = h5_files[0]
            else:
                selected_file = h5_files[0]
            
            logger.info(f"Found {len(h5_files)} HDF5 file(s) in {folder_path}: {[os.path.basename(f) for f in h5_files]}")
            logger.info(f"Using: {os.path.basename(selected_file)}")
            return selected_file
        return None

    def _validate_h5_structure(self, h5_file_path: str) -> bool:
        """Validate that the HDF5 file has the expected structure."""
        try:
            with h5py.File(h5_file_path, "r") as h5:
                if "data_split" not in h5:
                    logger.warning(f"HDF5 file {h5_file_path} does not contain 'data_split' group")
                    return False
                
                # Check if we can find the requested split
                data_split = h5["data_split"]
                if self.dataset_name and self.dataset_name in data_split:
                    if self.split not in data_split[self.dataset_name]:
                        logger.warning(f"Split '{self.split}' not found in dataset '{self.dataset_name}'")
                        return False
                
                logger.info(f"HDF5 file {h5_file_path} has valid structure")
                return True
        except Exception as e:
            logger.warning(f"Error validating HDF5 file {h5_file_path}: {e}")
            return False

    def __init__(
        self,
        split: str = "train",
        data_path: str = "data.hdf5",
        dataset_name: str = "",
        up_axis: str = "y",
        min_parts: int = 2,
        max_parts: int = 64,
        max_points_per_part: int = 2000,
        min_points_per_part: int = 20,
        poisson_sampling_radius: float = 0.02,
        random_scale_range: tuple[float, float] | None = None,
        multi_anchor: bool = False,
        multi_anchor_random_rate: float = 2.0,
        limit_val_samples: int = 0,
        min_dataset_size: int = 0,
        num_threads: int = 2,
        overlap_threshold: float = 1.0,
        overlap_threshold_in_meters: bool = True,
        preferred_h5_file: str = "",
        load_features: bool = True,
        use_random_split: bool = False,
        force_use_ply: bool = False,
        yaw_augmentation: bool = True,
        roll_pitch_range: float = 10.0,
    ):
        super().__init__()

        self.split = split
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.up_axis = up_axis.lower()
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.max_points_per_part = max_points_per_part
        self.poisson_sampling_radius = poisson_sampling_radius
        self.min_points_per_part = min_points_per_part
        self.random_scale_range = random_scale_range
        self.multi_anchor = multi_anchor
        self.multi_anchor_random_rate = multi_anchor_random_rate
        self.limit_val_samples = limit_val_samples
        self.min_dataset_size = min_dataset_size
        self.overlap_threshold = overlap_threshold
        self.overlap_threshold_in_meters = overlap_threshold_in_meters
        self.preferred_h5_file = preferred_h5_file
        self.load_features = load_features
        self.use_random_split = use_random_split
        self.force_use_ply = force_use_ply
        self.yaw_augmentation = yaw_augmentation
        self.roll_pitch_range = roll_pitch_range

        # Check if data_path is a directory and look for HDF5 files
        if os.path.isdir(self.data_path):
            if self.force_use_ply:
                # Force using PLY files, skip HDF5 detection
                logger.info(f"Force using PLY files for {self.data_path}")
                self.use_folder = True
            else:
                # Look for HDF5 files in the dataset folder
                h5_file = self._detect_h5_file_in_folder(self.data_path)
                if h5_file:
                    # Validate the HDF5 file structure before using it
                    if self._validate_h5_structure(h5_file):
                        # Use the HDF5 file found
                        self.data_path = h5_file
                        print("hdf5 file being loaded")
                        logger.info(f"Using HDF5 file from dataset folder: {self.data_path}")
                        self.use_folder = False
                    else:
                        logger.warning(f"HDF5 file {h5_file} has invalid structure, falling back to PLY files")
                        self.use_folder = True
                else:
                    logger.info(f"No HDF5 files found in {self.data_path}, using PLY files")
                    self.use_folder = True
        else:
            self.use_folder = False

        # self.use_folder = True
        
        # Determine split type for consistency across all splits
        self.split_type = self._determine_split_type(self.data_path, self.dataset_name, self.use_random_split)
        self.effective_use_random_split = (self.split_type == 'random')
            
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self._h5_file = None

        self.min_part_count = self.max_parts + 1
        self.max_part_count = 0
        
        self.part_counts = []
        self.precomputed_num_points = []

        self.fragments = self._build_fragment_list()

        collate_fn = "Variable Size"

        logger.info(
            f"| {self.dataset_name:16s} | {self.split:8s} | {len(self.fragments):8d} "
            f"| [{int(self.min_part_count):2d}, {int(self.max_part_count):2d}] | {collate_fn:16s} |"
        )

    def __len__(self):
        return len(self.fragments)

    def estimate_num_points(self, index: int) -> int:
        """Estimate the number of points in a sample."""
        num_points = self.precomputed_num_points[index]
        if num_points > 0:
            return num_points
        else:
            # fallback to the max possible number of points
            return self.part_counts[index] * self.max_points_per_part

    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Returns:
            A dictionary containing:
            - index (int): the index of the fragment
            - name (str): the name of the fragment
            - overlap_threshold (float): the overlap threshold
            - dataset_name (str): the name of the dataset
            - num_parts (int): the number of parts

            - pointclouds (N, 3) float32: Transformed point clouds.
            - pointclouds_gt (N, 3) float32: registered point clouds (ground truth).
            - pointclouds_normals (N, 3) float32: Transformed point cloud normals.
            - pointclouds_normals_gt (N, 3) float32: registered point cloud normals (ground truth).
            - features (N, D) float32: Pointwise features (if load_features=True and available).
            - rotations (P, 3, 3) float32: Rotation matrices.
            - translations (P, 3) float32: Translation vectors.
            - points_per_part (P) int64: Number of points per part.
            - scales (1, ) float32: Scale of the point clouds.
            - anchor_parts (P) bool: Boolean array indicating anchor parts.
            - anchor_indices (N, ) bool: Boolean array indicating anchor points.
            - init_rotation (3, 3) float32: Initial rotation matrix of the pointclouds_gt, used for recovering the original data.
            - is_pre_sampled (bool): Whether the data was already pre-sampled and didn't need resampling.

        Note:
            - For arrays rotations, translations, points_per_part, scale, anchor_part:
               - The first dimension is the maximum number of parts P.
               - We pad zeros to the array to make it of shape (P, ...).
               
            - For arrays pointclouds, pointclouds_gt, pointclouds_normals, pointclouds_normals_gt, features:
               - The first dimension is the number of points N.
               - We stack all parts into a single array.
               - The points_per_part can be used to unpack them.
               
            - The rotations and translations are followed by:
                pointclouds_gt[st:ed] = pointclouds[st:ed] @ rotations[i].T + translations[i]
        """
        
        frag = self.fragments[index]
        if self.use_folder:
            sample = self._load_from_folder(frag, index)
        else:
            sample = self._load_from_h5(frag, index)
        return self._transform(sample)

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, "r", libver='latest', swmr=True)
        return self._h5_file

    def _build_fragment_list(self) -> list[str]:
        """Read and filter fragment keys from hdf5 or folder."""
        
        fragments = []
        if self.use_folder:
            # Use the determined split type for consistency
            if self.effective_use_random_split:
                split_file = os.path.join(self.data_path, "data_split", f"{self.split}_random.txt")
                logger.info(f"Using random split file for consistency: {split_file}")
            else:
                split_file = os.path.join(self.data_path, "data_split", f"{self.split}.txt")
                logger.info(f"Using standard split file: {split_file}")
            
            # Verify the split file exists and has content, with fallback
            if not os.path.exists(split_file) or os.path.getsize(split_file) == 0:
                # Try fallback split type
                if self.effective_use_random_split:
                    # Try standard split as fallback
                    fallback_split_file = os.path.join(self.data_path, "data_split", f"{self.split}.txt")
                    if os.path.exists(fallback_split_file) and os.path.getsize(fallback_split_file) > 0:
                        logger.info(f"Random split file not available, using fallback standard split: {fallback_split_file}")
                        split_file = fallback_split_file
                        self.effective_use_random_split = False  # Update the effective split type
                    else:
                        logger.error(f"Neither random nor standard split file available for {self.split}")
                        return []
                else:
                    # Try random split as fallback
                    fallback_split_file = os.path.join(self.data_path, "data_split", f"{self.split}_random.txt")
                    if os.path.exists(fallback_split_file) and os.path.getsize(fallback_split_file) > 0:
                        logger.info(f"Standard split file not available, using fallback random split: {fallback_split_file}")
                        split_file = fallback_split_file
                        self.effective_use_random_split = True  # Update the effective split type
                    else:
                        logger.error(f"Neither standard nor random split file available for {self.split}")
                        return []
            
            with open(split_file, 'r') as f:
                frags = [line.strip() for line in f if line.strip()]
            
            if not frags:
                logger.warning(f"Split file {split_file} is empty or contains no valid fragments")
                return []

            # Use the same split type as determined for consistency
            if self.effective_use_random_split:
                num_points_split = f"{self.split}_random"
            else:
                num_points_split = self.split
            
            num_points_dir = os.path.join(self.data_path, "num_points")
            if not os.path.isdir(num_points_dir):
                num_points = [5000] * len(frags)
                logger.info(f"No 'num_points' directory found at {num_points_dir}. Defaulting all samples to 5000 points.")
            else:
                num_points_file = os.path.join(num_points_dir, f"{num_points_split}.txt")
                if os.path.exists(num_points_file):
                    with open(num_points_file, 'r') as f:
                        num_points = [int(line.strip()) for line in f if line.strip()]
                    assert len(num_points) == len(frags), f"Number of fragments and num_points do not match: {len(frags)} != {len(num_points)}"
                else:
                    # Try the other split type as fallback
                    fallback_split = self.split if self.effective_use_random_split else f"{self.split}_random"
                    fallback_num_points_file = os.path.join(num_points_dir, f"{fallback_split}.txt")
                    if os.path.exists(fallback_num_points_file):
                        logger.info(f"Primary num_points file not found at {num_points_file}, using fallback: {fallback_num_points_file}")
                        with open(fallback_num_points_file, 'r') as f:
                            num_points = [int(line.strip()) for line in f if line.strip()]
                        assert len(num_points) == len(frags), f"Number of fragments and num_points do not match: {len(frags)} != {len(num_points)}"
                    else:
                        num_points = [0] * len(frags)
                        logger.info(f"No precomputed num_points file at {num_points_file} or {fallback_num_points_file}. To precompute it: python -m rectified_point_flow.data.precompute_num_points --help")
                
            for frag, n_points in zip(frags, num_points):
                parts = glob.glob(
                    os.path.join(self.data_path, frag, "*.ply")
                )
                n_parts = len(parts)
                if self.min_parts <= n_parts <= self.max_parts:
                    self.min_part_count = min(self.min_part_count, n_parts)
                    self.max_part_count = max(self.max_part_count, n_parts)
                    self.part_counts.append(n_parts)
                    self.precomputed_num_points.append(n_points)
                    fragments.append(frag)

            # Log which split type was used
            if self.effective_use_random_split:
                logger.info(f"Successfully loaded {len(fragments)} fragments using random split")
            else:
                logger.info(f"Successfully loaded {len(fragments)} fragments using standard split")
                
            return fragments

        elif self.data_path.endswith('.hdf5'):
            h5 = self._get_h5_file()
            # Use the determined split type for consistency
            if self.effective_use_random_split:
                split_key = f"{self.split}_random"
                logger.info(f"Using random split for HDF5 dataset consistency: {split_key}")
            else:
                split_key = self.split
                logger.info(f"Using standard split for HDF5 dataset: {split_key}")
            
            # Access the determined split with fallback
            try:
                raw = h5["data_split"][self.dataset_name][split_key]
                logger.info(f"Successfully loaded {split_key} split for HDF5 dataset")
            except KeyError:
                # Try fallback split type
                if self.effective_use_random_split:
                    # Try standard split as fallback
                    fallback_split_key = self.split
                    try:
                        raw = h5["data_split"][self.dataset_name][fallback_split_key]
                        logger.info(f"Random split not available, using fallback standard split: {fallback_split_key}")
                        split_key = fallback_split_key
                        self.effective_use_random_split = False  # Update the effective split type
                    except KeyError:
                        logger.error(f"Neither random nor standard split available for {self.split} in HDF5 dataset '{self.dataset_name}'")
                        raise
                else:
                    # Try random split as fallback
                    fallback_split_key = f"{self.split}_random"
                    try:
                        raw = h5["data_split"][self.dataset_name][fallback_split_key]
                        logger.info(f"Standard split not available, using fallback random split: {fallback_split_key}")
                        split_key = fallback_split_key
                        self.effective_use_random_split = True  # Update the effective split type
                    except KeyError:
                        logger.error(f"Neither standard nor random split available for {self.split} in HDF5 dataset '{self.dataset_name}'")
                        raise
            
            frags = [r.decode() for r in raw[:]]

            # print(f"frags: {frags}")

            if "num_points" in h5 and self.dataset_name in h5["num_points"]:
                # Use the same split_key that was determined for fragments
                try:
                    num_points = h5["num_points"][self.dataset_name][split_key]
                    assert len(num_points) == len(frags), f"Number of fragments and num_points do not match: {len(frags)} != {len(num_points)}"
                except KeyError:
                    # Fallback to original split if the determined split_key doesn't exist for num_points
                    try:
                        fallback_split = self.split if split_key != self.split else f"{self.split}_random"
                        num_points = h5["num_points"][self.dataset_name][fallback_split]
                        logger.info(f"num_points for split '{split_key}' not found, using fallback split '{fallback_split}'")
                        assert len(num_points) == len(frags), f"Number of fragments and num_points do not match: {len(frags)} != {len(num_points)}"
                    except KeyError:
                        num_points = [0] * len(frags)
                        logger.info(f"No precomputed num_points found for splits '{split_key}' or '{fallback_split}' in {h5.filename}")
            else:
                num_points = [0] * len(frags)
                logger.info(f"No precomputed num_points file at {h5.filename}. To precompute it: python -m rectified_point_flow.data.precompute_num_points --help")

            fragments = []
            for name, n_points in zip(frags, num_points):
                try:
                    count = len(h5[name].keys())
                    if self.min_parts <= count <= self.max_parts:
                        self.min_part_count = min(self.min_part_count, count)
                        self.max_part_count = max(self.max_part_count, count)
                        self.part_counts.append(count)
                        self.precomputed_num_points.append(n_points)
                        fragments.append(name)
                except KeyError:
                    continue

        else:
            raise ValueError(
                f"Invalid data path: {self.data_path}. Please provide a folder path or a .hdf5 file."
            )     

        # limit or upsample
        if self.limit_val_samples > 0 and len(fragments) > self.limit_val_samples and self.split.startswith("val"):
            step = len(fragments) // self.limit_val_samples
            fragments = fragments[::step]
            self.part_counts = self.part_counts[::step]
            self.precomputed_num_points = self.precomputed_num_points[::step]

        return fragments

    def _load_from_h5(self, name: str, index: int) -> dict:
        group = self._get_h5_file()[name]
        parts = sorted(list(group.keys()))
        point_clouds = list(self.pool.map(lambda p: _load_point_cloud_from_h5(group, p), parts))
        pcs, pns, features, thr, is_pre_sampled = self._sample_points(point_clouds)
        return {
            "index": index,
            "name": name,
            "num_parts": len(parts),
            "pointclouds_gt": pcs,
            "pointclouds_normals_gt": pns,
            "features": features if self.load_features else None,
            "overlap_threshold": thr,
            "is_pre_sampled": is_pre_sampled,
        }

    def _load_from_folder(self, frag: str, index: int) -> dict:
        folder = os.path.join(self.data_path, frag)
        ply_files = sorted(glob.glob(os.path.join(folder, "*.ply")))
        point_clouds = [_load_point_cloud_from_ply(p) for p in ply_files]
        pcs, pns, features, overlap_thr, is_pre_sampled = self._sample_points(point_clouds)
        return {
            "index": index,
            "name": frag,
            "pointclouds_gt": pcs,
            "pointclouds_normals_gt": pns,
            "features": features if self.load_features else None,
            "overlap_threshold": overlap_thr,
            "num_parts": len(point_clouds),
            "is_pre_sampled": is_pre_sampled,
        }

    def _sample_points(self, meshes: list[trimesh.Trimesh]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None, float, bool]:
        """Sample points (and normals) from meshes, and handle features.  (simple version, input point cloud are already sampled and the features are already extracted)
        
        Returns:
            tuple of (point_clouds, normals, features, overlap_threshold, is_pre_sampled)
        """

        pcs, pns, features_list = [], [], []

        # now we would assume the input point cloud are already sampled and the features are already extracted
        # Use points directly without resampling
        for i, m in enumerate(meshes):
            pcs.append(m.vertices.copy())
            pns.append(np.zeros((len(m.vertices), 3)))  # Use zero normals for now
            
            # Handle features
            if self.load_features and hasattr(m, 'features') and m.features is not None:
                features_list.append(m.features.copy())
            else:
                features_list.append(None)
                
        overlap_thr = self.overlap_threshold
        is_pre_sampled = True

        # Consolidate features - only return if all parts have features
        if self.load_features and features_list and all(f is not None for f in features_list):
            features = features_list
        else:
            features = None
            if self.load_features and any(f is not None for f in features_list):
                logger.warning("Some parts have features while others don't. Returning None for features.")
            
        return pcs, pns, features, overlap_thr, is_pre_sampled
        
    
    def _make_y_up(self, pts_gt: np.ndarray, pns_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A simple transform to change the up axis of the point cloud. 

        Args:
            pts_gt: Point cloud coordinates of shape (N, 3).
            pns_gt: Point cloud normals of shape (N, 3).

        Returns:
            pts_gt: Point cloud coordinates of shape (N, 3).
            pns_gt: Point cloud normals of shape (N, 3).
        """
        if self.up_axis == "y":
            return pts_gt, pns_gt
        elif self.up_axis == "z":
            return pts_gt[:, [0, 2, 1]], pns_gt[:, [0, 2, 1]]
        elif self.up_axis == "x":
            return pts_gt[:, [1, 0, 2]], pns_gt[:, [1, 0, 2]]
        else:
            raise ValueError(f"Invalid up axis: {self.up_axis}")

    
    # v2, generate in gt gravity center coordinate system
    def _transform(self, data: dict) -> dict:
        """Apply scaling, rotation, centering, shuffling, and padding."""

        pcs_gt = data["pointclouds_gt"]
        pns_gt = data["pointclouds_normals_gt"]
        features = data.get("features", None)  # Get features if available
        n_parts = data["num_parts"]
        is_pre_sampled = data.get("is_pre_sampled", False)

        counts = np.array([len(pc) for pc in pcs_gt])
        offsets = np.concatenate([[0], np.cumsum(counts)])
        pts_gt = np.concatenate(pcs_gt)
        normals_gt = np.concatenate(pns_gt)

        total_point_count = pts_gt.shape[0]
        
        # Handle features concatenation
        if features is not None and self.load_features:
            features_concat = np.concatenate(features)
        else:
            features_concat = None

        # pts_gt, tran_global = center_pcd(pts_gt)
        tran_global = np.mean(pts_gt, axis=0)

        init_rot = np.eye(3) # deprecated for now
        
        # Figure out the primary part early
        primary_idx = np.argmax(counts)
        primary_st, primary_ed = offsets[primary_idx], offsets[primary_idx+1]
        
        # Center the primary part and apply its translation to all points
        primary_part_pts = pts_gt[primary_st:primary_ed]
        primary_part_centered, primary_trans = center_pcd(primary_part_pts)
        
        # Apply global rotation to the centered primary part if in train mode
        if self.split.startswith("train"):
            if self.yaw_augmentation:
                primary_part_rotated, primary_part_normals_rotated, rot_global = rotate_pcd_yaw(primary_part_centered, normals_gt[primary_st:primary_ed], roll_pitch_range=self.roll_pitch_range)
            else:
                primary_part_rotated, primary_part_normals_rotated, rot_global = rotate_pcd(primary_part_centered, normals_gt[primary_st:primary_ed])
        else:
            primary_part_rotated = primary_part_centered
            primary_part_normals_rotated = normals_gt[primary_st:primary_ed]
            rot_global = np.eye(3)

        # Calculate scale based on the extent of the globally rotated primary part
        scale = np.max(np.abs(primary_part_rotated)) * 1.5 # better to scale a factor of 2.0 because now we are only use one anchored part
        
        if self.split.startswith("train") and self.random_scale_range is not None:
            scale *= np.random.uniform(*self.random_scale_range)

        # print(f"scale: {scale}")

        # Apply the primary part's centering translation and global rotation to the entire point cloud
        pts_gt = (pts_gt - primary_trans) @ rot_global.T
        normals_gt = normals_gt @ rot_global.T

        pts_gt /= scale

        pts_gt, gt_trans = center_pcd(pts_gt)

        pts, normals = pts_gt.copy(), normals_gt.copy()
        part_indices = np.zeros(pts.shape[0], dtype=np.int64)
        
        # already scaled
        def _proc_part(i):
            """Process one part: center, rotate, and shuffle."""
            st, ed = offsets[i], offsets[i+1]
            
            # Center the point cloud
            part, trans = center_pcd(pts_gt[st:ed])
            
            if self.split.startswith("train"):
                # Apply random rotation to the centered part
                if self.yaw_augmentation:
                    part, norms, rot = rotate_pcd_yaw(part, normals_gt[st:ed], roll_pitch_range=self.roll_pitch_range)
                else:
                    part, norms, rot = rotate_pcd(part, normals_gt[st:ed])
            else:
                norms = normals_gt[st:ed]
                rot = np.eye(3)
            
            # Random shuffle point order within the part
            _order = np.random.permutation(len(part))
            pts[st: ed] = part[_order]
            normals[st: ed] = norms[_order]
            pts_gt[st:ed] = pts_gt[st:ed][_order]
            normals_gt[st:ed] = norms[_order]
            part_indices[st:ed] = i
            
            # Shuffle features with the same order
            if features is not None:
                features_concat[st:ed] = features_concat[st:ed][_order]

            return rot, trans

        results = list(self.pool.map(_proc_part, range(n_parts)))
        rots, trans = zip(*results)

        # Padding to max_parts (zero padding)
        pts_per_part = pad_data(counts, self.max_parts)
        rots = pad_data(np.stack(rots), self.max_parts)
        trans = pad_data(np.stack(trans), self.max_parts)

        # Use the largest part as the anchor part
        anchor = np.zeros(self.max_parts, bool)
        anchor[primary_idx] = True

        # Select extra parts (for some time, not always) if multi_anchor is enabled (deprecated for now)
        if self.split == "train" and self.multi_anchor and n_parts > 2 and np.random.rand() > self.multi_anchor_random_rate * 1.0 / n_parts:
            candidates = counts[:n_parts] > total_point_count * 0.05 # also not to be too few
            candidates[primary_idx] = False
            if candidates.any():
                extra_n = np.random.randint(
                    1, min(candidates.sum() + 1, n_parts - 1)
                )
                extra_idx = np.random.choice(
                    np.where(candidates)[0], extra_n, replace=False
                )
                anchor[extra_idx] = True
                # rots[extra_idx] = np.eye(3)
                # trans[extra_idx] = np.zeros(3)

        # Broadcast anchor part to points
        anchor_indices = np.zeros(total_point_count, bool)
        for i in range(n_parts):
            if anchor[i]:
                st, ed = offsets[i], offsets[i + 1]
                anchor_indices[st:ed] = True
                
                rots[i] = np.eye(3)
                trans[i] = -gt_trans

                pts[st:ed] = pts_gt[st:ed].copy() + gt_trans

        results = {}
        for key in ["index", "name", "overlap_threshold"]:
            results[key] = data[key]

        # print('trans: ', trans) # scaled translation

        # Results dictionary (N: number of points of the object, P: maximum number of parts)
        results["dataset_name"] = self.dataset_name                         # str
        results["data_path"] = data["name"]                                  # str, relative path from dataset root (fragment name/path)
        results["num_parts"] = n_parts                                      # int64
        results["pointclouds"] = pts.astype(np.float32)                     # (N, 3) float32
        results["pointclouds_gt"] = pts_gt.astype(np.float32)               # (N, 3) float32
        results["pointclouds_normals"] = normals.astype(np.float32)         # (N, 3) float32
        results["pointclouds_normals_gt"] = normals_gt.astype(np.float32)   # (N, 3) float32
        results["rotations"] = rots.astype(np.float32)                      # (P, 3, 3) float32, relative to pts_gt frame
        results["translations"] = trans.astype(np.float32)                  # (P, 3) float32  (in the scaled space), relative to pts_gt frame
        results["points_per_part"] = pts_per_part.astype(np.int64)          # (P, ) int64
        results["part_indices"] = part_indices.astype(np.int64)             # (N, ) int64
        results["scales"] = np.array(scale, dtype=np.float32)               # (1, ) float32
        results["anchor_parts"] = anchor.astype(bool)                       # (P, ) bool
        results["anchor_indices"] = anchor_indices.astype(bool)             # (N, ) bool
        results["init_rotation"] = init_rot.astype(np.float32)              # (3, 3) float32
        results["is_pre_sampled"] = is_pre_sampled                          # bool

        results["global_rotation"] = rot_global.astype(np.float32)          # (3, 3) float32, global rotation applied to all parts
        results["global_translation"] = tran_global.astype(np.float32)      # (3,) float32, global translation applied to all parts in the original unit
        
        # Add features to results if available
        if self.load_features and features_concat is not None:
            results["features"] = features_concat.astype(np.float32)        # (N, F) float32, F as the feature dimension
        
        return results

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()
        self.pool.shutdown()


if __name__ == "__main__":
    ds = PointCloudDataset(
        split="train",
        data_path="../dataset/kitti.hdf5",
        dataset_name="kitti",
    )
    sample = ds[0]
    for key, val in sample.items():
        if isinstance(val, np.ndarray):
            print(f"{key:<20} {val.shape}, {val.dtype}")
        else:
            print(f"{key:<20} {val}")

    # Sanity check for transformations
    n_parts = sample["num_parts"]
    pts_gt = sample["pointclouds_gt"]
    pts = sample["pointclouds"]
    pts_per_part = sample["points_per_part"]
    offsets = np.cumsum(pts_per_part)
    for i in range(n_parts):
        if not sample["anchor_part"][i]:
            st, ed = offsets[i], offsets[i + 1]
            rot, trans = sample["rotations"][i], sample["translations"][i]
            pts_recovered = (pts[st:ed] @ rot.T) + trans
            assert np.allclose(pts_recovered, pts_gt[st:ed], atol=1e-6)
    print("Sanity check passed!")