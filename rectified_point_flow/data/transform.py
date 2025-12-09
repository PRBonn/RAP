import logging
from typing import Tuple
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

logger = logging.getLogger("Data")
trimesh.util.log.setLevel(logging.ERROR)

if os.environ.get("USE_PCU", "0") == "1":
    try:
        import point_cloud_utils as pcu
        use_pcu = True
        logger.info("Using point_cloud_utils for point sampling.")
    except ImportError:
        logger.warning("point_cloud_utils not found, using trimesh.sample instead.")
        use_pcu = False
else:
    logger.info("Using trimesh.sample for point sampling.")
    use_pcu = False


def sample_points_poisson(mesh: trimesh.Trimesh, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points using Poisson disk sampling."""
    if use_pcu:
        v = mesh.vertices
        f = mesh.faces
        idx, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=count)
        pts = pcu.interpolate_barycentric_coords(f, idx, bc, v)
    else:
        pts, idx = trimesh.sample.sample_surface_even(mesh, count=count)
    return pts, idx

def sample_points_uniform(mesh: trimesh.Trimesh, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points using uniform sampling."""
    if use_pcu:
        v = mesh.vertices
        f = mesh.faces
        idx, bc = pcu.sample_mesh_uniform(v, f, num_samples=count)
    else:
        pts, idx = trimesh.sample.sample_surface(mesh, count=count)
    return pts, idx

def center_pcd(pcd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center point cloud at origin."""
    center = np.mean(pcd, axis=0)
    pcd = pcd - center
    return pcd, center


def rotate_pcd(pcd: np.ndarray, normals: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotate point cloud and normals."""
    rot = Rotation.random()
    pcd = rot.apply(pcd)
    
    if normals is not None:
        normals = rot.apply(normals)

    rot_inv = rot.inv()
    return pcd, normals, rot_inv.as_matrix()


def rotate_pcd_yaw(pcd: np.ndarray, normals: np.ndarray = None, 
                   yaw_range: float = 360.0, 
                   roll_pitch_range: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply random yaw rotation around z-axis with small roll/pitch perturbations.
    
    Args:
        pcd: Point cloud of shape (N, 3)
        normals: Normal vectors of shape (N, 3), optional
        yaw_range: Range for yaw rotation in degrees (default: 360.0 for full rotation)
        roll_pitch_range: Range for roll/pitch perturbations in degrees (default: 30.0)
        
    Returns:
        pcd: Rotated point cloud
        normals: Rotated normal vectors (if provided)
        rot_inv: Inverse rotation matrix (3, 3)
    """
    # Generate random angles
    yaw = np.random.uniform(-yaw_range/2, yaw_range/2)  # Random yaw around z-axis
    roll = np.random.uniform(-roll_pitch_range, roll_pitch_range)  # Small roll perturbation
    pitch = np.random.uniform(-roll_pitch_range, roll_pitch_range)  # Small pitch perturbation
    
    # Convert to radians
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    
    # Create rotation matrix: R = Rz(yaw) * Rx(roll) * Ry(pitch)
    # This applies yaw first, then small roll and pitch perturbations
    rot = Rotation.from_euler('zxy', [yaw_rad, roll_rad, pitch_rad])
    
    # Apply rotation to point cloud
    pcd = rot.apply(pcd)
    
    # Apply rotation to normals if provided
    if normals is not None:
        normals = rot.apply(normals)
    
    # Return inverse rotation matrix for later use
    rot_inv = rot.inv()
    return pcd, normals, rot_inv.as_matrix()


def pad_data(input_data: np.ndarray, max_parts: int) -> np.ndarray:
    """Pad zeros to data of shape (p, ...) to (max_parts, ...)"""
    d = np.array(input_data)
    pad_shape = (max_parts,) + tuple(d.shape[1:])
    pad_data = np.zeros(pad_shape, dtype=np.float32)
    pad_data[: d.shape[0]] = d
    return pad_data