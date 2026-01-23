import os
import sys
import shutil
import subprocess
import glob
import argparse
from pathlib import Path
from datetime import datetime

import gradio as gr

# Add paths for imports (same as demo.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'dataset_process'))

# Import color map
from dataset_process.utils.io_utils import CMAP_DEFAULT

# Parse command-line arguments
parser = argparse.ArgumentParser(description='RAP Gradio Demo')
parser.add_argument('--log_on', action='store_true', default=True,
                    help='Enable log window display (default: True)')
parser.add_argument('--log_off', dest='log_on', action='store_false',
                    help='Disable log window display')
parser.add_argument('--server_port', type=int, default=7860,
                    help='Server port for Gradio app (default: 7860)')
parser.add_argument('--flow_model_checkpoint', type=str, default='./weights/rap_model_12.ckpt',
                    help='Path to PRFM checkpoint')
parser.add_argument('--config', type=str, default='RAP_inference',
                    help='Config name for inference (default: RAP_inference)')
parser.add_argument('--model', type=str, default=None,
                    choices=['rap_10', 'rap_12'],
                    help='Model configuration to use (default: None, uses config default)')
parser.add_argument('--max_points_for_vis', type=int, default=500000,
                    help='Maximum number of points for visualization (default: 500000)')
args = parser.parse_args()
LOG_WINDOW_ENABLED = args.log_on
SERVER_PORT = args.server_port
FLOW_MODEL_CHECKPOINT = args.flow_model_checkpoint
CONFIG = args.config
MODEL = args.model
MAX_POINTS_FOR_VIS = args.max_points_for_vis


def is_mesh_file(file_path: str) -> bool:
    """Check if a file contains mesh data (faces/triangles). Supports PLY and OBJ formats."""
    try:
        import trimesh
        # Try to load as mesh first
        mesh = trimesh.load(str(file_path), process=False)
        if isinstance(mesh, trimesh.Trimesh):
            # Check if it has faces
            if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0:
                return True
        return False
    except Exception:
        # If loading fails, assume it's a point cloud
        return False


def is_ply_mesh(ply_path: str) -> bool:
    """Check if a PLY file contains mesh data (faces/triangles)."""
    return is_mesh_file(ply_path)


def convert_mesh_to_pointcloud(mesh_path: str, output_path: str, num_points: int = 100000) -> bool:
    """Convert a mesh file (PLY, OBJ, etc.) to point cloud PLY by sampling points from the surface."""
    try:
        import trimesh
        import open3d as o3d
        import numpy as np
        
        # Load mesh
        loaded = trimesh.load(str(mesh_path), process=False)
        
        # Handle Scene objects (multiple meshes)
        if isinstance(loaded, trimesh.Scene):
            # Combine all meshes in the scene
            meshes = []
            for name, geometry in loaded.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if not meshes:
                print(f"Error: Scene contains no valid meshes")
                return False
            # Merge all meshes
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            print(f"Error: Unsupported mesh type: {type(loaded)}")
            return False
        
        # Validate mesh
        if mesh.vertices is None or len(mesh.vertices) == 0:
            print(f"Error: Mesh has no vertices")
            return False
        
        if mesh.faces is None or len(mesh.faces) == 0:
            # No faces, use vertices as point cloud
            print(f"Warning: Mesh has no faces, using vertices as point cloud")
            points = np.array(mesh.vertices)
            face_indices = None
        else:
            # Validate faces
            if len(mesh.faces) == 0:
                points = np.array(mesh.vertices)
                face_indices = None
            else:
                # Sample points from mesh surface
                # Use uniform sampling which works well for most meshes
                face_indices = None
                try:
                    points, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
                except Exception as e1:
                    # Fallback to even sampling if uniform fails
                    try:
                        points, face_indices = trimesh.sample.sample_surface_even(mesh, count=num_points)
                    except Exception as e2:
                        # If sampling fails, use vertices
                        print(f"Warning: Surface sampling failed ({str(e1)}, {str(e2)}), using vertices")
                        points = np.array(mesh.vertices)
                        face_indices = None
        
        if len(points) == 0:
            print(f"Error: No points generated from mesh")
            return False
        
        # Compute normals from mesh if available
        normals = None
        if face_indices is not None:
            # We have sampled points with face indices
            try:
                # Get face normals
                face_normals = mesh.face_normals
                if len(face_normals) > 0 and len(face_indices) == len(points):
                    normals = face_normals[face_indices]
            except Exception:
                pass
        
        # If normals not available, try to use vertex normals
        if normals is None:
            try:
                # Try to get vertex normals from mesh
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    vertex_normals = mesh.vertex_normals
                    if len(vertex_normals) == len(mesh.vertices) and len(points) == len(mesh.vertices):
                        # Points are vertices, use vertex normals directly
                        normals = vertex_normals
                    elif face_indices is not None:
                        # Interpolate vertex normals based on face indices
                        # This is a simplified approach - in practice, you'd want barycentric interpolation
                        pass
            except Exception:
                pass
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except Exception as e:
        import traceback
        print(f"Error converting mesh to point cloud: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def convert_pts_to_ply(input_path: str, output_path: str) -> bool:
    """Convert PTS point cloud files to PLY format."""
    try:
        import open3d as o3d
        import numpy as np
        
        points = []
        colors = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # Split by whitespace
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                try:
                    # Parse x, y, z coordinates
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                    
                    # Check if RGB colors are present (columns 3, 4, 5)
                    if len(parts) >= 6:
                        try:
                            r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                            # Normalize to [0, 1] if values are in [0, 255]
                            if r > 1.0 or g > 1.0 or b > 1.0:
                                r, g, b = r / 255.0, g / 255.0, b / 255.0
                            colors.append([r, g, b])
                        except (ValueError, IndexError):
                            pass
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
        
        if len(points) == 0:
            return False
        
        points = np.array(points, dtype=np.float64)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if available
        if len(colors) == len(points):
            colors = np.array(colors, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except Exception as e:
        print(f"Error converting PTS file to PLY: {e}")
        return False


def convert_e57_to_ply(input_path: str, output_path: str) -> bool:
    """Convert E57 point cloud files to PLY format."""
    try:
        import pye57
        import open3d as o3d
        import numpy as np
        
        # Open E57 file
        e57 = pye57.E57(str(input_path))
        
        # Get number of scans
        num_scans = e57.scan_count
        if num_scans == 0:
            print("Error: E57 file contains no scans")
            return False
        
        # Collect points from all scans
        all_points = []
        all_colors = []
        has_colors = True  # Track if all scans have colors
        
        for scan_idx in range(num_scans):
            try:
                # Read scan data
                data = e57.read_scan(scan_idx)
                
                # Extract coordinates
                x = data['cartesianX']
                y = data['cartesianY']
                z = data['cartesianZ']
                
                # Stack coordinates
                points = np.vstack((x, y, z)).transpose()
                all_points.append(points)
                
                # Extract colors if available
                if 'colorRed' in data and 'colorGreen' in data and 'colorBlue' in data:
                    r = data['colorRed']
                    g = data['colorGreen']
                    b = data['colorBlue']
                    colors = np.vstack((r, g, b)).transpose()
                    # Normalize to [0, 1] if values are in [0, 255]
                    if colors.max() > 1.0:
                        colors = colors.astype(np.float64) / 255.0
                    all_colors.append(colors)
                else:
                    # This scan doesn't have colors
                    has_colors = False
            except Exception as e:
                print(f"Warning: Failed to read scan {scan_idx}: {e}")
                continue
        
        if not all_points:
            print("Error: No valid scans found in E57 file")
            return False
        
        # Combine all points
        combined_points = np.vstack(all_points)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        # Add colors if available for all scans
        if has_colors and all_colors and len(all_colors) == len(all_points):
            try:
                combined_colors = np.vstack(all_colors)
                if len(combined_colors) == len(combined_points):
                    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            except Exception as e:
                print(f"Warning: Failed to combine colors: {e}")
        
        if len(pcd.points) == 0:
            return False
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except ImportError:
        print("Error: pye57 is required for E57 files. Install with: pip install pye57")
        return False
    except Exception as e:
        print(f"Error converting E57 file to PLY: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def convert_to_ply(input_path: str, output_path: str) -> bool:
    """Convert PCD, LAS, PTS, or E57 point cloud files to PLY format."""
    try:
        import open3d as o3d
        import numpy as np
        
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.pcd':
            pcd = o3d.io.read_point_cloud(input_path)
            if len(pcd.points) == 0:
                return False
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
            return True
            
        elif file_ext in ['.las', '.laz']:
            try:
                import laspy
                las = laspy.read(input_path)
                points = np.vstack((las.x, las.y, las.z)).transpose()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if len(pcd.points) == 0:
                    return False
                o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
                return True
            except ImportError:
                print("Error: laspy is required for LAS/LAZ files. Install with: pip install laspy")
                return False
            except Exception as e:
                print(f"Error reading LAS file: {e}")
                return False
        
        elif file_ext == '.pts':
            return convert_pts_to_ply(input_path, output_path)
        
        elif file_ext == '.e57':
            return convert_e57_to_ply(input_path, output_path)
        
        return False
    except Exception as e:
        print(f"Error converting point cloud to PLY: {e}")
        return False


def downsample_points(points, colors, max_points):
    """Downsample points and colors if they exceed max_points."""
    import numpy as np
    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        return points[indices], colors[indices] if colors is not None else colors
    return points, colors

def combine_point_clouds(ply_files, output_path, max_points_count, use_original_colors=False):
    """Combine multiple PLY files into one, with color coding.
    
    Returns:
        tuple: (success: bool, point_cloud: o3d.geometry.PointCloud or None)
    """
    import open3d as o3d
    import numpy as np
    
    combined_pcd = o3d.geometry.PointCloud()
    
    for idx, ply_file in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(ply_file)
        if len(pcd.points) == 0:
            continue

        # random downsample pcd to max_points_count
        if len(pcd.points) > max_points_count:
            indices = np.random.choice(len(pcd.points), size=max_points_count, replace=False)
            pcd = pcd.select_by_index(indices)

        # Assign colors based on use_original_colors flag
        if not use_original_colors or not pcd.has_colors():
            # Assign color from CMAP_DEFAULT (overwrites any existing colors)
            rgb = CMAP_DEFAULT[idx % len(CMAP_DEFAULT)]
            pcd.paint_uniform_color(rgb)

        # Combine point clouds using Open3D's + operator
        combined_pcd += pcd
    
    if len(combined_pcd.points) == 0:
        return False, None
    
    o3d.io.write_point_cloud(str(output_path), combined_pcd, write_ascii=False)
    return True, combined_pcd


def create_glb_from_point_cloud(ply_path_or_pcd, output_glb_path: str, max_points_count: int) -> bool:
    """Convert a PLY point cloud (file path or Open3D PointCloud object) to GLB format using trimesh.Scene.
    
    Args:
        ply_path_or_pcd: Either a string path to a PLY file or an Open3D PointCloud object
        output_glb_path: Path to output GLB file
        max_points_count: Maximum number of points for visualization
    """
    try:
        import trimesh
        import open3d as o3d
        import numpy as np
        
        # Handle both file path string and PointCloud object
        if isinstance(ply_path_or_pcd, str):
            pcd = o3d.io.read_point_cloud(ply_path_or_pcd)
        else:
            pcd = ply_path_or_pcd
        
        if len(pcd.points) == 0:
            return False
        
        points = np.asarray(pcd.points)
        
        # Get colors
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            colors = (colors * 255).astype(np.uint8) if colors.max() <= 1.0 else colors.astype(np.uint8)
        else:
            rgb = CMAP_DEFAULT[0]
            colors = np.tile((np.array(rgb) * 255).astype(np.uint8), (len(points), 1))
        
        # Downsample if needed
        points, colors = downsample_points(points, colors, max_points_count)
        
        # Create trimesh PointCloud and export
        point_cloud = trimesh.PointCloud(vertices=points, colors=colors)
        scene = trimesh.Scene()
        scene.add_geometry(point_cloud)
        scene.export(file_obj=output_glb_path)
        return True
    except Exception as e:
        print(f"Error creating GLB from point cloud: {e}")
        return False


def detect_large_coordinates(ply_dir, threshold=1000.0):
    """Check if any point cloud has coordinates exceeding threshold."""
    import open3d as o3d
    import numpy as np
    
    ply_files = list(Path(ply_dir).glob("*.ply"))
    if not ply_files:
        return False
    
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(str(ply_file))
        if len(pcd.points) == 0:
            continue
        points = np.asarray(pcd.points)
        if np.any(np.abs(points) > threshold):
            return True
    return False


def calculate_global_shift(ply_dir):
    """Calculate global shift as the minimum of all points across all point clouds."""
    import open3d as o3d
    import numpy as np
    
    ply_files = list(Path(ply_dir).glob("*.ply"))
    if not ply_files:
        return None
    
    all_mins = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(str(ply_file))
        if len(pcd.points) == 0:
            continue
        points = np.asarray(pcd.points)
        all_mins.append(points.min(axis=0))
    
    if not all_mins:
        return None
    
    # Global shift is the minimum across all point clouds
    global_shift = np.minimum.reduce(all_mins)
    return global_shift


def apply_global_shift_to_ply(ply_path, global_shift):
    """Apply global shift to a PLY file."""
    import open3d as o3d
    import numpy as np
    
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return False
    
    points = np.asarray(pcd.points)
    points_shifted = points - global_shift
    
    pcd_shifted = o3d.geometry.PointCloud()
    pcd_shifted.points = o3d.utility.Vector3dVector(points_shifted)
    
    # Preserve colors if they exist
    if pcd.has_colors():
        pcd_shifted.colors = pcd.colors
    
    # Preserve normals if they exist
    if pcd.has_normals():
        pcd_shifted.normals = pcd.normals
    
    o3d.io.write_point_cloud(str(ply_path), pcd_shifted, write_ascii=False)
    return True


def apply_global_shift_to_directory(ply_dir, global_shift):
    """Apply global shift to all PLY files in a directory."""
    ply_files = list(Path(ply_dir).glob("*.ply"))
    success_count = 0
    for ply_file in ply_files:
        if apply_global_shift_to_ply(ply_file, global_shift):
            success_count += 1
    return success_count


def save_global_shift(global_shift, output_dir):
    """Save global shift to a text file."""
    shift_file = Path(output_dir) / "global_shift.txt"
    try:
        with open(shift_file, 'w') as f:
            f.write("# Global shift applied to input point clouds\n")
            f.write("# Format: shift_x shift_y shift_z\n")
            f.write("# To recover original coordinates, add this shift back\n")
            f.write(f"{global_shift[0]:.6f} {global_shift[1]:.6f} {global_shift[2]:.6f}\n")
        return True
    except Exception as e:
        print(f"Error saving global shift: {e}")
        return False


def normalize_file_paths(ply_files):
    """Normalize file paths from Gradio input."""
    if isinstance(ply_files, str):
        return [ply_files]
    elif not isinstance(ply_files, list):
        return list(ply_files) if ply_files else []
    return [str(f) for f in ply_files if f]


def get_file_path(src):
    """Get file path, handling Gradio file objects."""
    src = Path(src)
    if not src.exists():
        name = getattr(src, "name", None)
        if name:
            src = Path(name)
    return src if src.exists() else None


def calculate_total_file_size(file_paths):
    """Calculate total size of all files in bytes."""
    total_size = 0
    for file_path in file_paths:
        src = get_file_path(file_path)
        if src and src.exists():
            try:
                total_size += src.stat().st_size
            except (OSError, ValueError):
                # Skip files that can't be accessed
                pass
    return total_size


def build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio, 
                       apply_coordinate_transform, adaptive_parameters,
                       rigidity_forcing, n_generations, inference_sampling_steps,
                       save_trajectory, output_generated, use_original_colors):
    """Build command-line arguments for demo.py."""
    cmd = [
        "python", "demo.py",
        "--input", str(tmp_input_dir),
        "--output", str(tmp_output_dir),
        "--log_level", "INFO",
        "--flow_model_checkpoint", FLOW_MODEL_CHECKPOINT,
        "--config", CONFIG,
    ]
    
    if MODEL is not None:
        cmd += ["--model", MODEL]
    
    if voxel_size is not None:
        cmd += ["--voxel_size", str(float(voxel_size))]
    
    if voxel_ratio is not None:
        try:
            if float(voxel_ratio) > 0:
                cmd += ["--voxel_ratio", str(voxel_ratio)]
        except (ValueError, TypeError):
            pass
    
    if apply_coordinate_transform:
        cmd.append("--apply_coordinate_transform")
    
    if adaptive_parameters:
        cmd.append("--adaptive_parameters")
    
    if rigidity_forcing:
        cmd.append("--rigidity_forcing")
    else:
        cmd.append("--no_rigidity_forcing")
    
    if n_generations is not None:
        try:
            ng = int(n_generations)
            if ng > 0:
                cmd += ["--n_generations", str(ng)]
        except (ValueError, TypeError):
            pass
    
    if inference_sampling_steps is not None:
        try:
            iss = int(inference_sampling_steps)
            if iss > 0:
                cmd += ["--inference_sampling_steps", str(iss)]
        except (ValueError, TypeError):
            pass
    
    if save_trajectory:
        cmd.append("--save_trajectory")
        cmd.append("--save_merged_pointcloud_steps")
    else:
        cmd.append("--no_save_merged_pointcloud_steps")
    
    if output_generated:
        cmd.append("--output_generated")
    
    if use_original_colors:
        cmd.append("--use_original_colors")
    
    return cmd


def process_registered_files(log_dir, tmp_output_dir, max_points_count, use_original_colors=False):
    """Process registered PLY files and prepare for visualization.
    
    Returns:
        tuple: (point_cloud: o3d.geometry.PointCloud or None, file_path: str or None)
    """
    registered_pattern = str(log_dir / "**" / "registered" / "*_registered.ply")
    registered_files = sorted(glob.glob(registered_pattern, recursive=True))
    
    if not registered_files:
        return None, None
    
    try:
        import open3d as o3d
        
        if len(registered_files) > 1:
            # Combine multiple files
            combined_ply_path = tmp_output_dir / "downsampled_combined_registered.ply"
            success, combined_pcd = combine_point_clouds(registered_files, combined_ply_path, max_points_count, use_original_colors)
            if success:
                return combined_pcd, str(combined_ply_path.resolve())
        
        # Fallback: use first file as-is - load it and return both
        first_file_path = str(Path(registered_files[0]).resolve())
        pcd = o3d.io.read_point_cloud(first_file_path)
        if len(pcd.points) == 0:
            return None, None
        return pcd, first_file_path
    except ImportError:
        # Fallback: return file path only if Open3D not available
        first_file_path = str(Path(registered_files[0]).resolve())
        return None, first_file_path
    except Exception:
        # Fallback: return file path only on error
        first_file_path = str(Path(registered_files[0]).resolve())
        return None, first_file_path


def _yield_outputs(zip_path, registered_vis_file, log_output):
    """Helper function to yield outputs conditionally based on LOG_WINDOW_ENABLED."""
    if LOG_WINDOW_ENABLED:
        yield zip_path, registered_vis_file, log_output
    else:
        yield zip_path, registered_vis_file


def run_rap_demo(ply_files, voxel_size, voxel_ratio, apply_coordinate_transform,
                 adaptive_parameters, rigidity_forcing=True, n_generations=1, 
                 inference_sampling_steps=10, save_trajectory=False, output_generated=False,
                 use_original_colors=True):
    """Gradio callback to run the demo.py pipeline."""
    max_points_count = MAX_POINTS_FOR_VIS
    
    # Normalize inputs
    ply_files = normalize_file_paths(ply_files)
    
    if not ply_files or len(ply_files) < 2:
        error_msg = "Error: Please upload at least 2 point cloud files."
        yield from _yield_outputs(None, None, error_msg)
        return
    
    # Check total file size (5GB limit)
    MAX_TOTAL_SIZE = 5 * 1024 * 1024 * 1024  # 5GB in bytes
    total_size = calculate_total_file_size(ply_files)
    if total_size > MAX_TOTAL_SIZE:
        size_gb = total_size / (1024 * 1024 * 1024)
        error_msg = f"Error: Total input file size ({size_gb:.2f} GB) exceeds the maximum limit of 5 GB. Please reduce the number or size of files."
        yield from _yield_outputs(None, None, error_msg)
        return
    
    # Create temporary directories
    base_tmp = Path("./gradio_tmp")
    base_tmp.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_input_dir = base_tmp / f"input_{timestamp}"
    tmp_output_dir = base_tmp / f"output_{timestamp}"
    tmp_input_dir.mkdir(parents=True, exist_ok=True)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process and copy files
    log_output = "Processing input files...\n" if LOG_WINDOW_ENABLED else ""
    yield from _yield_outputs(None, None, log_output)
    
    for f in ply_files:
        src = get_file_path(f)
        if not src:
            error_msg = f"Error: Could not find file {f}\n"
            if LOG_WINDOW_ENABLED:
                log_output += error_msg
            yield from _yield_outputs(None, None, log_output)
            return
        
        file_ext = src.suffix.lower()
        dst = tmp_input_dir / (src.stem + '.ply')
        
        if file_ext == '.ply':
            # Check if PLY file is a mesh and convert if needed
            if is_ply_mesh(str(src)):
                if LOG_WINDOW_ENABLED:
                    log_output += f"Detected mesh in {src.name}, converting to point cloud...\n"
                yield from _yield_outputs(None, None, log_output)
                if convert_mesh_to_pointcloud(str(src), str(dst)):
                    if LOG_WINDOW_ENABLED:
                        log_output += f"Successfully converted mesh {src.name} to point cloud\n"
                else:
                    # Fallback: try to copy as point cloud
                    if LOG_WINDOW_ENABLED:
                        log_output += f"Warning: Mesh conversion failed, attempting to load as point cloud...\n"
                    try:
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(str(src))
                        if len(pcd.points) > 0:
                            o3d.io.write_point_cloud(str(dst), pcd, write_ascii=False)
                            if LOG_WINDOW_ENABLED:
                                log_output += f"Loaded {src.name} as point cloud (using vertices only)\n"
                        else:
                            error_msg = f"Error: Could not convert mesh {src.name} to point cloud\n"
                            if LOG_WINDOW_ENABLED:
                                log_output += error_msg
                            yield from _yield_outputs(None, None, log_output)
                            return
                    except Exception as e:
                        error_msg = f"Error: Failed to process {src.name}: {str(e)}\n"
                        if LOG_WINDOW_ENABLED:
                            log_output += error_msg
                        yield from _yield_outputs(None, None, log_output)
                        return
            else:
                # Regular point cloud PLY file, just copy
                shutil.copy(src, dst)
                if LOG_WINDOW_ENABLED:
                    log_output += f"Copied {src.name} to input directory\n"
        elif file_ext == '.obj':
            # OBJ files are typically meshes, convert to point cloud
            if LOG_WINDOW_ENABLED:
                log_output += f"Converting OBJ mesh {src.name} to point cloud...\n"
            yield from _yield_outputs(None, None, log_output)
            
            # Try to convert with detailed error reporting
            try:
                if convert_mesh_to_pointcloud(str(src), str(dst)):
                    if LOG_WINDOW_ENABLED:
                        log_output += f"Successfully converted OBJ mesh {src.name} to point cloud\n"
                else:
                    # Check if file exists and is readable
                    import os
                    if not os.path.exists(str(src)):
                        error_msg = f"Error: OBJ file {src.name} not found\n"
                    elif not os.access(str(src), os.R_OK):
                        error_msg = f"Error: Cannot read OBJ file {src.name}\n"
                    else:
                        error_msg = f"Error: Failed to convert OBJ mesh {src.name} to point cloud.\n"
                        error_msg += "Possible reasons: invalid mesh geometry, empty mesh, or unsupported OBJ format.\n"
                        error_msg += "Please check the file and try again.\n"
                    if LOG_WINDOW_ENABLED:
                        log_output += error_msg
                    yield from _yield_outputs(None, None, log_output)
                    return
            except Exception as e:
                error_msg = f"Error: Exception while converting OBJ mesh {src.name}: {str(e)}\n"
                if LOG_WINDOW_ENABLED:
                    log_output += error_msg
                yield from _yield_outputs(None, None, log_output)
                return
        elif file_ext in ['.pcd', '.las', '.laz', '.pts', '.e57']:
            if LOG_WINDOW_ENABLED:
                log_output += f"Converting {src.name} to PLY format...\n"
            yield from _yield_outputs(None, None, log_output)
            if not convert_to_ply(str(src), str(dst)):
                error_msg = f"Error: Failed to convert {src.name}\n"
                if file_ext == '.e57':
                    error_msg += "Note: E57 files require pye57 library. Install with: pip install pye57\n"
                if LOG_WINDOW_ENABLED:
                    log_output += error_msg
                yield from _yield_outputs(None, None, log_output)
                return
            if LOG_WINDOW_ENABLED:
                log_output += f"Successfully converted {src.name}\n"
        else:
            error_msg = f"Error: Unsupported file format {file_ext}\n"
            if LOG_WINDOW_ENABLED:
                log_output += error_msg
            yield from _yield_outputs(None, None, log_output)
            return
    
    # Check for large coordinates and apply global shift if needed
    if LOG_WINDOW_ENABLED:
        log_output += "\nChecking for large coordinates...\n"
    yield from _yield_outputs(None, None, log_output)
    
    global_shift = None
    if detect_large_coordinates(tmp_input_dir, threshold=100000.0):
        if LOG_WINDOW_ENABLED:
            log_output += "Large coordinates detected (likely UTM or global coordinates).\n"
            log_output += "Calculating and applying global shift...\n"
        yield from _yield_outputs(None, None, log_output)
        
        global_shift = calculate_global_shift(tmp_input_dir)
        if global_shift is not None:
            success_count = apply_global_shift_to_directory(tmp_input_dir, global_shift)
            if LOG_WINDOW_ENABLED:
                log_output += f"Applied global shift to {success_count} point cloud(s).\n"
                log_output += f"Global shift: [{global_shift[0]:.6f}, {global_shift[1]:.6f}, {global_shift[2]:.6f}]\n"
            
            # Save global shift to output directory
            if save_global_shift(global_shift, tmp_output_dir):
                if LOG_WINDOW_ENABLED:
                    log_output += "Global shift saved to global_shift.txt in output directory.\n"
            yield from _yield_outputs(None, None, log_output)
        else:
            if LOG_WINDOW_ENABLED:
                log_output += "Warning: Could not calculate global shift.\n"
            yield from _yield_outputs(None, None, log_output)
    else:
        if LOG_WINDOW_ENABLED:
            log_output += "Coordinates are within normal range, no shift needed.\n"
        yield from _yield_outputs(None, None, log_output)
    
    if LOG_WINDOW_ENABLED:
        log_output += f"\nStarting RAP registration process...\n"
    yield from _yield_outputs(None, None, log_output)
    
    # Build and run command
    cmd = build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio,
                            apply_coordinate_transform, adaptive_parameters,
                            rigidity_forcing, n_generations, inference_sampling_steps,
                            save_trajectory, output_generated, use_original_colors)
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, universal_newlines=True)
        
        # Stream output and display logs in real-time
        for line in proc.stdout:
            if LOG_WINDOW_ENABLED:
                log_output += line
            yield from _yield_outputs(None, None, log_output)
        
        proc.wait()
        
        if proc.returncode != 0:
            error_msg = f"\nProcess exited with error code {proc.returncode}\n"
            if LOG_WINDOW_ENABLED:
                log_output += error_msg
            yield from _yield_outputs(None, None, log_output)
            return
        
        if LOG_WINDOW_ENABLED:
            log_output += "\nRegistration completed successfully!\n"
            log_output += "Processing results...\n"
        yield from _yield_outputs(None, None, log_output)
        
        # Process registered files
        log_dir = tmp_output_dir / "logs"
        registered_pcd = None
        registered_ply_file = None
        if log_dir.exists():
            registered_pcd, registered_ply_file = process_registered_files(log_dir, tmp_output_dir, max_points_count, use_original_colors)
            if LOG_WINDOW_ENABLED:
                if registered_ply_file:
                    log_output += f"Found registered point cloud: {Path(registered_ply_file).name}\n"
                else:
                    log_output += "Warning: No registered point cloud files found\n"
        else:
            if LOG_WINDOW_ENABLED:
                log_output += "Warning: Log directory not found\n"
        
        yield from _yield_outputs(None, None, log_output)
        
        # Create zip archive
        if LOG_WINDOW_ENABLED:
            log_output += "Creating output zip archive...\n"
        yield from _yield_outputs(None, None, log_output)
        zip_path = shutil.make_archive(str(tmp_output_dir), "zip", tmp_output_dir)
        if LOG_WINDOW_ENABLED:
            log_output += f"Zip archive created: {Path(zip_path).name}\n"
        
        # Convert to GLB
        registered_vis_file = None
        if registered_pcd is not None or registered_ply_file:
            if LOG_WINDOW_ENABLED:
                log_output += "Converting to GLB format for visualization...\n"
            yield from _yield_outputs(None, None, log_output)
            glb_path = tmp_output_dir / "registered_pointcloud.glb"
            # Use point cloud object directly if available, otherwise fall back to file path
            pcd_or_path = registered_pcd if registered_pcd is not None else registered_ply_file
            if create_glb_from_point_cloud(pcd_or_path, str(glb_path), max_points_count):
                registered_vis_file = str(glb_path.resolve())
                if LOG_WINDOW_ENABLED:
                    log_output += "GLB file created successfully\n"
            else:
                if LOG_WINDOW_ENABLED:
                    log_output += "Warning: Failed to create GLB file\n"
        
        if LOG_WINDOW_ENABLED:
            log_output += "\n✓ All processing completed!\n"
        yield from _yield_outputs(zip_path, registered_vis_file, log_output)
    
    except Exception as e:
        error_msg = f"\nException occurred: {str(e)}\n"
        if LOG_WINDOW_ENABLED:
            log_output += error_msg
        yield from _yield_outputs(None, None, log_output)


# Prepare example datasets
example_data_dir = Path("demo_example_data").resolve()
examples, example_names = [], []

if example_data_dir.exists():
    for folder_path in sorted(example_data_dir.iterdir()):
        if folder_path.is_dir():
            all_files = sorted(list(folder_path.glob("*.ply")) + list(folder_path.glob("*.pcd")) + 
                               list(folder_path.glob("*.pts")) + list(folder_path.glob("*.obj")) +
                               list(folder_path.glob("*.e57")))
            if len(all_files) >= 2:
                examples.append([str(f.resolve()) for f in all_files])
                example_names.append(folder_path.name)


with gr.Blocks() as demo:
    gr.Markdown(
        "## Register Any Point (RAP) 🎤 [[code](https://github.com/PRBonn/RAP)] "
        "[[paper](https://arxiv.org/pdf/2512.01850)] [[project](https://register-any-point.github.io/)]\n"
        "🎤 RAP is a single-stage multi-view point cloud registration model that generates the registered point cloud by flow matching.\n\n"
        "☁️ Upload two or more point cloud / mesh files (`.ply`, `.pcd`, `.las`, `.laz`, `.pts`, `.e57`, or `.obj` format, at least two) for conducting the registration.\n"
        "📦 The results (including registered point clouds and logs) will be returned as a zip file.\n\n"
        "🚧 This demo is currently under construction and running on a local machine.\n"
        "⏳ Please be patient as it runs slower than usual due to gradio IO limitations.\n"
        "💡 You may need to enable the WebGPU for the visualization.\n\n"
        "🤔 Tips: If the results are not satisfactory, you can try to increase the number of generations or inference sampling steps and disable the adaptive parameters to try other settings.\n"
    )
    
    with gr.Row():
        ply_files = gr.File(label="Point cloud files", file_types=[".ply", ".pcd", ".las", ".laz", ".pts", ".e57", ".obj"],
                           file_count="multiple", type="filepath")
    
    # Example buttons
    if examples:
        gr.Markdown("### 📁 Example datasets (click buttons to load all files from folder)")
        buttons_per_row = 3
        for idx in range(0, len(examples), buttons_per_row):
            with gr.Row():
                for j in range(buttons_per_row):
                    if idx + j < len(examples):
                        example_file_list = examples[idx + j]
                        folder_name = example_names[idx + j]
                        gr.Button(f"📂 {folder_name} ({len(example_file_list)} files)",
                                variant="secondary", size="sm", scale=1).click(
                            fn=lambda files=example_file_list: files, outputs=ply_files)
    
    with gr.Row():
        n_generations = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                 label="Number of generations")
        inference_sampling_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                            label="Flow inference steps")

        # print(f"n_generations: {n_generations}, inference_sampling_steps: {inference_sampling_steps}")
    
    with gr.Row():
        voxel_size = gr.Slider(minimum=0.001, maximum=0.4, value=0.25, step=0.001,
                              label="Voxel size (meters) [overwritten by adaptive parameters]")
        voxel_ratio = gr.Slider(minimum=0.01, maximum=2.0, value=0.2, step=0.01,
                               label="Voxel ratio for sampling")
    
    with gr.Row():
        apply_coordinate_transform = gr.Checkbox(value=False,
            label="Apply frame transform (for 3DMatch-like data with Z-axis pointing forward)")
        adaptive_parameters = gr.Checkbox(value=True, label="Use adaptive parameters")
        rigidity_forcing = gr.Checkbox(value=True, label="Enable rigidity forcing")
    with gr.Row():
        output_generated = gr.Checkbox(value=False, label="Output generated keypoints (instead of transformed original points)")
        save_trajectory = gr.Checkbox(value=False, label="Save trajectory (in logs)")
        use_original_colors = gr.Checkbox(value=False, label="Visualize with original colors instead of index")

    
    run_button = gr.Button("Run RAP Demo")
    
    with gr.Row():
        output_zip = gr.File(label="Download output (zip)", interactive=False)
    
    registered_visualization = gr.Model3D(
        label="Registered Point Clouds (3D Viewer) [You may need to enable the WebGPU for the visualization]",
        visible=True)
    
    # Conditionally create log output component
    if LOG_WINDOW_ENABLED:
        log_output = gr.Textbox(
            label="Processing Logs",
            lines=15,
            max_lines=30,
            interactive=False,
            placeholder="Logs will appear here when processing starts...")
        outputs_list = [output_zip, registered_visualization, log_output]
    else:
        outputs_list = [output_zip, registered_visualization]
    
    run_button.click(
        fn=run_rap_demo,
        inputs=[ply_files, voxel_size, voxel_ratio, apply_coordinate_transform,
               adaptive_parameters, rigidity_forcing, n_generations, inference_sampling_steps,
               save_trajectory, output_generated, use_original_colors],
        outputs=outputs_list)


if __name__ == "__main__":
    import os
    share_mode = os.getenv("SHARE_URL", "temporary").lower()
    
    if share_mode == "permanent":
        print(f"Launching with permanent URL mode (no share=True) on port {SERVER_PORT}")
        demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT)
    else:
        print(f"Launching with temporary share URL (will change on restart) on port {SERVER_PORT}")
        print("For a permanent URL, deploy to Hugging Face Spaces or use a custom domain")
        demo.launch(share=True, server_name="0.0.0.0", server_port=SERVER_PORT)
