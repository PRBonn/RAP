import os
import sys
import shutil
import subprocess
import glob
import argparse
from pathlib import Path
from datetime import datetime

import gradio as gr
from natsort import natsorted

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
                    choices=['rap_8', 'rap_10', 'rap_12', 'rap_16'],
                    help='Model configuration to use (default: None, uses config default)')
parser.add_argument('--max_points_for_vis', type=int, default=1000000,
                    help='Maximum number of points for visualization (default: 1000000)')
args = parser.parse_args()
LOG_WINDOW_ENABLED = args.log_on
SERVER_PORT = args.server_port
FLOW_MODEL_CHECKPOINT = args.flow_model_checkpoint
CONFIG = args.config
MODEL = args.model
MAX_POINTS_FOR_VIS = args.max_points_for_vis

# Model selection mapping
MODEL_CONFIGS = {
    # "S (rap_8)": ("rap_8", "./weights/rap_model_8.ckpt"),
    "M (rap_10)": ("rap_10", "./weights/rap_model_10.ckpt"),
    "L (rap_12)": ("rap_12", "./weights/rap_model_12.ckpt"),
    # "Ls (rap_12)": ("rap_12", "./weights/rap_model_12_s.ckpt"),
    # "H (rap_16)": ("rap_16", "./weights/rap_model_16.ckpt"),
}


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


def convert_mesh_to_pointcloud(mesh_path: str, output_path: str, num_points: int = 100000) -> bool:
    """Convert a mesh file (PLY, OBJ, etc.) to point cloud PLY by sampling points from the surface."""
    try:
        import open3d as o3d
        
        # Load mesh with Open3D
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        if len(mesh.vertices) == 0:
            print(f"Error: Mesh has no vertices")
            return False
        
        # Sample points uniformly from mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # Ensure normals are computed
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except Exception as e:
        print(f"Error converting mesh to point cloud: {e}")
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
        
        if has_colors and all_colors and len(all_colors) == len(all_points):
            try:
                combined_colors = np.vstack(all_colors)
                if len(combined_colors) == len(combined_points):
                    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            except Exception:
                pass
        
        if len(pcd.points) == 0:
            return False
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except ImportError:
        print("Error: pye57 required for E57. pip install pye57")
        return False
    except Exception as e:
        print(f"Error converting E57: {e}")
        return False


def convert_ptx_to_ply(input_path: str, output_path: str) -> bool:
    """Convert PTX point cloud files to PLY format."""
    try:
        import open3d as o3d
        import numpy as np
        
        points = []
        colors = []
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
            i = 0
            
            # Skip header (typically 10-12 lines: columns, rows, transformation matrices)
            while i < len(lines) and i < 20:
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    i += 1
                    continue
                # Check if this looks like point data (numeric values)
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        float(parts[0])
                        float(parts[1])
                        float(parts[2])
                        break  # Found start of point data
                    except ValueError:
                        i += 1
                        continue
                i += 1
            
            # Parse point data (X Y Z Intensity R G B)
            for line in lines[i:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                    
                    # Extract RGB if available (columns 4, 5, 6)
                    if len(parts) >= 7:
                        try:
                            r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                            if r > 1.0 or g > 1.0 or b > 1.0:
                                r, g, b = r / 255.0, g / 255.0, b / 255.0
                            colors.append([r, g, b])
                        except (ValueError, IndexError):
                            pass
                except (ValueError, IndexError):
                    continue
        
        if len(points) == 0:
            return False
        
        points = np.array(points, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if len(colors) == len(points):
            colors = np.array(colors, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        return True
    except Exception as e:
        print(f"Error converting PTX file to PLY: {e}")
        return False


def convert_to_ply(input_path: str, output_path: str) -> bool:
    """Convert PCD, LAS, PTS, E57, or PTX point cloud files to PLY format."""
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
        
        elif file_ext == '.ptx':
            return convert_ptx_to_ply(input_path, output_path)
        
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
    
    # First pass: Load all point clouds and calculate total point count
    loaded_pcds = []
    total_points = 0
    
    for idx, ply_file in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(ply_file)
        if len(pcd.points) == 0:
            continue
        loaded_pcds.append((idx, pcd))
        total_points += len(pcd.points)
    
    # Calculate downsample ratio to achieve approximately max_points_count total
    if total_points > max_points_count:
        downsample_ratio = max_points_count / total_points
    else:
        downsample_ratio = 1.0
    
    # Second pass: Downsample each point cloud proportionally and combine
    combined_pcd = o3d.geometry.PointCloud()
    
    for idx, pcd in loaded_pcds:
        # Calculate target number of points for this point cloud
        target_points = max(1, int(len(pcd.points) * downsample_ratio))
        
        # Downsample if needed
        if len(pcd.points) > target_points:
            indices = np.random.choice(len(pcd.points), size=target_points, replace=False)
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
    
    if pcd.has_colors():
        pcd_shifted.colors = pcd.colors
    if pcd.has_normals():
        pcd_shifted.normals = pcd.normals
    
    o3d.io.write_point_cloud(str(ply_path), pcd_shifted, write_ascii=False)
    return True


def apply_global_shift_to_directory(ply_dir, global_shift):
    for ply_file in Path(ply_dir).glob("*.ply"):
        apply_global_shift_to_ply(ply_file, global_shift)


def save_global_shift(global_shift, output_dir):
    try:
        (Path(output_dir) / "global_shift.txt").write_text(
            f"{global_shift[0]:.6f} {global_shift[1]:.6f} {global_shift[2]:.6f}\n"
        )
        return True
    except OSError:
        return False


def normalize_file_paths(ply_files):
    if isinstance(ply_files, str):
        return [ply_files]
    if not isinstance(ply_files, (list, tuple)):
        ply_files = list(ply_files) if ply_files else []
    return [str(f) for f in ply_files if f]


def get_file_path(src):
    src = Path(src)
    return src if src.exists() else None


def calculate_total_file_size(file_paths):
    total = 0
    for f in file_paths:
        src = get_file_path(f)
        if src:
            try:
                total += src.stat().st_size
            except OSError:
                pass
    return total


def build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio,
                       apply_coordinate_transform, adaptive_parameters,
                       rigidity_forcing, n_generations, inference_sampling_steps,
                       save_trajectory, output_generated, use_original_colors,
                       model_name=None, model_checkpoint=None):
    checkpoint = model_checkpoint or FLOW_MODEL_CHECKPOINT
    model = model_name or MODEL
    cmd = [
        "python", "demo.py",
        "--input", str(tmp_input_dir),
        "--output", str(tmp_output_dir),
        "--log_level", "INFO",
        "--flow_model_checkpoint", checkpoint,
        "--config", CONFIG,
    ]
    
    if model is not None:
        cmd += ["--model", model]
    
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
    registered_pattern = str(log_dir / "**" / "registered" / "*_registered.ply")
    registered_files = natsorted(glob.glob(registered_pattern, recursive=True))
    if not registered_files:
        return None, None

    import open3d as o3d
    first_file = str(Path(registered_files[0]).resolve())
    if len(registered_files) > 1:
        combined_ply_path = tmp_output_dir / "downsampled_combined_registered.ply"
        success, combined_pcd = combine_point_clouds(registered_files, combined_ply_path, max_points_count, use_original_colors)
        if success:
            return combined_pcd, str(combined_ply_path.resolve())
    try:
        pcd = o3d.io.read_point_cloud(first_file)
        return (pcd, first_file) if len(pcd.points) > 0 else (None, None)
    except Exception:
        return None, first_file


def _yield_outputs(zip_path, registered_vis_file, log_output=""):
    if LOG_WINDOW_ENABLED:
        yield zip_path, registered_vis_file, log_output
    else:
        yield zip_path, registered_vis_file


def run_rap_demo(ply_files, model_selection, voxel_size, voxel_ratio, apply_coordinate_transform,
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
            if is_mesh_file(str(src)):
                if LOG_WINDOW_ENABLED:
                    log_output += f"Converting mesh {src.name} to point cloud...\n"
                yield from _yield_outputs(None, None, log_output)
                if not convert_mesh_to_pointcloud(str(src), str(dst)):
                    try:
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(str(src))
                        if len(pcd.points) > 0:
                            o3d.io.write_point_cloud(str(dst), pcd, write_ascii=False)
                        else:
                            error_msg = f"Error: Could not convert mesh {src.name}\n"
                            if LOG_WINDOW_ENABLED:
                                log_output += error_msg
                            yield from _yield_outputs(None, None, log_output)
                            return
                    except Exception as e:
                        error_msg = f"Error: Failed to process {src.name}: {e}\n"
                        if LOG_WINDOW_ENABLED:
                            log_output += error_msg
                        yield from _yield_outputs(None, None, log_output)
                        return
            else:
                shutil.copy(src, dst)
        elif file_ext == '.obj':
            if LOG_WINDOW_ENABLED:
                log_output += f"Converting OBJ {src.name}...\n"
            yield from _yield_outputs(None, None, log_output)
            if not convert_mesh_to_pointcloud(str(src), str(dst)):
                error_msg = f"Error: Failed to convert OBJ {src.name}\n"
                if LOG_WINDOW_ENABLED:
                    log_output += error_msg
                yield from _yield_outputs(None, None, log_output)
                return
        elif file_ext in ['.pcd', '.las', '.laz', '.pts', '.e57', '.ptx']:
            if LOG_WINDOW_ENABLED:
                log_output += f"Converting {src.name}...\n"
            yield from _yield_outputs(None, None, log_output)
            if not convert_to_ply(str(src), str(dst)):
                error_msg = f"Error: Failed to convert {src.name}"
                if file_ext == '.e57':
                    error_msg += " (pip install pye57)"
                error_msg += "\n"
                if LOG_WINDOW_ENABLED:
                    log_output += error_msg
                yield from _yield_outputs(None, None, log_output)
                return
        else:
            error_msg = f"Error: Unsupported file format {file_ext}\n"
            if LOG_WINDOW_ENABLED:
                log_output += error_msg
            yield from _yield_outputs(None, None, log_output)
            return
    
    if LOG_WINDOW_ENABLED:
        log_output += "\nChecking coordinates...\n"
    yield from _yield_outputs(None, None, log_output)

    if detect_large_coordinates(tmp_input_dir, threshold=100000.0):
        global_shift = calculate_global_shift(tmp_input_dir)
        if global_shift is not None:
            apply_global_shift_to_directory(tmp_input_dir, global_shift)
            save_global_shift(global_shift, tmp_output_dir)
            if LOG_WINDOW_ENABLED:
                log_output += f"Applied global shift [{global_shift[0]:.2f}, {global_shift[1]:.2f}, {global_shift[2]:.2f}]\n"
        yield from _yield_outputs(None, None, log_output)
    
    input_ply_files = natsorted(Path(tmp_input_dir).glob("*.ply"), key=lambda p: p.name)
    combined_input_ply_path = tmp_output_dir / "downsampled_combined_input.ply"
    combine_point_clouds([str(f) for f in input_ply_files], str(combined_input_ply_path),
                         max_points_count, use_original_colors)
    yield from _yield_outputs(None, None, log_output)
    
    model_name, model_checkpoint = MODEL_CONFIGS.get(model_selection, (None, None)) if model_selection else (None, None)
    
    cmd = build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio,
                            apply_coordinate_transform, adaptive_parameters,
                            rigidity_forcing, n_generations, inference_sampling_steps,
                            save_trajectory, output_generated, use_original_colors,
                            model_name=model_name, model_checkpoint=model_checkpoint)
    
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
            if LOG_WINDOW_ENABLED:
                log_output += f"\nProcess exited with code {proc.returncode}\n"
            yield from _yield_outputs(None, None, log_output)
            return
        
        log_dir = tmp_output_dir / "logs"
        registered_pcd, registered_ply_file = (None, None)
        if log_dir.exists():
            registered_pcd, registered_ply_file = process_registered_files(log_dir, tmp_output_dir, max_points_count, use_original_colors)
        yield from _yield_outputs(None, None, log_output)

        zip_path = shutil.make_archive(str(tmp_output_dir), "zip", tmp_output_dir)
        registered_vis_file = None
        if registered_pcd or registered_ply_file:
            glb_path = tmp_output_dir / "registered_pointcloud.glb"
            if create_glb_from_point_cloud(registered_pcd or registered_ply_file, str(glb_path), max_points_count):
                registered_vis_file = str(glb_path.resolve())
        yield from _yield_outputs(zip_path, registered_vis_file, log_output)
    
    except Exception as e:
        if LOG_WINDOW_ENABLED:
            log_output += f"\nError: {e}\n"
        yield from _yield_outputs(None, None, log_output)


# Prepare example datasets
example_data_dir = Path("demo_example_data").resolve()
examples, example_names = [], []

if example_data_dir.exists():
    for folder_path in natsorted(example_data_dir.iterdir(), key=lambda p: p.name):
        if folder_path.is_dir():
            all_files = (
                list(folder_path.glob("*.ply")) + list(folder_path.glob("*.pcd")) +
                list(folder_path.glob("*.pts")) + list(folder_path.glob("*.obj")) +
                list(folder_path.glob("*.e57"))
            )
            all_files = natsorted(all_files, key=lambda p: p.name)
            if len(all_files) >= 2:
                examples.append([str(f.resolve()) for f in all_files])
                example_names.append(folder_path.name)


with gr.Blocks() as demo:
    gr.Markdown(
        "## Register Any Point (RAP) 🎤 [[code](https://github.com/PRBonn/RAP)] "
        "[[paper](https://arxiv.org/pdf/2512.01850)] [[project](https://register-any-point.github.io/)]\n"
        "🎤 RAP is a single-stage multi-view point cloud registration model that generates the registered point cloud by flow matching.\n\n"
        "☁️ Upload two or more point cloud / mesh files (`.ply`, `.pcd`, `.las`, `.laz`, `.pts`, `.e57`, `.ptx`, or `.obj` format, at least two) for conducting the registration.\n"
        "📦 The results (including registered point clouds and logs) will be returned as a zip file.\n\n"
        "🚧 This demo is currently under construction and running on a local machine.\n"
        "⏳ Please be patient as it runs slower than usual due to gradio IO limitations.\n"
        "💡 You may need to enable the WebGPU for the visualization.\n\n"
        "🤔 Tips: If the results are not satisfactory, you can try to increase the number of generations or inference sampling steps and disable the adaptive parameters to try other settings.\n"
    )
    
    with gr.Row():
        ply_files = gr.File(label="Point cloud files", file_types=[".ply", ".pcd", ".las", ".laz", ".pts", ".e57", ".ptx", ".obj"],
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
                        button_text = f"📂 {folder_name} ({len(example_file_list)} files)"
                        gr.Button(button_text,
                                variant="secondary", size="sm", scale=1).click(
                            fn=lambda files=example_file_list: files, outputs=ply_files)
    
    with gr.Row():
        model_selection = gr.Radio(
            choices=list(MODEL_CONFIGS.keys()),
            value="L (rap_12)",  # Default to L (rap_12)
            label="Model Zoo",
        )
    
    with gr.Row():
        n_generations = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                 label="Number of generations")
        inference_sampling_steps = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                            label="Flow inference steps")

        # print(f"n_generations: {n_generations}, inference_sampling_steps: {inference_sampling_steps}")
    
    with gr.Row():
        voxel_size = gr.Slider(minimum=0.001, maximum=0.5, value=0.25, step=0.001,
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
        inputs=[ply_files, model_selection, voxel_size, voxel_ratio, apply_coordinate_transform,
               adaptive_parameters, rigidity_forcing, n_generations, inference_sampling_steps,
               save_trajectory, output_generated, use_original_colors],
        outputs=outputs_list)


if __name__ == "__main__":
    share = os.getenv("SHARE_URL", "temporary").lower() != "permanent"
    demo.launch(share=share, server_name="0.0.0.0", server_port=SERVER_PORT)
