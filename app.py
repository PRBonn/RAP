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
args = parser.parse_args()
LOG_WINDOW_ENABLED = args.log_on
SERVER_PORT = args.server_port


def convert_to_ply(input_path: str, output_path: str) -> bool:
    """Convert PCD or LAS point cloud files to PLY format."""
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


def prepare_colors(pcd, num_points):
    """Prepare colors for point cloud, using existing colors or default."""
    import numpy as np
    if pcd.has_colors() and len(np.asarray(pcd.colors)) > 0:
        return np.asarray(pcd.colors)
    rgb = CMAP_DEFAULT[0]
    return np.tile(rgb, (num_points, 1))


def process_point_cloud_for_visualization(ply_path, max_points, output_path):
    """Process a single PLY file for visualization: downsample and ensure colors."""
    import open3d as o3d
    import numpy as np
    
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        return False
    
    points = np.asarray(pcd.points)
    colors = prepare_colors(pcd, len(points))
    
    # Downsample if needed
    points, colors = downsample_points(points, colors, max_points)
    
    # Create new point cloud
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(points)
    processed_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(str(output_path), processed_pcd, write_ascii=True)
    return True


def combine_point_clouds(ply_files, max_points, output_path):
    """Combine multiple PLY files into one, with color coding."""
    import open3d as o3d
    import numpy as np
    
    all_points, all_colors = [], []
    
    for idx, ply_file in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(ply_file)
        if len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            all_points.append(points)
            
            # Assign color from CMAP_DEFAULT
            rgb = CMAP_DEFAULT[idx % len(CMAP_DEFAULT)]
            cloud_colors = np.tile(rgb, (len(points), 1))
            all_colors.append(cloud_colors)
    
    if not all_points:
        return False
    
    # Combine all points and colors
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Downsample if needed
    combined_points, combined_colors = downsample_points(combined_points, combined_colors, max_points)
    
    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    o3d.io.write_point_cloud(str(output_path), combined_pcd, write_ascii=True)
    return True


def create_glb_from_point_cloud(ply_path: str, output_glb_path: str, max_points: int) -> bool:
    """Convert a PLY point cloud to GLB format using trimesh.Scene."""
    try:
        import trimesh
        import open3d as o3d
        import numpy as np
        
        pcd = o3d.io.read_point_cloud(ply_path)
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
        points, colors = downsample_points(points, colors, max_points)
        
        # Create trimesh PointCloud and export
        point_cloud = trimesh.PointCloud(vertices=points, colors=colors)
        scene = trimesh.Scene()
        scene.add_geometry(point_cloud)
        scene.export(file_obj=output_glb_path)
        return True
    except Exception as e:
        print(f"Error creating GLB from point cloud: {e}")
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


def build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio, 
                       apply_coordinate_transform, adaptive_parameters,
                       rigidity_forcing, n_generations, inference_sampling_steps,
                       save_trajectory):
    """Build command-line arguments for demo.py."""
    cmd = [
        "python", "demo.py",
        "--input", str(tmp_input_dir),
        "--output", str(tmp_output_dir),
        "--log_level", "INFO",
    ]
    
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
    
    return cmd


def process_registered_files(log_dir, tmp_output_dir, max_points):
    """Process registered PLY files and prepare for visualization."""
    registered_pattern = str(log_dir / "**" / "registered" / "*_registered.ply")
    registered_files = sorted(glob.glob(registered_pattern, recursive=True))
    
    if not registered_files:
        return None
    
    try:
        import open3d as o3d
        
        if len(registered_files) > 1:
            # Combine multiple files
            combined_ply_path = tmp_output_dir / "downsampled_combined_registered.ply"
            if combine_point_clouds(registered_files, max_points, combined_ply_path):
                return str(combined_ply_path.resolve())
        else:
            # Process single file
            colored_ply_path = tmp_output_dir / "colored_registered.ply"
            if process_point_cloud_for_visualization(registered_files[0], max_points, colored_ply_path):
                return str(colored_ply_path.resolve())
        
        # Fallback: use first file as-is
        return str(Path(registered_files[0]).resolve())
    except ImportError:
        return str(Path(registered_files[0]).resolve())
    except Exception:
        return str(Path(registered_files[0]).resolve())


def _yield_outputs(zip_path, registered_vis_file, log_output):
    """Helper function to yield outputs conditionally based on LOG_WINDOW_ENABLED."""
    if LOG_WINDOW_ENABLED:
        yield zip_path, registered_vis_file, log_output
    else:
        yield zip_path, registered_vis_file


def run_rap_demo(ply_files, voxel_size, voxel_ratio, apply_coordinate_transform,
                 adaptive_parameters, rigidity_forcing=True, n_generations=1, 
                 inference_sampling_steps=10, save_trajectory=False, max_points=500000):
    """Gradio callback to run the demo.py pipeline."""
    # Normalize inputs
    ply_files = normalize_file_paths(ply_files)
    
    if not ply_files or len(ply_files) < 2:
        error_msg = "Error: Please upload at least 2 point cloud files."
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
            shutil.copy(src, dst)
            if LOG_WINDOW_ENABLED:
                log_output += f"Copied {src.name} to input directory\n"
        elif file_ext in ['.pcd', '.las', '.laz']:
            if LOG_WINDOW_ENABLED:
                log_output += f"Converting {src.name} to PLY format...\n"
            yield from _yield_outputs(None, None, log_output)
            if not convert_to_ply(str(src), str(dst)):
                error_msg = f"Error: Failed to convert {src.name}\n"
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
    
    if LOG_WINDOW_ENABLED:
        log_output += f"\nStarting RAP registration process...\n"
    yield from _yield_outputs(None, None, log_output)
    
    # Build and run command
    cmd = build_demo_command(tmp_input_dir, tmp_output_dir, voxel_size, voxel_ratio,
                            apply_coordinate_transform, adaptive_parameters,
                            rigidity_forcing, n_generations, inference_sampling_steps,
                            save_trajectory)
    
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
        registered_ply_file = None
        if log_dir.exists():
            registered_ply_file = process_registered_files(log_dir, tmp_output_dir, max_points)
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
        if registered_ply_file:
            if LOG_WINDOW_ENABLED:
                log_output += "Converting to GLB format for visualization...\n"
            yield from _yield_outputs(None, None, log_output)
            glb_path = tmp_output_dir / "registered_pointcloud.glb"
            if create_glb_from_point_cloud(registered_ply_file, str(glb_path), max_points):
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
            all_files = sorted(list(folder_path.glob("*.ply")) + list(folder_path.glob("*.pcd")))
            if len(all_files) >= 2:
                examples.append([str(f.resolve()) for f in all_files])
                example_names.append(folder_path.name)


with gr.Blocks() as demo:
    gr.Markdown(
        "## Register Any Point (RAP) 🎤 [[code](https://github.com/PRBonn/RAP)] "
        "[[paper](https://arxiv.org/pdf/2512.01850)] [[project](https://register-any-point.github.io/)]\n"
        "🎤 RAP is a single-stage multi-view point cloud registration model that generates the registered point cloud by flow matching.\n\n"
        "☁️ Upload two or more point cloud files (`.ply`, `.pcd`, `.las`, or `.laz` format, at least two) for conducting the registration.\n"
        "📦 The results (including registered point clouds and logs) will be returned as a zip file.\n\n"
        "🚧 This demo is currently under construction and running on a local machine.\n"
        "⏳ Please be patient as it runs slower than usual due to gradio IO limitations.\n"
        "💡 You may need to enable the WebGPU for the visualization.\n\n"
        "🤔 Tips: If the results are not satisfactory, you can try to increase the number of generations or inference sampling steps and disable the adaptive parameters to try other settings.\n"
    )
    
    with gr.Row():
        ply_files = gr.File(label="Point cloud files", file_types=[".ply", ".pcd", ".las", ".laz"],
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
        voxel_ratio = gr.Slider(minimum=0.02, maximum=2.0, value=0.2, step=0.01,
                               label="Voxel ratio for sampling")
    
    with gr.Row():
        apply_coordinate_transform = gr.Checkbox(value=False,
            label="Apply frame transform (for 3DMatch-like data with Z-axis pointing forward)")
        adaptive_parameters = gr.Checkbox(value=True, label="Use adaptive parameters")
        rigidity_forcing = gr.Checkbox(value=True, label="Enable rigidity forcing")
        save_trajectory = gr.Checkbox(value=False, label="Save trajectory as GIF (in logs/visualizations/)")
    
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
               save_trajectory],
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
