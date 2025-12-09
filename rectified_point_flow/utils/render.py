"""Point cloud rendering using PyTorch3D or Mitsuba."""

import colorsys
import math

import numpy as np
import matplotlib.cm as cm
import torch
from PIL import Image
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)

try:
    import mitsuba
    mitsuba.set_variant('scalar_rgb')
    from mitsuba import load_dict
    from mitsuba import Transform4f, Point3f, Vector3f
    import drjit as dr
    mitsuba_available = True
except ImportError:
    mitsuba_available = False

# Default color map for point cloud visualization (64 distinct colors)
CMAP_DEFAULT = [
    [0.99, 0.55, 0.38],  # Coral/Orange
    [0.52, 0.75, 0.90],  # Sky Blue
    [0.65, 0.85, 0.33],  # Lime Green
    [0.91, 0.54, 0.76],  # Pink
    [0.79, 0.38, 0.69],  # Purple
    [1.00, 0.85, 0.18],  # Yellow
    [0.90, 0.77, 0.58],  # Tan/Beige
    [0.84, 0.00, 0.00],  # Red
    [0.00, 0.65, 0.93],  # Blue
    [0.55, 0.24, 1.00],  # Violet
    [0.00, 0.80, 0.40],  # Green
    [1.00, 0.50, 0.00],  # Orange
    [0.20, 0.60, 0.80],  # Cyan Blue
    [0.90, 0.20, 0.30],  # Rose Red
    [0.40, 0.70, 0.40],  # Forest Green
    [0.70, 0.30, 0.60],  # Magenta
    [0.30, 0.50, 0.70],  # Steel Blue
    [0.80, 0.60, 0.20],  # Brown/Gold
    [0.50, 0.80, 0.70],  # Mint Green
    [0.90, 0.40, 0.50],  # Salmon
    [0.20, 0.40, 0.60],  # Navy Blue
    [0.60, 0.80, 0.20],  # Chartreuse
    [0.80, 0.30, 0.40],  # Crimson
    [0.40, 0.60, 0.80],  # Light Blue
    [0.70, 0.50, 0.30],  # Sienna
    [0.30, 0.70, 0.50],  # Teal
    [0.90, 0.60, 0.30],  # Peach
    [0.50, 0.30, 0.70],  # Indigo
    [0.60, 0.40, 0.20],  # Dark Brown
    [0.20, 0.80, 0.60],  # Turquoise
    [0.95, 0.70, 0.85],  # Lavender Pink
    [0.35, 0.45, 0.55],  # Slate Gray
    [0.85, 0.45, 0.65],  # Hot Pink
    [0.25, 0.75, 0.85],  # Aqua Blue
    [0.70, 0.85, 0.50],  # Light Green
    [0.45, 0.25, 0.55],  # Dark Purple
    [0.95, 0.80, 0.40],  # Light Gold
    [0.15, 0.50, 0.35],  # Dark Green
    [0.80, 0.50, 0.70],  # Orchid
    [0.55, 0.65, 0.85],  # Periwinkle
    [0.90, 0.30, 0.60],  # Deep Pink
    [0.40, 0.80, 0.60],  # Emerald
    [0.65, 0.25, 0.45],  # Burgundy
    [0.30, 0.85, 0.75],  # Aquamarine
    [0.75, 0.35, 0.25],  # Rust Red
    [0.50, 0.50, 0.70],  # Blue Gray
    [0.85, 0.65, 0.40],  # Amber
    [0.20, 0.30, 0.50],  # Midnight Blue
    [0.95, 0.50, 0.30],  # Coral Red
    [0.60, 0.70, 0.30],  # Olive Green
    [0.70, 0.20, 0.50],  # Deep Rose
    [0.35, 0.60, 0.45],  # Sea Green
    [0.80, 0.40, 0.20],  # Burnt Orange
    [0.45, 0.55, 0.75],  # Powder Blue
    [0.90, 0.50, 0.70],  # Rose Pink
    [0.25, 0.65, 0.40],  # Jade Green
    [0.65, 0.45, 0.25],  # Coffee Brown
    [0.40, 0.30, 0.60],  # Deep Indigo
    [0.85, 0.75, 0.50],  # Khaki
    [0.50, 0.40, 0.30],  # Taupe
    [0.75, 0.60, 0.45],  # Caramel
    [0.30, 0.40, 0.55],  # Charcoal Blue
]


cmap_dict = {
    "default": CMAP_DEFAULT
}

def part_ids_to_colors(part_ids: torch.Tensor, colormap: str = "default", part_order: str = "random") -> torch.Tensor:
    """Generate colors for parts based on part IDs.
    
    Args:
        part_ids: Tensor of shape (N,) containing part IDs for each point.
        colormap: Name of matplotlib colormap to use:
            - "matplotlib:<name>": Use a matplotlib colormap.
            - "hue:<lum>:<sat>": Use a hue-based colormap.
            - "default": Use a default colormap.
        part_order: Reordering of parts to use for colormap:
            - "size": Sort parts by size (default).
            - "id": Sort parts by ID.
            - "random": Randomly shuffle parts.
        
    Returns:
        RGB colors in float tensor of shape (N, 3).
    """
    device = part_ids.device

    if part_order == "size":
        # Order parts by size (descending) so that the largest part is first
        unique_parts, counts = torch.unique(part_ids, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True, stable=True)
        unique_parts = unique_parts[sorted_indices]
    elif part_order == "id":
        # Keep original order
        unique_parts = torch.unique(part_ids)
    elif part_order == "random":
        # Randomly shuffle parts
        unique_parts = torch.randperm(part_ids.max().item() + 1)
    else:
        raise ValueError(f"Invalid part order: {part_order}")
    
    num_parts = len(unique_parts)
    
    if colormap.startswith("matplotlib:"):
        colormap = colormap.split(":")[1]
        cmap = cm.get_cmap(colormap)
        if num_parts == 1:
            color_indices = np.array([0.5])
        else:
            # 0.1-0.9 to avoid extreme colors
            color_indices = np.linspace(0.1, 0.9, num_parts)
        colors_rgba = torch.tensor([cmap(idx) for idx in color_indices], device=device)
        unique_colors = colors_rgba[:, :3].float()     # (num_parts, 3), remove alpha channel
    
    elif colormap.startswith("hue"): # e.g. hue:0.5:0.5
        if ":" in colormap:
            _, lum, sat = colormap.split(":")
            lum, sat = float(lum), float(sat)
        else:
            lum, sat = 0.5, 0.5
        offset = 0.5
        unique_colors = torch.stack([
            torch.tensor(colorsys.hls_to_rgb((offset + float(i) / num_parts) % 1.0, lum, sat)) 
            for i in range(num_parts)
        ], dim=0).to(device)
    
    elif colormap in cmap_dict:
        color_list = cmap_dict[colormap]
        unique_colors = torch.stack([
            torch.tensor(color_list[i % len(color_list)])
            for i in range(num_parts)
        ], dim=0).to(device)

    else:
        raise ValueError(f"Invalid colormap: {colormap}")
    
    # Create mapping from part_id to color index
    max_part_id = unique_parts.max().item()
    part_id_to_color_idx = torch.full((max_part_id + 1,), -1, dtype=torch.long, device=device)
    part_id_to_color_idx[unique_parts] = torch.arange(len(unique_parts), device=device)
    part_indices = part_id_to_color_idx[part_ids]
    colors = unique_colors[part_indices]  # (N, 3)
    return colors


def probs_to_colors(
        probs: torch.Tensor,
        colormap: str = "matplotlib:Blues",
        remap_range: tuple[float, float] = (0.15, 0.95),
    ) -> torch.Tensor:
    """Convert probabilities [0, 1] to RGB colors using matplotlib colormap.
    
    Args:
        probs: Tensor of shape (N,) containing probabilities in [0, 1].
        colormap: Name of matplotlib colormap to use (e.g., "viridis", "plasma", etc).
        remap_range: Remap the probability range before applying the colormap, avoiding extreme colors.

    Returns:
        RGB colors tensor of shape (N, 3) with values in [0, 1].
    """
    device = probs.device
    if colormap.startswith("matplotlib:"):
        colormap = colormap.split(":")[1]
        cmap = cm.get_cmap(colormap)
    elif colormap == "default":
        cmap = cm.get_cmap("Blues")
    else:
        raise ValueError(f"Invalid colormap: {colormap}")

    # Remap probability from [0, 1] to [remap_range[0], remap_range[1]]
    probs = remap_range[0] + (remap_range[1] - remap_range[0]) * probs
    probs = probs.clamp(0, 1)

    probs_np = probs.detach().cpu().numpy()
    colors_rgba = cmap(probs_np)         # (N, 4)
    colors_rgb = colors_rgba[:, :3]      # (N, 3)
    return torch.tensor(colors_rgb, dtype=torch.float32, device=device)


def img_tensor_to_pil(image_tensor: torch.Tensor) -> Image: 
    """Tensor to PIL Image (H, W, C) and scale to [0, 255]."""
    image_np = (image_tensor.cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(image_np)


@torch.inference_mode()
def visualize_point_clouds_pytorch3d(
    points: torch.Tensor,
    colors: torch.Tensor,
    center_points: bool = False,
    image_size: int = 512,
    point_radius: float = 0.015,
    camera_dist: float = 2.0,
    camera_elev: float = 1.0,
    camera_azim: float = 0.0,
    camera_fov: float = 45.0,
) -> torch.Tensor:
    """
    Render point cloud(s) as either flat disks or true mesh spheres.

    Args:
        points: Point cloud coordinates of shape (N, 3) or (B, N, 3).
        colors: Colors for each point of shape (N, 3).
        center_points: If True, centers the point cloud around the origin.
        image_size: Output image resolution (square).
        point_radius: Radius of each rendered point in world units.
        camera_dist: Distance of camera from point cloud center.
        camera_elev: Camera elevation angle in degrees.
        camera_azim: Camera azimuth angle in degrees.
        camera_fov: Camera field of view in degrees.

    Returns:
        Rendered image(s) of shape (H, W, 3) for single input or
        (B, H, W, 3) for batched input, with values in [0, 1].
    """
    device = points.device

    # -- Batch handling --
    if points.dim() == 2:
        batch_size = 1
        single_input = True
        points = points.unsqueeze(0)                # (1, N, 3)
    elif points.dim() == 3:
        batch_size = points.shape[0]
        single_input = False
    else:
        raise ValueError(f"Expected points to have 2 or 3 dims, got {points.dim()}")

    # Prepare lists for batched inputs
    pts_list, cols_list = [], []
    for i in range(batch_size):
        pts = points[i]  # (N, 3)
        if center_points:
            pts = pts - pts.mean(dim=0, keepdim=True)
        pts_list.append(pts.float())
        cols_list.append(colors.float())

    # Camera setup
    R, T = look_at_view_transform(
        dist=camera_dist,
        elev=camera_elev,
        azim=camera_azim,
        device=device
    )
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=camera_fov, device=device)
    pointclouds = Pointclouds(points=pts_list, features=cols_list)
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=point_radius,
        points_per_pixel=40,
    )
    rasterizer = PointsRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    compositor = AlphaCompositor(background_color=(1.0, 1.0, 1.0))
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    images = renderer(pointclouds)               # (B, H, W, 3)

    return images[0] if single_input else images


@torch.inference_mode()
def visualize_point_clouds_mitsuba(
    points: torch.Tensor,
    colors: torch.Tensor,
    center_points: bool = False,
    image_size: int = 512,
    point_radius: float = 0.015,
    camera_dist: float = 2.0,
    camera_elev: float = 20.0,
    camera_azim: float = 45.0,
    camera_fov: float = 45.0,
) -> torch.Tensor:
    """Render point cloud(s) as tiny spheres via Mitsuba 3.

    Args:
        points: Point cloud coordinates of shape (N, 3) or (B, N, 3).
        colors: Colors for each point of shape (N, 3).
        center_points: If True, centers the point cloud around the origin.
        image_size: Output image resolution (square).
        point_radius: Radius of each rendered point in world units.
        camera_dist: Distance of camera from point cloud center.
        camera_elev: Camera elevation angle in degrees.
        camera_azim: Camera azimuth angle in degrees.
        camera_fov: Camera field of view in degrees.

    Returns: (H, W, 3) or (B, H, W, 3), values in [0, 1].
    """
    device = points.device

    # -- Batch handling --
    if points.ndim == 2:
        points = points.unsqueeze(0)
        single = True
    elif points.ndim == 3:
        single = False
    else:
        raise ValueError("points must be (N,3) or (B,N,3)")

    B, N = points.shape[0], points.shape[1]

    # Mitsuba expects points to be in y-up coordinate system

    # common camera transform
    theta = math.radians(camera_azim)
    phi   = math.radians(camera_elev)
    origin = Point3f(
        camera_dist * math.cos(theta) * math.cos(phi),
        camera_dist * math.sin(phi),
        camera_dist * math.sin(theta) * math.cos(phi),
    )
    to_world_cam = Transform4f().look_at(
        origin, Point3f(0, 0, 0), Point3f(0, 1, 0)
    )
    colors = colors.cpu().tolist()
    outputs = []
    for b in range(B):
        pts = points[b]
        if center_points:
            pts = pts - pts.mean(dim=0, keepdim=True)

        scene_spheres = {
            'type': 'scene',
            'integrator': {'type': 'path'},
            'sensor': {
                'type': 'perspective',
                'fov': camera_fov,
                'to_world': to_world_cam,
                'film': {
                    'type': 'hdrfilm',
                    'width': image_size,
                    'height': image_size,
                    'rfilter': {'type': 'gaussian'}
                }
            },
            'env_light': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            },
            'light': {
                'type': 'directional',
                'direction': [0.1, 0.5, 1.0],
                'irradiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            }
        }
        for i, (p, c) in enumerate(zip(pts.cpu().tolist(), colors)):
            scene_spheres[f'sph{i}'] = {
                'type': 'sphere',
                'radius': point_radius,
                'to_world': Transform4f().translate(Vector3f(*p)),
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': c},
                }
            }
        scene = load_dict(scene_spheres, parallel=False)
        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)
        film = sensor.film()
        img_sph = film.develop(raw=False)  # (H,W,3) TensorXf
        arr_sph = np.array(img_sph)        # (H,W,3) ndarray
        sph = torch.from_numpy(arr_sph).to(device).clamp(0, 1)
        outputs.append(sph)
        # clean up
        del scene, sensor, film, img_sph, arr_sph, sph
        dr.sync_thread()

    out = torch.stack(outputs, dim=0)  # (B, H, W, 3)
    return out[0] if single else out


def visualize_point_clouds(
    renderer: str = "pytorch3d",
    **kwargs,
) -> torch.Tensor:
    if renderer == "none":
        # Return a dummy tensor for renderer="none" case
        # This allows the visualizer to skip actual rendering but still function
        points = kwargs.get("points", torch.zeros(1, 3))
        if points.dim() == 2:
            # Single point cloud
            return torch.zeros(1, 1, 1, 3)
        else:
            # Multiple point clouds
            return torch.zeros(points.shape[0], 1, 1, 3)
    elif renderer == "mitsuba":
        if mitsuba_available:
            return visualize_point_clouds_mitsuba(**kwargs)
        else:
            raise ImportError("Mitsuba not found, set visualizer.renderer to 'pytorch3d' to use PyTorch3D for rendering.")
    elif renderer == "pytorch3d":
        return visualize_point_clouds_pytorch3d(**kwargs)
    else:
        raise ValueError(f"Invalid renderer: {renderer}")
