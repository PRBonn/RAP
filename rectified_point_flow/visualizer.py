"""Visualization utilities for point cloud registration."""

from pathlib import Path
import logging
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from .utils.render import visualize_point_clouds, img_tensor_to_pil, part_ids_to_colors, probs_to_colors
from .utils.point_clouds import ppp_to_ids

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import pandas as pd

logger = logging.getLogger("Visualizer")


class VisualizationCallback(Callback):
    """Base Lightning callback for visualizing point clouds during evaluation."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        renderer: str = "mitsuba",
        colormap: str = "default",
        scale_to_original_size: bool = False,
        center_points: bool = False,
        image_size: int = 512,
        point_radius: float = 0.015,
        camera_dist: float = 4.0,
        camera_elev: float = 20.0,
        camera_azim: float = 30.0,
        camera_fov: float = 45.0,
        max_samples_per_batch: Optional[int] = None,
    ):
        """Initialize base visualization callback.

        Args:
            save_dir (str): Directory to save images. If None, uses trainer.log_dir/visualizations.
            renderer (str): Renderer to use, can be "mitsuba" or "pytorch3d". Default: "mitsuba".
            colormap (str): Colormap to use. Default: "default".
            scale_to_original_size (bool): If True, scales the point clouds to the original size. 
                Otherwise, keep the scaling, i.e. [-1, 1]. Default: False.
            center_points: If True, centers the point cloud around the origin. Default: False.
            image_size (int): Output image resolution (square). Default: 512.
            point_radius (float): Radius of each rendered point in world units. Default: 0.015.
            camera_dist (float): Distance (m) of camera from origin. Default: 4.0.
            camera_elev (float): Elevation angle (deg). Default: 20.0.
            camera_azim (float): Azimuth angle (deg). Default: 30.0.
            camera_fov (float): Field of view (deg). Default: 45.0.
            max_samples_per_batch (int): Maximum samples to visualize per batch. None means all.
        """
        super().__init__()
        self.save_dir = save_dir
        self.renderer = renderer
        self.colormap = colormap
        self.scale_to_original_size = scale_to_original_size
        self.max_samples_per_batch = max_samples_per_batch

        self.vis_dir = None
        self._vis_kwargs = {
            "renderer": self.renderer,
            "center_points": center_points,
            "image_size": image_size,
            "point_radius": point_radius,
            "camera_dist": camera_dist,
            "camera_elev": camera_elev,
            "camera_azim": camera_azim,
            "camera_fov": camera_fov,
        }

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test" and self.renderer != "none":
            if self.save_dir is None:
                self.vis_dir = Path(trainer.log_dir) / "visualizations"
            else:
                self.vis_dir = Path(self.save_dir) / "visualizations"
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _save_sample_images(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save visualization images for a single sample.
        
        Args:
            points (torch.Tensor): Point cloud of shape (N, 3).
            colors (torch.Tensor): Colors of shape (N, 3).
            sample_name (str): sample name for filename.
            subfolder (str | None): Optional subfolder under visualization directory.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
            
        try:
            image = visualize_point_clouds(
                points=points,
                colors=colors,
                **self._vis_kwargs
            )
            image_pil = img_tensor_to_pil(image)
            sample_name = sample_name.replace('/', '_')
            target_dir = self.vis_dir if subfolder is None else (self.vis_dir / str(subfolder))
            target_dir.mkdir(parents=True, exist_ok=True)
            image_pil.save(target_dir / f"{sample_name}.png")
        except Exception as e:
            logger.error(f"Error saving visualization for sample {sample_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Override this method in subclasses for specific visualization logic."""
        raise NotImplementedError("Subclasses must implement on_test_batch_end")


class FlowVisualizationCallback(VisualizationCallback):
    """Visualization callback for rectified point flow models."""

    def __init__(
        self,
        save_trajectory: bool = True,
        trajectory_gif_fps: int = 25,
        trajectory_gif_pause_last_frame: float = 1.0,
        filter_failures: bool = False,
        failure_metric: Optional[str] = None,
        failure_generation_idx: int = 0,
        folder_suffix: Optional[str] = None,
        save_individual_parts: bool = False,
        colorize_input_parts_with_pca: bool = False,
        colorize_generated_parts_with_pca: bool = False,
        **kwargs
    ):
        """Initialize flow visualization callback.

        Args:
            save_trajectory (bool): Whether to save trajectory as GIF. Default: True.
            trajectory_gif_fps (int): Frames per second for the GIF.
            trajectory_gif_pause_last_frame (float): Pause time for the last frame in seconds.
            filter_failures (bool): If True, only visualize samples that fail the specified metric.
                Default: False (visualize all samples).
            failure_metric (str | None): Metric name to check for failures. For recall metrics,
                a failure means the value is 0 (or < threshold). For error metrics, a failure
                means the value exceeds the threshold. Examples: "recall_at_5deg_2m (outdoor_bufferx)".
                If None and filter_failures=True, uses "recall_at_5deg_2m (outdoor_bufferx)".
            failure_generation_idx (int): Which generation index to use for failure checking.
                Default: 0 (first generation).
            folder_suffix (str | None): Optional suffix to append to dataset name in folder path.
                If provided, folders will be named like "{dataset_name}_{folder_suffix}".
                Default: None (use dataset name only).
            save_individual_parts (bool): If True, save individual visualizations for each part
                colored by part ID. Default: False.
            colorize_input_parts_with_pca (bool): If True and save_individual_parts=True, colorize
                input parts using PCA of point-wise local features from batch["features"].
                Requires features to be available in the batch. Default: False.
            colorize_generated_parts_with_pca (bool): If True and save_individual_parts=True, colorize
                generated parts using PCA of transformer features (after transformer, before MLP head).
                Requires transformer_features to be available in outputs. Default: False.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(**kwargs)
        self.save_trajectory = save_trajectory
        self.trajectory_gif_fps = trajectory_gif_fps
        self.trajectory_gif_pause_last_frame = trajectory_gif_pause_last_frame
        self.filter_failures = filter_failures
        self.failure_metric = failure_metric or "recall_at_5deg_2m (outdoor_bufferx)"
        self.failure_generation_idx = failure_generation_idx
        self.folder_suffix = folder_suffix
        self.save_individual_parts = save_individual_parts
        self.colorize_input_parts_with_pca = colorize_input_parts_with_pca
        self.colorize_generated_parts_with_pca = colorize_generated_parts_with_pca
        
        # PCA basis storage (computed from first batch)
        self.input_features_pca_basis = None  # Dict with 'mean', 'eigenvecs', 'pca_min', 'pca_max'
        self.transformer_features_pca_basis = None  # Dict with 'mean', 'eigenvecs', 'pca_min', 'pca_max'
        self.pca_basis_computed = False  # Track if PCA basis has been computed

    def _compute_pca_basis(self, features: torch.Tensor) -> dict:
        """Compute PCA basis from features.
        
        Args:
            features: Point-wise features of shape (N, F) where F is feature dimension.
            
        Returns:
            Dict with 'mean', 'eigenvecs', 'pca_min', 'pca_max' for PCA transformation.
        """
        if features.shape[0] < 2:
            return None
        
        # Center the features
        features_mean = features.mean(dim=0, keepdim=True)  # (1, F)
        features_centered = features - features_mean
        
        # Compute covariance matrix
        cov = torch.mm(features_centered.t(), features_centered) / (features.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            # Sort in descending order
            idx = eigenvals.argsort(descending=True)
            eigenvecs = eigenvecs[:, idx]
        except Exception as e:
            logger.warning(f"PCA basis computation failed: {e}.")
            return None
        
        # Project features onto first 3 principal components
        pca_features = torch.mm(features_centered, eigenvecs[:, :3])  # (N, 3)
        
        # Compute normalization parameters
        pca_min = pca_features.min(dim=0, keepdim=True)[0]  # (1, 3)
        pca_max = pca_features.max(dim=0, keepdim=True)[0]  # (1, 3)
        
        return {
            'mean': features_mean,
            'eigenvecs': eigenvecs[:, :3],  # (F, 3)
            'pca_min': pca_min,
            'pca_max': pca_max,
        }

    def _features_to_pca_colors(
        self, 
        features: torch.Tensor, 
        pca_basis: Optional[dict] = None
    ) -> torch.Tensor:
        """Convert point-wise features to RGB colors using PCA.
        
        Args:
            features: Point-wise features of shape (N, F) where F is feature dimension.
            pca_basis: Optional pre-computed PCA basis dict with 'mean', 'eigenvecs', 'pca_min', 'pca_max'.
                      If None, computes PCA per-part (old behavior).
            
        Returns:
            RGB colors of shape (N, 3) with values in [0, 1].
        """
        if features.shape[0] < 2:
            # Not enough points for PCA, return uniform color
            return torch.ones(features.shape[0], 3, device=features.device) * 0.5
        
        if pca_basis is not None:
            # Use pre-computed PCA basis
            features_mean = pca_basis['mean']  # (1, F)
            eigenvecs = pca_basis['eigenvecs']  # (F, 3)
            pca_min = pca_basis['pca_min']  # (1, 3)
            pca_max = pca_basis['pca_max']  # (1, 3)
            
            # Center the features using stored mean
            features_centered = features - features_mean
            
            # Project features onto first 3 principal components
            pca_features = torch.mm(features_centered, eigenvecs)  # (N, 3)
        else:
            # Compute PCA per-part (old behavior)
            # Center the features
            features_centered = features - features.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = torch.mm(features_centered.t(), features_centered) / (features.shape[0] - 1)
            
            # Compute eigenvalues and eigenvectors
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(cov)
                # Sort in descending order
                idx = eigenvals.argsort(descending=True)
                eigenvecs = eigenvecs[:, idx]
            except Exception as e:
                logger.warning(f"PCA computation failed: {e}. Using uniform colors.")
                return torch.ones(features.shape[0], 3, device=features.device) * 0.5
            
            # Project features onto first 3 principal components
            pca_features = torch.mm(features_centered, eigenvecs[:, :3])  # (N, 3)
            
            # Normalize to [0, 1] range for RGB
            pca_min = pca_features.min(dim=0, keepdim=True)[0]
            pca_max = pca_features.max(dim=0, keepdim=True)[0]
        
        # Normalize to [0, 1] range for RGB
        pca_range = pca_max - pca_min
        
        # Avoid division by zero
        pca_range = torch.where(pca_range < 1e-6, torch.ones_like(pca_range), pca_range)
        
        colors_normalized = (pca_features - pca_min) / pca_range
        
        # Clamp to [0, 1] to ensure valid RGB values
        colors_normalized = torch.clamp(colors_normalized, 0.0, 1.0)
        
        return colors_normalized

    def _save_trajectory_gif(
        self,
        trajectory: torch.Tensor,
        colors: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save trajectory as GIF.
        
        Args:
            trajectory: Point clouds representing the trajectory steps of shape (num_steps, N, 3).
            colors: Colors of shape (N, 3). Same for all trajectory steps.
            sample_name (str): sample name for filename.
            subfolder (str | None): Optional subfolder under visualization directory.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
            
        try:
            target_dir = self.vis_dir if subfolder is None else (self.vis_dir / str(subfolder))
            target_dir.mkdir(parents=True, exist_ok=True)
            gif_path = target_dir / f"{sample_name}.gif"

            # Render trajectory steps
            rendered_images = visualize_point_clouds(
                points=trajectory,                                          # (num_steps, N, 3)
                colors=colors,                                              # (N, 3)
                **self._vis_kwargs,
            )                                                               # (num_steps, H, W, 3)
            frames = []
            num_steps = trajectory.shape[0]
            for step in range(num_steps):
                frame_pil = img_tensor_to_pil(rendered_images[step])        # (H, W, 3)
                frames.append(frame_pil)
            
            # Frame duration and pause on last frame in ms
            duration = int(1000 / self.trajectory_gif_fps)
            durations = [duration] * len(frames)
            durations[-1] = int(duration + self.trajectory_gif_pause_last_frame * 1000)

            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,  # Infinite loop
                optimize=True
            )
        except Exception as e:
            logger.error(f"Error saving trajectory GIF for sample {sample_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Save flow visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        points_per_part = batch["points_per_part"]                            # (bs, max_parts)
        B, _ = points_per_part.shape
        part_ids = ppp_to_ids(points_per_part)                                # (bs, N)
        
        # Handle dynamic batching - check if we have cu_seqlens_batch
        if "cu_seqlens_batch" in batch:
            # Dynamic batching format (TP, 3)
            cu_seqlens_batch = batch["cu_seqlens_batch"]                      # (B + 1,)
            pts_flat = batch["pointclouds"]                                   # (TP, 3)
            pts_gt_flat = batch["pointclouds_gt"]                            # (TP, 3)
            
            # Convert from dynamic batching to fixed batching
            pts_batches = []
            pts_gt_batches = []
            
            for b in range(B):
                start_idx = cu_seqlens_batch[b]
                end_idx = cu_seqlens_batch[b + 1]
                pts_batches.append(pts_flat[start_idx:end_idx])
                pts_gt_batches.append(pts_gt_flat[start_idx:end_idx])
            
            # Find max points for padding
            max_points = max(batch.shape[0] for batch in pts_batches)
            
            # Create padded tensors
            device = pts_flat.device
            pts = torch.zeros(B, max_points, 3, device=device, dtype=pts_flat.dtype)
            pts_gt = torch.zeros(B, max_points, 3, device=device, dtype=pts_gt_flat.dtype)
            
            for b, (pts_batch, pts_gt_batch) in enumerate(zip(pts_batches, pts_gt_batches)):
                pts[b, :pts_batch.shape[0]] = pts_batch
                pts_gt[b, :pts_gt_batch.shape[0]] = pts_gt_batch
        else:
            # Fixed batching format (B, N, 3)
            pts = batch["pointclouds"].view(B, -1, 3)                         # (bs, N, 3)
            pts_gt = batch["pointclouds_gt"].view(B, -1, 3)                   # (bs, N, 3)
        
        # K generations
        trajectories_list = outputs['trajectories']                           # (K, num_steps, num_points, 3)
        K = len(trajectories_list)
        
        # Handle trajectory reshaping for dynamic batching
        if "cu_seqlens_batch" in batch:
            # For dynamic batching, trajectories are in (num_steps, TP, 3) format
            pointclouds_pred_list = []
            for traj in trajectories_list:
                # Convert each trajectory from (num_steps, TP, 3) to (B, N, 3) for the final step
                final_step = traj[-1]  # (TP, 3)
                
                # Split into batches like we did above
                pred_batches = []
                for b in range(B):
                    start_idx = cu_seqlens_batch[b]
                    end_idx = cu_seqlens_batch[b + 1]
                    pred_batches.append(final_step[start_idx:end_idx])
                
                # Pad to max_points
                pred_padded = torch.zeros(B, max_points, 3, device=final_step.device, dtype=final_step.dtype)
                for b, pred_batch in enumerate(pred_batches):
                    pred_padded[b, :pred_batch.shape[0]] = pred_batch
                
                pointclouds_pred_list.append(pred_padded)
        else:
            # Fixed batching
            pointclouds_pred_list = [traj[-1].view(B, -1, 3) for traj in trajectories_list]

        if self.scale_to_original_size:
            scale = batch["scales"]                                           # (B,)
            pts = pts * scale[:, None, None]                                  # (bs, N, 3)
            pts_gt = pts_gt * scale[:, None, None]                           # (bs, N, 3)
            pointclouds_pred_list = [pred * scale[:, None, None] for pred in pointclouds_pred_list]

        # Extract features if available and PCA coloring is enabled
        features_batch = None
        if self.colorize_input_parts_with_pca and "features" in batch and batch["features"] is not None:
            features_raw = batch["features"]  # (B, N, F) or (TP, F) for dynamic batching
            if "cu_seqlens_batch" in batch:
                # Dynamic batching: features are in (TP, F) format
                # Reuse max_points computed earlier
                feature_batches = []
                for b in range(B):
                    start_idx = cu_seqlens_batch[b]
                    end_idx = cu_seqlens_batch[b + 1]
                    feature_batches.append(features_raw[start_idx:end_idx])
                
                # Pad to max_points (already computed above)
                F = features_raw.shape[-1]
                device = features_raw.device
                features_batch = torch.zeros(B, max_points, F, device=device, dtype=features_raw.dtype)
                
                for b, feat_batch in enumerate(feature_batches):
                    features_batch[b, :feat_batch.shape[0]] = feat_batch
            else:
                # Fixed batching: features are in (B, N, F) format
                features_batch = features_raw.view(B, -1, features_raw.shape[-1])

        # Extract transformer features if available and PCA coloring for generated parts is enabled
        transformer_features_list = None
        if self.colorize_generated_parts_with_pca and 'transformer_features' in outputs:
            transformer_features_list = outputs['transformer_features']  # List of (TP, embed_dim), one per generation

        # Compute PCA basis from first batch if not already computed
        if not self.pca_basis_computed:
            # Compute PCA basis for input features
            if self.colorize_input_parts_with_pca and features_batch is not None:
                # Collect all valid features from the batch
                all_input_features = []
                for b in range(B):
                    valid_n = int(points_per_part[b].sum().item())
                    if valid_n > 0:
                        all_input_features.append(features_batch[b][:valid_n])
                
                if all_input_features:
                    all_input_features = torch.cat(all_input_features, dim=0)  # (total_points, F)
                    self.input_features_pca_basis = self._compute_pca_basis(all_input_features)
                    if self.input_features_pca_basis is not None:
                        logger.info(f"Computed input features PCA basis from first batch: {all_input_features.shape[0]} points")
            
            # Compute PCA basis for transformer features
            if self.colorize_generated_parts_with_pca and transformer_features_list is not None and len(transformer_features_list) > 0:
                # Use first generation's transformer features
                transformer_features_first = transformer_features_list[0]  # (TP, embed_dim)
                
                if "cu_seqlens_batch" in batch:
                    # Dynamic batching: collect all features
                    all_transformer_features = transformer_features_first  # Already (TP, embed_dim)
                else:
                    # Fixed batching: collect all features
                    all_transformer_features = transformer_features_first.view(-1, transformer_features_first.shape[-1])  # (B*N, embed_dim)
                
                self.transformer_features_pca_basis = self._compute_pca_basis(all_transformer_features)
                if self.transformer_features_pca_basis is not None:
                    logger.info(f"Computed transformer features PCA basis from first batch: {all_transformer_features.shape[0]} points")
            
            self.pca_basis_computed = True

        # Get failure mask if filtering is enabled
        failure_mask = None
        if self.filter_failures and 'eval_results' in outputs:
            eval_results_list = outputs.get('eval_results', [])
            if eval_results_list and len(eval_results_list) > self.failure_generation_idx:
                eval_results = eval_results_list[self.failure_generation_idx]
                if self.failure_metric in eval_results:
                    metric_values = eval_results[self.failure_metric]  # (B,)
                    # For recall metrics: failure means value == 0 (or < 1.0)
                    # For error metrics: failure means value > threshold (but we'll handle recalls here)
                    if 'recall' in self.failure_metric.lower():
                        failure_mask = metric_values < 1.0  # Recall < 1.0 means failure
                    else:
                        # For error metrics, we'd need a threshold, but for now assume 0 means success
                        failure_mask = metric_values > 0
                else:
                    logger.warning(f"Metric '{self.failure_metric}' not found in eval_results. Available metrics: {list(eval_results.keys())}")

        # If filtering failures, count how many failures we've visualized
        failures_visualized = 0
        
        for i in range(B):
            # Check if we've reached the maximum number of samples to visualize
            # If filtering failures, count actual failures visualized; otherwise count samples checked
            if self.max_samples_per_batch is not None:
                if self.filter_failures and failure_mask is not None:
                    # When filtering failures, limit by number of failures visualized
                    if failures_visualized >= self.max_samples_per_batch:
                        break
                else:
                    # When not filtering, limit by number of samples checked
                    if i >= self.max_samples_per_batch:
                        break
            
            # Check if we should skip this sample based on failure filtering
            if failure_mask is not None and not failure_mask[i].item():
                continue  # Skip visualization for non-failure cases
            
            # Increment failure counter if we're visualizing a failure
            if failure_mask is not None:
                failures_visualized += 1
                
            dataset_name = str(batch["dataset_name"][i])
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"
            
            # Construct subfolder name with optional suffix
            subfolder_name = dataset_name
            if self.folder_suffix is not None:
                subfolder_name = f"{dataset_name}_{self.folder_suffix}"

            colors = part_ids_to_colors(
                part_ids[i], colormap=self.colormap, part_order="id"
            )
            # Trim padded zeros if any
            valid_n = int(points_per_part[i].sum().item())
            pts_i = pts[i][:valid_n]
            pts_gt_i = pts_gt[i][:valid_n]
            part_ids_i = part_ids[i][:valid_n]
            colors_i = colors[:valid_n]
            self._save_sample_images(
                points=pts_i,
                colors=colors_i,
                sample_name=f"{sample_name}_input",
                subfolder=subfolder_name,
            )
            self._save_sample_images(
                points=pts_gt_i,
                colors=colors_i,
                sample_name=f"{sample_name}_gt",
                subfolder=subfolder_name,
            )

            for n in range(K):
                pointclouds_pred = pointclouds_pred_list[n]
                pred_i = pointclouds_pred[i][:valid_n]
                self._save_sample_images(
                    points=pred_i,
                    colors=colors_i,
                    sample_name=f"{sample_name}_generation{n+1:02d}",
                    subfolder=subfolder_name,
                )

                if self.save_trajectory:
                    # Save end_point_trajectory (primary trajectory)
                    trajectory = trajectories_list[n]  # (num_steps, TP, 3) or (num_steps, B*N, 3)
                    
                    if "cu_seqlens_batch" in batch:
                        # Dynamic batching: convert (num_steps, TP, 3) to (num_steps, B, N, 3)
                        num_steps = trajectory.shape[0]
                        trajectory_batched = torch.zeros(num_steps, B, max_points, 3, 
                                                       device=trajectory.device, dtype=trajectory.dtype)
                        
                        for step in range(num_steps):
                            step_points = trajectory[step]  # (TP, 3)
                            for b in range(B):
                                start_idx = cu_seqlens_batch[b]
                                end_idx = cu_seqlens_batch[b + 1]
                                batch_points = step_points[start_idx:end_idx]
                                trajectory_batched[step, b, :batch_points.shape[0]] = batch_points
                        
                        trajectory_for_sample = trajectory_batched[:, i]  # (num_steps, N, 3)
                    else:
                        # Fixed batching: reshape and extract
                        num_steps = trajectory.shape[0]
                        trajectory = trajectory.reshape(num_steps, B, -1, 3)  # (num_steps, B, N, 3)
                        trajectory_for_sample = trajectory[:, i]  # (num_steps, N, 3)
                    
                    if self.scale_to_original_size:
                        trajectory_for_sample = trajectory_for_sample * scale[i, None, None]
                    
                    self._save_trajectory_gif(
                        trajectory=trajectory_for_sample,
                        colors=colors,
                        sample_name=f"{sample_name}_generation{n+1:02d}_end_point",
                        subfolder=subfolder_name,
                    )
                    
                    # Save original trajectory (x_t) if available
                    if 'trajectories_x_t' in outputs and outputs['trajectories_x_t'][n] is not None:
                        trajectory_x_t = outputs['trajectories_x_t'][n]  # (num_steps, TP, 3) or (num_steps, B*N, 3)
                        
                        if "cu_seqlens_batch" in batch:
                            # Dynamic batching: convert (num_steps, TP, 3) to (num_steps, B, N, 3)
                            num_steps = trajectory_x_t.shape[0]
                            trajectory_batched = torch.zeros(num_steps, B, max_points, 3, 
                                                           device=trajectory_x_t.device, dtype=trajectory_x_t.dtype)
                            
                            for step in range(num_steps):
                                step_points = trajectory_x_t[step]  # (TP, 3)
                                for b in range(B):
                                    start_idx = cu_seqlens_batch[b]
                                    end_idx = cu_seqlens_batch[b + 1]
                                    batch_points = step_points[start_idx:end_idx]
                                    trajectory_batched[step, b, :batch_points.shape[0]] = batch_points
                            
                            trajectory_x_t_for_sample = trajectory_batched[:, i]  # (num_steps, N, 3)
                        else:
                            # Fixed batching: reshape and extract
                            num_steps = trajectory_x_t.shape[0]
                            trajectory_x_t = trajectory_x_t.reshape(num_steps, B, -1, 3)  # (num_steps, B, N, 3)
                            trajectory_x_t_for_sample = trajectory_x_t[:, i]  # (num_steps, N, 3)
                        
                        if self.scale_to_original_size:
                            trajectory_x_t_for_sample = trajectory_x_t_for_sample * scale[i, None, None]
                        
                        self._save_trajectory_gif(
                            trajectory=trajectory_x_t_for_sample,
                            colors=colors,
                            sample_name=f"{sample_name}_generation{n+1:02d}_original",
                            subfolder=subfolder_name,
                        )
            
            # Save individual parts visualization if enabled
            if self.save_individual_parts:
                # Get unique part IDs for this sample
                unique_part_ids = torch.unique(part_ids_i)
                
                # Extract features for this sample if PCA coloring is enabled
                features_i = None
                if self.colorize_input_parts_with_pca and features_batch is not None:
                    features_i = features_batch[i][:valid_n]  # (valid_n, F)
                
                for part_id in unique_part_ids:
                    part_id_int = int(part_id.item())
                    # Create mask for this part
                    part_mask = part_ids_i == part_id
                    
                    if part_mask.sum() == 0:
                        continue  # Skip if no points for this part
                    
                    # Extract points for this part
                    part_points = pts_i[part_mask]
                    
                    # For generations and GT: always use part ID colors
                    part_color_id = colors_i[part_mask][0:1]  # Get first color (all same)
                    part_colors_id = part_color_id.expand(part_points.shape[0], -1)
                    
                    # Always save visualization with uniform ID colors for input
                    self._save_sample_images(
                        points=part_points,
                        colors=part_colors_id,
                        sample_name=f"{sample_name}_input_part{part_id_int:02d}",
                        subfolder=subfolder_name,
                    )
                    
                    # Additionally save PCA-colored version if enabled
                    if self.colorize_input_parts_with_pca and features_i is not None:
                        # Use PCA-based coloring from features
                        part_features = features_i[part_mask]  # (num_points_in_part, F)
                        part_colors_input_pca = self._features_to_pca_colors(
                            part_features, 
                            pca_basis=self.input_features_pca_basis
                        )
                        self._save_sample_images(
                            points=part_points,
                            colors=part_colors_input_pca,
                            sample_name=f"{sample_name}_input_part{part_id_int:02d}_pca",
                            subfolder=subfolder_name,
                        )
                    
                    # # Also save GT for this part
                    # part_points_gt = pts_gt_i[part_mask]
                    # self._save_sample_images(
                    #     points=part_points_gt,
                    #     colors=part_colors_id,
                    #     sample_name=f"{sample_name}_gt_part{part_id_int:02d}",
                    #     subfolder=subfolder_name,
                    # )
                    
                    # Save each generation for this part
                    for n in range(K):
                        pointclouds_pred = pointclouds_pred_list[n]
                        pred_i = pointclouds_pred[i][:valid_n]
                        part_points_pred = pred_i[part_mask]
                        
                        # Always save visualization with uniform ID colors for generated part
                        self._save_sample_images(
                            points=part_points_pred,
                            colors=part_colors_id,
                            sample_name=f"{sample_name}_generation{n+1:02d}_part{part_id_int:02d}",
                            subfolder=subfolder_name,
                        )
                        
                        # Additionally save PCA-colored version if enabled
                        if self.colorize_generated_parts_with_pca and transformer_features_list is not None and n < len(transformer_features_list):
                            # Extract transformer features for this generation and sample
                            transformer_features_n = transformer_features_list[n]  # (TP, embed_dim)
                            
                            if "cu_seqlens_batch" in batch:
                                # Dynamic batching: split transformer features by batch
                                start_idx = cu_seqlens_batch[i]
                                end_idx = cu_seqlens_batch[i + 1]
                                transformer_features_i = transformer_features_n[start_idx:end_idx]  # (N_i, embed_dim)
                            else:
                                # Fixed batching: reshape and extract
                                transformer_features_i = transformer_features_n.view(B, -1, transformer_features_n.shape[-1])[i][:valid_n]  # (valid_n, embed_dim)
                            
                            # Extract features for this part
                            part_transformer_features = transformer_features_i[part_mask]  # (num_points_in_part, embed_dim)
                            
                            if part_transformer_features.shape[0] > 0:
                                # Use PCA-based coloring from transformer features
                                part_colors_gen_pca = self._features_to_pca_colors(
                                    part_transformer_features,
                                    pca_basis=self.transformer_features_pca_basis
                                )
                                self._save_sample_images(
                                    points=part_points_pred,
                                    colors=part_colors_gen_pca,
                                    sample_name=f"{sample_name}_generation{n+1:02d}_part{part_id_int:02d}_pca",
                                    subfolder=subfolder_name,
                                )



class OverlapVisualizationCallback(VisualizationCallback):
    """Visualization callback for overlap prediction models."""

    def __init__(
        self,
        save_pairwise_overlap: bool = False,
        save_pairwise_csv: bool = False,
        save_pairwise_pointclouds: bool = True,
        **kwargs
    ):
        """Initialize overlap visualization callback.

        Args:
            save_pairwise_overlap (bool): Whether to save pairwise overlap visualizations (histograms and heatmaps). Default: False.
            save_pairwise_csv (bool): Whether to save pairwise overlap data as CSV. Default: False.
            save_pairwise_pointclouds (bool): Whether to save pairwise overlap point cloud visualizations. Default: True.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(**kwargs)
        self.save_pairwise_overlap = save_pairwise_overlap
        self.save_pairwise_csv = save_pairwise_csv
        self.save_pairwise_pointclouds = save_pairwise_pointclouds

    def _save_pairwise_overlap_plot(
        self,
        overlap_probs: torch.Tensor,
        part_ids: torch.Tensor,
        points_per_part: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save pairwise overlap probability plot for each part combination.
        
        Args:
            overlap_probs: Overlap probabilities of shape (N, C) where C is number of parts.
            part_ids: Part IDs for each point of shape (N,).
            points_per_part: Number of points per part of shape (max_parts,).
            sample_name (str): Sample name for filename.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
            
        try:
            # Move all tensors to CPU to avoid device mismatch issues
            overlap_probs = overlap_probs.cpu()
            part_ids = part_ids.cpu()
            points_per_part = points_per_part.cpu()
            
            # Get valid parts (parts with non-zero point counts)
            valid_parts = points_per_part > 0
            valid_part_indices = torch.where(valid_parts)[0]
            num_valid_parts = len(valid_part_indices)
            
            if num_valid_parts < 2:
                logger.warning(f"Sample {sample_name} has less than 2 valid parts, skipping pairwise overlap plot")
                return
            
            # Calculate number of pairwise combinations
            num_combinations = num_valid_parts * (num_valid_parts - 1) // 2
            
            # Create subplot grid
            cols = min(3, num_combinations)  # Max 3 columns
            rows = (num_combinations + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            # Ensure axes is always a list
            if num_combinations == 1:
                axes = [axes]
            elif rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, list) else [axes]
            else:
                axes = axes.flatten()
            
            # Convert to list if it's a numpy array
            if hasattr(axes, 'flatten'):
                axes = axes.flatten().tolist()
            elif not isinstance(axes, list):
                axes = [axes]
            
            # Generate all pairwise combinations
            part_combinations = list(combinations(valid_part_indices.tolist(), 2))
            
            for idx, (part_i, part_j) in enumerate(part_combinations):
                ax = axes[idx]
                
                # Get points that belong to part_i
                part_i_mask = part_ids == part_i
                part_i_points = overlap_probs[part_i_mask]  # (N_i, C)
                
                if len(part_i_points) == 0:
                    continue
                
                # Get overlap probabilities for part_j from part_i points
                overlap_probs_i_to_j = part_i_points[:, part_j]  # (N_i,)
                
                # Create histogram of overlap probabilities
                ax.hist(overlap_probs_i_to_j.numpy(), bins=20, alpha=0.7, 
                       color='blue', edgecolor='black')
                ax.set_xlabel(f'Overlap Probability (Part {part_i} → Part {part_j})')
                ax.set_ylabel('Number of Points')
                ax.set_title(f'Part {part_i} → Part {part_j}\nMean: {overlap_probs_i_to_j.mean():.3f}')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis limits to [0, 1]
                ax.set_xlim(0, 1)
            
            # Hide unused subplots
            for idx in range(num_combinations, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            target_dir = self.vis_dir if subfolder is None else (self.vis_dir / str(subfolder))
            target_dir.mkdir(parents=True, exist_ok=True)
            plot_path = target_dir / f"{sample_name}_pairwise_overlap.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error saving pairwise overlap plot for sample {sample_name}: {e}")

    def _save_pairwise_overlap_heatmap(
        self,
        overlap_probs: torch.Tensor,
        part_ids: torch.Tensor,
        points_per_part: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save pairwise overlap probability heatmap for all part combinations.
        
        Args:
            overlap_probs: Overlap probabilities of shape (N, C) where C is number of parts.
            part_ids: Part IDs for each point of shape (N,).
            points_per_part: Number of points per part of shape (max_parts,).
            sample_name (str): Sample name for filename.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
        try:
            # Move all tensors to CPU to avoid device mismatch issues
            overlap_probs = overlap_probs.cpu()
            part_ids = part_ids.cpu()
            points_per_part = points_per_part.cpu()
            
            # Get valid parts (parts with non-zero point counts)
            valid_parts = points_per_part > 0
            valid_part_indices = torch.where(valid_parts)[0]
            num_valid_parts = len(valid_part_indices)
            
            if num_valid_parts < 2:
                logger.warning(f"Sample {sample_name} has less than 2 valid parts, skipping pairwise overlap heatmap")
                return
            
            # Create overlap matrix
            overlap_matrix = torch.zeros(num_valid_parts, num_valid_parts)
            
            # Calculate mean overlap probability for each pair
            for i, part_i in enumerate(valid_part_indices):
                for j, part_j in enumerate(valid_part_indices):
                    if i == j:
                        # Self-overlap is always 1.0 (diagonal)
                        overlap_matrix[i, j] = 1.0
                    else:
                        # Get points that belong to part_i
                        part_i_mask = part_ids == part_i
                        part_i_points = overlap_probs[part_i_mask]  # (N_i, C)
                        
                        if len(part_i_points) > 0:
                            # Get overlap probabilities for part_j from part_i points
                            overlap_probs_i_to_j = part_i_points[:, part_j]  # (N_i,)
                            overlap_matrix[i, j] = overlap_probs_i_to_j.mean()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Convert to numpy for plotting
            overlap_matrix_np = overlap_matrix.numpy()
            
            # Create heatmap
            im = ax.imshow(overlap_matrix_np, cmap='Blues', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Overlap Probability', rotation=270, labelpad=15)
            
            # Set labels
            ax.set_xlabel('Part ID')
            ax.set_ylabel('Part ID')
            ax.set_title(f'Pairwise Overlap Probability Matrix\n{sample_name}')
            
            # Set tick labels
            ax.set_xticks(range(num_valid_parts))
            ax.set_yticks(range(num_valid_parts))
            ax.set_xticklabels([f'Part {valid_part_indices[i]}' for i in range(num_valid_parts)])
            ax.set_yticklabels([f'Part {valid_part_indices[i]}' for i in range(num_valid_parts)])
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add text annotations
            for i in range(num_valid_parts):
                for j in range(num_valid_parts):
                    text = ax.text(j, i, f'{overlap_matrix_np[i, j]:.3f}',
                                 ha='center', va='center', fontsize=8,
                                 color='white' if overlap_matrix_np[i, j] > 0.5 else 'black')
            
            plt.tight_layout()
            
            # Save the plot
            target_dir = self.vis_dir if subfolder is None else (self.vis_dir / str(subfolder))
            target_dir.mkdir(parents=True, exist_ok=True)
            plot_path = target_dir / f"{sample_name}_pairwise_overlap_heatmap.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error saving pairwise overlap heatmap for sample {sample_name}: {e}")

    def _save_pairwise_overlap_csv(
        self,
        overlap_probs: torch.Tensor,
        part_ids: torch.Tensor,
        points_per_part: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save pairwise overlap probability data as CSV for further analysis.
        
        Args:
            overlap_probs: Overlap probabilities of shape (N, C) where C is number of parts.
            part_ids: Part IDs for each point of shape (N,).
            points_per_part: Number of points per part of shape (max_parts,).
            sample_name (str): Sample name for filename.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
            
        try:
            # Move all tensors to CPU to avoid device mismatch issues
            overlap_probs = overlap_probs.cpu()
            part_ids = part_ids.cpu()
            points_per_part = points_per_part.cpu()
            
            # Get valid parts (parts with non-zero point counts)
            valid_parts = points_per_part > 0
            valid_part_indices = torch.where(valid_parts)[0]
            num_valid_parts = len(valid_part_indices)
            
            if num_valid_parts < 2:
                logger.warning(f"Sample {sample_name} has less than 2 valid parts, skipping pairwise overlap CSV")
                return
            
            # Collect all pairwise overlap data
            pairwise_data = []
            
            for i, part_i in enumerate(valid_part_indices):
                for j, part_j in enumerate(valid_part_indices):
                    if i == j:
                        # Self-overlap is always 1.0
                        continue
                    
                    # Get points that belong to part_i
                    part_i_mask = part_ids == part_i
                    part_i_points = overlap_probs[part_i_mask]  # (N_i, C)
                    
                    if len(part_i_points) > 0:
                        # Get overlap probabilities for part_j from part_i points
                        overlap_probs_i_to_j = part_i_points[:, part_j]  # (N_i,)
                        
                        # Add each point's overlap probability to the data
                        for point_idx, prob in enumerate(overlap_probs_i_to_j):
                            pairwise_data.append({
                                'source_part': int(part_i),
                                'target_part': int(part_j),
                                'point_idx': point_idx,
                                'overlap_probability': float(prob),
                                'source_part_points': len(part_i_points)
                            })
            
            # Create DataFrame and save to CSV
            if pairwise_data:
                df = pd.DataFrame(pairwise_data)
                target_dir = self.vis_dir if subfolder is None else (self.vis_dir / str(subfolder))
                target_dir.mkdir(parents=True, exist_ok=True)
                csv_path = target_dir / f"{sample_name}_pairwise_overlap.csv"
                df.to_csv(csv_path, index=False)
                
                # Also save summary statistics
                summary_data = []
                for i, part_i in enumerate(valid_part_indices):
                    for j, part_j in enumerate(valid_part_indices):
                        if i == j:
                            continue
                        
                        part_data = df[(df['source_part'] == part_i) & (df['target_part'] == part_j)]
                        if len(part_data) > 0:
                            summary_data.append({
                                'source_part': int(part_i),
                                'target_part': int(part_j),
                                'mean_overlap': float(part_data['overlap_probability'].mean()),
                                'std_overlap': float(part_data['overlap_probability'].std()),
                                'min_overlap': float(part_data['overlap_probability'].min()),
                                'max_overlap': float(part_data['overlap_probability'].max()),
                                'num_points': len(part_data)
                            })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_csv_path = target_dir / f"{sample_name}_pairwise_overlap_summary.csv"
                    summary_df.to_csv(summary_csv_path, index=False)
            
        except Exception as e:
            logger.error(f"Error saving pairwise overlap CSV for sample {sample_name}: {e}")

    def _save_pairwise_overlap_pointclouds(
        self,
        overlap_probs: torch.Tensor,
        part_ids: torch.Tensor,
        points_per_part: torch.Tensor,
        pts_gt: torch.Tensor,
        sample_name: str,
        subfolder: Optional[str] = None,
    ):
        """Save point cloud visualizations for each pairwise overlap.
        
        For each pair of parts (A, B), creates a single visualization showing both parts:
        - Points of part A colored with gradient based on part A's color
        - Points of part B colored with gradient based on part B's color
        
        Args:
            overlap_probs: Overlap probabilities of shape (N, C) where C is number of parts.
            part_ids: Part IDs for each point of shape (N,).
            points_per_part: Number of points per part of shape (max_parts,).
            pts_gt: Ground truth point cloud of shape (N, 3).
            sample_name (str): Sample name for filename.
        """
        # Skip saving if renderer is set to "none"
        if self.renderer == "none":
            return
            
        try:
            # Move all tensors to CPU to avoid device mismatch issues
            overlap_probs = overlap_probs.cpu()
            part_ids = part_ids.cpu()
            points_per_part = points_per_part.cpu()
            pts_gt = pts_gt.cpu()
            
            # Get valid parts (parts with non-zero point counts)
            valid_parts = points_per_part > 0
            valid_part_indices = torch.where(valid_parts)[0]
            num_valid_parts = len(valid_part_indices)
            
            if num_valid_parts < 2:
                logger.warning(f"Sample {sample_name} has less than 2 valid parts, skipping pairwise overlap pointclouds")
                return
            
            # Generate all pairwise combinations
            part_combinations = list(combinations(valid_part_indices.tolist(), 2))
            
            for part_i, part_j in part_combinations:
                # Get points for both parts
                part_i_mask = part_ids == part_i
                part_j_mask = part_ids == part_j
                
                if part_i_mask.sum() > 0 and part_j_mask.sum() > 0:
                    # Get points for part_i
                    part_i_points = pts_gt[part_i_mask]  # (N_i, 3)
                    
                    # Get points for part_j
                    part_j_points = pts_gt[part_j_mask]  # (N_j, 3)
                    
                    # Combine points for both parts
                    combined_points = torch.cat([part_i_points, part_j_points], dim=0)  # (N_i + N_j, 3)
                    
                    # Get overlap probabilities for coloring
                    part_i_overlap_probs = overlap_probs[part_i_mask, part_j]  # (N_i,) - overlap from part_i to part_j
                    part_j_overlap_probs = overlap_probs[part_j_mask, part_i]  # (N_j,) - overlap from part_j to part_i
                    
                    # Use probs_to_colors with different colormaps for each part
                    part_i_colors = probs_to_colors(part_i_overlap_probs, colormap="matplotlib:Blues")  # Part A uses Blues
                    part_j_colors = probs_to_colors(part_j_overlap_probs, colormap="matplotlib:Reds")   # Part B uses Reds
                    
                    # Combine colors
                    combined_colors = torch.cat([part_i_colors, part_j_colors], dim=0)  # (N_i + N_j, 3)
                    
                    # Save combined visualization
                    self._save_sample_images(
                        points=combined_points,
                        colors=combined_colors,
                        sample_name=f"{sample_name}_pair{part_i}_pair{part_j}",
                        subfolder=subfolder,
                    )
            
        except Exception as e:
            logger.error(f"Error saving pairwise overlap pointclouds for sample {sample_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Save overlap visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        overlap_probs = outputs["overlap_probs"]                              # (total_points, C)
        points_per_part = batch["points_per_part"]                            # (bs, max_parts)
        B, _ = points_per_part.shape
        part_ids = ppp_to_ids(points_per_part)                                # (bs, N)
        
        # Handle dynamic batching - check if we have cu_seqlens_batch
        if "cu_seqlens_batch" in batch:
            # Dynamic batching format (TP, 3)
            cu_seqlens_batch = batch["cu_seqlens_batch"]                      # (B + 1,)
            pts_flat = batch["pointclouds"]                                   # (TP, 3)
            pts_gt_flat = batch["pointclouds_gt"]                            # (TP, 3)
            
            # Convert from dynamic batching to fixed batching
            pts_batches = []
            pts_gt_batches = []
            
            for b in range(B):
                start_idx = cu_seqlens_batch[b]
                end_idx = cu_seqlens_batch[b + 1]
                pts_batches.append(pts_flat[start_idx:end_idx])
                pts_gt_batches.append(pts_gt_flat[start_idx:end_idx])
            
            # Find max points for padding
            max_points = max(batch.shape[0] for batch in pts_batches)
            N = max_points
            
            # Create padded tensors
            device = pts_flat.device
            pts = torch.zeros(B, max_points, 3, device=device, dtype=pts_flat.dtype)
            pts_gt = torch.zeros(B, max_points, 3, device=device, dtype=pts_gt_flat.dtype)
            
            for b, (pts_batch, pts_gt_batch) in enumerate(zip(pts_batches, pts_gt_batches)):
                pts[b, :pts_batch.shape[0]] = pts_batch
                pts_gt[b, :pts_gt_batch.shape[0]] = pts_gt_batch
        else:
            # Fixed batching format (B, N, 3)
            pts = batch["pointclouds"].view(B, -1, 3)                         # (bs, N, 3)
            pts_gt = batch["pointclouds_gt"].reshape(B, -1, 3)                # (bs, N, 3)
            N = pts_gt.shape[1]
        
        # Reshape overlap_probs to (B, N, C)
        overlap_probs = overlap_probs.reshape(B, N, -1)

        # print(part_ids[0])
        # print("# Parts: ", max(part_ids[0]).item()+1)
        # # print(overlap_probs.shape)
        # print(overlap_probs[0])
        # check passed, make sense

        overlap_prob, _ = torch.max(overlap_probs, dim=2)                     # (bs, N)

        # Scale to original size
        if self.scale_to_original_size:
            scale = batch["scales"]                                           # (bs,)
            pts = pts * scale[:, None, None]                                  # (bs, N, 3)
            pts_gt = pts_gt * scale[:, None, None]

        for i in range(B):
            # Check if we've reached the maximum number of samples to visualize
            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break
                
            dataset_name = str(batch["dataset_name"][i])
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"
            
            # Save input image
            colors = part_ids_to_colors(
                part_ids[i], colormap="default", part_order="id"
            )
            
            # Save gt image
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_gt",
                subfolder=dataset_name,
            )
            
            # Save overlap visualization
            colors = probs_to_colors(overlap_prob[i], colormap="matplotlib:Blues")
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_overlap",
                subfolder=dataset_name,
            )
            
            # Save pairwise overlap plots
            if self.save_pairwise_overlap:
                self._save_pairwise_overlap_plot(
                    overlap_probs=overlap_probs[i],  # (N, C)
                    part_ids=part_ids[i],           # (N,)
                    points_per_part=points_per_part[i],  # (max_parts,)
                    sample_name=sample_name,
                    subfolder=dataset_name,
                )

            # Save pairwise overlap heatmap
            if self.save_pairwise_overlap:
                self._save_pairwise_overlap_heatmap(
                    overlap_probs=overlap_probs[i],  # (N, C)
                    part_ids=part_ids[i],           # (N,)
                    points_per_part=points_per_part[i],  # (max_parts,)
                    sample_name=sample_name,
                    subfolder=dataset_name,
                )

            # Save pairwise overlap CSV
            if self.save_pairwise_csv:
                self._save_pairwise_overlap_csv(
                    overlap_probs=overlap_probs[i],  # (N, C)
                    part_ids=part_ids[i],           # (N,)
                    points_per_part=points_per_part[i],  # (max_parts,)
                    sample_name=sample_name,
                    subfolder=dataset_name,
                )

            # Save pairwise overlap pointclouds
            if self.save_pairwise_pointclouds:
                self._save_pairwise_overlap_pointclouds(
                    overlap_probs=overlap_probs[i],  # (N, C)
                    part_ids=part_ids[i],           # (N,)
                    points_per_part=points_per_part[i],  # (max_parts,)
                    pts_gt=pts_gt[i],               # (N, 3)
                    sample_name=sample_name,
                    subfolder=dataset_name,
                )
