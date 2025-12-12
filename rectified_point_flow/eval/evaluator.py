import json
from pathlib import Path
from typing import Any, Dict

import torch
import lightning as L
try:
    import open3d as o3d
except ImportError:
    o3d = None
import numpy as np

from .metrics import compute_cd, compute_transform_errors, compute_transform_errors_direct, compute_correspondence_rmse, compute_approximate_transform_error, compute_rigidity_rmse, compute_ecdf
from ..utils.point_clouds import repeat_by_cu_seqlens, ppp_to_ids, split_parts
from ..utils.render import part_ids_to_colors

class Evaluator:
    """Evaluator for Rectified Point Flow model. """
    
    def __init__(self, model: L.LightningModule, save_pointcloud_parts: bool = False, save_merged_pointcloud_steps: bool = False, max_samples_per_batch: int = None, rmse_eval_on: bool = False, rmse_eval_on_transformed: bool = True, folder_suffix: str | None = None, save_json: bool = True):
        self.model = model
        self.save_pointcloud_parts = save_pointcloud_parts
        self.save_merged_pointcloud_steps = save_merged_pointcloud_steps
        self.max_samples_per_batch = max_samples_per_batch
        self.rmse_eval_on = rmse_eval_on
        self.rmse_eval_on_transformed = rmse_eval_on_transformed
        self.folder_suffix = folder_suffix
        self.save_json = save_json

    def _compute_metrics(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        rotations_pred: torch.Tensor | None = None,
        translations_pred: torch.Tensor | None = None, # still in the scaled space
    ):
        """Compute evaluation metrics."""
        pts = data["pointclouds"]                       # (TP, 3)
        pts_gt = data["pointclouds_gt"]                 # (TP, 3)
        points_per_part = data["points_per_part"]       # (B, P)
        anchor_parts = data["anchor_parts"]             # 
        anchor_indices = data["anchor_indices"]         # (TP, )
        scales = data["scales"]                         # (B,)
        rots_gt = data["rotations"]                     # (B, P, 3, 3)
        trans_gt = data["translations"]                 # (B, P, 3)
        cu_seqlens_batch = data["cu_seqlens_batch"]     # (B+1, )
        cu_seqlens_part = data["cu_seqlens_part"]       # 

        object_cd = compute_cd(pts_gt, pointclouds_pred, cu_seqlens_batch)
        
        object_cd_m = object_cd * scales

        metrics = {
            "chamfer_l2 (m)": object_cd_m,
            "object_chamfer": object_cd, # unit scale
        }

        if rotations_pred is not None and translations_pred is not None:

            rot_errors, trans_errors = compute_transform_errors(
                pts, pts_gt, rots_gt, trans_gt, rotations_pred, translations_pred, points_per_part, anchor_parts, None, scales, cu_seqlens_batch,
            )
            
            # Compute direct transform errors (without anchor normalization, all parts treated equally)
            rot_errors_direct, trans_errors_direct = compute_transform_errors_direct(
                rots_gt, trans_gt, rotations_pred, translations_pred, points_per_part, None, scales,
            )
            
            rot_recalls = self._recall_at_thresholds(rot_errors, [5, 10, 15])
            trans_recalls = self._recall_at_thresholds(trans_errors, [0.2, 0.3, 1.0, 2.0, 5.0])
            
            # Calculate combined recalls for commonly used threshold pairs
            combined_recalls = self._combined_recall_at_specific_pairs(
                rot_errors, trans_errors, 
                [(10, 0.2), (15, 0.3), (1, 0.3), (2, 0.3), (5, 2.0), (10, 5.0)]
            )

            chamfer_recalls = self._recall_at_thresholds(object_cd_m, [0.2])
            
            # Compute rigidity RMSE: compare generated point cloud with transformed input point cloud
            rigidity_rmse = compute_rigidity_rmse(
                pts, pointclouds_pred, rotations_pred, translations_pred, 
                points_per_part, cu_seqlens_batch, scales
            )
            
            # # Compute ECDF for rotation and translation errors
            # ecdf_r, mean_r, med_r, ecdf_t, mean_t, med_t = compute_ecdf(
            #     rot_errors_direct, trans_errors_direct,
            #     r_splits=[3, 5, 10, 30, 45],
            #     t_splits=[0.05, 0.1, 0.25, 0.5, 0.75]
            # )
            
            # Expand ECDF values and statistics to per-sample format (repeat same value for all samples)
            B = rot_errors.shape[0]
            device = rot_errors.device
            
            # print(rot_errors.shape)
            metrics.update({
                "average_rotation_error (deg)": rot_errors,
                "average_translation_error (m)": trans_errors,
                "recall_at_10deg_0.2m (nss)": combined_recalls[0],
                "recall_at_15deg_0.3m (indoor_bufferx)": combined_recalls[1],
                "recall_at_5deg_2m (outdoor_bufferx)": combined_recalls[4],
                "recall_at_10deg_5m (map)": combined_recalls[5],
                "recall_at_chamfer_0.2m": chamfer_recalls[0],
                "rigidity_rmse (m)": rigidity_rmse,
                # ECDF metrics (expanded to per-sample format)
                # "ecdf_rotation_at_3deg": torch.full((B,), float(ecdf_r[0]), device=device, dtype=torch.float32),
                # "ecdf_rotation_at_5deg": torch.full((B,), float(ecdf_r[1]), device=device, dtype=torch.float32),
                # "ecdf_rotation_at_10deg": torch.full((B,), float(ecdf_r[2]), device=device, dtype=torch.float32),
                # "ecdf_rotation_at_30deg": torch.full((B,), float(ecdf_r[3]), device=device, dtype=torch.float32),
                # "ecdf_rotation_at_45deg": torch.full((B,), float(ecdf_r[4]), device=device, dtype=torch.float32),
                # "mean_rotation_error (deg)": torch.full((B,), mean_r, device=device, dtype=torch.float32),
                # "median_rotation_error (deg)": torch.full((B,), med_r, device=device, dtype=torch.float32),
                # "ecdf_translation_at_0.05m": torch.full((B,), float(ecdf_t[0]), device=device, dtype=torch.float32),
                # "ecdf_translation_at_0.1m": torch.full((B,), float(ecdf_t[1]), device=device, dtype=torch.float32),
                # "ecdf_translation_at_0.25m": torch.full((B,), float(ecdf_t[2]), device=device, dtype=torch.float32),
                # "ecdf_translation_at_0.5m": torch.full((B,), float(ecdf_t[3]), device=device, dtype=torch.float32),
                # "ecdf_translation_at_0.75m": torch.full((B,), float(ecdf_t[4]), device=device, dtype=torch.float32),
                # "mean_translation_error (m)": torch.full((B,), mean_t, device=device, dtype=torch.float32),
                # "median_translation_error (m)": torch.full((B,), med_t, device=device, dtype=torch.float32),
            })

        # Compute correspondence RMSE if enabled and point cloud pairs are detected
        if self.rmse_eval_on and points_per_part.shape[1] == 2:
            # Check if rotations and translations are available (only needed when evaluating on transformed inputs)
            if self.rmse_eval_on_transformed and (rotations_pred is None or translations_pred is None):
                return metrics
            
            # Scale point clouds back to original scale (in meters)
            scale_per_point = repeat_by_cu_seqlens(scales, cu_seqlens_batch).view(-1, 1)
            pts_gt_scaled = pts_gt * scale_per_point
            pts_input_scaled = pts * scale_per_point
            pts_pred_scaled = pointclouds_pred * scale_per_point
            
            # Split scaled point clouds into parts (source=part 0, target=part 1)
            parts_gt_scaled = split_parts(pts_gt_scaled, points_per_part, cu_seqlens_batch)
            parts_input_scaled = split_parts(pts_input_scaled, points_per_part, cu_seqlens_batch)
            
            # Split predicted point clouds if not using transformed inputs
            if not self.rmse_eval_on_transformed:
                parts_pred_scaled = split_parts(pts_pred_scaled, points_per_part, cu_seqlens_batch)
            
            B = points_per_part.shape[0]
            device = pts_gt.device
            rmse_values = torch.zeros(B, device=device)
            correspondence_ratios = torch.zeros(B, device=device)
            transform_error_values = torch.zeros(B, device=device)
            
            # Identity covariance matrix for transform error calculation
            identity_covariance = np.eye(6, dtype=np.float32)
            
            for b in range(B):
                # Check if both parts have valid points
                if points_per_part[b, 0] > 0 and points_per_part[b, 1] > 0:
                    source_gt = parts_gt_scaled[b][0]  # Part 0 is source (scaled)
                    target_gt = parts_gt_scaled[b][1]  # Part 1 is target (scaled)
                    
                    # Compute GT relative rotation and translation: T_rel = T_target @ inv(T_source)
                    # R_rel = R_target @ R_source.T
                    # t_rel = R_target @ (-R_source.T @ t_source) + t_target = t_target - R_target @ R_source.T @ t_source
                    R_source_gt = rots_gt[b, 0]  # (3, 3) - keep on device
                    t_source_gt = trans_gt[b, 0] * scales[b]  # (3,) scaled to meters - keep on device
                    R_target_gt = rots_gt[b, 1]  # (3, 3) - keep on device
                    t_target_gt = trans_gt[b, 1] * scales[b]  # (3,) scaled to meters - keep on device
                    
                    R_rel_gt = R_target_gt @ R_source_gt.T
                    t_rel_gt = t_target_gt - R_rel_gt @ t_source_gt
                    
                    if self.rmse_eval_on_transformed:
                        # Get input point clouds for source and target parts
                        source_input = parts_input_scaled[b][0]  # Part 0 is source (scaled)
                        target_input = parts_input_scaled[b][1]  # Part 1 is target (scaled)
                        
                        # Get predicted rotations and translations for this batch
                        R_source = rotations_pred[b, 0]  # (3, 3) rotation for source part
                        t_source = translations_pred[b, 0] * scales[b]  # (3,) translation for source part (scale back to meters)
                        R_target = rotations_pred[b, 1]  # (3, 3) rotation for target part
                        t_target = translations_pred[b, 1] * scales[b]  # (3,) translation for target part (scale back to meters)
                        
                        # Apply transformations to input point clouds
                        # Transform: x' = x @ R.T + t
                        source_transformed = source_input @ R_source.T + t_source
                        target_transformed = target_input @ R_target.T + t_target
                        
                        # Compute RMSE for this pair using transformed input point clouds
                        rmse, _, correspondence_ratio = compute_correspondence_rmse(
                            source_gt, target_gt, source_transformed, target_transformed,
                            distance_threshold=0.05 # 5cm
                        )
                        
                        # Compute estimated relative rotation and translation from predicted transforms
                        # Same method as compute_transform_errors: compute relative transforms, then delta
                        R_rel_est = R_target @ R_source.T
                        t_rel_est = t_target - R_rel_est @ t_source
                        
                        # Compute rotation and translation errors using the same method as compute_transform_errors
                        # delta_R = R_rel_gt.T @ R_rel_est (relative rotation error)
                        # delta_t = t_rel_est - t_rel_gt (translation difference, already scaled)
                        delta_R = R_rel_gt.T @ R_rel_est
                        delta_t = t_rel_est - t_rel_gt
                    else:
                        # Use predicted point clouds directly without transformation
                        source_pred = parts_pred_scaled[b][0]  # Part 0 is source (scaled)
                        target_pred = parts_pred_scaled[b][1]  # Part 1 is target (scaled)
                        
                        # Compute RMSE for this pair using predicted point clouds directly
                        rmse, _, correspondence_ratio = compute_correspondence_rmse(
                            source_gt, target_gt, source_pred, target_pred,
                            distance_threshold=0.05 # 5cm
                        )
                        
                        # When rmse_eval_on_transformed=False, we don't have predicted transforms
                        # so we cannot compute transform errors - set to inf
                        delta_R = None
                        delta_t = None
                    
                    # Compute transform error only if we have delta transforms
                    if delta_R is not None and delta_t is not None:
                        # Convert to numpy for compute_approximate_transform_error
                        R_error = delta_R.cpu().numpy()
                        t_error = delta_t.cpu().numpy()
                        
                        # Compute transform error using identity covariance
                        transform_error = compute_approximate_transform_error(R_error, t_error, identity_covariance)
                        transform_error_values[b] = torch.tensor(np.sqrt(transform_error), device=device, dtype=torch.float32)  # Use sqrt as RMSE
                    else:
                        # Cannot compute transform error without predicted transforms
                        transform_error_values[b] = float('inf')
                    
                    rmse_values[b] = rmse
                    correspondence_ratios[b] = correspondence_ratio
                else:
                    rmse_values[b] = float('inf')
                    correspondence_ratios[b] = 0.0
                    transform_error_values[b] = float('inf')
            
            # Compute recall at RMSE 0.2m
            rmse_recalls = self._recall_at_thresholds(rmse_values, [0.2])
            transform_error_recalls = self._recall_at_thresholds(transform_error_values, [0.2])
            
            metrics.update({
                "correspondence_rmse (m)": rmse_values,
                "correspondence_ratio": correspondence_ratios,
                "recall_at_rmse_0.2m": rmse_recalls[0],
                "transform_error_rmse (m)": transform_error_values,
                "recall_at_transform_error_rmse_0.2m": transform_error_recalls[0],
            })

        return metrics
    
    @staticmethod
    def _recall_at_thresholds(metrics: torch.Tensor, thresholds: list[float]):
        """Compute metrics of shape (B,) at thresholds."""
        return [(metrics <= threshold).float() for threshold in thresholds]
    
    @staticmethod
    def _combined_recall_at_specific_pairs(
        rot_errors: torch.Tensor, 
        trans_errors: torch.Tensor, 
        threshold_pairs: list[tuple[float, float]]
    ):
        """Compute combined recall where both rotation and translation thresholds are met for specific pairs.
        
        Args:
            rot_errors: Rotation errors of shape (B,) in degrees
            trans_errors: Translation errors of shape (B,) in meters
            threshold_pairs: List of (rotation_threshold, translation_threshold) tuples
            
        Returns:
            List of combined recalls for each threshold pair
        """
        combined_recalls = []
        for rot_thresh, trans_thresh in threshold_pairs:
            # Both conditions must be satisfied
            rot_satisfied = rot_errors <= rot_thresh
            trans_satisfied = trans_errors <= trans_thresh
            combined_satisfied = rot_satisfied & trans_satisfied
            combined_recalls.append(combined_satisfied.float())
        return combined_recalls
    


    def _save_single_result(
        self,
        data: Dict[str, Any],
        metrics: Dict[str, torch.Tensor],
        pointclouds_pred: torch.Tensor,
        idx: int,
        generation_idx: int | str = 0,
        batch_idx: int = 0,
        rotations_pred: torch.Tensor = None,
        translations_pred: torch.Tensor = None,
        trajectory: torch.Tensor | None = None,
        original_trajectory: torch.Tensor | None = None,
    ):
        """Save a single evaluation result to JSON and PLY files.

        Args:
            data: Input data dictionary.
            metrics: Computed metrics dictionary.
            pointclouds_pred: Predicted point clouds tensor.
            idx: Index of the sample in the batch.
            generation_idx: Generation index for the result file name (int or str like "selected").
            batch_idx: Index within the batch for limiting point cloud saving.
            rotations_pred: Predicted rotation matrices (B, P, 3, 3).
            translations_pred: Predicted translation vectors (B, P, 3).
        """
        dataset_name = data["dataset_name"][idx]
        sample_idx = int(data['index'][idx])
        cu_seqlens_batch = data["cu_seqlens_batch"]

        # Handle special generation_idx values
        if isinstance(generation_idx, str):
            generation_idx_str = f"generation_{generation_idx}"
            generation_idx_num = -1  # Use -1 for special cases in JSON
        else:
            generation_idx_str = f"generation{generation_idx:02d}"
            generation_idx_num = generation_idx

        entry = {
            "name": data["name"][idx],
            "dataset": dataset_name,
            "num_parts": int(data["num_parts"][idx]),
            "generation_idx": generation_idx_num,
            "scales": float(data["scales"][idx]),
        }
        entry.update({k: float(v[idx]) for k, v in metrics.items()})

        # Get data path to preserve hierarchical structure
        data_path = data.get("data_path", data["name"][idx]) if "data_path" in data else data["name"][idx]
        if isinstance(data_path, (list, tuple)):
            data_path = data_path[idx]
        
        # Construct base folder name with optional suffix (matching visualizer format)
        dataset_name_for_folder = str(dataset_name)
        if self.folder_suffix is not None:
            dataset_name_for_folder = f"{dataset_name}_{self.folder_suffix}"
        
        # Create sample-specific subfolder in results directory, preserving input dataset hierarchy
        # Convert data_path to Path and create the same structure in results
        data_path_parts = Path(str(data_path)).parts
        sample_dir = Path(self.model.trainer.log_dir) / "results" / dataset_name_for_folder / Path(*data_path_parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON file (if enabled)
        if self.save_json:
            json_filepath = sample_dir / f"{dataset_name}_sample{sample_idx:05d}_{generation_idx_str}.json"
            json_filepath.write_text(json.dumps(entry, indent=2))
        
        # Save PLY files (only for first k samples per batch if limit is set)
        if self.max_samples_per_batch is None or batch_idx < self.max_samples_per_batch:
            self._save_pointclouds_as_ply(data, pointclouds_pred, idx, sample_dir, dataset_name, sample_idx, generation_idx_str, rotations_pred, translations_pred)
            # Create generation subfolder for merged point clouds if enabled
            if self.save_merged_pointcloud_steps:
                generation_dir = sample_dir / "generation"
                generation_dir.mkdir(parents=True, exist_ok=True)
                # Save merged input point cloud with colors if enabled
                self._save_merged_input_pointcloud(data, idx, generation_dir)
                # Save merged point clouds at each step if enabled
                if trajectory is not None:
                    endpoint_dir = generation_dir / "endpoint"
                    endpoint_dir.mkdir(parents=True, exist_ok=True)
                    self._save_merged_pointcloud_steps(data, trajectory, idx, endpoint_dir, trajectory_type="endpoint")
                # Save original trajectory (x_t) if available
                if original_trajectory is not None:
                    midpoint_dir = generation_dir / "midpoint"
                    midpoint_dir.mkdir(parents=True, exist_ok=True)
                    self._save_merged_pointcloud_steps(data, original_trajectory, idx, midpoint_dir, trajectory_type="midpoint")
        
        # Extract global transformations if available
        global_rotation = None
        global_translation = None
        if "global_rotation" in data:
            global_rotation = data["global_rotation"][idx].cpu() if isinstance(data["global_rotation"], torch.Tensor) else torch.tensor(data["global_rotation"][idx])
        if "global_translation" in data:
            global_translation = data["global_translation"][idx].cpu() if isinstance(data["global_translation"], torch.Tensor) else torch.tensor(data["global_translation"][idx])
        
        # Save transformation files for all samples (not limited by max_samples_per_batch)
        self._save_transformation_files(data, idx, sample_dir, dataset_name, sample_idx, generation_idx_str, rotations_pred, translations_pred,
                                        global_rotation, global_translation)

    def _save_transformation_files(
        self,
        data: Dict[str, Any],
        idx: int,
        sample_dir: Path,
        dataset_name: str,
        sample_idx: int,
        generation_idx: str | int = 0,
        rotations_pred: torch.Tensor = None,
        translations_pred: torch.Tensor = None,
        global_rotation: torch.Tensor = None,
        global_translation: torch.Tensor = None,
    ):
        """Save relative transformation files (predicted relative to GT) for all parts of a sample, and global transformations.
        
        Args:
            data: Input data dictionary.
            idx: Index of the sample in the batch.
            sample_dir: Directory to save transformation files.
            dataset_name: Name of the dataset.
            sample_idx: Sample index.
            generation_idx: Generation index (int or str like "selected").
            rotations_pred: Predicted rotation matrices (B, P, 3, 3).
            translations_pred: Predicted translation vectors (B, P, 3).
            global_rotation: Global rotation matrix (3, 3) applied to all parts during preprocessing.
            global_translation: Global translation vector (3,) in meters.
        
        Files saved:
            - {suffix}_part{pid}_transform.txt: Part-specific transformation matrices (4x4)
            - {suffix}_global_transform.txt: Global transformation matrix (4x4) if available
        """
        if rotations_pred is None or translations_pred is None:
            return
            
        try:
            # Move to CPU to avoid device mismatch issues
            rotations_pred_cpu = rotations_pred[idx].cpu()  # (P, 3, 3)
            translations_pred_cpu = translations_pred[idx].cpu()  # (P, 3)
            
            # Get GT transformations
            rotations_gt = data["rotations"][idx].cpu()  # (P, 3, 3)
            translations_gt = data["translations"][idx].cpu()  # (P, 3)
            points_per_part = data["points_per_part"][idx].cpu()  # (max_parts,)
            
            # Get valid parts (parts with non-zero point counts)
            valid_parts = points_per_part > 0
            valid_part_indices = torch.where(valid_parts)[0]
            
            # Handle generation_idx (can be int or already formatted string like "generation00" or "generation_selected")
            if isinstance(generation_idx, str):
                # Check if it already starts with "generation"
                if generation_idx.startswith("generation"):
                    suffix = generation_idx
                else:
                    suffix = f"generation_{generation_idx}"
            else:
                suffix = f"generation{generation_idx:02d}"
            
            # Get scale for this sample to convert translations from scaled space to meters
            scale = float(data["scales"][idx])
            
            for pid in valid_part_indices:
                pid_int = int(pid)
                if pid_int < len(rotations_pred_cpu) and pid_int < len(rotations_gt):
                    # Get predicted and GT transformations for this part
                    R_pred = rotations_pred_cpu[pid_int].numpy().astype(float)  # (3, 3)
                    t_pred = translations_pred_cpu[pid_int].numpy().astype(float)  # (3,) - in scaled space
                    R_gt = rotations_gt[pid_int].numpy().astype(float)  # (3, 3)
                    t_gt = translations_gt[pid_int].numpy().astype(float)  # (3,) - in scaled space
                    
                    # Scale translations back to meters (original scale)
                    t_pred_m = t_pred * scale  # (3,) - in meters
                    t_gt_m = t_gt * scale  # (3,) - in meters
                    
                    # Compute relative transformation: transforms from GT frame to predicted frame
                    # Given: pts_gt = pts @ R_gt^T + t_gt_m, pts_pred = pts @ R_pred^T + t_pred_m
                    # We want: pts_pred = pts_gt @ R_relative^T + t_relative
                    # Solving: R_relative^T = R_gt @ R_pred^T, so R_relative = R_pred @ R_gt^T
                    # t_relative = t_pred_m - t_gt_m @ R_gt @ R_pred^T = t_pred_m - t_gt_m @ R_relative^T
                    R_relative_T = R_gt @ R_pred.T
                    R_relative = R_relative_T.T  # R_relative = R_pred @ R_gt^T
                    t_relative = t_pred_m - t_gt_m @ R_relative_T  # (3,) - in meters
                    
                    # Construct 4x4 relative transformation matrix
                    transformation_matrix = np.eye(4, dtype=np.float32)
                    transformation_matrix[:3, :3] = R_relative
                    transformation_matrix[:3, 3] = t_relative

                    # Save global transformation if available (same for all parts)
                    if global_rotation is not None and global_translation is not None:
                        R_global = global_rotation.cpu().numpy().astype(float)  # (3, 3)
                        t_global_m = global_translation.cpu().numpy().astype(float)  # (3,) in meters
                        
                        # Construct 4x4 global transformation matrix
                        global_transformation_matrix = np.eye(4, dtype=np.float32)
                        global_transformation_matrix[:3, :3] = R_global
                        global_transformation_matrix[:3, 3] = t_global_m  # Use scaled translation in meters

                        transformation_matrix = transformation_matrix @ np.linalg.inv(global_transformation_matrix) #  (4, 4)
                        
                    # Save transformation matrix as txt file
                    transform_filename = f"{dataset_name}_sample{sample_idx:05d}_{suffix}_part{pid_int:02d}_transform.txt"
                    transform_path = sample_dir / transform_filename
                    
                    # Save with header for clarity
                    with open(transform_path, 'w') as f:
                        for row in transformation_matrix:
                            f.write(" ".join(f"{val:12.8f}" for val in row) + "\n")
                
                            
        except Exception as e:
            import logging
            logger = logging.getLogger("Evaluator")
            logger.error(f"Error saving transformation files for sample {sample_idx}: {e}")

    def _save_pointclouds_as_ply(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        idx: int,
        sample_dir: Path,
        dataset_name: str,
        sample_idx: int,
        generation_idx: str | int = 0,
        rotations_pred: torch.Tensor = None,
        translations_pred: torch.Tensor = None,
    ):
        """Save input, ground truth, and predicted point clouds as PLY files.
        
        Args:
            data: Input data dictionary.
            pointclouds_pred: Predicted point clouds tensor.
            idx: Index of the sample in the batch.
            sample_dir: Directory to save PLY files.
            dataset_name: Name of the dataset.
            sample_idx: Sample index.
            generation_idx: Generation index (int or str like "selected").
            rotations_pred: Predicted rotation matrices (B, P, 3, 3).
            translations_pred: Predicted translation vectors (B, P, 3).
        """
        try:
            # Get scale for rescaling
            scale = float(data["scales"][idx])
            
            # Handle dynamic vs fixed batching for input and GT point clouds
            if "cu_seqlens_batch" in data:
                # Dynamic batching
                cu_seqlens_batch = data["cu_seqlens_batch"]
                start_idx = cu_seqlens_batch[idx]
                end_idx = cu_seqlens_batch[idx + 1]
                
                pts_input = data["pointclouds"][start_idx:end_idx].detach().cpu()
                pts_gt = data["pointclouds_gt"][start_idx:end_idx].detach().cpu()
                pts_pred = pointclouds_pred[start_idx:end_idx].detach().cpu()
                
                # For dynamic batching, we need to get part info differently
                # Get the batch size to extract part information
                B = data["points_per_part"].shape[0]
                points_per_part = data["points_per_part"][idx]  # (max_parts,)
                valid_n = int(pts_input.shape[0])  # All points are valid in dynamic batching
                
            else:
                # Fixed batching - need to determine the valid number of points
                points_per_part = data["points_per_part"][idx]
                valid_n = int(points_per_part.sum().item())
                
                # Reshape from (B, N, 3) and take valid points
                B = data["pointclouds"].shape[0]
                N = data["pointclouds"].shape[1] // B if len(data["pointclouds"].shape) == 2 else data["pointclouds"].shape[1]
                
                pts_input = data["pointclouds"].view(B, N, 3)[idx][:valid_n].detach().cpu()
                pts_gt = data["pointclouds_gt"].view(B, N, 3)[idx][:valid_n].detach().cpu()
                pts_pred = pointclouds_pred.view(B, N, 3)[idx][:valid_n].detach().cpu()
            
            # Rescale to original size
            pts_input_scaled = pts_input * scale
            pts_gt_scaled = pts_gt * scale
            pts_pred_scaled = pts_pred * scale
            
            # Save individual part point clouds if enabled
            if self.save_pointcloud_parts:
                self._save_parts_pointclouds(
                    pts_input_scaled, None, points_per_part, sample_dir, dataset_name, sample_idx, "input", None, None, scale
                )
                self._save_parts_pointclouds(
                    pts_gt_scaled, None, points_per_part, sample_dir, dataset_name, sample_idx, "gt", None, None, scale
                )
                # Pass rotation and translation for predicted point clouds
                rots_idx = rotations_pred[idx] if rotations_pred is not None else None
                trans_idx = translations_pred[idx] if translations_pred is not None else None
                # generation_idx_str already includes "generation" prefix
                self._save_parts_pointclouds(
                    pts_pred_scaled, pts_gt_scaled, points_per_part, sample_dir, dataset_name, sample_idx, generation_idx, rots_idx, trans_idx, scale
                )
            
        except Exception as e:
            import logging
            logger = logging.getLogger("Evaluator")
            logger.error(f"Error saving PLY files for sample {sample_idx}: {e}")



    def _save_parts_pointclouds(
        self,
        points: torch.Tensor,
        target_points: torch.Tensor = None,
        points_per_part: torch.Tensor = None,
        sample_dir: Path = None,
        dataset_name: str = None,
        sample_idx: int = None,
        suffix: str = None,
        rotations: torch.Tensor = None,
        translations: torch.Tensor = None,
        scale: float = 1.0,
    ):
        """Save individual part point clouds with part-wise colors as PLY files and transformations.
        
        Args:
            points: Point cloud tensor of shape (N, 3).
            target_points: Target point cloud tensor of shape (N, 3). If provided, compute transformation.
            points_per_part: Number of points per part of shape (max_parts,).
            sample_dir: Directory to save PLY files.
            dataset_name: Name of the dataset.
            sample_idx: Sample index.
            suffix: Suffix for the filename (e.g., "input", "gt", "generation01").
            rotations: Predicted rotation matrices for this sample (P, 3, 3).
            translations: Predicted translation vectors for this sample (P, 3) - in scaled space.
            scale: Scale factor to convert translations from scaled space to meters. Default is 1.0.
        """
        try:
            # Ensure all tensors are on CPU
            points = points.cpu()
            points_per_part = points_per_part.cpu()
            if target_points is not None:
                target_points = target_points.cpu()
            
            # Generate part IDs from points_per_part
            part_ids = ppp_to_ids(points_per_part.unsqueeze(0))[0]  # Add batch dim and remove it
            part_ids = part_ids[:len(points)].cpu()  # Trim to actual number of points and ensure CPU
            
            # Get part-wise colors
            colors = part_ids_to_colors(part_ids, colormap="default", part_order="id").cpu()
            
            # Get unique parts
            unique_parts = torch.unique(part_ids)
            
            for pid in unique_parts.tolist():
                mask = part_ids == pid
                num_pts = int(mask.sum().item())
                if num_pts == 0:
                    continue
                
                if o3d is not None:
                    # Extract points and colors for this part
                    pts_part = points[mask].numpy().astype(float)
                    cols_part = colors[mask].numpy().astype(float)
                    
                    # Create point cloud with colors
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_part)
                    pcd.colors = o3d.utility.Vector3dVector(cols_part)
                    
                    # Save part PLY file directly in sample folder
                    part_filename = f"{dataset_name}_sample{sample_idx:05d}_{suffix}_part{int(pid):02d}.ply"
                    part_path = sample_dir / part_filename
                    o3d.io.write_point_cloud(str(part_path), pcd, write_ascii=False)
                    
                # Save transformation if rotations and translations are provided
                if rotations is not None and translations is not None and int(pid) < len(rotations):
                    # Get rotation and translation for this part
                    R = rotations[int(pid)].cpu().numpy().astype(float)  # (3, 3)
                    t = translations[int(pid)].cpu().numpy().astype(float)  # (3,) - in scaled space
                    
                    # Scale translation back to meters (original scale)
                    t_m = t * scale  # (3,) - in meters
                    
                    # Construct 4x4 transformation matrix
                    transformation_matrix = np.eye(4, dtype=np.float32)
                    transformation_matrix[:3, :3] = R
                    transformation_matrix[:3, 3] = t_m  # Use scaled translation in meters
                    
                    # Save transformation matrix as txt file
                    transform_filename = f"{dataset_name}_sample{sample_idx:05d}_{suffix}_part{int(pid):02d}_transform.txt"
                    transform_path = sample_dir / transform_filename
                    
                    # Save with header for clarity
                    with open(transform_path, 'w') as f:
                        for row in transformation_matrix:
                            f.write(" ".join(f"{val:12.8f}" for val in row) + "\n")
                
        except Exception as e:
            import logging
            logger = logging.getLogger("Evaluator")
            logger.error(f"Error saving part PLY files for sample {sample_idx}: {e}")

    def _save_merged_input_pointcloud(
        self,
        data: Dict[str, Any],
        idx: int,
        generation_dir: Path,
    ):
        """Save merged input point cloud with part-wise colors as input.pcd.
        
        Args:
            data: Input data dictionary.
            idx: Index of the sample in the batch.
            generation_dir: Generation directory to save PCD files.
        """
        try:
            if o3d is None:
                import logging
                logger = logging.getLogger("Evaluator")
                logger.warning("Open3D not available, skipping merged input point cloud saving")
                return
            
            # Handle dynamic vs fixed batching for input point clouds
            if "cu_seqlens_batch" in data:
                # Dynamic batching
                cu_seqlens_batch = data["cu_seqlens_batch"]
                start_idx = cu_seqlens_batch[idx]
                end_idx = cu_seqlens_batch[idx + 1]
                
                pts_input = data["pointclouds"][start_idx:end_idx].detach().cpu()
                num_points = end_idx - start_idx
            else:
                # Fixed batching
                points_per_part = data["points_per_part"][idx]
                num_points = int(points_per_part.sum().item())
                
                # Reshape from (B, N, 3) and take valid points
                B = data["pointclouds"].shape[0]
                N = data["pointclouds"].shape[1] // B if len(data["pointclouds"].shape) == 2 else data["pointclouds"].shape[1]
                
                pts_input = data["pointclouds"].view(B, N, 3)[idx][:num_points].detach().cpu()
            
            # Keep points in canonical frame (no scaling)
            
            # Get points_per_part for this sample to generate part IDs and colors
            points_per_part = data["points_per_part"][idx].cpu()
            
            # Generate part IDs from points_per_part
            part_ids = ppp_to_ids(points_per_part.unsqueeze(0))[0]  # Add batch dim and remove it
            part_ids = part_ids[:num_points].cpu()  # Trim to actual number of points
            
            # Get part-wise colors (same as used in _save_parts_pointclouds)
            colors = part_ids_to_colors(part_ids, colormap="default", part_order="id").cpu()
            
            # Create merged point cloud with part-wise colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_input.numpy().astype(float))
            pcd.colors = o3d.utility.Vector3dVector(colors.numpy().astype(float))
            
            # Save merged input point cloud as input.pcd
            input_path = generation_dir / "input.pcd"
            o3d.io.write_point_cloud(str(input_path), pcd, write_ascii=False)
                
        except Exception as e:
            import logging
            logger = logging.getLogger("Evaluator")
            logger.error(f"Error saving merged input point cloud: {e}")

    def _save_merged_pointcloud_steps(
        self,
        data: Dict[str, Any],
        trajectory: torch.Tensor,
        idx: int,
        output_dir: Path,
        trajectory_type: str = "endpoint",
    ):
        """Save merged point cloud at each step of the generation trajectory as step_0.pcd, step_1.pcd, etc.
        
        Args:
            data: Input data dictionary.
            trajectory: Trajectory tensor of shape (num_steps, TP, 3) containing point clouds at each step.
            idx: Index of the sample in the batch.
            output_dir: Directory to save PCD files (endpoint or midpoint subfolder).
            trajectory_type: Type of trajectory - "endpoint" for x_0_hat or "midpoint" for x_t.
        """
        try:
            if o3d is None:
                import logging
                logger = logging.getLogger("Evaluator")
                logger.warning("Open3D not available, skipping merged point cloud step saving")
                return
            
            # Handle dynamic vs fixed batching
            if "cu_seqlens_batch" in data:
                # Dynamic batching
                cu_seqlens_batch = data["cu_seqlens_batch"]
                start_idx = cu_seqlens_batch[idx]
                end_idx = cu_seqlens_batch[idx + 1]
                num_points = end_idx - start_idx
            else:
                # Fixed batching
                points_per_part = data["points_per_part"][idx]
                num_points = int(points_per_part.sum().item())
            
            # Get points_per_part for this sample to generate part IDs and colors
            points_per_part = data["points_per_part"][idx].cpu()
            
            # Generate part IDs from points_per_part
            part_ids = ppp_to_ids(points_per_part.unsqueeze(0))[0]  # Add batch dim and remove it
            part_ids = part_ids[:num_points].cpu()  # Trim to actual number of points
            
            # Get part-wise colors (same as used in _save_parts_pointclouds)
            colors = part_ids_to_colors(part_ids, colormap="default", part_order="id").cpu()
            
            # Extract trajectory steps for this sample
            # trajectory shape: (num_steps, TP, 3) for dynamic batching or (num_steps, B, N, 3) for fixed batching
            num_steps = trajectory.shape[0]
            
            for step_idx in range(num_steps):
                # Extract points for this sample at this step
                if "cu_seqlens_batch" in data:
                    # Dynamic batching: trajectory shape is (num_steps, TP, 3)
                    step_points = trajectory[step_idx, start_idx:end_idx].detach().cpu()
                else:
                    # Fixed batching: trajectory shape might be (num_steps, B*N, 3) or (num_steps, B, N, 3)
                    B = data["points_per_part"].shape[0]
                    if len(trajectory.shape) == 3:
                        # Shape is (num_steps, B*N, 3) - need to reshape
                        N = trajectory.shape[1] // B
                        step_points = trajectory[step_idx].view(B, N, 3)[idx][:num_points].detach().cpu()
                    else:
                        # Shape is already (num_steps, B, N, 3)
                        step_points = trajectory[step_idx, idx, :num_points].detach().cpu()
                
                # Keep points in canonical frame (no scaling)
                
                # Create merged point cloud with part-wise colors
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(step_points.numpy().astype(float))
                pcd.colors = o3d.utility.Vector3dVector(colors.numpy().astype(float))
                
                # Save merged point cloud as step_{step_idx}.pcd
                step_filename = f"step_{step_idx}.pcd"
                step_path = output_dir / step_filename
                o3d.io.write_point_cloud(str(step_path), pcd, write_ascii=False)
                
        except Exception as e:
            import logging
            logger = logging.getLogger("Evaluator")
            logger.error(f"Error saving merged point cloud steps: {e}")

    def run(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        rotations_pred: torch.Tensor | None = None,
        translations_pred: torch.Tensor | None = None,
        save_results: bool = False,
        generation_idx: int | str = 0,
        trajectory: torch.Tensor | None = None,
        original_trajectory: torch.Tensor | None = None,
    ):
        """Run evaluation and optionally save results.

        Args:
            data: Input data dictionary, containing:
                pointclouds_gt (B, N, 3): Ground truth point clouds.
                scales (B,): scales factors.
                points_per_part (B, P): Points per part.
                name (B,): Object names.
                dataset_name (B,): Dataset names.
                index (B,): Object indices.
                num_parts (B,): Number of parts.

            pointclouds_pred (B, N, 3) or (B*N, 3): Model output samples.
            rotations_pred (B, P, 3, 3), optional: Estimated rotation matrices.
            translations_pred (B, P, 3), optional: Estimated translation vectors. (here it's still in the scaled space?)
            save_results (bool): If True, save each result to log_dir/results.
            generation_idx (int | str): The index of the generation (mainly for best-of-n generations).
                Can be an int (0, 1, 2, ...) or a string like "selected" for special cases.
            trajectory (num_steps, TP, 3), optional: End-point trajectory tensor (x_0_hat) containing point clouds at each generation step.
                Used for saving merged point clouds at each step when save_merged_pointcloud_steps is enabled.
            original_trajectory (num_steps, TP, 3), optional: Original trajectory tensor (x_t) containing point clouds at each generation step.
                Used for saving merged point clouds at each step when save_merged_pointcloud_steps is enabled.

        Returns:
            A dictionary with:

                object_chamfer_dist (B,): Object Chamfer distance in meters.
                part_accuracy (B,): Part accuracy.

            If rotations_pred and translations_pred are provided, also return:

                rotation_error (B,): Rotation errors in degrees.
                translation_error (B,): Translation errors in meters.
                recall_at_5deg (B,): Recall at 5 degrees.
                recall_at_10deg (B,): Recall at 10 degrees.
                recall_at_15deg (B,): Recall at 15 degrees.
                recall_at_0.2m (B,): Recall at 0.2 meters.
                recall_at_0.3m (B,): Recall at 0.3 meters.
                recall_at_1m (B,): Recall at 1 meter.
                recall_at_2m (B,): Recall at 2 meters.
                recall_at_5m (B,): Recall at 5 meters.
                combined_recall_at_5deg_0.2m (B,): Combined recall at 5° and 0.2m.
                combined_recall_at_5deg_0.3m (B,): Combined recall at 5° and 0.3m.
                combined_recall_at_10deg_0.3m (B,): Combined recall at 10° and 0.3m.
                combined_recall_at_10deg_1m (B,): Combined recall at 10° and 1m.
                combined_recall_at_15deg_1m (B,): Combined recall at 15° and 1m.
                combined_recall_at_15deg_2m (B,): Combined recall at 15° and 2m.
        """
        metrics = self._compute_metrics(data, pointclouds_pred, rotations_pred, translations_pred)
        if save_results:
            B = data["points_per_part"].size(0)
            for i in range(B):
                self._save_single_result(data, metrics, pointclouds_pred, i, generation_idx, batch_idx=i, rotations_pred=rotations_pred, translations_pred=translations_pred, trajectory=trajectory, original_trajectory=original_trajectory)
        return metrics
