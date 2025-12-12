"""Rectified Flow for Point Cloud Registration"""

import math
import time
from functools import partial
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval.evaluator import Evaluator
from .procrustes import fit_transformations
from .sampler import get_sampler
from .utils.checkpoint import get_rng_state, load_checkpoint_for_module, set_rng_state
from .utils.logging import MetricsMeter, log_metrics_on_step, log_metrics_on_epoch
from .utils.point_clouds import repeat_by_cu_seqlens


def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
    return time.time()


def compute_linearity(trajectory: torch.Tensor):
    """Compute the linearity of the trajectory of shape (FlowSteps, TP, 3).
    Return the linearity of each point in the trajectory (TP,).

    The linearity is defined as the ratio of the distance between the first and last point in the trajectory (straight line) to the total distance of the trajectory (actual segments).

    Args:
        trajectory: (FlowSteps, TP, 3)

    Returns:
        linearity: (TP,)
    """
    straight_line_distance = torch.norm(trajectory[0] - trajectory[-1], dim=-1)
    actual_distance = torch.zeros_like(straight_line_distance)
    for i in range(1, trajectory.shape[0]):
        segment_distance = torch.norm(trajectory[i] - trajectory[i-1], dim=-1)
        actual_distance += segment_distance
    return straight_line_distance / actual_distance

class RectifiedPointFlow(L.LightningModule):
    """Rectified Flow model for point cloud registration."""
    
    def __init__(
        self,
        feature_extractor: L.LightningModule | None = None,
        flow_model: nn.Module = None,
        optimizer: "partial[torch.optim.Optimizer]" = None,
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        encoder_ckpt: str = None,
        flow_model_ckpt: str = None,
        loss_type: str = "mse",
        timestep_sampling: str = "u-shaped",
        inference_sampling_steps: int = 20,
        inference_sampler: str = "euler",
        n_generations: int = 1,
        pred_proc_fn: Callable | None = None,
        save_results: bool = False,
        save_json: bool = True,
        save_pointcloud_parts: bool = False,
        save_merged_pointcloud_steps: bool = False,
        max_samples_per_batch: int = None,
        encoder_on: bool = True,
        encoder_freeze: bool = True,
        rigidity_forcing: bool = False,
        return_end_point_trajectory: bool = True,
        rmse_eval_on: bool = False,
        folder_suffix: str | None = None,
        use_average_rigidity_rmse: bool = True,
    ):
        if flow_model is None:
            raise ValueError("flow_model is required")
        if optimizer is None:
            raise ValueError("optimizer is required")
        if encoder_on and feature_extractor is None:
            raise ValueError("feature_extractor is required when encoder_on=True")
        super().__init__()
        self.feature_extractor = feature_extractor if encoder_on else None
        self.flow_model = flow_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_type = loss_type
        self.timestep_sampling = timestep_sampling
        self.inference_sampling_steps = inference_sampling_steps
        self.inference_sampler = inference_sampler
        self.n_generations = n_generations
        self.pred_proc_fn = pred_proc_fn
        self.save_results = save_results
        self.save_json = save_json
        self.save_pointcloud_parts = save_pointcloud_parts
        self.save_merged_pointcloud_steps = save_merged_pointcloud_steps
        self.max_samples_per_batch = max_samples_per_batch
        self.encoder_on = encoder_on
        self.encoder_freeze = encoder_freeze
        self.rigidity_forcing = rigidity_forcing
        self.return_end_point_trajectory = return_end_point_trajectory
        self.rmse_eval_on = rmse_eval_on
        self.folder_suffix = folder_suffix
        self.use_average_rigidity_rmse = use_average_rigidity_rmse

        # Load checkpoints
        if flow_model_ckpt is not None:
            load_checkpoint_for_module(
                self.flow_model,
                flow_model_ckpt,
                prefix_to_remove="flow_model.",
                strict=False,
            )

        # Initialize
        self.evaluator = Evaluator(self, save_pointcloud_parts=self.save_pointcloud_parts, save_merged_pointcloud_steps=self.save_merged_pointcloud_steps, max_samples_per_batch=self.max_samples_per_batch, rmse_eval_on=self.rmse_eval_on, folder_suffix=self.folder_suffix, save_json=self.save_json)
        self.meter = MetricsMeter(self)
        self.last_sample_counts = {}  # Store sample counts from last test epoch
        self.last_part_count_ranges = {}  # Store part count ranges from last test epoch

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()

    def _sample_timesteps(
        self,
        batch_size: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 2.0,
        a: float = 4.0,
        eps: float = 0.01,
    ):
        """Sample timesteps based on weighting scheme."""
        device = self.device
        if self.timestep_sampling == "u_shaped":
            u = torch.rand(batch_size, device=device) * 2 - 1
            u = torch.asinh(u * math.sinh(a)) / a
            u = (u + 1) / 2
        elif self.timestep_sampling == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
            u = torch.sigmoid(u)
        elif self.timestep_sampling == "mode":
            u = torch.rand(size=(batch_size,), device=device)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        elif self.timestep_sampling == "uniform":
            u = torch.rand(size=(batch_size,), device=device)
        # TODO: add more timestep sampling modes (beta, cosmap, etc.)
        # elif self.timestep_sampling == "beta":
        #     u = torch.rand(size=(batch_size,), device=device)
        #     u = u ** 0.5
        else:
            raise ValueError(f"Invalid timestep sampling mode: {self.timestep_sampling}")
        
        # Clamp small t to reduce loss spikes
        u = u.clamp(eps, 1.0)
        return u
    
    def _encode(self, x: torch.Tensor):
        """Encode point clouds using the feature extractor. (from PTV3)
        not used for now
        
        Args:
            x: (TP, 3) condition point clouds

        Returns:
            features: (TP, dim_feat) encoded point features
        """

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                # We can also implement feature extraction here, but it's not used for now
                # features = self.feature_extractor(x)
                latent_features = torch.zeros(x.shape[0], 64, device=x.device)
        return latent_features

    def _compute_flow_target(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple:
        """Compute the learning target of rectified flow.

        Args:
            x_0 (TP, 3): Ground truth point cloud
            x_1 (TP, 3): Noise point cloud
            t (TP, ): Timesteps

        Returns:
            x_t: Linear interpolation point cloud
            v_t: Velocity field
        """
        t = t.unsqueeze(-1)
        x_t = (1 - t) * x_0 + t * x_1     # interpolated point cloud
        v_t = x_1 - x_0                   # velocity field
        return x_t, v_t

    def _prepare_data(self, data_dict: dict):
        """Prepare data for training."""

        # TP = Total points
        # B = Batch size
        # P = Max parts per batch (64)
        # VP = Total number of valid parts

        x_0 = data_dict["pointclouds_gt"]                               # (TP, 3)
        cond = data_dict["pointclouds"]                                 # (TP, 3)
        local_features = data_dict["features"]                          # (TP, dim_feat)
        scales = data_dict["scales"]                                    # (TP, )
        anchor_indices = data_dict["anchor_indices"]                    # (TP, )
        points_per_part = data_dict["points_per_part"]                  # (B, P)

        B = scales.shape[0]

        valid_part = points_per_part > 0
        cu_seqlens_part = torch.cumsum(points_per_part[valid_part], 0)
        cu_seqlens_part = nn.functional.pad(cu_seqlens_part, (1, 0))   
        cu_seqlens_part = cu_seqlens_part.to(cond.device, dtype=torch.int32)           # (VP + 1, )
        cu_seqlens_batch = data_dict["cu_seqlens"].to(cond.device, dtype=torch.int32)  # (B + 1, )

        # Sample timesteps
        timesteps = self._sample_timesteps(batch_size=B)                # (B, )

        # print("anchor_indices: ", anchor_indices)

        return x_0, cond, local_features, scales, timesteps, anchor_indices, points_per_part, cu_seqlens_batch, cu_seqlens_part

    def forward(self, data_dict: dict):
        """Forward pass for training using rectified flow."""
        x_0, cond, local_features, scales, timesteps, anchor_indices, points_per_part, cu_seqlens_batch, cu_seqlens_part = self._prepare_data(data_dict)
        
        # Encode point clouds
        if self.encoder_on:
            latent_features = self._encode(cond)                               # (TP, dim_feat)
        else:
            latent_features = None

        t = repeat_by_cu_seqlens(timesteps, cu_seqlens_batch)
        x_1 = torch.randn_like(x_0)                                 # (TP, 3)
        x_t, v_t = self._compute_flow_target(x_0, x_1, t)           # (TP, 3) each 

        # Apply anchor part constraints # anchor-free operation
        # x_t[anchor_indices] = x_0[anchor_indices]
        # v_t[anchor_indices] = 0.0

        # Predict velocity field
        v_pred = self.flow_model(
            x=x_t,
            timesteps=timesteps,
            cond_coord=cond,
            local_features=local_features,
            latent_features=latent_features,
            scales=scales,
            anchor_indices=anchor_indices,
            cu_seqlens_batch=cu_seqlens_batch,
            cu_seqlens_part=cu_seqlens_part,
        )

        output_dict = {
            "t": timesteps,
            "v_pred": v_pred,
            "v_t": v_t,
            "x_0": x_0,
            "x_1": x_1,
            "x_t": x_t,
            "scales": scales,
            "cond": cond,
            "latent_features": latent_features,
            "cu_seqlens_batch": cu_seqlens_batch,
            "cu_seqlens_part": cu_seqlens_part,
            "points_per_part": points_per_part,
        }

        if self.pred_proc_fn is not None:
            output_dict = self.pred_proc_fn(output_dict)
        return output_dict


    def loss(self, output_dict: dict):
        """Compute rectified flow loss."""
        v_pred = output_dict["v_pred"]
        v_t = output_dict["v_t"]

        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "huber":
            loss = F.huber_loss(v_pred, v_t, reduction="mean")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return {
            "loss": loss,
            "norm_v_pred": v_pred.norm(dim=-1).mean(),
            "norm_v_t": v_t.norm(dim=-1).mean(),
        }

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Training step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        log_metrics_on_step(self, loss_dict, prefix="train")
        return loss_dict["loss"]

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Validation step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        pointclouds_pred = self.sample_rectified_flow(data_dict, output_dict["latent_features"])

        # Evaluate the final predicted point clouds
        # breakpoint()
        data_dict["scales"] = output_dict["scales"]
        data_dict["pointclouds"] = output_dict["cond"]
        data_dict["pointclouds_gt"] = output_dict["x_0"]
        data_dict["cu_seqlens_batch"] = output_dict["cu_seqlens_batch"]
        data_dict["cu_seqlens_part"] = output_dict["cu_seqlens_part"]
        data_dict["points_per_part"] = output_dict["points_per_part"]
        eval_results = self.evaluator.run(data_dict, pointclouds_pred)
        self.meter.add_metrics(
            dataset_names=data_dict["dataset_name"], 
            num_parts=data_dict["num_parts"],
            **eval_results
        )
        return loss_dict["loss"]

    def test_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Test step with support for multiple generations."""
        
        # Start timing for inference (excluding data loading)
        inference_start_time = get_time()

        output_dict = self.forward(data_dict)

        n_trajectories = []
        n_trajectories_x_t = []  # Store original trajectory (x_t) for each generation
        n_rotations_pred = []
        n_translations_pred = []
        n_eval_results = []
        n_transformer_features = []  # Store transformer features for each generation
        generation_times = []  # Track time for each generation

        points_per_part = data_dict["points_per_part"]

        for gen_idx in range(self.n_generations):
            # Start timing for this generation
            gen_start_time = get_time()
            
            # Always request transformer features during test (visualizer can decide whether to use them)
            sample_result = self.sample_rectified_flow(
                data_dict, 
                output_dict["latent_features"], 
                return_tarjectory=True,
                return_transformer_features=True,
            )
            # Handle both trajectory types: end_point_trajectory and trajectory (x_t)
            if isinstance(sample_result['trajectory'], dict):
                # Both trajectories are available
                trajs = sample_result['trajectory']['end_point_trajectory']  # Use end_point_trajectory as primary
                trajs_x_t = sample_result['trajectory']['trajectory']  # Original trajectory
            else:
                # Backward compatibility: single trajectory
                trajs = sample_result['trajectory']
                trajs_x_t = None
            transformer_features = sample_result['transformer_features']
            pointclouds_pred = trajs[-1]
            
            # Store both trajectories
            n_trajectories.append(trajs)
            if trajs_x_t is not None:
                n_trajectories_x_t.append(trajs_x_t)
            else:
                n_trajectories_x_t.append(None)  
            data_dict["scales"] = output_dict["scales"]
            data_dict["pointclouds"] = output_dict["cond"]
            data_dict["pointclouds_gt"] = output_dict["x_0"]
            data_dict["cu_seqlens_batch"] = output_dict["cu_seqlens_batch"]
            data_dict["cu_seqlens_part"] = output_dict["cu_seqlens_part"]
            data_dict["points_per_part"] = output_dict["points_per_part"]
            # data_dict["linearity"] = sample_result['linearity']
            
            # fit transformation is done here
            rotations_pred, translations_pred = fit_transformations(
                output_dict["cond"], pointclouds_pred, points_per_part, output_dict["cu_seqlens_batch"]
            )
            # here it's still in the scaled space

            eval_results = self.evaluator.run(
                data_dict, 
                pointclouds_pred, 
                rotations_pred, 
                translations_pred, 
                save_results=self.save_results, 
                generation_idx=gen_idx,
                trajectory=trajs if self.save_merged_pointcloud_steps else None,
                original_trajectory=trajs_x_t if self.save_merged_pointcloud_steps else None,
            )
            
            # Add metrics to meter for the first generation (to track sample counts)
            if gen_idx == 0:
                self.meter.add_metrics(
                    dataset_names=data_dict["dataset_name"], 
                    num_parts=data_dict["num_parts"],
                    **eval_results
                )
            
            # End timing for this generation
            gen_end_time = get_time()
            gen_time = gen_end_time - gen_start_time
            generation_times.append(gen_time)
            
            n_rotations_pred.append(rotations_pred)
            n_translations_pred.append(translations_pred)
            n_eval_results.append(eval_results)
            n_transformer_features.append(transformer_features)
        
        # End timing for inference
        inference_end_time = get_time()
        inference_time = inference_end_time - inference_start_time
        
        # Compute average metrics
        avg_results = {}
        for key in n_eval_results[0].keys():
            avg = sum(result[key] for result in n_eval_results) / len(n_eval_results)
            # Ensure scalar values for logging
            if isinstance(avg, torch.Tensor) and avg.numel() > 1:
                avg = avg.mean()
            avg_results[f'avg/{key}'] = avg
        self.log_dict(avg_results, prog_bar=False)

        # Compute best of N (BoN) metrics
        if self.n_generations > 1:
            best_results = {}
            for key in n_eval_results[0].keys():
                values = [result[key] for result in n_eval_results]
                
                # Stack values and compute best across generations for each batch element
                stacked_values = torch.stack(values)  # (n_generations, B) or (n_generations,)
                if ('acc' in key or 'recall' in key or 'success' in key or 'ecdf' in key): # you may need to change here for your new metrics
                    best_val = torch.max(stacked_values, dim=0)[0]  # Max across generations
                else: # like error, chamfer distance, etc.
                    best_val = torch.min(stacked_values, dim=0)[0]  # Min across generations
                
                # Ensure scalar values for logging
                if isinstance(best_val, torch.Tensor) and best_val.numel() > 1:
                    best_val = best_val.mean()
                best_results[f'best_of_{self.n_generations}/{key}'] = best_val
            self.log_dict(best_results, prog_bar=False)
        
        # Compute rigidity-selected metrics (select generation with smallest rigidity RMSE)
        rigidity_selected_results = {}
        if self.n_generations > 1 and 'rigidity_rmse (m)' in n_eval_results[0]:
            # Compute rigidity RMSE for selection
            # Note: use_average_rigidity_rmse requires return_end_point_trajectory=True
            # because we need x_0_hat (end points) not x_t (intermediate states) for rigidity RMSE
            if self.use_average_rigidity_rmse and self.return_end_point_trajectory:
                # Compute average rigidity RMSE over all trajectory steps for each generation
                from .eval.metrics import compute_rigidity_rmse
                
                rigidity_rmses = []
                for gen_idx in range(self.n_generations):
                    trajs = n_trajectories[gen_idx]  # (num_steps, TP, 3)
                    num_steps = trajs.shape[0]
                    step_rmses = []
                    
                    # Compute RMSE for each trajectory step
                    for step_idx in range(num_steps):
                        step_pointclouds = trajs[step_idx]  # (TP, 3)
                        
                        # Fit transformations for this step
                        step_rotations_pred, step_translations_pred = fit_transformations(
                            output_dict["cond"], step_pointclouds, points_per_part, output_dict["cu_seqlens_batch"]
                        )
                        
                        # Compute rigidity RMSE for this step
                        step_rigidity_rmse = compute_rigidity_rmse(
                            output_dict["cond"], step_pointclouds, step_rotations_pred, step_translations_pred,
                            points_per_part, output_dict["cu_seqlens_batch"], output_dict["scales"]
                        )
                        # print(f"step_rigidity_rmse: {step_rigidity_rmse} for generation {gen_idx} and step {step_idx}")
                        step_rmses.append(step_rigidity_rmse)
                    
                    # Average RMSE across all steps for this generation
                    if len(step_rmses) > 0:
                        avg_rigidity_rmse = torch.stack(step_rmses).mean(dim=0)  # (B,)
                    else:
                        # Fallback: use infinity if no steps (shouldn't happen in practice)
                        avg_rigidity_rmse = torch.full(
                            (points_per_part.shape[0],), 
                            float('inf'), 
                            device=output_dict["cond"].device
                        )
                    # print(f"avg_rigidity_rmse across steps: {avg_rigidity_rmse} for generation {gen_idx}")
                    rigidity_rmses.append(avg_rigidity_rmse)
                
                stacked_rigidity = torch.stack(rigidity_rmses)  # (n_generations, B)
                
                # Log average rigidity RMSE across all generations (for metric table)
                avg_rigidity_rmse_across_gens = stacked_rigidity.mean(dim=0)  # (B,)
                if isinstance(avg_rigidity_rmse_across_gens, torch.Tensor) and avg_rigidity_rmse_across_gens.numel() > 1:
                    avg_rigidity_rmse_scalar = avg_rigidity_rmse_across_gens.mean()
                else:
                    avg_rigidity_rmse_scalar = avg_rigidity_rmse_across_gens.item() if isinstance(avg_rigidity_rmse_across_gens, torch.Tensor) else avg_rigidity_rmse_across_gens
                
                # Log in avg/ section (average across all generations
                # Log avg_results again to include the new metric
                # self.log_dict(avg_results, prog_bar=False)
                # # Also log in rigidity_selected section for reference
                # rigidity_selected_results['rigidity_selected/avg_rigidity_rmse (m)'] = avg_rigidity_rmse_scalar
            else:
                # Use final step's rigidity RMSE (original behavior)
                rigidity_rmses = [result['rigidity_rmse (m)'] for result in n_eval_results]
                stacked_rigidity = torch.stack(rigidity_rmses)  # (n_generations, B)
            
            # Find best generation index for each batch element (smallest rigidity RMSE)
            best_gen_indices = torch.argmin(stacked_rigidity, dim=0)  # (B,)
            
            # Select metrics from the best generation for each batch element
            B = best_gen_indices.shape[0]
            for key in n_eval_results[0].keys():
                values = [result[key] for result in n_eval_results]
                stacked_values = torch.stack(values)  # (n_generations, B) or (n_generations,)
                
                if stacked_values.ndim == 1:
                    # Scalar per generation (shouldn't happen with per-batch metrics, but handle it)
                    # Use the best generation index from the first batch element
                    selected_val = stacked_values[best_gen_indices[0]]
                elif stacked_values.shape[1] == B:
                    # Per-batch values - select from best generation for each batch element
                    # Use advanced indexing: stacked_values[best_gen_indices, torch.arange(B)]
                    selected_val = stacked_values[best_gen_indices, torch.arange(B, device=stacked_values.device)]
                else:
                    # Unexpected shape, skip this metric
                    continue
                
                # Ensure scalar values for logging
                if isinstance(selected_val, torch.Tensor) and selected_val.numel() > 1:
                    selected_val = selected_val.mean()
                rigidity_selected_results[f'rigidity_selected/{key}'] = selected_val
            
            self.log_dict(rigidity_selected_results, prog_bar=False)
            
            # Save rigidity-selected generation results
            if self.save_results:
                B = best_gen_indices.shape[0]
                # Build selected pointclouds, rotations, and translations for the entire batch
                # Each batch element may have a different selected generation
                cu_seqlens_batch = output_dict["cu_seqlens_batch"]
                device = n_trajectories[0][-1].device
                
                # Collect selected pointclouds for each batch element
                selected_pointclouds_list = []
                for b in range(B):
                    best_gen_idx = int(best_gen_indices[b].item())
                    pointclouds_selected = n_trajectories[best_gen_idx][-1]  # Last trajectory step
                    if cu_seqlens_batch is not None:
                        start_idx = cu_seqlens_batch[b]
                        end_idx = cu_seqlens_batch[b + 1]
                        selected_pointclouds_list.append(pointclouds_selected[start_idx:end_idx])
                    else:
                        selected_pointclouds_list.append(pointclouds_selected[b])
                
                # Concatenate selected pointclouds
                if cu_seqlens_batch is not None:
                    pointclouds_selected_batch = torch.cat(selected_pointclouds_list, dim=0)
                else:
                    pointclouds_selected_batch = torch.stack(selected_pointclouds_list, dim=0)
                
                # Build selected rotations and translations
                # For each batch element, select from the best generation
                rotations_selected_batch = torch.zeros_like(n_rotations_pred[0])
                translations_selected_batch = torch.zeros_like(n_translations_pred[0])
                for b in range(B):
                    best_gen_idx = int(best_gen_indices[b].item())
                    rotations_selected_batch[b] = n_rotations_pred[best_gen_idx][b]
                    translations_selected_batch[b] = n_translations_pred[best_gen_idx][b]
                
                # Save the selected generation using the evaluator
                self.evaluator.run(
                    data_dict,
                    pointclouds_selected_batch,
                    rotations_selected_batch,
                    translations_selected_batch,
                    save_results=True,
                    generation_idx="selected",
                )

        # Compute overlap ratio selected metrics (select generation with largest overlap ratio)
        overlap_ratio_selected_results = {}
        if self.n_generations > 1 and 'overlap_ratio_at_1%' in n_eval_results[0]:
            # Extract overlap ratio for each generation
            overlap_ratios = [result['overlap_ratio_at_1%'] for result in n_eval_results]
            stacked_overlap_ratios = torch.stack(overlap_ratios)  # (n_generations, B)
            best_gen_indices = torch.argmax(stacked_overlap_ratios, dim=0)  # (B,)
            for key in n_eval_results[0].keys():
                values = [result[key] for result in n_eval_results]
                stacked_values = torch.stack(values)  # (n_generations, B) or (n_generations,)
                if stacked_values.ndim == 1:
                    # Scalar per generation (shouldn't happen with per-batch metrics, but handle it)
                    # Use the best generation index from the first batch element
                    selected_val = stacked_values[best_gen_indices[0]]
                elif stacked_values.shape[1] == B:
                    # Per-batch values - select from best generation for each batch element
                    # Use advanced indexing: stacked_values[best_gen_indices, torch.arange(B)]
                    selected_val = stacked_values[best_gen_indices, torch.arange(B, device=stacked_values.device)]
                else:
                    # Unexpected shape, skip this metric
                    continue
                # Ensure scalar values for logging
                if isinstance(selected_val, torch.Tensor) and selected_val.numel() > 1:
                    selected_val = selected_val.mean()
                overlap_ratio_selected_results[f'overlap_ratio_selected/{key}'] = selected_val
            self.log_dict(overlap_ratio_selected_results, prog_bar=False)
                
        return {
            'trajectories': n_trajectories,
            'trajectories_x_t': n_trajectories_x_t,  # Original trajectory (x_t) for each generation
            'rotations_pred': n_rotations_pred,
            'translations_pred': n_translations_pred,
            'eval_results': n_eval_results,  # List of dicts, one per generation
            'transformer_features': n_transformer_features,  # List of transformer features, one per generation
            'inference_time': inference_time,
            'generation_times': generation_times,
        }
    
    @torch.inference_mode()
    def sample_rectified_flow(
        self, 
        data_dict: dict,
        latent_features: torch.Tensor, 
        x_1: torch.Tensor | None = None,
        return_tarjectory: bool = False,
        return_transformer_features: bool = False,
    ) -> torch.Tensor | list[torch.Tensor] | dict:
        """Sample from rectified flow using configurable integration methods.
        
        Args:
            data_dict: Input data dictionary
            latent: Feature latent dictionary
            x_1: Optional initial noise. If None, generates random Gaussian noise.
            return_tarjectory: Whether to return the trajectory
            return_transformer_features: Whether to return transformer features from final step
            
        Returns:
            If return_tarjectory == False and return_transformer_features == False:
                (num_points, 3) final sampled points
            If return_tarjectory == True and return_transformer_features == False:
                (num_steps, num_points, 3) trajectory
            If return_transformer_features == True:
                Dict with 'trajectory' (or 'points') and 'transformer_features' (TP, embed_dim)
        """

        _, cond, local_features, scales, timesteps, anchor_indices, points_per_part, cu_seqlens_batch, cu_seqlens_part = self._prepare_data(data_dict)
        
        # x_0_anchored_parts = torch.zeros_like(x_0)
        # x_0_anchored_parts[anchor_indices] = x_0[anchor_indices]
        
        x_1 = torch.randn_like(cond) if x_1 is None else x_1

        # Track if we need to capture features on the last call
        transformer_features = None
        call_count = [0]  # Use list to allow modification in nested function
        total_steps = self.inference_sampling_steps
        last_call_captured = [False]  # Track if we've captured features

        def _flow_model_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            B = cu_seqlens_batch.shape[0] - 1
            timesteps_tensor = torch.full((B,), t, device=x.device)
            
            # Capture transformer features on the last call (when t is very close to 0 or on final step)
            nonlocal transformer_features
            is_last_call = (t < 1e-6) or (call_count[0] >= total_steps - 1)
            
            if return_transformer_features and is_last_call and not last_call_captured[0]:
                result = self.flow_model(
                    x=x,
                    timesteps=timesteps_tensor,
                    cond_coord=cond,
                    local_features=local_features,
                    latent_features=latent_features,
                    scales=scales,
                    anchor_indices=anchor_indices,
                    cu_seqlens_batch=cu_seqlens_batch,
                    cu_seqlens_part=cu_seqlens_part,
                    return_transformer_features=True,
                )
                transformer_features = result['transformer_features']
                last_call_captured[0] = True
                return result['velocity']
            else:
                call_count[0] += 1
                return self.flow_model(
                    x=x,
                    timesteps=timesteps_tensor,
                    cond_coord=cond,
                    local_features=local_features,
                    latent_features=latent_features,
                    scales=scales,
                    anchor_indices=anchor_indices,
                    cu_seqlens_batch=cu_seqlens_batch,
                    cu_seqlens_part=cu_seqlens_part,
                )
        
        result = get_sampler(self.inference_sampler)(
            flow_model_fn=_flow_model_fn,
            x_1=x_1,
            x_0=cond,
            condition=cond,
            points_per_part=points_per_part,
            cu_seqlens_batch=cu_seqlens_batch,
            anchor_indices=anchor_indices,
            num_steps=self.inference_sampling_steps,
            return_trajectory=True,
            rigidity_forcing=self.rigidity_forcing,
            return_end_point_trajectory=self.return_end_point_trajectory,
        )

        # compute the linearity of the trajectory
        # linearity = compute_linearity(result)

        if return_transformer_features:
            if return_tarjectory:
                # result is already a dict with 'end_point_trajectory' and 'trajectory' keys
                return {
                    'trajectory': result,
                    'transformer_features': transformer_features,
                    # 'linearity': linearity,
                }
            else:
                return {
                    'points': result[-1] if isinstance(result, dict) else result[-1],
                    'transformer_features': transformer_features,
                    # 'linearity': linearity,
                }
        return result

    def on_validation_epoch_end(self):
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="val")
        return metrics
    
    def on_test_epoch_end(self):
        # Capture sample counts and part count ranges before meter is reset
        self.last_sample_counts = self.meter.get_sample_counts()
        self.last_part_count_ranges = self.meter.get_part_count_ranges()
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="test")
        return metrics
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["rng_state"] = get_rng_state()
        return super().on_save_checkpoint(checkpoint)
    
    def on_load_checkpoint(self, checkpoint):
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        else:
            print("No RNG state found in checkpoint.")
        super().on_load_checkpoint(checkpoint)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
    

if __name__ == "__main__":
    # Test the model
    from .flow_model import PointCloudDiT

    lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.1)


    flow_model = PointCloudDiT(
        in_dim=64,
        out_dim=3,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
    )
    rectified_point_flow = RectifiedPointFlow(
        feature_extractor=None,
        flow_model=flow_model,
        optimizer=torch.optim.AdamW,
        lr_scheduler=lr_scheduler,
        inference_sampler="euler",  # Can be "euler", "rk2", or "rk4"
    )

    print(rectified_point_flow)
