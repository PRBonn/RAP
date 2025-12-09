"""Sampling utilities for Rectified Point Flow."""

import torch
from typing import Callable
from functools import partial

from .procrustes import rigidify_prediction_with_procrustes

# Base sampler

def flow_sampler(
    step_fn: Callable,
    flow_model_fn: Callable,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    anchor_indices: torch.Tensor,
    num_steps: int = 20,
    points_per_part: torch.Tensor | None = None,
    cu_seqlens_batch: torch.Tensor | None = None,
    condition: torch.Tensor | None = None,
    return_trajectory: bool = False,
    rigidity_forcing: bool = False,
    return_end_point_trajectory: bool = True
):
    """Base sampler for rectified point flow.
    
    Args:
        step_fn: Integration step function that takes (x_t, t, dt, flow_model_fn, anchor_indices, x_0).
        flow_model_fn: Partial flow model function that takes (x, timesteps) and returns velocity.
        x_1: Initial noise (B, N, 3).
        x_0: Ground truth anchor points (B, N, 3).
        anchor_indices: Anchor point indices (B, N).
        num_steps: Number of integration steps, default 20.
        return_trajectory: Whether to return full trajectory, default False.
        rigidity_forcing: Whether to use rigidity guidance, default False. If True, 
           then, we will the x_0_hat := x_t - v * t, we fit the best rigid transformation from cond to x_0_hat, Rot(x_0_hat, cond) and Trans(x_0_hat, cond)
           then, we rebuild the x_t_rigid := (Rot(x_0_hat, cond) @ cond + Trans(x_0_hat, cond)) * (1 - t) + x_1 * t
        
    Returns:
        Final sampled points or trajectory
    """
    dt = 1.0 / num_steps
    x_t = x_1.clone()
    # x_t[anchor_indices] = x_0[anchor_indices] # anchor-free operation

    if return_trajectory:
        trajectory = torch.empty((num_steps, *x_1.shape), device=x_1.device)
        # Always store both trajectories when return_trajectory is True
        trajectory_x_t = torch.empty((num_steps, *x_1.shape), device=x_1.device)
    else:
        trajectory = None
        trajectory_x_t = None

    for step in range(num_steps):
        t = 1 - step * dt
        x_t, x_0_hat = step_fn(x_t, t, dt, flow_model_fn, anchor_indices, x_0)
        
        if rigidity_forcing:
            x_0_hat_rigidified = rigidify_prediction_with_procrustes(x_0_hat, condition, points_per_part, cu_seqlens_batch)
            x_t = x_0_hat_rigidified * (1 - t + dt) + x_1 * (t - dt)

        if return_trajectory:
            # Store both trajectories
            trajectory[step] = x_0_hat.detach().clone()  # end_point_trajectory
            trajectory_x_t[step] = x_t.detach().clone()  # original trajectory

    if return_trajectory:
        # Return both trajectories as a dictionary
        return {
            'end_point_trajectory': trajectory,
            'trajectory': trajectory_x_t,
        }
    else:
        return x_t.detach()


# Integration step functions

def euler_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
):
    """Euler integration step."""
    v = flow_model_fn(x_t, t)
    x_0_hat = x_t - v * t
    x_t = x_t - dt * v
    # x_t[anchor_indices] = x_0[anchor_indices] # anchor-free operation
    return x_t, x_0_hat

def rk2_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
):
    """RK2 (midpoint method) integration step."""
    # K1
    v1 = flow_model_fn(x_t, t)

    # K2
    x_mid = x_t - 0.5 * dt * v1
    x_mid[anchor_indices] = x_0[anchor_indices]
    t_next = max(0, t - 0.5 * dt)
    v2 = flow_model_fn(x_mid, t_next)

    # RK2 update
    x_t = x_t - dt * (v1 + v2) / 2
    x_t[anchor_indices] = x_0[anchor_indices]
    return x_t

def rk4_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
):
    """RK4 (4th order Runge-Kutta) integration step."""
    # K1
    v1 = flow_model_fn(x_t, t)
    
    # K2
    x_temp = x_t - dt * v1 / 2
    x_temp[anchor_indices] = x_0[anchor_indices]
    t_half = max(0, t - dt / 2)
    v2 = flow_model_fn(x_temp, t_half)
    
    # K3
    x_temp = x_t - dt * v2 / 2
    x_temp[anchor_indices] = x_0[anchor_indices]
    v3 = flow_model_fn(x_temp, t_half)
    
    # K4
    x_temp = x_t - dt * v3
    x_temp[anchor_indices] = x_0[anchor_indices]
    t_next = max(0, t - dt)
    v4 = flow_model_fn(x_temp, t_next)
    
    # RK4 update
    x_t = x_t - dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    x_t[anchor_indices] = x_0[anchor_indices]
    return x_t


# Sampler factory

def get_sampler(sampler_name: str):
    """Get sampler function by name.
    
    Args:
        sampler_name: Name of the sampler ('euler', 'rk2', 'rk4')
        
    Returns:
        Sampler function
    """
    step_fns = {
        'euler': euler_step,
        # 'rk2': rk2_step,
        # 'rk4': rk4_step,
    }
    if sampler_name not in step_fns:
        raise ValueError(f"Unknown sampler: {sampler_name}. Available: {list(step_fns.keys())}")
    
    return partial(flow_sampler, step_fns[sampler_name])
