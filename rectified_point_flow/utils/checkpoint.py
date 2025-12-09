import logging
import random
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger("Utils")

# Checkpoint Loading

def load_checkpoint_for_module(
    module: torch.nn.Module,
    checkpoint_path: str,
    prefix_to_remove: str = None,
    keys_to_substitute: dict = None,
    prefix_to_add: str = None,
    strict: bool = False,
) -> dict:
    """Load checkpoint for a PyTorch module with prefix editing.
    
    Args:
        module: The PyTorch module to load the checkpoint into
        checkpoint_path: Path to the checkpoint file
        prefix_to_remove: Prefix to remove from checkpoint keys (e.g., "module.")
        keys_to_substitute: Prefix to substitute in checkpoint keys (e.g., {"encoder.": "feature_extractor."})
        prefix_to_add: Prefix to add to checkpoint keys (e.g., "encoder.")
        strict: Whether to strictly enforce that keys match
        
    Returns:
        Result dictionary from load_state_dict operation
    """        
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )["state_dict"]
    
    # Handle prefix removal
    if prefix_to_remove is not None:
        state_dict = {
            k.replace(prefix_to_remove, ""): v 
            for k, v in state_dict.items()
            if k.startswith(prefix_to_remove)
        }
    
    # Handle keys substitution
    if keys_to_substitute is not None:
        for old_prefix, new_prefix in keys_to_substitute.items():
            state_dict = {
                k.replace(old_prefix, new_prefix): v 
                for k, v in state_dict.items()
            }
    
    # Handle prefix addition
    if prefix_to_add is not None:
        state_dict = {f"{prefix_to_add}{k}": v for k, v in state_dict.items()}
    
    # Load the state dict
    load_result = module.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}. {load_result}")        
    return load_result


def download_rap_checkpoint(filename: str, local_dir: str) -> str:
    """Download RAP checkpoint from Hugging Face."""
    os.makedirs(local_dir, exist_ok=True)  # Ensure the directory exists
    repo_id = "PRBonn/RAP"
    ckpt_path = os.path.join(local_dir, filename)
    if not os.path.exists(ckpt_path):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    return ckpt_path

# RNG State Saving and Loading

def get_rng_state():
    """Get the RNG state."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }

def set_rng_state(state: dict):
    """Set the RNG state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states_num = len(torch.cuda.get_rng_state_all())
    if len(state["torch_cuda"]) != cuda_states_num:
        print(
            f"Warning: Expected {cuda_states_num} CUDA states, but found {len(state['torch_cuda'])} in checkpoint."
        )
        used_state_num = min(cuda_states_num, len(state["torch_cuda"]))
    else:
        used_state_num = cuda_states_num
    torch.cuda.set_rng_state_all(state["torch_cuda"][:used_state_num])
    print("RNG state restored from checkpoint.")