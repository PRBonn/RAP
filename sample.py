"""Sampling from a trained Rectified Point Flow."""

import logging
from pathlib import Path
import os
import warnings
import time
import numpy as np

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module, download_rap_checkpoint, print_eval_table
from rectified_point_flow.visualizer import VisualizationCallback

logger = logging.getLogger("Sample")
warnings.filterwarnings("ignore", module="lightning")  # ignore warning from lightning' connectors
warnings.filterwarnings("ignore", category=FutureWarning)

# Optimize for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_CKPT_PATH_HF = "rap_model.ckpt"


def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
    return time.time()


def setup(cfg: DictConfig):
    """Setup evaluation components."""

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        ckpt_path = download_rap_checkpoint(DEFAULT_CKPT_PATH_HF, './weights')
    elif not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Please provide a valid checkpoint in the config or via ckpt_path='...' argument")
        exit(1)

    seed = cfg.get("seed", None)
    if seed is not None:
        L.seed_everything(seed, workers=True, verbose=False)
        logger.info(f"Seed set to {seed} for sampling")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()

    vis_config = cfg.get("visualizer", {})
    callbacks = []
    if vis_config:
        vis_callback: VisualizationCallback = hydra.utils.instantiate(vis_config)
        callbacks.append(vis_callback)
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
    )
    return model, datamodule, trainer


@hydra.main(version_base="1.3", config_path="./config", config_name="RAP_base")
def main(cfg: DictConfig):
    """Entry point for evaluating the model on validation set.
    
    Visualization Options:
    - To disable all visualization saving: visualizer.renderer=none
    - To limit visualization to first N samples per batch: visualizer.max_samples_per_batch=N
    - To combine both: visualizer.renderer=none visualizer.max_samples_per_batch=10
    
    Examples:
    - Evaluation only (no visualizations): python sample.py visualizer.renderer=none
    - First 5 samples only: python sample.py visualizer.max_samples_per_batch=5
    - First 10 samples with no saving: python sample.py visualizer.renderer=none visualizer.max_samples_per_batch=10
    """

    model, datamodule, trainer = setup(cfg)
    
    # Add timing callback to track inference times
    class TimingCallback(L.Callback):
        def __init__(self):
            self.inference_times = []
            self.generation_times = []
            self.batch_sizes = []  # Track batch sizes for per-sample calculations
            self.generation_batch_sizes = []  # Track batch sizes for each generation time
        
        def _get_batch_size(self, batch):
            """Get the batch size from the batch data."""
            if isinstance(batch, dict):
                # For dynamic batching, check if we have cu_seqlens_batch
                if "cu_seqlens_batch" in batch:
                    cu_seqlens_batch = batch["cu_seqlens_batch"]
                    batch_size = len(cu_seqlens_batch) - 1  # cu_seqlens_batch has shape (B + 1,)
                    return batch_size
                # For fixed batching, use points_per_part shape
                elif "points_per_part" in batch:
                    return batch["points_per_part"].shape[0]
                else:
                    # Fallback: try to get batch size from any tensor in the batch
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.ndim >= 1:
                            return value.shape[0]
            return 1  # Default fallback
            
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            # Record start time for this batch
            self.batch_start_time = get_time()
            
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            # Record end time and calculate inference time
            batch_end_time = get_time()
            inference_time = batch_end_time - self.batch_start_time
            self.inference_times.append(inference_time)
            
            # Get dataset name from dataloader index
            dataset_names = getattr(trainer.datamodule, 'dataset_names', [])
            dataset_name = dataset_names[dataloader_idx] if dataloader_idx < len(dataset_names) else f"dataset_{dataloader_idx}"
            
            # Get batch size for per-sample timing calculations
            batch_size = self._get_batch_size(batch)
            self.batch_sizes.append(batch_size)
            
            # Extract generation times and overlap ratios
            if outputs:
                # Handle generation times
                if 'generation_times' in outputs:
                    gen_times = outputs['generation_times']
                    self.generation_times.extend(gen_times)
                    # Track batch size for each generation time
                    self.generation_batch_sizes.extend([batch_size] * len(gen_times))
                    
                    # Calculate per-sample generation times
                    per_sample_gen_times = [t / batch_size for t in gen_times]
                    
                    # Print generation times for this batch
                    if len(gen_times) > 1:
                        avg_gen_time = np.mean(gen_times)
                        avg_per_sample_gen_time = np.mean(per_sample_gen_times)
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Generation times = {[f'{t:.4f}s' for t in gen_times]}")
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Per-sample generation times = {[f'{t:.4f}s' for t in per_sample_gen_times]}")
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Average generation time = {avg_gen_time:.4f}s")
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Average per-sample generation time = {avg_per_sample_gen_time:.4f}s")
                    else:
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Generation time = {gen_times[0]:.4f}s")
                        logger.info(f"Test Dataloader {dataloader_idx} ({dataset_name}) - Batch {batch_idx}: Per-sample generation time = {per_sample_gen_times[0]:.4f}s")
            
            # Calculate per-sample inference time (but don't log by default)
            per_sample_inference_time = inference_time / batch_size
            
        def on_test_end(self, trainer, pl_module):
            if self.inference_times and self.batch_sizes:
                # Calculate batch-level statistics
                avg_time = np.mean(self.inference_times)
                std_time = np.std(self.inference_times)
                total_time = sum(self.inference_times)
                
                # Calculate per-sample statistics
                per_sample_times = [inf_time / batch_size for inf_time, batch_size in zip(self.inference_times, self.batch_sizes)]
                avg_per_sample_time = np.mean(per_sample_times)
                std_per_sample_time = np.std(per_sample_times)
                total_samples = sum(self.batch_sizes)
                
                # Inference time summary removed by default
                # logger.info(f"=== INFERENCE TIME (include IO, visualization, etc.) SUMMARY ===")
                # logger.info(f"Average inference time per batch: {avg_time:.4f}s ± {std_time:.4f}s")
                # logger.info(f"Average inference time per sample: {avg_per_sample_time:.4f}s ± {std_per_sample_time:.4f}s")
                # logger.info(f"Total inference time: {total_time:.4f}s")
                # logger.info(f"Total samples processed: {total_samples}")
                
            if self.generation_times and self.generation_batch_sizes:
                # Calculate batch-level statistics
                avg_gen_time = np.mean(self.generation_times)
                std_gen_time = np.std(self.generation_times)
                total_gen_time = sum(self.generation_times)
                
                # Calculate per-sample statistics using the correct batch sizes for each generation
                per_sample_gen_times = [gen_time / batch_size for gen_time, batch_size in zip(self.generation_times, self.generation_batch_sizes)]
                avg_per_sample_gen_time = np.mean(per_sample_gen_times)
                std_per_sample_gen_time = np.std(per_sample_gen_times)
                
                logger.info(f"=== GENERATION TIME SUMMARY ===")
                logger.info(f"Average generation time per batch: {avg_gen_time:.4f}s ± {std_gen_time:.4f}s")
                logger.info(f"Average generation time per sample: {avg_per_sample_gen_time:.4f}s ± {std_per_sample_gen_time:.4f}s")
                logger.info(f"Total generation time: {total_gen_time:.4f}s")
    
    # Add timing callback to trainer
    timing_callback = TimingCallback()
    trainer.callbacks.append(timing_callback)
    
    eval_results = trainer.test(
        model=model,
        datamodule=datamodule, 
        verbose=False,
    )
    
    # Extract sample counts from model and convert to list in dataset order
    sample_counts = []
    part_count_ranges = []
    for dataset_name in datamodule.dataset_names:
        count = model.last_sample_counts.get(dataset_name, 0)
        sample_counts.append(count)
        
        part_range = model.last_part_count_ranges.get(dataset_name, (0, 0))
        part_count_ranges.append(part_range)
    
    print_eval_table(eval_results, datamodule.dataset_names, sample_counts=sample_counts, part_count_ranges=part_count_ranges)
    logger.info("Visualizations saved to:" + str(Path(cfg.get('log_dir')) / "visualizations"))
    logger.info("Evaluation results saved to:" + str(Path(cfg.get('log_dir')) / "results"))


if __name__ == "__main__":
    main()