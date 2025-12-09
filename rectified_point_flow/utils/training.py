"""Training utilities for Rectified Point Flow."""

import glob
import logging
import os
from typing import List

import hydra
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("Train")


def find_wandb_run_id(ckpt_path: str) -> str | None:
    """Find the latest wandb run ID from the checkpoint path."""
    ckpt_dir = os.path.dirname(ckpt_path)
    wandb_dir = os.path.join(ckpt_dir, "wandb")
    if os.path.exists(os.path.join(wandb_dir, "latest-run")):
        run_log_path = glob.glob(os.path.join(wandb_dir, "latest-run", "run-*.wandb"))[0]
        run_id = os.path.basename(run_log_path).split(".")[0].split("-")[-1]
        if len(run_id) == 8:
            return run_id
    return None


def setup_wandb_resume(cfg: DictConfig) -> None:
    """Setup wandb resume configuration if checkpoint exists."""
    if "wandb" in cfg.get("loggers", dict()).keys():
        run_id = find_wandb_run_id(cfg.get("ckpt_path"))
        if run_id:
            cfg.loggers.wandb.id = run_id
            cfg.loggers.wandb.resume = "allow"
            print(f"Found the latest wandb run ID: {run_id}. Continue logging to this run.")
        else:
            print("No previous wandb run ID found. Logging to a new run.")


def setup_loggers(cfg: DictConfig) -> List[Logger]:
    """Initialize and setup loggers."""
    loggers: List[Logger] = [
        hydra.utils.instantiate(logger)
        for logger in cfg.get("loggers", dict()).values()
    ]
    return loggers

@rank_zero_only
def log_code_to_wandb(loggers: List[Logger]) -> None:
    """Log code to wandb if available."""
    for log in loggers:
        if isinstance(log, WandbLogger):
            original_cwd = hydra.utils.get_original_cwd()
            log.experiment.log_code(
                root=original_cwd,
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml")
            )
            print(f"Codes logged to wandb from {original_cwd}")

@rank_zero_only
def log_config_to_wandb(loggers: List[Logger], cfg: DictConfig):
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        for log in loggers:
            if isinstance(log, WandbLogger):
                log.experiment.config.update(config_dict, allow_val_change=True)
    except Exception as e:
        logger.warning(f"Failed to update wandb config: {e}")