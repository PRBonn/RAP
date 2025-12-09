from __future__ import annotations

from functools import partial
from typing import Iterable, Callable

import torch


def _separate_params_by_dim(parameters: Iterable[torch.nn.Parameter]) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into matrix-like (ndim>=2) and vector-like (ndim<2)."""
    matrix_like: list[torch.nn.Parameter] = []
    vector_like: list[torch.nn.Parameter] = []
    for p in parameters:
        if not p.requires_grad:
            continue
        (matrix_like if p.ndim >= 2 else vector_like).append(p)
    return matrix_like, vector_like


def build_adamw(lr: float, weight_decay: float = 0.0, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, lr_muon: float = 0.0) -> Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]:
    """Return a factory that creates AdamW on provided parameters."""
    return partial(torch.optim.AdamW, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def build_muon_with_aux_adam(
    parameters: Iterable[torch.nn.Parameter],
    lr_muon: float = 0.02,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Construct MuonWithAuxAdam with two param groups from an iterable of parameters:
    - matrix-like parameters (ndim>=2) use Muon with lr_muon and weight decay
    - vector-like parameters (bias/gain etc.) use AdamW with lr_adam/betas/weight_decay
    Designed to be used with Hydra as a partial, e.g., self.optimizer(self.parameters()).
    """
    try:
        from muon import MuonWithAuxAdam  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Muon is not installed. Install with: pip install git+https://github.com/KellerJordan/Muon") from e

    matrix_like, vector_like = _separate_params_by_dim(parameters)
    param_groups = [
        dict(params=matrix_like, use_muon=True, lr=lr_muon, weight_decay=weight_decay*0.1), # according to Jianlin Su's blog, the rule of thumb is to set lr_muon as 10*lr_adam and set weight_decay_muon as 0.1 * weight_decay_adam
        dict(params=vector_like, use_muon=False, lr=lr, betas=betas, weight_decay=weight_decay),
    ]
    return MuonWithAuxAdam(param_groups)


