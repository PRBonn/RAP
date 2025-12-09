from collections import defaultdict
from itertools import chain
import re
from typing import Dict, List, Any

import lightning as L
import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table


def log_metrics_on_step(
    module: L.LightningModule,
    metrics: Dict[str, float],
    prefix: str = "train",
):
    """Log per‐step scalars only on rank 0."""
    for name, value in metrics.items():
        module.log(
            f"{prefix}/{name}",
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            rank_zero_only=True,
        )


def log_metrics_on_epoch(
    module: L.LightningModule,
    metrics: Dict[str, torch.Tensor],
    prefix: str = "val",
):
    """Log per‐epoch scalars, automatically synced & averaged across ranks.
    
    For single dataset: only log overall metrics to avoid redundancy.
    For multiple datasets: log both per-dataset and overall metrics.
    """
    # Separate dataset-specific and overall metrics
    dataset_metrics = {}
    overall_metrics = {}
    
    for name, value in metrics.items():
        if name.startswith("overall/"):
            overall_metrics[name.replace("overall/", "")] = value
        else:
            dataset_metrics[name] = value
    
    # Determine if we have multiple datasets
    dataset_names = set()
    for name in dataset_metrics.keys():
        if "/" in name:
            dataset_name = name.split("/")[0]
            dataset_names.add(dataset_name)
    
    # Always log overall metrics with "overall/" prefix to maintain consistency
    # This ensures ModelCheckpoint can always find "val/overall/object_chamfer"
    for name, value in overall_metrics.items():
        module.log(
            f"{prefix}/overall/{name}",
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            rank_zero_only=True,
        )
    
    # For multiple datasets, also log per-dataset metrics
    if len(dataset_names) > 1:
        for name, value in dataset_metrics.items():
            module.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                rank_zero_only=True,
            )

def print_eval_table(
    results: list[dict[str, float]],
    dataset_names: list[str],
    digits: int = 4,
    sample_counts: list[int] = None,
    part_count_ranges: list[tuple[int, int]] = None,
):
    """
    Pretty-print a list of evaluation-result dicts with Rich, split into Avg and BoN sections.
    For single dataset, use a more concise format.

    Args:
        results: List of dicts from trainer.test(), each dict corresponds to one dataloader.
        dataset_names: List of dataset names corresponding to each dataloader.
        digits: Number of decimal places for floats.
        sample_counts: Optional list of sample counts for each dataset.
        part_count_ranges: Optional list of (min, max) part count tuples for each dataset.
    """
    if not results:
        print("No results to display")
        return
    
    # PyTorch Lightning test() returns a list of dicts, one per dataloader
    # Each dict contains metrics like "avg/object_chamfer", "best_of_n/rotation_error", etc.
    
    # Collect all unique metric names across all dataloaders
    all_metrics = set()
    for result_dict in results:
        all_metrics.update(result_dict.keys())
    
    # Split into avg and best-of-n metrics
    avg_metrics = sorted(m for m in all_metrics if m.startswith("avg/"))
    bon_metrics = sorted(m for m in all_metrics if re.match(r"best_of_\w+/", m))
    rigidity_selected_metrics = sorted(m for m in all_metrics if m.startswith("rigidity_selected/"))
    
    fmt = f"{{:.{digits}f}}"
    
    # For single dataset, use a more concise format
    if len(results) == 1 and len(dataset_names) >= 1:
        dataset_name = dataset_names[0]
        result_dict = results[0]
        
        table = Table()
        table.add_column("Metric", style="bold magenta", justify="left", no_wrap=True)
        table.add_column(dataset_name, style="cyan", justify="right")
        
        # Add sample count row if provided
        if sample_counts and len(sample_counts) >= 1:
            table.add_row("Sample Count", str(sample_counts[0]))
        
        # Add part count range row if provided
        if part_count_ranges and len(part_count_ranges) >= 1:
            min_parts, max_parts = part_count_ranges[0]
            if min_parts == max_parts:
                table.add_row("Part Count", str(min_parts))
            else:
                table.add_row("Part Count Range", f"[{min_parts}, {max_parts}]")
        
        # Add section separator if we added any info rows
        if (sample_counts and len(sample_counts) >= 1) or (part_count_ranges and len(part_count_ranges) >= 1):
            table.add_section()
        
        # Add average metrics
        for metric in avg_metrics:
            val = result_dict.get(metric)
            if val is not None:
                display_name = metric.replace("avg/", "")
                if isinstance(val, (int, float)):
                    table.add_row(display_name, fmt.format(float(val)))
                else:
                    table.add_row(display_name, str(val))
        
        # Add section separator if both avg and bon metrics exist
        if avg_metrics and bon_metrics:
            table.add_section()
        
        # Add best-of-N metrics
        for metric in bon_metrics:
            val = result_dict.get(metric)
            if val is not None:
                # Extract the actual n value from best_of_n/ or best_of_<number>/
                match = re.match(r"best_of_(\d+)/", metric)
                if match:
                    n_value = match.group(1)
                    display_name = metric.replace(f"best_of_{n_value}/", f"best_{n_value}_")
                else:
                    # Fallback to original behavior if no number found
                    display_name = metric.replace("best_of_n/", "best_")
                if isinstance(val, (int, float)):
                    table.add_row(display_name, fmt.format(float(val)))
                else:
                    table.add_row(display_name, str(val))
        
        # Add section separator if rigidity-selected metrics exist
        if rigidity_selected_metrics:
            table.add_section()
        
        # Add rigidity-selected metrics
        for metric in rigidity_selected_metrics:
            val = result_dict.get(metric)
            if val is not None:
                display_name = metric.replace("rigidity_selected/", "selected_")
                if isinstance(val, (int, float)):
                    table.add_row(display_name, fmt.format(float(val)))
                else:
                    table.add_row(display_name, str(val))
    else:
        # Multiple datasets:
        table = Table()
        table.add_column("Metrics", style="bold magenta", justify="left", no_wrap=True)
        
        # Add columns for each dataset
        for dataset_name in dataset_names:
            table.add_column(dataset_name, style="cyan")

        # Add sample count row if provided
        if sample_counts and len(sample_counts) == len(dataset_names):
            count_row = ["Sample Count"]
            for count in sample_counts:
                count_row.append(str(count))
            table.add_row(*count_row)
        
        # Add part count range row if provided
        if part_count_ranges and len(part_count_ranges) == len(dataset_names):
            parts_row = ["Part Count Range"]
            for min_parts, max_parts in part_count_ranges:
                if min_parts == max_parts:
                    parts_row.append(str(min_parts))
                else:
                    parts_row.append(f"[{min_parts}, {max_parts}]")
            table.add_row(*parts_row)
        
        # Add section separator if we added any info rows
        if ((sample_counts and len(sample_counts) == len(dataset_names)) or 
            (part_count_ranges and len(part_count_ranges) == len(dataset_names))):
            table.add_section()

        # Collect all unique base metric names (e.g., "avg/object_chamfer")
        # and their original full names with dataloader index suffix
        base_to_full_metrics = {}
        for i, result_dict in enumerate(results):
            dataloader_idx_suffix = f"/dataloader_idx_{i}" if len(results) > 1 else ""
            for full_metric_key in result_dict.keys():
                if full_metric_key.endswith(dataloader_idx_suffix):
                    base_metric_key = full_metric_key[:-len(dataloader_idx_suffix)]
                else:
                    base_metric_key = full_metric_key
                
                if base_metric_key not in base_to_full_metrics:
                    base_to_full_metrics[base_metric_key] = {}
                base_to_full_metrics[base_metric_key][i] = full_metric_key
        
        # Split into avg and best-of-n base metrics
        base_avg_metrics = sorted([m for m in base_to_full_metrics if m.startswith("avg/")])
        base_bon_metrics = sorted([m for m in base_to_full_metrics if re.match(r"best_of_\w+/", m)])
        base_rigidity_selected_metrics = sorted([m for m in base_to_full_metrics if m.startswith("rigidity_selected/")])
        
        # Add average metrics
        for base_metric in base_avg_metrics:
            row = [base_metric.replace("avg/", "")]
            for i, dataset_name in enumerate(dataset_names):
                full_metric_key = base_to_full_metrics[base_metric].get(i)
                if full_metric_key and i < len(results):
                    val = results[i].get(full_metric_key)
                    if val is None:
                        row.append("-")
                    elif isinstance(val, (int, float)):
                        row.append(fmt.format(float(val)))
                    else:
                        row.append(str(val))
                else:
                    row.append("-")
            table.add_row(*row)

        if base_avg_metrics and base_bon_metrics:
            table.add_section()

        # Add best-of-N metrics
        for base_metric in base_bon_metrics:
            # Extract the actual n value from best_of_n/ or best_of_<number>/
            match = re.match(r"best_of_(\d+)/", base_metric)
            if match:
                n_value = match.group(1)
                display_name = base_metric.replace(f"best_of_{n_value}/", f"best_{n_value}_")
            else:
                # Fallback to original behavior if no number found
                display_name = base_metric.replace("best_of_n/", "best_")
            row = [display_name]
            for i, dataset_name in enumerate(dataset_names):
                full_metric_key = base_to_full_metrics[base_metric].get(i)
                if full_metric_key and i < len(results):
                    val = results[i].get(full_metric_key)
                    if val is None:
                        row.append("-")
                    elif isinstance(val, (int, float)):
                        row.append(fmt.format(float(val)))
                    else:
                        row.append(str(val))
                else:
                    row.append("-")
            table.add_row(*row)

        if base_bon_metrics and base_rigidity_selected_metrics:
            table.add_section()

        # Add rigidity-selected metrics
        for base_metric in base_rigidity_selected_metrics:
            row = [base_metric.replace("rigidity_selected/", "selected_")]
            for i, dataset_name in enumerate(dataset_names):
                full_metric_key = base_to_full_metrics[base_metric].get(i)
                if full_metric_key and i < len(results):
                    val = results[i].get(full_metric_key)
                    if val is None:
                        row.append("-")
                    elif isinstance(val, (int, float)):
                        row.append(fmt.format(float(val)))
                    else:
                        row.append(str(val))
                else:
                    row.append("-")
            table.add_row(*row)
    console = Console()
    console.print(table)


class MetricsMeter:
    """Helper class for accumulating metrics for each dataset.

    Example:
        >>> metrics_meter = MetricsMeter(module)
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "A"], 
                loss=torch.tensor([0.1, 0.2, 0.3]),
                acc=torch.tensor([0.9, 0.8, 0.7]),
            )
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "C"], 
                loss=torch.tensor([0.4, 0.5, 0.6]),
                acc=torch.tensor([0.6, 0.5, 0.4]),
            )
        >>> results = metrics_meter.log_on_epoch_end()
        >>> print(results)
        {
            "A/loss": 0.2667,
            "A/acc": 0.7333,
            "B/loss": 0.35,
            "B/acc": 0.65,
            "C/loss": 0.6,
            "C/acc": 0.4,
            "overall/loss": 0.35,
            "overall/acc": 0.65,
        }
    """

    def __init__(self, module: L.LightningModule):
        self.module = module
        self.reset()

    def reset(self):
        self._sums = defaultdict(lambda: defaultdict(float))
        self._counts = defaultdict(lambda: defaultdict(int))
        self._metrics_seen = set()
        self._part_counts = defaultdict(lambda: defaultdict(list))  # Track part counts per dataset

    def add_metrics(self, dataset_names: List[str], num_parts: torch.Tensor = None, **metrics: torch.Tensor):
        """Accumulate a batch of per-sample metrics."""
        if not metrics:
            return
        
        if any(ds == "overall" for ds in dataset_names):
            raise ValueError("'overall' is a reserved dataset name and cannot be used.")

        B = next(iter(metrics.values())).shape[0]
        if len(dataset_names) != B:
            raise ValueError(f"len(dataset_names)={len(dataset_names)} != batch size {B}")
        for k, t in metrics.items():
            if t.shape[0] != B:
                raise ValueError(f"metric '{k}' has shape {t.shape} != ({B},)")
            self._metrics_seen.add(k)

        for i, ds in enumerate(dataset_names):
            for k, t in metrics.items():
                v = t[i].item()
                self._sums[k][ds] += v
                self._counts[k][ds] += 1
                self._sums[k]["_overall"] += v
                self._counts[k]["_overall"] += 1
            
            # Track part counts if provided
            if num_parts is not None:
                try:
                    if isinstance(num_parts, torch.Tensor):
                        part_count = int(num_parts[i].item())
                    else:
                        # Handle case where num_parts is a list or single value
                        if isinstance(num_parts, (list, tuple)):
                            part_count = int(num_parts[i])
                        else:
                            part_count = int(num_parts)
                    
                    self._part_counts[ds]["parts"].append(part_count)
                    self._part_counts["_overall"]["parts"].append(part_count)
                except Exception as e:
                    print(f"Error processing num_parts: {e}")
                    print(f"num_parts type: {type(num_parts)}, value: {num_parts}")
                    print(f"dataset: {ds}, sample index: {i}")

    def compute_average(self) -> Dict[str, torch.Tensor]:
        """Gather per-dataset sums/counts, and compute global averages."""
        # local dataset list
        local_ds = sorted(
            set(chain.from_iterable(self._counts[k].keys() for k in self._metrics_seen))
            - {"_overall"}
        )

        # gather global dataset list
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_ds)
            global_ds = sorted(set(chain.from_iterable(gathered)))
        else:
            global_ds = local_ds

        # flatten sums and counts with fixed order
        metrics = sorted(self._metrics_seen)
        N = len(global_ds) + 1
        flat_sums = []
        flat_counts = []
        for k in metrics:
            for ds in global_ds:
                flat_sums.append(self._sums[k].get(ds, 0.0))
                flat_counts.append(self._counts[k].get(ds, 0))
            flat_sums.append(self._sums[k].get("_overall", 0.0))
            flat_counts.append(self._counts[k].get("_overall", 0))

        device = getattr(self.module, "device", torch.device("cpu")) or torch.device("cpu")
        sums_t = torch.tensor(flat_sums, dtype=torch.float64, device=device)
        counts_t = torch.tensor(flat_counts, dtype=torch.float64, device=device)

        # all-reduce
        if world_size > 1:
            dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)

        # compute average metrics
        results: Dict[str, torch.Tensor] = {}
        for idx, k in enumerate(metrics):
            for j, ds in enumerate(global_ds + ["overall"]):
                pos = idx * N + j
                total_sum = sums_t[pos]
                total_count = counts_t[pos]
                avg = (
                    total_sum / total_count if total_count > 0 
                    else torch.tensor(float("nan"), device=device)
                )
                results[f"{ds}/{k}"] = avg

        # reset for next epoch
        self.reset()
        return results

    def get_sample_counts(self) -> Dict[str, int]:
        """Get sample counts per dataset."""
        # local dataset list
        local_ds = sorted(
            set(chain.from_iterable(self._counts[k].keys() for k in self._metrics_seen))
            - {"_overall"}
        )

        # gather global dataset list
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_ds)
            global_ds = sorted(set(chain.from_iterable(gathered)))
        else:
            global_ds = local_ds

        if not self._metrics_seen:
            return {ds: 0 for ds in global_ds}

        # Use the first metric to get counts (all metrics should have same counts per dataset)
        first_metric = sorted(self._metrics_seen)[0]
        
        # flatten counts with fixed order
        flat_counts = []
        for ds in global_ds:
            flat_counts.append(self._counts[first_metric].get(ds, 0))

        device = getattr(self.module, "device", torch.device("cpu")) or torch.device("cpu")
        counts_t = torch.tensor(flat_counts, dtype=torch.int64, device=device)

        # all-reduce
        if world_size > 1:
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)

        # return sample counts per dataset
        sample_counts = {}
        for j, ds in enumerate(global_ds):
            sample_counts[ds] = int(counts_t[j].item())
        
        return sample_counts

    def get_part_count_ranges(self) -> Dict[str, tuple[int, int]]:
        """Get part count ranges (min, max) per dataset."""
        # local dataset list
        local_ds = sorted(
            set(self._part_counts.keys()) - {"_overall"}
        )

        # gather global dataset list
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_ds)
            global_ds = sorted(set(chain.from_iterable(gathered)))
        else:
            global_ds = local_ds

        if not self._part_counts:
            return {ds: (0, 0) for ds in global_ds}

        # Gather part counts from all processes
        part_count_ranges = {}
        for ds in global_ds:
            local_parts = self._part_counts[ds].get("parts", [])
            
            if world_size > 1:
                # Gather part counts from all processes
                gathered_parts = [None] * world_size
                dist.all_gather_object(gathered_parts, local_parts)
                all_parts = []
                for parts_list in gathered_parts:
                    all_parts.extend(parts_list)
            else:
                all_parts = local_parts
            
            if all_parts:
                part_count_ranges[ds] = (min(all_parts), max(all_parts))
            else:
                part_count_ranges[ds] = (0, 0)
        
        return part_count_ranges
