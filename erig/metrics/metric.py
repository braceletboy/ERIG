"""Load saved attributions, compute metrics, and store results."""

from __future__ import annotations

from typing import Any, Self
from collections.abc import Mapping, Sequence
from os import PathLike
from types import TracebackType
from erig.methods.method import Method

import json
import logging
import os
import random
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from rich.progress import (
    Progress,
    TextColumn,
    TimeRemainingColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    SpinnerColumn
)
from torch.utils.data import Dataset, DataLoader

from erig.methods.method_config import get_method_class
from erig.methods.meta_method_config import (
    get_meta_method_class, list_available_meta_methods
)
from erig.methods.model_config import get_model_id
from erig.methods.utils import (
    set_global_seed, get_device,
    is_transformers_model,
    load_timm_model,
    load_transformers_model,
    move_collated_batch_to_device,
)


logger = logging.getLogger("rich")
torch._C._debug_only_display_vmap_fallback_warnings(True)


def open_hdf5_for_reading(hdf5_path: str | PathLike[str]) -> h5py.File:
  """Open a saved attribution file for reading.

  Args:
      hdf5_path: Path to the HDF5 file

  Returns:
      An open read-only h5py.File handle.
  """
  try:
    return h5py.File(hdf5_path, 'r', libver='latest', swmr=True)
  except OSError as error:
    logger.warning(
        f"SWMR read of {hdf5_path} failed ({error}); falling back to a "
        f"plain read-only open")
    return h5py.File(hdf5_path, 'r', libver='latest')


class HDF5Dataset(Dataset):
  """Read saved examples, attributions, and predictions.

  Use within a ``with`` statement to manage the file lifetime.
  """

  def __init__(self, hdf5_path: str | PathLike[str]) -> None:
    """
    Args:
      hdf5_path: Path to the HDF5 file containing examples and attributions.
    """
    self.hdf5_path = hdf5_path
    self.file_handle = None
    logger.info(f"Reading keys from HDF5 file: {hdf5_path}")
    with open_hdf5_for_reading(self.hdf5_path) as f:
      self.keys = list(sorted(f.keys(), key=lambda x: int(x.split('_')[-1])))

  def __len__(self) -> int:
    """Return the number of stored examples."""
    return len(self.keys)

  def __enter__(self) -> Self:
    """Open the HDF5 file when entering context."""
    self.file_handle = open_hdf5_for_reading(self.hdf5_path)
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: TracebackType | None,
  ) -> None:
    """Close the HDF5 file when exiting context."""
    if self.file_handle is not None:
      self.file_handle.close()
      self.file_handle = None

  def __getitem__(self, idx: int) -> tuple[str, dict[str, Any], np.ndarray, np.ndarray]:
    """Read an example and its saved attribution and prediction.

    Args:
        idx: Position in the dataset.

    Returns:
        The example key, input dictionary, attribution array, and prediction array.

    Raises:
        RuntimeError: If the dataset has not been opened as a context manager."""
    if self.file_handle is None:
      raise RuntimeError("HDF5Dataset must be used within a context manager")

    key = self.keys[idx]
    example_json_string = self.file_handle[key]["example"][()]
    if isinstance(example_json_string, bytes):
      example_json_string = example_json_string.decode('utf-8')
    loaded_example = json.loads(example_json_string)
    example = {}
    for k, v in loaded_example.items():
      if isinstance(v, dict) and set(v) == {'__tensor__'}:
        value = np.asarray(self.file_handle[key]['inputs'][v['__tensor__']][()])
        # Match the original JSON reader's scalar types and tensor dtypes.
        # In particular, uint8 masks need int64 for the -1 label sentinel.
        if value.ndim == 0:
          example[k] = value.item()
        else:
          dtype = (torch.int64 if value.dtype.kind in 'iu' else
                   torch.get_default_dtype() if value.dtype.kind == 'f' else None)
          example[k] = torch.as_tensor(value, dtype=dtype)
      elif isinstance(v, list):
        try:
          v = torch.tensor(v)
        finally:
          example[k] = v
      else:
        example[k] = v
    attributions = self.file_handle[key]["attributions"][()]
    original_prediction = self.file_handle[key]["original_prediction"][()]
    return key, example, attributions, original_prediction


class Metric(ABC):
  """Base class for metrics computed from saved attributions."""

  # Metrics that never evaluate the model (pure attribution-map computations
  # such as sparsity, LINV and localization) override this to False: the
  # checkpoint is then neither downloaded nor loaded onto the GPU, so those
  # metrics cannot fail on model load and do not hold model weights in GPU
  # memory under contention. Batch tensors still move to the device, so the
  # computed values are bit-identical to the uses_model=True behaviour.
  uses_model = True

  def __init__(self, name: str, config: dict[str, Any]) -> None:
    """Initialize a metric from its run configuration.

    Args:
        name: Name of the metric.
        config: Model, data, output, and metric settings.
    """
    self.name = name
    self.config = config
    self.attributions_dir = config["attributions_dir"]
    self.output_dir = config.get("output_dir", self.attributions_dir)

    # Set global random seed for reproducibility
    self.seed = config.get("seed", 42)
    set_global_seed(self.seed)

    self.dataset_name = config["dataset_name"]

    # Load model. Model-free metrics (uses_model=False) skip the load
    # entirely: self.model stays None, and the hasattr guards below and in
    # compute_metrics fall through naturally.
    self.model_name = config["model_name"]
    if self.uses_model:
      self.model = self._get_model()
      if config.get("compile", False):
        logger.info("Compiling model for performance optimization")
        self.model = torch.compile(self.model)
    else:
      self.model = None
      logger.info(
          f"{self.name} metric does not evaluate the model; skipping model "
          f"load for {self.model_name}")

    # Set up device
    self.device = get_device(self.config)
    logger.info(
        f"Initialized {self.model_name} method on device: {self.device}")
    if hasattr(self.model, 'to'):
      self.model.to(self.device)

    self.input_key = config.get("input_key", "input_ids")
    self.batch_size = config.get("metric_batch_size", config.get("eval_batch_size", 128))
    self.dataset = HDF5Dataset(os.path.join(
        self.attributions_dir, "attributions.hdf5"))
    if 0 <= config.get("max_examples", -1) <= len(self.dataset.keys):
      # Limit to a random subset of examples if max_examples is set
      self.dataset.keys = list(sorted(random.sample(
          self.dataset.keys,
          min(config["max_examples"], len(self.dataset.keys))
      ), key=lambda x: int(x.split('_')[-1])))
      logger.info(f"[bold red]Limiting to {len(self.dataset.keys)} "
                  f"examples for {self.name} metric(s)[/bold red]")
    self.save_path = os.path.join(config["output_dir"], f'{self.name}.hdf5')
    logger.info(f"Metrics will be saved to: {self.save_path}")

    self.progress = Progress(
        SpinnerColumn(finished_text="[DONE]"),
        TextColumn(
            "[progress.description]{task.description}({task.percentage:05.2f}%)"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        expand=True
    )
    self.aggregate_metrics = {}

  def _load_method(self) -> tuple[str, Method]:
    """Load the attribution method associated with the saved run.

    Returns:
        method_name: Name of the method
        method_instance: Instance of the method class
    """
    with open(os.path.join(self.attributions_dir, 'config.json')) as f:
      method_config = json.load(f)['attributions']
    method_name = method_config['method_name']
    method_config["dataset_name"] = "attributions"
    method_config["dataset_config"]["dataset_name"] = "attributions"
    method_config["no_progress_bar"] = True
    method_config["output_dir"] = self.output_dir
    logger.info(f"Initializing {method_name} method...")

    if method_name not in list_available_meta_methods():
      method_class = get_method_class(method_name)
    else:
      method_class = get_meta_method_class(method_name)
    method_config["_model"] = self.model
    method_config["eval_batch_size"] = self.config.get("eval_batch_size", 1)
    method_instance = method_class(method_config)
    method_instance.progress = self.progress
    return method_name, method_instance

  def _get_model(self) -> torch.nn.Module:
    """Load the model used to evaluate attributions.

    Returns:
        The model, returning logits from a plain tensor input
    """
    model_id = get_model_id(self.model_name, self.dataset_name)
    if is_transformers_model(model_id):
      return load_transformers_model(model_id)
    return load_timm_model(model_id)

  def compute_metrics(self) -> None:
    """
    Compute and store metrics for each batch of examples in the dataset.
    """
    if hasattr(self.model, "eval"):
      self.model.eval()
    with (self.dataset, h5py.File(
            self.save_path, 'w', libver='latest') as metrics_group):
      individual_group = metrics_group.create_group("individual")
      # Process examples in batches using DataLoader
      dataloader = DataLoader(
          self.dataset, batch_size=self.batch_size, shuffle=False,
          num_workers=self.config.get("num_workers", 0)
      )
      with self.progress:
        examples_bar = self.progress.add_task(
            f"[cyan]Examples", total=len(self.dataset)
        )

        for batch_data in dataloader:
          (batch_keys, batch_examples,
           batch_attributions, batch_original_predictions) = batch_data
          batch_examples = move_collated_batch_to_device(
              batch_examples, self.device)
          batch_attributions = batch_attributions.to(self.device)
          batch_original_predictions = batch_original_predictions.to(
              self.device)

          batch_metrics = self.compute_metric_per_batch(
              batch_keys, batch_examples, batch_attributions,
              batch_original_predictions
          )

          # Save metrics for each example in the batch
          for i, example_key in enumerate(batch_keys):
            example_group = individual_group.create_group(example_key)
            for metric_name, metric_values in batch_metrics.items():
              example_group.create_dataset(metric_name, data=metric_values[i])
              if not metric_name.startswith("probs"):
                if metric_name not in self.aggregate_metrics:
                  self.aggregate_metrics[metric_name] = []
                self.aggregate_metrics[metric_name].append(metric_values[i])

          # Update progress bar
          self.progress.update(examples_bar, advance=len(batch_keys))

      # Compute and save aggregate metrics
      aggregate_group = metrics_group.create_group("aggregate")
      aggregate_metrics = self.compute_aggregate_metrics()
      for metric_name, metric_value in aggregate_metrics.items():
        aggregate_group.create_dataset(metric_name, data=metric_value)

  @abstractmethod
  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_predictions: torch.Tensor,
  ) -> Mapping[str, np.ndarray | Sequence[float | np.floating[Any]]]:
    """Compute per-example metric scores for a batch.

    Args:
        keys: List of example keys in the batch
        examples: Batched input examples (dict with batched tensors)
        attributions: Batched attribution scores (tensor)
        original_predictions: Model predictions for the unperturbed inputs.

    Returns:
        Dictionary of computed metrics where each value is a list/array with one
        value per example in the batch
    """
    raise NotImplementedError(
        "Subclasses must implement compute_metric_per_batch")

  def compute_aggregate_metrics(self) -> dict[str, float | np.floating[Any]]:
    """Summarize metric scores across examples."""
    return {
        key: np.nanmean(np.stack(values, axis=0))
        for key, values in self.aggregate_metrics.items()
    }
