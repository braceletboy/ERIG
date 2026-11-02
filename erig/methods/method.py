"""Model loading, input transforms, and storage for attribution methods."""

from __future__ import annotations

from typing import Any, Self
from collections.abc import Callable, Sequence
from types import TracebackType
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

import json
import logging
import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from datasets import load_dataset
from erig.methods.model_config import get_model_id
from erig.methods.utils import (
    set_global_seed, get_device,
    move_example_to_device,
    unsqueeze_example,
    squeeze_example,
    detach_example,
    get_timm_transforms,
    get_transformers_transforms,
    is_transformers_model,
    load_timm_model,
    load_transformers_model,
)
from rich.progress import (
    Progress,
    TextColumn,
    TimeRemainingColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    SpinnerColumn
)


logger = logging.getLogger("rich")
torch._C._debug_only_display_vmap_fallback_warnings(True)


class _DummyProgress:
  """No-op implementation of the Rich progress interface."""

  def __enter__(self) -> Self:
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: TracebackType | None,
  ) -> bool:
    return False

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def add_task(
      self,
      description: str,
      total: float | None = None,
      **kwargs: Any,
  ) -> int:
    """Return a dummy task ID."""
    return 1000

  def update(self, task_id: int, advance: float = 1, **kwargs: Any) -> None:
    """Do nothing on update."""
    pass

  def remove_task(self, task_id: int) -> None:
    """Do nothing on remove."""
    pass


class Method(ABC):
  """Base class for attribution methods.

  Subclasses implement ``_compute_attrs_per_example``. The base class handles
  tensor attribution, model loading, preprocessing, dataset iteration, and
  output storage.
  """

  def __init__(self, name: str, config: dict[str, Any]) -> None:
    """Initialize the explainability method.

    Args:
        name: Name of the method (e.g., "ig" or "erig")
        config: Configuration dictionary having method and dataset parameters
    """

    if "_model" in config:
      self.name = name
      self.config = config
      self.model = config["_model"]
      self.device = str(next(self.model.parameters()).device)
      self.input_key = config.get("input_key", "image")
      self.batch_size = config.get("eval_batch_size", 1)
      self.progress = _DummyProgress()
      self.seed = config.get("seed", 42)
      self.output_dir = config.get("output_dir", ".")
      self.model_name = config.get("model_name", "provided")
      self.dataset_name = "attributions"
      self.dataset_config = {"dataset_name": "attributions"}
      return
    self.name = name
    logger.info(f"Initializing method: `{self.name}` with config: {config}")

    self.config = config
    self.output_dir = config["output_dir"]

    self.dataset_config = config["dataset_config"]
    self.seed = config.get("seed", 42)

    set_global_seed(self.seed)

    self.model_name = config["model_name"]
    self.dataset_name = config["dataset_name"]

    self.input_key = self.config.get("input_key", "image")

    self.batch_size = self.config.get("eval_batch_size", 1)
    self.model, self.input_transform = self._get_model_and_input_transform()
    if config.get("compile", False):
      logger.info("Compiling model for performance optimization")
      self.model = torch.compile(self.model)

    self.device = get_device(self.config)
    logger.info(
        f"Initialized {self.model_name} method on device: {self.device}")
    if hasattr(self.model, 'to'):
      self.model.to(self.device)

    logger.info(f"Dataset configuration for {self.dataset_name}: "
                f"{self.dataset_config}")
    self._load_dataset()

    if not config.get("no_progress_bar", False):
      logger.info(f"Progress bar enabled for {self.name.upper()}")
      self.progress = Progress(
          SpinnerColumn(finished_text="[DONE]"),
          TextColumn(
              "[progress.description]{task.description}"
              "({task.percentage:05.2f}%)"
          ),
          BarColumn(),
          MofNCompleteColumn(),
          TimeRemainingColumn(),
          TimeElapsedColumn(),
          expand=True
      )
    else:
      logger.info(f"Progress bar disabled for {self.name.upper()}")
      self.progress = _DummyProgress()

    self.all_metrics = {}

  def _load_dataset(self) -> None:
    """
    Load dataset from Hugging Face datasets.

    Returns:
        Raw dataset ready for processing
    """
    if self.dataset_config.get(
            "dataset_name", self.dataset_name) == "attributions":
      return  # Dataset will be loaded from attributions.hdf5 in metric class
    dataset = load_dataset(
        self.dataset_config.get("dataset_name", self.dataset_name),
        self.dataset_config.get("task_name"),
        split=self.dataset_config["validation_key"],
        data_files=self.dataset_config.get("data_files"),
        revision=self.dataset_config.get("revision"),
        token=os.environ.get("HF_TOKEN")
    )
    num_samples = self.config["num_samples"]
    self.dataset = dataset.shuffle(seed=self.seed).select(
        torch.arange(min(num_samples, len(dataset)))
        if num_samples > 0 else torch.arange(len(dataset)))

    self.sample_indices = (self.dataset._indices.column(0).to_pylist()
                           if self.dataset._indices is not None
                           else list(range(len(self.dataset))))

    task_suffix = (
        f"/{self.dataset_config['task_name']}"
        if self.dataset_config.get('task_name') else ""
    )
    logger.info(f"Loaded split: {self.dataset_config['validation_key']} of "
                f"dataset: {self.dataset_config['dataset_name']}{task_suffix}")

  def _get_model_and_input_transform(
      self,
  ) -> tuple[torch.nn.Module, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Get the model and input transformation function.

    timm serves the official ImageNet checkpoints; transformers serves the
    domain fine-tuned ones, which are named by Hub repo id.

    Returns:
        Tuple containing the model and input transformation function
    """
    model_id = get_model_id(self.model_name, self.dataset_name)
    if is_transformers_model(model_id):
      model = load_transformers_model(model_id)
      transform, spatial_transform = get_transformers_transforms(
          model_id)
    else:
      model = load_timm_model(model_id)
      transform, spatial_transform = get_timm_transforms(model_id)

    def image_transform(example: dict[str, Any]) -> dict[str, Any]:
      """Prepare an image and any associated label or segmentation mask."""
      example[self.input_key] = transform(
          example[self.input_key].convert("RGB"))
      if self.dataset_name == "imagenet-1k-wds":
        if 'json' in example and 'label' in example['json']:
          example['label'] = example['json']['label']
      if self.dataset_name == "oxford-iiit-pet":
        # The trimap lives in a differently named column. Move it under the
        # "mask" key the rest of the pipeline expects, and drop the original
        # so no undecoded PIL image reaches json.dumps in
        # compute_attributions.
        if "segmentation_mask" in example:
          example["mask"] = example.pop("segmentation_mask")
      if "mask" in example:  # For segmentation datasets
        example["mask"] = spatial_transform(
            example["mask"].convert("L"))

        # The localization metric compares the mask against the attribution
        # map pixel for pixel, so a geometry mismatch between the image and
        # spatial pipelines would silently misalign every score.
        image_size = example[self.input_key].shape[-2:]
        mask_size = example["mask"].shape[-2:]
        if image_size != mask_size:
          raise ValueError(
              f"Mask/image spatial mismatch for {self.model_name}: image is "
              f"{tuple(image_size)} but mask is {tuple(mask_size)}"
          )
      return example

    return model, image_transform

  def compute_attributions_for_dataset(
      self,
      tensor_dataset: (
          Dataset[dict[str, Any]] | HFDataset | Sequence[dict[str, Any]]
      ),
      skip_transform: bool = False,
      skip_progress_init: bool = False,
  ) -> list[dict[str, Any]]:
    """Compute attributions and return results in memory (no HDF5 write).

    Args:
        tensor_dataset: Any ``torch.utils.data.Dataset`` yielding examples.
        skip_transform: Whether to skip input transformation (if already done).
        skip_progress_init: Whether to skip progress bar initialization
                            (if already done).
    Returns:
        List of output dicts (attributions, predictions, labels) with tensors
        detached and moved to CPU.
    """
    tensor_dataloader = torch.utils.data.DataLoader(
        tensor_dataset, batch_size=None
    )

    self.model.eval()

    outputs = []

    if not skip_progress_init:
      self.progress.start()

    examples_bar = self.progress.add_task(
        description="[green]Tensor Examples", total=len(tensor_dataloader),
    )
    for example_idx, example in enumerate(tensor_dataloader):
      if not skip_transform:  # Transform example
        example = self.input_transform(example)

      example = move_example_to_device(
          unsqueeze_example(example, dim=0), self.device)
      output = self._compute_attrs_per_example(example_idx, example)
      outputs.append({
          k: v.detach().cpu().squeeze(0) if torch.is_tensor(v) else v
          for k, v in output.items()
      })
      self.progress.update(examples_bar, advance=1)

    if skip_progress_init:  # examples are not the main task
      self.progress.remove_task(examples_bar)

    if not skip_progress_init:
      self.progress.stop()

    return outputs

  def compute_attributions(self) -> None:
    """Generate and save attribution scores for the tokenized dataset.

    The generated attributions will be saved in an HDF5 file. The structure of
    the file will be as follows:
    - /example_{save_idx}/
        - example: Original input example
        - attributions: Attribution scores for the example
    """
    self.model.eval()

    logger.info(f"Starting prediction on {len(self.dataset)} examples")

    with h5py.File(
        os.path.join(self.output_dir, "attributions.hdf5"), 'w', libver='latest'
    ) as root_group:
      root_group.attrs["config"] = json.dumps(self.config)
      root_group.attrs["num_samples"] = len(self.dataset)

      with self.progress:
        examples_bar = self.progress.add_task(
            description="[green]Examples", total=len(self.dataset),
        )

        save_idx = 0
        for example_idx, example in enumerate(self.dataset):
          example = self.input_transform(example)
          example = move_example_to_device(
              unsqueeze_example(example, dim=0), self.device)
          output = self._compute_attrs_per_example(example_idx, example)
          if output["attributions"] is None:
            self.progress.update(examples_bar, advance=1)
            continue

          example_group = root_group.create_group(f"example_{save_idx}")
          save_idx += 1

          # Store numeric inputs natively instead of duplicating them as JSON.
          saved_example = move_example_to_device(
              detach_example(squeeze_example(example, dim=0)), 'cpu')
          tensor_group = example_group.create_group('inputs')
          metadata = {}
          for key, value in saved_example.items():
            if torch.is_tensor(value):
              tensor_group.create_dataset(
                  key, data=value.numpy(),
                  compression='lzf' if value.ndim else None
              )
              metadata[key] = {'__tensor__': key}
            else:
              metadata[key] = value
          example_group.create_dataset('example', data=json.dumps(metadata))

          for key, value in output.items():
            if value is not None:
              # Handle list outputs (e.g., poly_terms) by creating a subgroup
              if isinstance(value, list):
                subgrp = example_group.create_group(key)
                for idx, v in enumerate(value):
                  data = (v.detach().cpu().squeeze(0).numpy()
                          if torch.is_tensor(v) else np.asarray(v))
                  subgrp.create_dataset(f"term_{idx}", data=data)
              else:
                example_group.create_dataset(
                    key, data=(value.detach().cpu().squeeze(0).numpy()
                               if torch.is_tensor(value) else value)
                )

          self.progress.update(examples_bar, advance=1)

  def attribute(self, inputs: torch.Tensor) -> torch.Tensor:
    """Return BCHW attributions on the model's device.

    CHW inputs receive a batch dimension. Inputs are moved to the model's
    device automatically and the model is set to evaluation mode. The model's
    training state is restored after attribution computation.
    """
    if inputs.ndim == 3:
      inputs = inputs.unsqueeze(0)
    if (
        inputs.ndim != 4
        or not inputs.is_floating_point() or inputs.numel() == 0
    ):
      raise ValueError(
          'inputs must be a nonempty floating-point CHW or BCHW tensor'
      )
    device = next(self.model.parameters()).device
    inputs = inputs.detach().to(device)
    training = self.model.training
    self.model.eval()
    try:
      with torch.enable_grad():
        values = [self._compute_attrs_per_example(i, {self.input_key: x[None]})
                  ['attributions'].detach() for i, x in enumerate(inputs)]
      return torch.cat(values, dim=0)
    finally:
      self.model.train(training)

  @abstractmethod
  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute attribution scores for each example.

    Args:
        example_idx: Index of the example
        example: Input example

    Returns:
        Dictionary containing attributions, prediction, and label (if available)
    """
    raise NotImplementedError(
        f"_compute_attrs_per_example_vision method must be implemented "
        f"by `{self.name}`"
    )

  def get_config(self) -> dict[str, Any]:
    """
    Get the current configuration.

    Returns:
        Copy of the configuration dictionary
    """
    return self.config.copy()

  def __str__(self) -> str:
    """
    String representation of the method.

    Returns:
        Human-readable string describing the method
    """
    return f"Method: `{self.name}` (Device: {self.device})"

  def __repr__(self) -> str:
    """
    Detailed representation of the method.

    Returns:
        Detailed string representation for debugging
    """
    return (
        f"Method: `{self.name}` (Device: {self.device}, Config: {self.config})"
    )
