"""Integrated Gradients.

References:
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep
  networks. In Proceedings of the 34th International Conference on Machine
  Learning-Volume 70 (pp. 3319-3328).
"""

from __future__ import annotations

from typing import Any

import logging

import torch
from torch.utils.data import Dataset, DataLoader

from erig.methods.method import Method
from erig.methods.utils import get_baseline, squeeze_example


logger = logging.getLogger("rich")


class IntegratedGradientsDataset(Dataset):
  """Interpolate between a baseline and an input along a straight path."""

  def __init__(
      self,
      example: dict[str, Any],
      input_key: str,
      num_steps: int,
      baseline: torch.Tensor,
  ) -> None:
    """
    Initialize the dataset.

    Args:
        example: Input example
        input_key: Key for the input tensor
        num_steps: Number of interpolation steps
        baseline: Baseline input for Integrated Gradients
    """
    self.example = example
    self.input_key = input_key
    self.num_steps = num_steps
    self.baseline = baseline

  def __len__(self) -> int:
    """
    Get the length of the dataset.

    Returns:
        Length of the dataset
    """
    return self.num_steps + 1

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """
    Get an item from the dataset.

    Args:
        idx: Index of the item

    Returns:
        Interpolated input example
    """
    alpha = idx / self.num_steps
    interpolated = self.example.copy()
    interpolated[self.input_key] = self.baseline + alpha * (
        self.example[self.input_key] - self.baseline)
    return interpolated


class IntegratedGradients(Method):
  """Compute Integrated Gradients attributions."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the Integrated Gradients method.

    Args:
        config: Configuration dictionary
    """
    super().__init__("ig", config)

    # IG-specific initialization
    compute_config = config["compute_config"]
    self.baseline = compute_config.get("ig_baseline", "zero")
    self.num_steps = compute_config.get("ig_num_steps", 50)  # integration steps

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute Integrated Gradients attributions for a single example.
    Args:
        example_idx: Index of the example
        example: Input example

    Returns:
        Dictionary containing attributions, prediction, and label (if available)
    """
    # Compute original prediction
    with torch.inference_mode():
      logits = self.model(example[self.input_key])
      original_pred = torch.softmax(logits, dim=-1)
    original_pred_class = torch.argmax(original_pred, dim=-1).item()
    if example.get("label") is not None:
      label = example["label"]
      if original_pred_class != label:
        logger.debug(
            f"Example {example_idx}: Model prediction {original_pred_class} "
            f"does not match true label {label}. Skipping Attribution."
        )
        return {
            "original_example_idx": example_idx,
            "attributions": None,
            "original_prediction": original_pred,
            "label": label,
        }

    # Remove batch dimension
    example = squeeze_example(example, dim=0)

    # Get baseline input
    baseline_input = get_baseline(example[self.input_key], self.baseline)
    baseline_input = baseline_input.to(self.device)

    dataloader = DataLoader(
        IntegratedGradientsDataset(
            example, self.input_key, self.num_steps, baseline_input
        ),
        batch_size=self.batch_size,
        shuffle=False,
    )

    integrated_grads = torch.zeros_like(example[self.input_key].unsqueeze(0))
    ig_bar = self.progress.add_task(
        f"[green]Computing IG - Example {example_idx}", total=self.num_steps
    )
    for batch in dataloader:
      batch[self.input_key].requires_grad = True
      logits = self.model(batch[self.input_key])
      pred = torch.softmax(logits, dim=-1)

      # Compute loss - cross sample gradients are zero - so sum is fine here
      loss = pred[:, original_pred_class].sum()

      # Backward pass
      gradients = torch.autograd.grad(
          outputs=loss,
          inputs=batch[self.input_key],
          create_graph=False
      )[0]  # use autograd instead of backward to save memory

      # Accumulate gradients
      integrated_grads += gradients.sum(0, keepdim=True) / self.num_steps
      self.progress.update(ig_bar, advance=batch[self.input_key].size(0))
    self.progress.remove_task(ig_bar)

    # Scale by input difference
    integrated_grads *= (example[self.input_key] - baseline_input)

    return {
        "original_example_idx": example_idx,
        "attributions": integrated_grads,
        "original_prediction": original_pred,
        "label": example.get("label", None),
    }
