"""Blur Integrated Gradients (BIG).

References:
- Xu S, Venugopalan S, Sundararajan M. Attribution in scale and space.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition 2020 (pp. 9680-9689).

Implementation adapted from:
- https://github.com/PAIR-code/saliency/
"""

from __future__ import annotations

from typing import Any

import logging

import torch
from torch.utils.data import Dataset, DataLoader

from erig.methods.method import Method
from erig.methods.utils import (
    apply_gaussian_blur, squeeze_example
)

logger = logging.getLogger("rich")


class BlurIntegratedGradientsDataset(Dataset):
  """Generate blurred inputs along the BIG path."""

  def __init__(
      self,
      example: dict[str, Any],
      input_key: str,
      num_steps: int,
      max_sigma: float = 50.0,
      use_sqrt: bool = False,
      grad_step: float = 0.01,
  ) -> None:
    """
    Initialize the dataset.

    Args:
        example: Input example
        input_key: Key for the input tensor
        num_steps: Number of interpolation steps
        max_sigma: Maximum sigma for Gaussian blur
        use_sqrt: Whether to use square root scaling for sigma
        grad_step: Step size for numerical gradient approx of gaussian input
    """
    self.example = example
    self.input_key = input_key
    self.num_steps = num_steps
    self.max_sigma = max_sigma
    self.use_sqrt = use_sqrt
    self.grad_step = grad_step
    self.sigmas = torch.linspace(
        0, max_sigma, num_steps + 1, dtype=torch.float32)
    if use_sqrt:
      self.sigmas = torch.sqrt(self.sigmas)
    self.sigma_diffs = self.sigmas[1:] - self.sigmas[:-1]

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
    sigma = self.sigmas[idx]
    sigma_diff = (
        self.sigma_diffs[idx] if idx < self.num_steps else torch.tensor(0.0)
    )
    blurred = self.example.copy()
    blurred[self.input_key] = apply_gaussian_blur(
        self.example[self.input_key], sigma
    )
    # For numerical gradient approximation, add a small perturbation
    gaussian_grad = apply_gaussian_blur(
        self.example[self.input_key], sigma + self.grad_step
    ) - blurred[self.input_key]
    blurred["gaussian_grad"] = gaussian_grad / self.grad_step
    blurred["sigma_diff"] = sigma_diff
    return blurred


class BlurIntegratedGradients(Method):
  """Compute Blur Integrated Gradients attributions."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the Blur Integrated Gradients method.

    Args:
        config: Configuration dictionary
    """
    super().__init__("big", config)

    # BIG-specific initialization
    compute_config = config["compute_config"]
    self.num_steps = compute_config.get("big_num_steps", 50)
    self.max_sigma = compute_config.get("big_max_sigma", 50.0)
    self.use_sqrt = compute_config.get("big_use_sqrt", False)
    self.gaussian_grad_step = compute_config.get("big_gaussian_grad_step", 1e-2)

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute Blur Integrated Gradients attributions for an image.

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

    dataloader = DataLoader(
        BlurIntegratedGradientsDataset(
            example, self.input_key, self.num_steps, self.max_sigma,
            self.use_sqrt, self.gaussian_grad_step
        ),
        batch_size=self.batch_size, shuffle=False,
    )

    blur_integrated_grads = torch.zeros_like(
        example[self.input_key].unsqueeze(0))
    big_bar = self.progress.add_task(
        f"[green]Computing BIG - Example {example_idx}", total=self.num_steps
    )
    for batch in dataloader:
      gaussian_grad = batch.pop("gaussian_grad").to(self.device)
      sigma_diff = batch.pop("sigma_diff").view(-1, 1, 1, 1).to(self.device)
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

      # Scale gradients by sigma difference and gaussian gradient
      gradients *= gaussian_grad
      gradients *= sigma_diff

      # Accumulate gradients
      blur_integrated_grads += gradients.sum(0, keepdim=True)
      self.progress.update(big_bar, advance=batch[self.input_key].size(0))
    self.progress.remove_task(big_bar)

    # Because we summed in reverse order of max to 0
    blur_integrated_grads *= -1.0

    return {
        "original_example_idx": example_idx,
        "attributions": blur_integrated_grads,
        "original_prediction": original_pred,
        "label": example.get("label", None),
    }
