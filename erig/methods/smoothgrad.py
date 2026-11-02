"""SmoothGrad averages attributions over a clean input and noisy draws.

References:
- Smilkov D, Thorat N, Kim B, Viégas F, Wattenberg M. Smoothgrad: removing noise
  by adding noise. arXiv preprint arXiv:1706.03825. 2017 Jun 12.
"""

from __future__ import annotations

from typing import Any

import logging

import torch

from erig.methods.method import Method
from erig.methods.method_config import get_method_class

logger = logging.getLogger("rich")


class SmoothGrad(Method):
  """Average a base method over one clean draw and noisy draws."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the SmoothGrad method.

    Args:
        config: Configuration dictionary
    """
    super().__init__("smoothgrad", config)

    compute_config = config["compute_config"]
    self.num_samples = compute_config.get("smoothgrad_num_samples", 50)
    self.noise_level = compute_config.get("smoothgrad_noise_level", 0.1)
    self.base_method_name = compute_config.get("smoothgrad_base_method", "ig")
    logger.info(
        f"Initialized SmoothGrad with base method: {self.base_method_name}"
    )

    # Override the number of steps key for the base method
    if self.base_method_name in ["gig", "gidgi", "gerig"]:
      base_num_steps_key = f"{self.base_method_name}_num_segments"
    else:
      base_num_steps_key = f"{self.base_method_name}_num_steps"
    base_method_compute_config = {
        **compute_config.get("smoothgrad_base_method_compute_config", {}),
        base_num_steps_key: compute_config.get("smoothgrad_base_num_steps", 100),
    }
    base_method_config = {
        **config, "compute_config": base_method_compute_config
    }
    base_method_config["no_progress_bar"] = True
    base_method_class = get_method_class(self.base_method_name)
    self.base_method = base_method_class(base_method_config)

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute SmoothGrad attributions for a given example.

    Args:
        example_idx: Index of the example
        example: Input example
    Returns:
        Attributions dictionary
    """
    input_tensor = example[self.input_key]

    # Filter the clean input only; noisy draws keep their own predicted targets.
    if example.get("label") is not None:
      label = example["label"]
      with torch.no_grad():
        pred = torch.softmax(self.model(input_tensor), dim=-1)
      pred_class = torch.argmax(pred, dim=-1).item()
      if pred_class != label:
        logger.debug(
            f"Example {example_idx}: Model prediction {pred_class} "
            f"does not match true label {label}. Skipping Attribution."
        )
        return {
            "original_example_idx": example_idx,
            "attributions": None,
            "original_prediction": pred,
            "label": label,
        }

    samples_bar = self.progress.add_task(
        f"[green]Computing SmoothGrad - {self.base_method.name.upper()}",
        total=self.num_samples
    )
    self.base_method.model.eval()
    original_outputs = self.base_method._compute_attrs_per_example(
        example_idx, example)
    total_attributions = original_outputs["attributions"]
    self.progress.update(samples_bar, advance=1)
    for _ in range(self.num_samples - 1):
      noise = torch.randn_like(input_tensor) * self.noise_level * (
          input_tensor.max() - input_tensor.min())  # Scale noise by input range
      noisy_example = example.copy()
      noisy_example.pop("label", None)
      noisy_example[self.input_key] = input_tensor + noise

      outputs = self.base_method._compute_attrs_per_example(
          example_idx, noisy_example)
      total_attributions += outputs["attributions"]
      self.progress.update(samples_bar, advance=1)
    self.progress.remove_task(samples_bar)

    avg_attributions = total_attributions / self.num_samples

    return original_outputs | {"attributions": avg_attributions}
