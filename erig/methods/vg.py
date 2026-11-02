"""Vanilla gradient attribution.

References:
- Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks:
  Visualising image classification models and saliency maps. arXiv preprint
  arXiv:1312.6034. 2013 Dec 20.
"""

from __future__ import annotations

from typing import Any

import logging

import torch

from erig.methods.method import Method


logger = logging.getLogger("rich")


class VanillaGradients(Method):
  """Compute input gradients for the predicted-class probability."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the Vanilla Gradients method.

    V

    Args:
        config: Configuration dictionary
    """
    super().__init__("vg", config)

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute Vanilla Gradients attributions for a single example.
    Args:
        example_idx: Index of the example
        example: Input example

    Returns:
        Dictionary containing attributions, prediction, and label (if available)
    """
    # Compute original prediction and gradients
    example[self.input_key].requires_grad_(True)
    logits = self.model(example[self.input_key])
    pred = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(pred, dim=-1).item()
    if example.get("label") is not None:
      label = example["label"]
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
    gradients = torch.autograd.grad(
        outputs=pred[:, pred_class],
        inputs=example[self.input_key],
        create_graph=False
    )[0]  # gradients shape: (input_dim,)

    return {
        "original_example_idx": example_idx,
        "attributions": gradients,
        "original_prediction": pred,
        "label": example.get("label", None),
    }
