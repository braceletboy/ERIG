"""Important Direction Gradient Integration (IDGI).

References:
- Yang, R., Wang, B., & Bilgic, M. (2023). IDGI: A framework to eliminate
  explanation noise from integrated gradients. In Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (pp. 23725-23734).

Implementation adapted from:
- https://github.com/ShreeSinghi/TMLR-IDGI/
"""

from __future__ import annotations

from typing import Any

import logging

import torch
from torch.utils.data import DataLoader

from erig.methods.method import Method
from erig.methods.utils import get_baseline, squeeze_example
from erig.methods.ig import (
    IntegratedGradientsDataset as ImportantDirectionGradientIntegrationDataset
)


logger = logging.getLogger("rich")


class ImportantDirectionGradientIntegration(Method):
  """Compute Important Direction Gradient Integration attributions."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the IDGI method.

    Args:
        config: Configuration dictionary
    """
    super().__init__("idgi", config)

    # IDGI-specific initialization
    compute_config = config["compute_config"]
    self.baseline = compute_config.get("idgi_baseline", "zero")
    self.num_steps = compute_config.get(
        "idgi_num_steps", 50)  # integration steps

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute IDGI attributions for a single example.
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

    # Prepare baseline example
    baseline_example = example.copy()

    # Remove batch dimension
    example = squeeze_example(example, dim=0)

    # Get baseline input
    baseline_input = get_baseline(example[self.input_key], self.baseline)
    baseline_input = baseline_input.to(self.device)

    baseline_example[self.input_key] = baseline_input.unsqueeze(0)
    baseline_example[self.input_key].requires_grad = True
    baseline_logits = self.model(baseline_example[self.input_key])
    baseline_pred = torch.softmax(baseline_logits, dim=-1)

    dataloader = DataLoader(
        ImportantDirectionGradientIntegrationDataset(
            example, self.input_key, self.num_steps, baseline_input
        ),
        batch_size=self.batch_size,
        shuffle=False,
    )

    idgi = torch.zeros_like(
        example[self.input_key].unsqueeze(0))
    idgi_bar = self.progress.add_task(
        f"[green]Computing IDGI - Example {example_idx}", total=self.num_steps
    )

    # Keep track of previous prediction for output difference calculation
    prev_pred = None

    for i, batch in enumerate(dataloader):
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

      # Compute norm scaled gradients
      bsz = batch[self.input_key].size(0)
      flat_grads = gradients.view(bsz, -1)
      squared_flat_grads = flat_grads*flat_grads
      flat_norm_grads = (
          squared_flat_grads /
          (squared_flat_grads.sum(-1, keepdim=True) + 1e-10)
      )

      # Scale by output difference - First sample in IDGI dataset uses baseline,
      # rest use previous sample
      current_batch_preds = pred[:, original_pred_class]
      if i == 0:
        if bsz == 1:
          output_diff = (
              current_batch_preds - baseline_pred[0, original_pred_class]
          )
        else:
          ref_preds = torch.zeros_like(current_batch_preds)
          ref_preds[0] = baseline_pred[0, original_pred_class]
          ref_preds[1:] = current_batch_preds[:-1]
          output_diff = current_batch_preds - ref_preds
      else:
        # Previous sample is last of previous batch for first, then previous
        # samples of current batch
        if bsz == 1:
          output_diff = (
              current_batch_preds - prev_pred[-1, original_pred_class]
          )
        else:
          ref_preds = torch.zeros_like(current_batch_preds)
          ref_preds[0] = prev_pred[-1, original_pred_class]
          ref_preds[1:] = current_batch_preds[:-1]
          output_diff = current_batch_preds - ref_preds

      flat_norm_grads *= output_diff.unsqueeze(-1)

      # Store current prediction for next iteration
      prev_pred = pred.detach().clone()

      # Accumulate exponential gradients
      norm_grads = flat_norm_grads.view_as(gradients)
      idgi += norm_grads.sum(dim=0, keepdim=True)
      self.progress.update(idgi_bar, advance=bsz)
    self.progress.remove_task(idgi_bar)

    return {
        "original_example_idx": example_idx,
        "attributions": idgi,
        "original_prediction": original_pred,
        "label": example.get("label", None),
    }
