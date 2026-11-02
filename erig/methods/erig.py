"""Entropy-Regularized Integrated Gradients (ERIG)."""

from __future__ import annotations

from typing import Any

import logging

import torch
from torch.utils.data import DataLoader

from erig.methods.method import Method
from erig.methods.utils import (
    get_baseline,
    squeeze_example,
)
from erig.methods.ig import IntegratedGradientsDataset as ERIGDataset

logger = logging.getLogger("rich")


class EntropyRegularizedIntegratedGradients(Method):
  """ERIG attribution with softmax weights over squared path gradients.

  AERIG and SERIG subclass this implementation to change the temperature or
  weighting. Each variant reads its own prefixed configuration keys and can
  extend the uniform path dataset.
  """

  # Variants may use a subclass exposing more per-sample
  # information (e.g. the path position for a path-dependent temperature).
  path_dataset_cls = ERIGDataset

  def __init__(self, config: dict[str, Any], name: str = 'erig') -> None:
    """
    Initialize the Entropy-Regularized Integrated Gradients method.

    Args:
        config: Configuration dictionary
        name: Method name, also the prefix of this method's keys in
            ``config["compute_config"]`` (``erig_*`` by default), so variants
            registered under another name read their own keys
    """

    super().__init__(name, config)

    compute_config = config["compute_config"]

    def key(suffix: str) -> str:
      return f"{name}_{suffix}"

    self.exp_temperature = compute_config.get(key("exp_temperature"), 3.0)
    self.baseline = compute_config.get(key("baseline"), "zero")
    self.num_steps = compute_config.get(
        key("num_steps"), 50)  # integration steps
    self.baseline_kwargs = {
        "sigma": compute_config.get(key("baseline_blur_sigma"), 20.0),
        "kernel_size": compute_config.get(
            key("baseline_blur_kernel_size"), 121),
        "gaussian_sigma": compute_config.get(
            key("baseline_gaussian_sigma"), 0.1),
        "window_size": compute_config.get(
            key("baseline_perm_window"), 8),
    }

  def _batch_temperature(
      self,
      batch: dict[str, Any],
      flat_grads: torch.Tensor,
  ) -> float | torch.Tensor:
    """Temperature for the current path batch: the fixed ``exp_temperature``.

    Variants with a path-dependent temperature override this and return a
    tensor broadcastable against ``flat_grads`` (one value per sample).
    """
    return self.exp_temperature  # fixed temperature

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute Integrated Inverse Gradients attributions for a single example.

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
    baseline_input = get_baseline(
        example[self.input_key], self.baseline, **self.baseline_kwargs)
    baseline_input = baseline_input.to(self.device)

    baseline_example[self.input_key] = baseline_input.unsqueeze(0)
    baseline_example[self.input_key].requires_grad = True
    baseline_logits = self.model(baseline_example[self.input_key])
    baseline_pred = torch.softmax(baseline_logits, dim=-1)

    dataloader = DataLoader(
        self.path_dataset_cls(
            example, self.input_key, self.num_steps, baseline_input
        ),
        batch_size=self.batch_size,
        shuffle=False,
    )

    integrated_sensitivity_grads = torch.zeros_like(
        example[self.input_key].unsqueeze(0))
    erig_bar = self.progress.add_task(
        f"[green]Computing ERIG - Example {example_idx}", total=self.num_steps
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

      # Compute sensitivity weights
      bsz = batch[self.input_key].size(0)
      flat_grads = gradients.view(bsz, -1)
      flat_sensitivity_grads = self._sensitivity_weights(
          flat_grads, self._batch_temperature(batch, flat_grads)
      )

      # Scale each weight by its successive output difference.
      current_batch_preds = pred[:, original_pred_class]

      # First sample in ERIG dataset uses baseline, rest use previous sample
      if i == 0:
        if bsz == 1:
          output_diff = (
              current_batch_preds - baseline_pred[0:1, original_pred_class]
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
              current_batch_preds - prev_pred[-1:, original_pred_class]
          )
        else:
          ref_preds = torch.zeros_like(current_batch_preds)
          ref_preds[0] = prev_pred[-1, original_pred_class]
          ref_preds[1:] = current_batch_preds[:-1]
          output_diff = current_batch_preds - ref_preds

      flat_sensitivity_grads *= output_diff.unsqueeze(-1)

      # Store current prediction for next iteration
      prev_pred = pred.detach().clone()

      # Accumulate sensitivity gradients
      sensitivity_grads = flat_sensitivity_grads.view_as(gradients)
      integrated_sensitivity_grads += sensitivity_grads.sum(dim=0, keepdim=True)
      self.progress.update(erig_bar, advance=bsz)
    self.progress.remove_task(erig_bar)

    return {
        "original_example_idx": example_idx,
        "attributions": integrated_sensitivity_grads,
        "original_prediction": original_pred,
        "label": example.get("label", None),
    }

  @staticmethod
  def _sensitivity_weights(
      gradients: torch.Tensor,
      temperature: float | torch.Tensor,
      dim: int = -1,
  ) -> torch.Tensor:
    """Compute the paper's maximum-entropy squared-gradient weights."""
    weights = torch.softmax((gradients * gradients) / temperature, dim=dim)
    if weights.isnan().any():
      raise ValueError('NaN in sensitivity weights')
    return weights
