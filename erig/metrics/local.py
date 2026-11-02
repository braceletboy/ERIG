"""Measure agreement between attribution maps and foreground masks."""

from __future__ import annotations

from typing import Any

import logging

import torch
import numpy as np

from erig.metrics.metric import Metric
from erig.metrics.utils import map_attr_values_to_linear_space

logger = logging.getLogger("rich")


def binarize_and_filter_masks(
    flat_mask: torch.Tensor,
    dataset_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Prepare foreground masks and identify examples that can be scored.

  Args:
      flat_mask: Integer tensor of shape (batch, pixels)
      dataset_name: Dataset whose masks are being evaluated.

  Returns:
      Tuple ``(flat_mask, valid_mask_filter)`` -- the mask after any
      binarization, and a bool tensor selecting the scorable examples
  """
  # Pets trimaps distinguish the object from both background and boundary.
  if dataset_name in ('oxford-iiit-pet', 'dpdl-benchmark/oxford_iiit_pet'):
    flat_mask = (flat_mask == 1).to(flat_mask.dtype)
  elif dataset_name not in ('imagenet-s', 'braceletboy/imagenet-s'):
    raise ValueError(f'Localization masks are not supported for {dataset_name}')

  # Count unique nonzero values in each mask
  sorted_masks, _ = torch.sort(flat_mask, dim=1)
  # Pad with -1 at start to ensure first element is counted as new
  padded = torch.cat([
      torch.full((flat_mask.shape[0], 1), -1,
                 device=flat_mask.device, dtype=flat_mask.dtype),
      sorted_masks
  ], dim=1)
  # Find where consecutive values differ and are nonzero
  is_new = sorted_masks != padded[:, :-1]
  is_nonzero = sorted_masks != 0
  unique_nonzero_count = (is_new & is_nonzero).sum(dim=1)

  # Only consider examples with exactly 1 foreground region
  return flat_mask, unique_nonzero_count == 1


class Local(Metric):
  """Measure pixel accuracy, F1, and intersection over union."""

  # Pure attribution-map computation: the model is never evaluated.
  uses_model = False

  def __init__(self, config: dict[str, Any]) -> None:
    '''
    Initialize the Local class.

    Args:
        config: Configuration dictionary containing parameters
    '''
    super().__init__('local', config)
    compute_config = config.get('compute_config', {})
    self.batch_size = compute_config.get("local_batch_size", 128)

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor | None,
  ) -> dict[str, np.ndarray]:
    """Compute localization scores for a batch of examples.

    Args:
        keys: List of example keys
        examples: Batch of input examples (dict of tensors)
        attributions: Corresponding attributions for the examples (tensor)
        original_probs: Original model pred probs for the examples (tensor)
    Returns:
        Dictionary containing the computed localization metrics for the batch
    """
    attributions = attributions.abs().sum(dim=1)
    flat_attributions = attributions.flatten(start_dim=1)
    flat_mask = examples["mask"].flatten(start_dim=1)

    flat_mask, valid_mask_filter = binarize_and_filter_masks(
        flat_mask, self.dataset_name)
    skip_keys = [
        key for key, valid in zip(keys, valid_mask_filter) if not valid
    ]
    if len(skip_keys) > 0:
      logger.warning(
          f"Skipping {len(skip_keys)} examples with invalid masks: "
          f"{skip_keys}"
      )

    valid_flat_attributions = flat_attributions[valid_mask_filter]
    valid_masks = flat_mask[valid_mask_filter]
    valid_binary_masks = (valid_masks > 0).float()

    # map attributions to [0, 1] for thresholding
    mapped_attributions = map_attr_values_to_linear_space(
        valid_flat_attributions)

    # Store pixel counts for each threshold
    tp_curve = []
    tn_curve = []
    fp_curve = []
    fn_curve = []

    for threshold in torch.linspace(0, 1, 21):
      attribution_binary_masks = (
          (mapped_attributions >= threshold).float()
      )
      true_positive_pixels = (attribution_binary_masks * valid_binary_masks)
      false_positive_pixels = (
          attribution_binary_masks * (1 - valid_binary_masks)
      )
      false_negative_pixels = (
          (1 - attribution_binary_masks) * valid_binary_masks
      )
      true_negative_pixels = (
          (1 - attribution_binary_masks) * (1 - valid_binary_masks)
      )

      # Sum over pixels, keeping batch dimension
      tp_curve.append(true_positive_pixels.sum(dim=1))
      tn_curve.append(true_negative_pixels.sum(dim=1))
      fp_curve.append(false_positive_pixels.sum(dim=1))
      fn_curve.append(false_negative_pixels.sum(dim=1))

    # Stack to create tensors of shape (valid_batch_size, 21)
    tp_curve = torch.stack(tp_curve, dim=1)
    tn_curve = torch.stack(tn_curve, dim=1)
    fp_curve = torch.stack(fp_curve, dim=1)
    fn_curve = torch.stack(fn_curve, dim=1)

    # Pad with nan for invalid masks to match full batch size
    tp_curve_full = torch.full(
        (len(keys), 21),
        float('nan'), device=tp_curve.device
    )
    tn_curve_full = torch.full(
        (len(keys), 21),
        float('nan'), device=tn_curve.device
    )
    fp_curve_full = torch.full(
        (len(keys), 21),
        float('nan'), device=fp_curve.device
    )
    fn_curve_full = torch.full(
        (len(keys), 21),
        float('nan'), device=fn_curve.device
    )
    tp_curve_full[valid_mask_filter] = tp_curve
    tn_curve_full[valid_mask_filter] = tn_curve
    fp_curve_full[valid_mask_filter] = fp_curve
    fn_curve_full[valid_mask_filter] = fn_curve
    tp = tp_curve_full.detach().cpu().numpy()
    tn = tn_curve_full.detach().cpu().numpy()
    fp = fp_curve_full.detach().cpu().numpy()
    fn = fn_curve_full.detach().cpu().numpy()
    total_pixels = tp + tn + fp + fn
    pixel_accuracy = (tp + tn) / total_pixels
    recall = tp / (tp + fn)
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)

    # Handle empty predictions and invalid masks.
    pixel_f1 = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall), 0.0)
    iou = tp / (tp + fp + fn)
    thresholds = np.linspace(0, 1, 21)
    indices = np.where((thresholds >= 0.05) & (thresholds <= 0.95))[0]
    return {
        name: np.mean(values[:, indices], axis=1)
        for name, values in (('pixel_accuracy', pixel_accuracy),
                             ('pixel_f1', pixel_f1), ('iou', iou))
    }
