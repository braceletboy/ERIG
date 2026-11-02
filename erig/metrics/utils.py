'''
Utility functions for computing metrics.
'''

from __future__ import annotations


import torch


def compute_threshold(attributions: torch.Tensor, percentage: float) -> torch.Tensor:
  """Find an attribution-mass threshold separately for each example.

  Args:
      attributions: Tensor with examples along the first dimension.
      percentage: Percentage of total attribution mass to retain.

  Returns:
      One threshold per example.

  Raises:
      ValueError: If percentage is outside [0, 100].
  """
  if percentage < 0 or percentage > 100:
    raise ValueError('percentage must be in [0, 100]')

  batch_size = attributions.shape[0]
  # Flatten all dimensions except batch
  flat_shape = (batch_size, -1)
  flat_attributions = attributions.reshape(flat_shape)

  # For percentage == 100, return min values per batch
  if percentage == 100:
    thresholds = torch.min(flat_attributions, dim=1)[0]
    return thresholds

  # Compute sum per batch item
  attribution_sum = torch.sum(flat_attributions, dim=1, keepdim=True)

  # Sort attributions per batch item (descending)
  sorted_attributions, _ = torch.sort(
      torch.abs(flat_attributions), dim=1, descending=True)

  # Compute normalized cumulative sum per batch
  cum_sum = 100.0 * torch.cumsum(sorted_attributions, dim=1) / attribution_sum

  # Find first index where cum_sum >= percentage for each batch item
  mask = cum_sum >= percentage
  # For each batch item, find the first True value
  threshold_indices = torch.argmax(mask.float(), dim=1)

  # Gather the threshold values
  thresholds = torch.gather(sorted_attributions, 1,
                            threshold_indices.unsqueeze(1)).squeeze(1)

  return thresholds


def map_attr_values_to_linear_space(
    attributions: torch.Tensor,
    clip_below_percentile: float = 5.0,
) -> torch.Tensor:
  """Rescale attribution maps for visualization.

  Args:
      attributions: Tensor with examples along the first dimension.
      clip_below_percentile: The metric's fixed lower clipping setting.

  Returns:
      A tensor with the same shape, mapped to [0, 1].
  """
  m = compute_threshold(attributions, percentage=2.0)
  e = compute_threshold(attributions, percentage=100-clip_below_percentile)

  # Reshape m and e to enable broadcasting: (batch_size, 1, 1, ...)
  batch_size = attributions.shape[0]
  view_shape = [batch_size] + [1] * (attributions.ndim - 1)
  m = m.view(*view_shape)
  e = e.view(*view_shape)

  transformed = (torch.abs(attributions) - e) / (m - e + 1e-10)

  # Recover the original sign of the attributions
  transformed *= torch.sign(attributions)

  # Preserve the sign and clipping order of the research implementation.
  transformed *= (transformed >= 0.0)

  # Clip values above and below
  transformed = torch.clamp(transformed, 0.0, 1.0)

  return transformed
