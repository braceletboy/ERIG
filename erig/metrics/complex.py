"""Concentration, gray-level entropy, and compressibility of attribution maps."""

from __future__ import annotations

from typing import Any
import math
import cv2
import numpy as np
import torch
from erig.metrics.metric import Metric
from erig.metrics.utils import map_attr_values_to_linear_space


def _batch_percentile(sorted_flat: torch.Tensor, percentile: float) -> torch.Tensor:
  """Find a percentile in each sorted row.

  Args:
      sorted_flat: Ascending-sorted values shaped (batch, pixels).
      percentile: Position in the distribution, expressed from 0 to 100.

  Returns:
      One interpolated value per row.
  """
  position = percentile / 100.0 * (sorted_flat.size(1) - 1)
  lo, hi = math.floor(position), math.ceil(position)
  return (sorted_flat[:, lo]
          + (sorted_flat[:, hi] - sorted_flat[:, lo]) * (position - lo))


def percentile_clip_to_uint8(values: torch.Tensor) -> torch.Tensor:
  """Render attribution maps as grayscale images.

  Args:
      values: Maps with examples along the first dimension.

  Returns:
      A uint8 tensor with the same shape and device as the input.
  """
  arr = torch.nan_to_num(values.double(), nan=0.0, posinf=0.0, neginf=0.0)
  sorted_flat = arr.flatten(start_dim=1).sort(dim=1).values
  view = (-1,) + (1,) * (arr.dim() - 1)
  lower = _batch_percentile(sorted_flat, 0.5).view(view)
  upper = _batch_percentile(sorted_flat, 99.5).view(view)
  scale = torch.where(upper > lower, 1.0 / (upper - lower),
                      torch.zeros_like(upper))
  unit = (arr.clamp(min=lower, max=upper) - lower) * scale  # [0, 1]
  return torch.round(unit * 255.0).clamp(0, 255).to(torch.uint8)


def batch_histogram_entropy(attributions: torch.Tensor) -> torch.Tensor:
  """Measure how varied the gray levels are in each attribution image.

  Args:
      attributions: Floating-point attribution tensor shaped (batch, C, H, W).

  Returns:
      One float64 entropy in bits per example, on the input device.

  References:
      Apóstolo D. et al. pycol-vis: Python Image Complexity Library.
      https://github.com/DiogoApostolo/pycol-vis
  """
  maps_u8 = percentile_clip_to_uint8(attributions.sum(dim=1))
  values = maps_u8.flatten(start_dim=1)
  histogram = torch.zeros(
      values.size(0), 256, dtype=torch.float64, device=values.device)
  histogram.scatter_add_(
      1, values.long(), torch.ones(values.shape, dtype=torch.float64,
                                   device=values.device))
  probs = histogram / histogram.sum(dim=1, keepdim=True)
  terms = torch.where(probs > 0, probs * torch.log2(probs),
                      torch.zeros_like(probs))
  return -terms.sum(dim=1)


def batch_bhatt_entropy(attributions: torch.Tensor) -> torch.Tensor:
  """Measure how widely attribution mass is spread across pixels.

  Args:
      attributions: Floating-point attribution tensor shaped (batch, C, H, W).

  Returns:
      One float64 normalized entropy per example, on the input device.

  References:
      Bhatt U., Weller A., Moura J. M. F. Evaluating and Aggregating
      Feature-based Model Explanations. IJCAI 2020, Section 3.3.
      https://doi.org/10.24963/ijcai.2020/417
  """
  spatial = attributions.sum(dim=1)
  finite = torch.isfinite(spatial).flatten(1).all(dim=1)
  flat_u8 = percentile_clip_to_uint8(spatial).flatten(1)
  weights = flat_u8.double()
  log_n = math.log(flat_u8.size(1))
  nan = torch.full_like(finite, float("nan"), dtype=torch.float64)
  total = weights.sum(dim=1, keepdim=True)
  probs = torch.where(total > 0, weights / total, torch.zeros_like(weights))
  terms = torch.where(probs > 0, probs * torch.log(probs),
                      torch.zeros_like(probs))
  entropy_nats = -terms.sum(dim=1)
  bad = ~finite | (total.squeeze(1) == 0)
  entropy_nats = torch.where(bad, nan, entropy_nats)
  return entropy_nats / log_n


def jpeg_metrics(attributions: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
  """Measure JPEG compressibility and reconstruction error.

  Args:
      attributions: Floating-point attribution tensor shaped (batch, C, H, W).

  Returns:
      Two float64 NumPy arrays, each with one value per example: encoded/raw
      byte ratio and reconstruction RMSE in gray-level units.

  Raises:
      RuntimeError: If JPEG encoding fails.

  References:
      Apóstolo D. et al. pycol-vis: Python Image Complexity Library.
      https://github.com/DiogoApostolo/pycol-vis
  """
  maps_u8 = percentile_clip_to_uint8(attributions.sum(dim=1)).cpu().numpy()
  ratios = []
  errors = []
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
  for map_u8 in maps_u8:
    ok, encoded = cv2.imencode(".jpg", map_u8, encode_param)
    if not ok:
      raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    ratio = len(encoded) / (map_u8.size * map_u8.itemsize)
    rmse = float(np.sqrt(np.mean(
        (map_u8.astype(np.float32) - decoded.astype(np.float32)) ** 2)))
    ratios.append(float(ratio))
    errors.append(rmse)
  return np.array(ratios), np.array(errors)


def batch_gini(attributions: torch.Tensor) -> torch.Tensor:
  """Measure visual sparsity of attribution maps.

  Args:
      attributions: Floating-point attribution tensor shaped (batch, C, H, W).

  Returns:
      One Gini coefficient per example, with the input dtype and device.

  References:
      Chalasani P. et al. Concise Explanations of Neural Networks using
      Adversarial Training. ICML 2020, Section 6.2.
      https://proceedings.mlr.press/v119/chalasani20a.html
  """
  visualization = map_attr_values_to_linear_space(
      attributions.abs().sum(dim=1, keepdim=True), clip_below_percentile=2.0)
  values = visualization.flatten(1)
  sorted_values, _ = torch.sort(torch.abs(values), dim=1)
  n = torch.tensor(values.size(1), dtype=values.dtype, device=values.device)
  indices = torch.arange(1, values.size(1) + 1, dtype=values.dtype,
                         device=values.device)
  numerator = torch.sum(
      (2 * indices.unsqueeze(0) - n - 1) * sorted_values, dim=1)
  denominator = n * torch.sum(sorted_values, dim=1)
  return numerator / (denominator + 1e-30)


class Complexity(Metric):
  """Compute and collect the concentration measures for saved attributions."""
  uses_model = False

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__('complex', config)

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor | None,
  ) -> dict[str, np.ndarray]:
    jpeg_ratio, jpeg_rmse = jpeg_metrics(attributions)
    return {
        'entropy_complexity': batch_bhatt_entropy(attributions).cpu().numpy(),
        'histogram_entropy': batch_histogram_entropy(attributions).cpu().numpy(),
        'jpeg_ratio': jpeg_ratio,
        'jpeg_rmse': jpeg_rmse,
        'gini_sparsity': batch_gini(attributions).cpu().numpy(),
    }
