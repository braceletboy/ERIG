"""Accuracy Information Curve (AIC).

Measure prediction accuracy as attribution-selected features reveal more
image information.
"""

from __future__ import annotations

from typing import Any

import logging

import torch
import numpy as np
from scipy.interpolate import interp1d
from pytorch_msssim import MS_SSIM

from erig.metrics.metric import Metric
from erig.methods.utils import get_baseline

logger = logging.getLogger("rich")


class AIC(Metric):
  """Measure prediction accuracy as image information is revealed."""

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__("aic", config)
    compute_config = config.get("compute_config", {})
    self.num_partitions = compute_config.get("aic_num_partitions", 10)

    self.ms_ssim = MS_SSIM(data_range=1.0, size_average=False, channel=3)

  def _compute_ms_ssim(
      self, img1: torch.Tensor, img2: torch.Tensor
  ) -> torch.Tensor:
    """Measure image similarity with MS-SSIM."""
    # normalize to [0, 1] because ssim/ms_ssim expect images in this range
    batch_size = img1.shape[0]
    img1_flat = img1.view(batch_size, -1)
    img2_flat = img2.view(batch_size, -1)

    img1_min = img1_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    img1_max = img1_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    img2_min = img2_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    img2_max = img2_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)

    img1 = (img1 - img1_min) / (img1_max - img1_min + 1e-30)
    img2 = (img2 - img2_min) / (img2_max - img2_min + 1e-30)
    return self.ms_ssim(img1, img2)

  def compute_aggregate_metrics(self) -> dict[str, float | np.floating[Any]]:
    """Summarize accuracy information curves across examples."""
    curve_x = np.linspace(0.0, 1.0, num=101, endpoint=True)
    curves = np.stack(self.aggregate_metrics['aic_curve'], axis=0)
    curve_mean = np.nanmean(curves, axis=0)
    return {'aic': np.trapezoid(curve_mean, curve_x)}

  def _compute_aic_curves(
      self,
      aic_scores: torch.Tensor,
      x_metrics: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-example accuracy information curves and their areas."""
    blur_aic_scores = aic_scores[:, 0]
    blur_x_metrics = x_metrics[:, 0]
    original_aic_scores = aic_scores[:, -1]
    original_x_metrics = x_metrics[:, -1]
    keep_samples = original_x_metrics > blur_x_metrics
    keep_samples_aic = keep_samples & (original_aic_scores > blur_aic_scores)
    if keep_samples_aic.sum() < keep_samples_aic.size(0):
      num_excluded = keep_samples_aic.size(0) - keep_samples_aic.sum()
      logger.warning(
          f"Excluding {num_excluded} samples from AIC metric computation due to"
          " higher accuracy/MS-SSIM for blurred input than original input.")

    keep_aic_scores = aic_scores[keep_samples_aic, :]
    keep_x_metrics_aic = x_metrics[keep_samples_aic, :]
    keep_aic_scores, _ = torch.cummax(keep_aic_scores, dim=1)
    keep_normalized_aic_scores = (
        keep_aic_scores - keep_aic_scores[:, 0:1]
    ) / (keep_aic_scores[:, -1:] - keep_aic_scores[:, 0:1] + 1e-30)
    keep_normalized_aic_scores = torch.clamp(
        keep_normalized_aic_scores, 0.0, 1.0)
    keep_normalized_x_metrics_aic = (
        keep_x_metrics_aic - keep_x_metrics_aic[:, 0:1]
    ) / (keep_x_metrics_aic[:, -1:] - keep_x_metrics_aic[:, 0:1] + 1e-30)
    keep_normalized_x_metrics_aic = torch.clamp(
        keep_normalized_x_metrics_aic, 0.0, 1.0)

    keep_curve_y_aic = torch.zeros(
        keep_samples_aic.sum(), 101,
        device=keep_aic_scores.device, dtype=keep_aic_scores.dtype)
    curve_x = np.linspace(
        start=0.0, stop=1.0, num=100, endpoint=False)
    for i in range(keep_aic_scores.size(0)):
      interp_func = interp1d(
          keep_normalized_x_metrics_aic[i].detach().cpu().numpy(),
          keep_normalized_aic_scores[i].detach().cpu().numpy())
      keep_curve_y_aic[i] = torch.tensor(
          np.append(interp_func(curve_x), 1.0),
          device=keep_aic_scores.device, dtype=keep_aic_scores.dtype)
    curve_x = torch.tensor(
        np.append(curve_x, 1.0), device=keep_aic_scores.device,
        dtype=keep_aic_scores.dtype)
    curve_y_aic = torch.zeros(
        aic_scores.size(0), 101,
        device=keep_curve_y_aic.device, dtype=keep_curve_y_aic.dtype)
    curve_y_aic[keep_samples_aic, :] = keep_curve_y_aic
    # A non-improving accuracy curve contributes zero; invalid information
    # ranges contribute NaN. Keep this distinction in the dataset aggregate.
    curve_y_aic[~keep_samples, :] = torch.nan
    keep_aic = torch.trapezoid(keep_curve_y_aic, curve_x.unsqueeze(0), dim=1)
    aic = torch.zeros(aic_scores.size(
        0), device=keep_aic.device, dtype=keep_aic.dtype)
    aic[keep_samples_aic] = keep_aic
    aic[~keep_samples] = torch.nan
    return curve_y_aic, aic

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor,
  ) -> dict[str, np.ndarray]:
    """Compute per-example AIC values and curves needed for aggregation."""

    images = examples[self.input_key]
    original_ms_ssim = torch.ones(images.size(
        0), device=images.device, dtype=images.dtype)
    pred_class = torch.argmax(original_probs, dim=-1, keepdim=True)
    original_scores = original_probs.gather(-1, pred_class).squeeze(-1)
    aic_baseline = get_baseline(
        images, baseline_type="pil_blur",
        sigma=20.0, kernel_size=121
    )
    flattened_input = torch.flatten(images, 1)
    sorted_indices = torch.argsort(attributions.reshape(
        attributions.shape[0], -1), descending=True)
    total_positions = sorted_indices.size(1)
    aic_bar = self.progress.add_task(
        f"[green]AIC {keys[0]}-{keys[-1]}", total=self.num_partitions)
    aic_scores = []
    ms_ssim_list = []
    for keep_percent in torch.linspace(0, 100, self.num_partitions + 1)[:-1]:
      num_to_keep = int((keep_percent / 100) * total_positions)
      indices_to_keep = sorted_indices[:, :num_to_keep]
      bokeh_inputs = aic_baseline.clone()
      flattened_bokeh = torch.flatten(bokeh_inputs, 1)
      flattened_bokeh.scatter_(
          1, indices_to_keep, flattened_input.gather(1, indices_to_keep))
      bokeh_inputs = flattened_bokeh.view_as(bokeh_inputs)
      batch_ms_ssim = self._compute_ms_ssim(images, bokeh_inputs)
      with torch.inference_mode():
        bokeh_logits = self.model(bokeh_inputs)
        bokeh_probs = torch.softmax(bokeh_logits, dim=-1)
      bokeh_accuracies = (bokeh_probs.argmax(dim=-1) ==
                          pred_class.squeeze(-1)).float()
      aic_scores.append(bokeh_accuracies.cpu())
      ms_ssim_list.append(batch_ms_ssim.cpu())
      self.progress.advance(aic_bar, 1)
    self.progress.remove_task(aic_bar)
    aic_scores.append(1.0 * torch.ones_like(original_scores).cpu())
    ms_ssim_list.append(original_ms_ssim.cpu())
    curves, aic = self._compute_aic_curves(
        torch.stack(aic_scores, dim=1), torch.stack(ms_ssim_list, dim=1))
    return {
        'aic': aic.detach().cpu().numpy(),
        'aic_curve': curves.detach().cpu().numpy(),
    }
