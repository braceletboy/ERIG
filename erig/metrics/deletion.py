"""Iterative Removal of Features (DEL).

Measure how model confidence changes as important image regions are removed.
"""

from __future__ import annotations

from typing import Any

import logging
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.ndimage import mean as ndimage_mean
from skimage.segmentation import slic

from erig.metrics.metric import Metric


logger = logging.getLogger("rich")

EPS = 1e-10


def _slic_single(img_np: np.ndarray) -> np.ndarray:
  """Segment one image into superpixels."""
  img_hwc = np.transpose(img_np, (1, 2, 0))
  return slic(img_hwc, n_segments=50, compactness=10.0,
              channel_axis=2, start_label=0)


class DEL(Metric):
  """Measure area under the confidence curve as image regions are removed."""

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__("del", config)
    compute_config = config.get("compute_config", {})
    self.num_workers = compute_config.get(
        "del_num_workers", min(os.cpu_count(), 32))

  def _compute_slic_segments(
      self,
      images: torch.Tensor,
  ) -> tuple[list[np.ndarray], list[int]]:
    """Return superpixel label maps and segment counts for each image."""
    images_np = images.detach().cpu().numpy()
    with Pool(self.num_workers) as pool:
      results = list(pool.map(_slic_single, images_np))
    n_segments_list = [int(seg.max()) + 1 for seg in results]
    return results, n_segments_list

  def _compute_segment_importance(
      self,
      processed_attrs: torch.Tensor,
      segments_list: list[np.ndarray],
      n_segments_list: list[int],
  ) -> list[np.ndarray]:
    """Rank each image's segments by decreasing mean attribution."""
    attr_spatial = processed_attrs.squeeze(1).detach().cpu().numpy()
    sorted_segments = []
    for i in range(attr_spatial.shape[0]):
      n_seg = n_segments_list[i]
      importance = ndimage_mean(
          attr_spatial[i], labels=segments_list[i], index=np.arange(n_seg))
      importance = np.array(importance)
      sorted_segments.append(np.argsort(-importance))
    return sorted_segments

  def _build_segment_mean_baseline(
      self,
      images: torch.Tensor,
      segments_list: list[np.ndarray],
      n_segments_list: list[int],
  ) -> torch.Tensor:
    """Replace each segment with its mean color to form the deletion baseline."""
    images_np = images.detach().cpu().numpy()
    B, C, H, W = images_np.shape
    baseline_np = np.empty_like(images_np)
    for i in range(B):
      seg = segments_list[i]
      n_seg = n_segments_list[i]
      seg_indices = np.arange(n_seg)
      for c in range(C):
        channel = images_np[i, c]
        seg_means = ndimage_mean(channel, labels=seg, index=seg_indices)
        seg_means = np.array(seg_means)
        baseline_np[i, c] = seg_means[seg]
    return torch.from_numpy(baseline_np).to(images.device, dtype=images.dtype)

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor,
  ) -> dict[str, list[float | np.floating[Any]]]:

    pred_class = torch.argmax(original_probs, dim=-1, keepdim=True)
    original_scores = original_probs.gather(-1, pred_class).squeeze(-1)

    start_idx = keys[0].split('_')[-1]
    end_idx = keys[-1].split('_')[-1]
    bar_label = f"DEL {start_idx}-{end_idx}"

    processed = attributions.sum(dim=1, keepdim=True)

    images = examples[self.input_key]
    B, C, H, W = images.shape

    segments_list, n_segments_list = self._compute_slic_segments(images)
    sorted_segments = self._compute_segment_importance(
        processed, segments_list, n_segments_list)
    baseline = self._build_segment_mean_baseline(
        images, segments_list, n_segments_list)

    seg_tensors = [torch.from_numpy(seg.astype(np.int64)).to(images.device)
                   for seg in segments_list]

    max_segments = max(n_segments_list)

    del_bar = self.progress.add_task(
        f"[green]{bar_label}",
        total=max_segments
    )

    scores = [original_scores.cpu()]
    cumulative_masks = torch.zeros(B, H, W, dtype=torch.bool,
                                   device=images.device)

    for step in range(max_segments):
      modified = images.clone()

      # delete the next segment
      for i in range(B):
        if step < n_segments_list[i]:
          s = sorted_segments[i][step]
          cumulative_masks[i] |= (seg_tensors[i] == s)
        modified[i, :, cumulative_masks[i]] = baseline[
            i, :, cumulative_masks[i]]

      with torch.inference_mode():
        logits = self.model(modified)
        modified_probs = torch.softmax(logits, dim=-1)

      modified_scores = modified_probs.gather(-1, pred_class).squeeze(-1)
      scores.append(modified_scores.cpu())
      self.progress.update(del_bar, advance=1)

    self.progress.remove_task(del_bar)

    all_scores = torch.stack(scores, dim=1).detach().cpu().numpy()
    original_scores_np = original_scores.detach().cpu().numpy()

    values = []
    for i in range(B):
      steps = n_segments_list[i]
      values.append(np.trapezoid(
          all_scores[i, :steps + 1] / (original_scores_np[i] + EPS),
          dx=1 / steps))
    return {'deletion_auc': values}
