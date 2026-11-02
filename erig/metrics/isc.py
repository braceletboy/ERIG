"""Area under the insertion score curve."""

from __future__ import annotations

from typing import Any

import logging

import numpy as np
import torch

from erig.metrics.metric import Metric
from erig.methods.utils import get_baseline


logger = logging.getLogger("rich")


class ISC(Metric):
  """Measure model scores as features are revealed in attribution order."""

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__("isc", config)
    compute_config = config.get("compute_config", {})
    self.num_partitions = compute_config.get("isc_num_partitions", 10)

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor,
  ) -> dict[str, np.ndarray]:

    pred_class = torch.argmax(original_probs, dim=-1, keepdim=True)
    original_scores = original_probs.gather(-1, pred_class).squeeze(-1)

    start_idx = keys[0].split('_')[-1]
    end_idx = keys[-1].split('_')[-1]
    bar_label = f"ISC {start_idx}-{end_idx}"

    sorted_indices = torch.argsort(
        attributions.reshape(attributions.shape[0], -1), descending=True)

    isc_baseline = get_baseline(examples[self.input_key], "zero")
    flattened_input = torch.flatten(examples[self.input_key], 1)
    total_positions = sorted_indices.size(1)

    isc_bar = self.progress.add_task(
        f"[green]{bar_label}",
        total=self.num_partitions
    )

    scores = []
    for keep_percent in torch.linspace(0, 100, self.num_partitions+1)[:-1]:
      num_to_keep = int((keep_percent / 100) * total_positions)

      indices_to_keep = sorted_indices[:, :num_to_keep]

      modified_inputs = isc_baseline.clone()
      flattened_modified = torch.flatten(modified_inputs, 1)
      flattened_modified.scatter_(
          1, indices_to_keep,
          flattened_input.gather(1, indices_to_keep)
      )
      modified_inputs = flattened_modified.view_as(modified_inputs)
      with torch.inference_mode():
        logits = self.model(modified_inputs)
        modified_probs = torch.softmax(logits, dim=-1)
      modified_scores = modified_probs.gather(-1, pred_class).squeeze(-1)
      scores.append(modified_scores.cpu())
      self.progress.update(isc_bar, advance=1)

    self.progress.remove_task(isc_bar)
    scores.append(original_scores.cpu())

    scores = torch.stack(scores, dim=1)

    isc_val = torch.trapezoid(scores, dx=1/self.num_partitions, dim=1)
    return {"isc": isc_val.detach().cpu().numpy()}
