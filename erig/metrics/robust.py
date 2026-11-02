"""Attribution stability under sampled input perturbations.

The metrics measure how much an attribution changes when the input changes.

References:

- Local Lipschitz Estimate: Alvarez-Melis, D., & Jaakkola, T. S. (2018).
                            On the robustness of interpretability methods.
                            ICML Workshop on Human Interpretability in Machine
                            Learning (WHI 2018)
- RIS: Agarwal C, Johnson N, Pawelczyk M, Krishna S, Saxena E, Zitnik M,
            Lakkaraju H. Rethinking stability for attribution-based
            explanations. ICLR 2022.
- Max-Sensitivity - Yeh CK, Hsieh CY, Suggala A, Inouye DI,
                                      Ravikumar PK. On the (in) fidelity and
                                      sensitivity of explanations. Advances in
                                      neural information processing systems.
                                      2019;32.
"""

from __future__ import annotations

from typing import Any

import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from erig.metrics.metric import Metric
from erig.methods.utils import index_into_collated_batch

logger = logging.getLogger("rich")


class PerturbationDataset(Dataset):
  """Generate perturbed input examples."""

  def __init__(
      self,
      num_perturbations: int,
      example: dict[str, Any],
      input_key: str,
      seed: int = 0,
  ) -> None:
    """
    Initialize the PerturbationDataset.

    Args:
        num_perturbations: Number of perturbations to generate
        example: Input example (dict of tensors)
        input_key: Key to access the input tensor in examples
        seed: Base seed; perturbation idx uses seed + idx
    """
    self.input_key = input_key
    self.num_perturbations = num_perturbations
    self.example = example
    self.seed = seed

  def __len__(self) -> int:
    """Return the number of sampled perturbations."""
    return self.num_perturbations

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Return a reproducibly perturbed copy of the input example."""
    perturbed_example = self.example.copy()
    # Evaluate every perturbation, including prediction-changing ones.
    perturbed_example.pop("label", None)
    input_ = perturbed_example[self.input_key]
    # Deterministic per index: the metric reads each perturbation twice
    # (once for the perturbed attributions, once for the input deltas) and
    # both reads must see the same noise
    generator = torch.Generator()
    generator.manual_seed(self.seed + idx)
    noise = 2 * torch.rand(input_.shape, generator=generator) * 0.1 - 0.1
    perturbed_example[self.input_key] = input_ + noise
    return perturbed_example


class Robust(Metric):
  """Measure attribution changes under sampled input perturbations."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the Robustness class.

    Args:
        config: Configuration dictionary containing parameters
    """
    super().__init__("robust", config)
    compute_config = config.get("compute_config", {})
    self.num_perturbations = compute_config.get("robust_num_perturbations", 50)
    robust_max_examples = compute_config.get("robust_max_examples", -1)
    if ((0 <= robust_max_examples) and
            (robust_max_examples <= len(self.dataset.keys))):
      # Limit to a random subset of examples if max_examples is set
      self.dataset.keys = list(sorted(random.sample(
          self.dataset.keys,
          min(robust_max_examples, len(self.dataset.keys))
      ), key=lambda x: int(x.split('_')[-1])))
      logger.info(f"[bold red]Finally limiting to {len(self.dataset.keys)} "
                  f"examples for robust metric(s)[/bold red]")

    self.method_name, self.method = self._load_method()

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_predictions: torch.Tensor,
  ) -> dict[str, np.ndarray]:
    """Compute sensitivity, local Lipschitz, and input stability scores.

    Args:
        keys: List of example keys
        examples: Batch of input examples (dict of tensors)
        attributions: Corresponding attributions for the examples (tensor)
    Returns:
        Dict of robustness scores
    """
    start_idx = keys[0].split('_')[-1]
    end_idx = keys[-1].split('_')[-1]

    # move to CPU to save GPU memory
    examples = {k: v.cpu() if torch.is_tensor(v) else v
                for k, v in examples.items()}
    torch.cuda.empty_cache()
    attributions = attributions.cpu()

    # Compute robustness metrics for each example in the batch
    bsz = attributions.shape[0]
    local_lipschitz_scores = []
    ris_scores = []
    max_sensitivity_scores = []
    batch_bar = self.progress.add_task(
        description=f"[green]Computing Robustness - Examples {start_idx} "
        f"to {end_idx}",
        total=bsz
    )
    for i in range(bsz):
      example_i = index_into_collated_batch(examples, i)

      # Compute perturbation values; seed by example index so streams are
      # unique per example and reproducible across runs
      perturbation_dataset = PerturbationDataset(
          self.num_perturbations, example_i, self.input_key,
          seed=int(keys[i].split('_')[-1]) * self.num_perturbations
      )
      perturbation_outputs = self.method.compute_attributions_for_dataset(
          perturbation_dataset, skip_transform=True, skip_progress_init=True
      )
      perturbation_attributions = torch.stack(
          [out["attributions"] for out in perturbation_outputs], dim=0)
      orig_attr = attributions[i]
      orig_input = example_i[self.input_key]

      # Compute differences
      delta_attr = perturbation_attributions - orig_attr.unsqueeze(0)
      delta_input = torch.stack(
          [
              perturbation_dataset[i][self.input_key]
              for i in range(len(perturbation_outputs))
          ],
          dim=0
      ) - example_i[self.input_key].unsqueeze(0)

      # Local Lipschitz Estimate
      attr_norm = torch.linalg.vector_norm(
          delta_attr.reshape(delta_attr.shape[0], -1), ord=2, dim=-1
      )
      input_norm = torch.linalg.vector_norm(
          delta_input.reshape(delta_input.shape[0], -1), ord=2, dim=1
      )
      norm_ratios = attr_norm / (input_norm + 1e-15)
      local_lipschitz = torch.max(norm_ratios).item()
      local_lipschitz_scores.append(local_lipschitz)

      # Relative Input Stability
      attr_ratio = attr_norm / (torch.linalg.vector_norm(
          orig_attr.reshape(1, -1), ord=2, dim=-1
      ) + 1e-15)
      input_ratio = input_norm / (torch.linalg.vector_norm(
          orig_input.reshape(1, -1), ord=2, dim=-1
      ) + 1e-15)
      ris_ratios = attr_ratio / (input_ratio + 1e-15)
      ris = torch.max(ris_ratios).item()

      ris_scores.append(ris)

      # Max-Sensitivity
      max_sensitivity = torch.max(attr_norm).item()
      max_sensitivity_scores.append(max_sensitivity)
      self.progress.update(batch_bar, advance=1)
    self.progress.remove_task(batch_bar)

    return {
        "local_lipschitz": np.array(local_lipschitz_scores),
        "ris": np.array(ris_scores),
        "max_sensitivity": np.array(max_sensitivity_scores),
    }
