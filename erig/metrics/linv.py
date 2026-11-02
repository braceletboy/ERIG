"""Log-Inverse Noise Variance (LINV) for attribution maps.

Higher values indicate lower estimated noise variance. Noise reduction alone
does not establish faithfulness or localization accuracy.

References:
- Immerkaer J. Fast noise variance estimation. Computer vision and image
  understanding. 1996 Sep 1;64(2):300-2.
- Chen G, Zhu F, Ann Heng P. An efficient statistical method for image noise
  level estimation. In Proceedings of the IEEE international conference on
  computer vision 2015 (pp. 477-485).
"""

from __future__ import annotations

from typing import Any

import logging

import numpy as np
import torch
import torch.nn.functional as F

from erig.metrics.metric import Metric


logger = logging.getLogger("rich")


class LINV(Metric):
  """Estimate spatial noise in attribution maps."""

  # Pure attribution-map computation: the model is never evaluated.
  uses_model = False

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the LINV class.

    Args:
        config: Configuration dictionary containing parameters
    """
    super().__init__("linv", config)
    compute_config = config.get("compute_config", {})
    self.batch_size = compute_config.get("linv_batch_size", 512)

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_probs: torch.Tensor | None,
  ) -> dict[str, np.ndarray]:
    """Compute convolution and PCA noise scores for a batch of examples.

    Args:
        keys: List of example keys
        examples: Batch of input examples (dict of tensors)
        attributions: Corresponding attributions for the examples (tensor)
        original_probs: Original model pred probs for the examples (tensor)
    Returns:
        Dict of LINV scores
    """
    results = {}

    num_channels = attributions.size(1)
    noise_estimation_kernel = torch.tensor(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=attributions.dtype,
        device=attributions.device
        # Shape: (C, 1, 3, 3)
    ).unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

    # Compute conv variant
    estimated_noise = F.conv2d(
        attributions, noise_estimation_kernel.flip(2, 3), groups=num_channels)
    avg_noise_variance = 1/36 * torch.mean(estimated_noise**2, dim=(1, 2, 3))
    linv_conv = -1 * torch.log10(avg_noise_variance + 1e-30)
    results["linv_conv"] = linv_conv.cpu().numpy()

    # Compute PCA variant
    patch_size = 8
    patches = attributions.unfold(
        2, patch_size, 1).unfold(3, patch_size, 1)
    patches = patches.contiguous().view(attributions.size(
        0), attributions.size(1), -1, patch_size, patch_size)
    patches = patches.view(attributions.size(0), -1, patch_size**2)
    patches = patches.permute(0, 2, 1)  # (bsz, patch_size^2, num_patches)

    patches_centered = patches - patches.mean(dim=-1, keepdim=True)
    cov_matrix = torch.matmul(
        patches_centered, patches_centered.transpose(-2, -1)
    ) / patches.size(-1)
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    eigenvals, _ = eigenvals.sort(dim=-1, descending=True)

    batch_size = eigenvals.size(0)
    tau_values = -1 * torch.ones(
        batch_size, device=eigenvals.device, dtype=eigenvals.dtype)

    for k in range(eigenvals.size(1)):
      tau = torch.mean(eigenvals[:, k:], dim=-1)  # Shape: (batch_size)

      below_tau = torch.sum(eigenvals[:, k:] < tau.unsqueeze(-1), dim=-1)
      above_tau = torch.sum(eigenvals[:, k:] > tau.unsqueeze(-1), dim=-1)

      median_condition = (below_tau == above_tau)

      # Update tau_values for elements that satisfy condition and
      # haven't been set yet
      mask = median_condition & (tau_values == -1)
      tau_values[mask] = tau[mask]

      # If all batch elements have found their tau, break
      if torch.all(tau_values > 0):
        break

    # For any remaining elements that didn't satisfy condition, raise error
    remaining_mask = (tau_values == -1)
    if torch.any(remaining_mask):
      raise ValueError(
          "Some batch elements did not satisfy the median condition."
          "This shouldn't happen with typical attribution maps."
      )

    linv_pca = -1 * torch.log10(tau_values + 1e-30).cpu().numpy()
    results["linv_pca"] = linv_pca

    return results
