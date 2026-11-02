"""Signed ERIG: Entropy-Regularized Integrated Gradients with per-feature sign.

ERIG (``erig.methods.erig``) weights the path integrand with the maximum-entropy
softmax over squared path gradients, w*_i = softmax(g^2 / tau)_i, so the
per-feature sign of the gradient is discarded inside the weighting; the sign
of an attribution comes only from the signed output increments along the
path. This variant multiplies each weight by the gradient sign,

    w~_i = w*_i * sign(g_i),

which keeps the magnitude profile of the weights (features with a zero
gradient get weight 0 instead of the softmax floor). Path, baseline,
and output-difference scaling are inherited unchanged from
ERIG. The configuration keys are ``serig_*`` with the same meaning as ``erig_*``
(registry name ``serig``).
"""

from __future__ import annotations

from typing import Any


import torch

from erig.methods.erig import EntropyRegularizedIntegratedGradients


class SignedERIG(EntropyRegularizedIntegratedGradients):
  """ERIG whose sensitivity weights carry the sign of the path gradient."""

  def __init__(self, config: dict[str, Any]) -> None:
    super().__init__(config, name="serig")

  @staticmethod
  def _sensitivity_weights(
      gradients: torch.Tensor,
      temperature: float | torch.Tensor,
      dim: int = -1,
  ) -> torch.Tensor:
    """Multiply ERIG's sensitivity weights by the sign of the raw gradient."""
    weights = EntropyRegularizedIntegratedGradients._sensitivity_weights(
        gradients, temperature, dim)
    return weights * torch.sign(gradients)
