"""Adaptive ERIG: Entropy-Regularized Integrated Gradients with a path-dependent
temperature tau(t).

ERIG (``erig.methods.erig``) uses same temperature tau at every point of the
path. This variant evaluates the maximum-entropy weighting at the path point
Gamma(t), t in [0, 1], with tau = tau_fn(t). ``t`` is the interpolation
coefficient k / K of the uniform path sample, exposed by AlphaPathDataset.

The schedule is any callable ``t -> tau > 0`` (pass ``tau_fn`` to the
constructor from code). For the CLI, named schedules are built from
(``tau_start``, ``tau_end``), the temperatures at t = 0 and t = 1:

    constant     tau(t) = tau_start
    linear       tau(t) = tau_start + (tau_end - tau_start) * t
    exponential  tau(t) = tau_start * (tau_end / tau_start) ** t

Configuration keys (registry name ``aerig``): ``aerig_*`` with the meaning
of ``erig_*``, plus ``aerig_tau_schedule`` / ``aerig_tau_start`` /
``aerig_tau_end``; ``aerig_exp_temperature`` is ignored (the schedule defines
the temperature).
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import math

import torch

from erig.methods.erig import (
    ERIGDataset,
    EntropyRegularizedIntegratedGradients,
)


def _constant(tau_start: float, tau_end: float) -> Callable[[float], float]:
  """Return a schedule that holds the starting temperature constant."""
  del tau_end
  return lambda t: tau_start


def _linear(tau_start: float, tau_end: float) -> Callable[[float], float]:
  """Return a schedule that interpolates linearly between the temperatures."""
  return lambda t: tau_start + (tau_end - tau_start) * t


def _exponential(tau_start: float, tau_end: float) -> Callable[[float], float]:
  """Return a schedule that interpolates exponentially between the temperatures."""
  ratio = tau_end / tau_start
  return lambda t: tau_start * math.pow(ratio, t)


# name -> factory(tau_start, tau_end) -> schedule(t)
TAU_SCHEDULES = {
    "constant": _constant,
    "linear": _linear,
    "exponential": _exponential,
}


def make_tau_schedule(
    name: str,
    tau_start: float,
    tau_end: float,
) -> Callable[[float], float]:
  """Build the named temperature schedule t in [0, 1] -> tau(t).

  Both endpoint temperatures must be positive.
  """
  if name not in TAU_SCHEDULES:
    raise ValueError(
        f"Unknown tau schedule {name!r}; available: {sorted(TAU_SCHEDULES)}")
  tau_start, tau_end = float(tau_start), float(tau_end)
  if not (math.isfinite(tau_start) and math.isfinite(tau_end)
          and tau_start > 0 and tau_end > 0):
    raise ValueError(
        "tau_start and tau_end must be positive and finite, got "
        f"{tau_start} and {tau_end}")
  return TAU_SCHEDULES[name](tau_start, tau_end)


class AlphaPathDataset(ERIGDataset):
  """Uniform path dataset that also returns the path position ``alpha``."""

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Return a path sample together with its interpolation coefficient."""
    item = super().__getitem__(idx)
    item["alpha"] = torch.tensor(idx / self.num_steps, dtype=torch.float32)
    return item


class AdaptiveERIG(EntropyRegularizedIntegratedGradients):
  """ERIG with a path-dependent temperature tau(t)."""

  path_dataset_cls = AlphaPathDataset

  def __init__(
      self,
      config: dict[str, Any],
      tau_fn: Callable[[float], float] | None = None,
  ) -> None:
    """
    Args:
        config: Configuration dictionary (``aerig_*`` keys)
        tau_fn: Optional schedule ``t -> tau``; overrides the named schedule
            from the config (kept out of the config so it stays serializable)
    """
    # Resolve the schedule first so a bad one fails before the base class
    # loads the model and dataset.
    compute_config = config["compute_config"]
    self.tau_schedule = compute_config.get("aerig_tau_schedule", "linear")
    self.tau_start = float(compute_config.get("aerig_tau_start", 3.0))
    self.tau_end = float(compute_config.get("aerig_tau_end", 3.0))
    self.tau_fn = tau_fn if tau_fn is not None else make_tau_schedule(
        self.tau_schedule, self.tau_start, self.tau_end)
    super().__init__(config, name="aerig")

  def _batch_temperature(
      self,
      batch: dict[str, Any],
      flat_grads: torch.Tensor,
  ) -> float | torch.Tensor:
    """Per-sample temperature tau(alpha), broadcastable against flat_grads."""
    alphas = batch["alpha"]
    taus = torch.tensor(
        [float(self.tau_fn(float(alpha))) for alpha in alphas.tolist()],
        dtype=flat_grads.dtype, device=flat_grads.device)
    if (not torch.isfinite(taus).all()) or (taus <= 0).any():
      raise ValueError(
          f"tau(t) must be positive and finite; got {taus.tolist()} for "
          f"t = {alphas.tolist()}")
    return taus.view(-1, *([1] * (flat_grads.dim() - 1)))
