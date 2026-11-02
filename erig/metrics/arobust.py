"""Adversarial attribution stability.

Search for input perturbations that change an explanation while preserving
the prediction. Supported methods are IG, IDGI, and ERIG.

References:

- Ghorbani A, Abid A, Zou J. Interpretation of neural networks is fragile.
  AAAI 2019.
- Dombrowski AK, Alber M, Anders C, Ackermann M, Müller KR, Kessel P.
  Explanations can be manipulated and geometry is to blame. NeurIPS 2019.
- Madry A, Makelov A, Schmidt L, Tsipras D, Vladu A. Towards deep learning
  models resistant to adversarial attacks. ICLR 2018.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Generator

import logging
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import Dataset

from erig.metrics.metric import Metric
from erig.methods.erig import EntropyRegularizedIntegratedGradients
from erig.methods.utils import get_baseline, index_into_collated_batch

logger = logging.getLogger("rich")

# Methods whose attributions are differentiable in the input
SUPPORTED_METHODS = ("ig", "idgi", "erig")


class AdversarialPerturbationDataset(Dataset):
  """Input examples with precomputed adversarial perturbations."""

  def __init__(
      self,
      example: dict[str, Any],
      input_key: str,
      deltas: torch.Tensor,
  ) -> None:
    """
    Initialize the AdversarialPerturbationDataset.

    Args:
        example: Input example (dict of tensors), unbatched
        input_key: Key to access the input tensor in examples
        deltas: Tensor of perturbations, shape [num_deltas, *input_shape]
    """
    self.example = example
    self.input_key = input_key
    self.deltas = deltas

  def __len__(self) -> int:
    """Return the number of adversarial perturbations."""
    return self.deltas.shape[0]

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Return a copy of the example with the indexed adversarial perturbation."""
    perturbed_example = self.example.copy()
    # Evaluate every perturbation, including prediction-changing ones.
    perturbed_example.pop("label", None)
    perturbed_example[self.input_key] = (
        self.example[self.input_key] + self.deltas[idx]
    )
    return perturbed_example


@contextmanager
def softplus_smoothed(
    model: torch.nn.Module, beta: float
) -> Generator[torch.nn.Module, None, None]:
  """Temporarily smooth model activations for attribution attacks.

  Args:
      model: Model whose activations should be smoothed
      beta: Softplus sharpness; larger values approximate ReLU more closely

  Yields:
      The model with its ReLU modules replaced.
  """
  replaced = []
  for module in model.modules():
    for child_name, child in module.named_children():
      if isinstance(child, nn.ReLU):
        setattr(module, child_name, nn.Softplus(beta=beta))
        replaced.append((module, child_name, child))
  if not replaced:
    logger.debug("No ReLU modules found to smooth; attacking as is")
  try:
    yield model
  finally:
    for module, child_name, child in replaced:
      setattr(module, child_name, child)


class ARobust(Metric):
  """Search for perturbations that change attributions adversarially."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the Adversarial Robustness class.

    Args:
        config: Configuration dictionary containing parameters
    """
    super().__init__("arobust", config)
    compute_config = config.get("compute_config", {})
    self.num_restarts = compute_config.get("arobust_num_restarts", 3)
    self.num_iterations = compute_config.get("arobust_num_iterations", 100)
    self.num_path_points = compute_config.get("arobust_num_path_points", 20)
    arobust_max_examples = compute_config.get("arobust_max_examples", -1)
    if (
        (0 <= arobust_max_examples) and
        (arobust_max_examples <= len(self.dataset.keys))
    ):
      # Limit to a random subset of examples if max_examples is set
      self.dataset.keys = list(sorted(random.sample(
          self.dataset.keys,
          min(arobust_max_examples, len(self.dataset.keys))
      ), key=lambda x: int(x.split('_')[-1])))
      logger.info(f"[bold red]Finally limiting to {len(self.dataset.keys)} "
                  f"examples for arobust metric(s)[/bold red]")

    self.method_name, self.method = self._load_method()
    if self.method_name not in SUPPORTED_METHODS:
      raise ValueError(
          f"Adversarial robustness is only defined for methods with "
          f"input-differentiable attributions {SUPPORTED_METHODS}; got "
          f"'{self.method_name}'. Path methods with discrete, "
          f"input-dependent path construction (e.g. gig, big) are excluded "
          f"by design."
      )
    self.baseline_type = getattr(self.method, "baseline", "zero")
    self.baseline_kwargs = getattr(self.method, "baseline_kwargs", {})
    self.num_steps = self.method.num_steps

  def _sample_path_points(
      self,
      path_alphas: torch.Tensor,
      generator: torch.Generator,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample path points for estimating the attribution.

    Args:
        path_alphas: Interpolation coefficients of the full path
        generator: Random generator used to draw the subsample

    Returns:
        Tuple of (arrival coefficients, preceding coefficients or None).
    """
    num_points = min(self.num_path_points, self.num_steps)
    # Point 0 is the baseline itself and contributes a zero increment
    indices = torch.randperm(
        self.num_steps, generator=generator, device=path_alphas.device) + 1
    indices = indices[:num_points].sort().values
    if self.method_name in ("idgi", "erig"):
      return path_alphas[indices], path_alphas[indices - 1]
    return path_alphas[indices], None

  def _surrogate_attribution(
      self,
      input_tensor: torch.Tensor,
      baseline_input: torch.Tensor,
      target_class: int,
      arrival_alphas: torch.Tensor,
      previous_alphas: torch.Tensor | None,
      create_graph: bool = True,
  ) -> torch.Tensor:
    """Estimate attributions with input derivatives available for the attack.

    Args:
        input_tensor: Input tensor of shape [1, C, H, W], carrying the graph
        baseline_input: Baseline tensor of shape [1, C, H, W]
        target_class: Class index the attributions are computed for
        arrival_alphas: Interpolation coefficients of the arrival points
        previous_alphas: Coefficients of the preceding points, or None
        create_graph: Whether to retain the graph through the path gradients,
                      which is needed to differentiate the attribution itself

    Returns:
        Attribution tensor of shape [1, C, H, W].
    """
    difference = input_tensor - baseline_input
    num_arrivals = arrival_alphas.numel()
    if previous_alphas is None:
      all_alphas = arrival_alphas
    else:
      all_alphas = torch.cat([arrival_alphas, previous_alphas], dim=0)

    # The sampled point count is unbounded configuration (doubled again by
    # output-difference pairing), so path points go through the model one
    # metric batch at a time. Cross-sample gradients are zero, so the
    # per-chunk sum over the arrival prefix gives per-point gradients.
    prediction_chunks = []
    gradient_chunks = []
    consumed = 0
    for alpha_chunk in torch.split(all_alphas, self.batch_size, dim=0):
      chunk = baseline_input + alpha_chunk.view(-1, 1, 1, 1) * difference
      if not chunk.requires_grad:
        # The path points are leaves when the input carries no perturbation
        chunk.requires_grad_(True)
      chunk_predictions = torch.softmax(
          self.model(chunk), dim=-1)[:, target_class]
      prediction_chunks.append(chunk_predictions)
      arrivals_in_chunk = min(
          max(num_arrivals - consumed, 0), chunk.shape[0])
      if arrivals_in_chunk > 0:
        gradient_chunks.append(torch.autograd.grad(
            outputs=chunk_predictions[:arrivals_in_chunk].sum(),
            inputs=chunk,
            create_graph=create_graph
        )[0][:arrivals_in_chunk])
      consumed += chunk.shape[0]
    predictions = torch.cat(prediction_chunks, dim=0)
    gradients = torch.cat(gradient_chunks, dim=0)

    if self.method_name == "ig":
      return gradients.mean(dim=0, keepdim=True) * difference

    flat_gradients = gradients.view(num_arrivals, -1)
    if self.method_name == "idgi":
      squared = flat_gradients * flat_gradients
      weights = squared / (squared.sum(dim=-1, keepdim=True) + 1e-10)
    else:
      weights = EntropyRegularizedIntegratedGradients._sensitivity_weights(
          flat_gradients, self.method.exp_temperature
      )

    if previous_alphas is not None:
      increments = predictions[:num_arrivals] - predictions[num_arrivals:]
      weights = weights * increments.unsqueeze(-1)
    return weights.view_as(gradients).sum(dim=0, keepdim=True)

  def _attack_objective(
      self,
      perturbation: torch.Tensor,
      input_tensor: torch.Tensor,
      baseline_input: torch.Tensor,
      target_class: int,
      reference_attribution: torch.Tensor,
      arrival_alphas: torch.Tensor,
      previous_alphas: torch.Tensor | None,
  ) -> torch.Tensor:
    """Measure attribution change while penalizing changes to the prediction.

    Args:
        perturbation: Perturbation tensor carrying the graph
        input_tensor: Unperturbed input tensor of shape [1, C, H, W]
        baseline_input: Baseline tensor of shape [1, C, H, W]
        target_class: Class index the attributions are computed for
        reference_attribution: Surrogate attribution at the unperturbed input
        arrival_alphas: Interpolation coefficients of the arrival points
        previous_alphas: Coefficients of the preceding points, or None

    Returns:
        Scalar objective tensor.
    """
    perturbed_input = input_tensor + perturbation
    attribution = self._surrogate_attribution(
        perturbed_input, baseline_input, target_class, arrival_alphas,
        previous_alphas
    )
    distortion = 1 - F.cosine_similarity(
        attribution.reshape(1, -1), reference_attribution.reshape(1, -1),
        dim=-1, eps=1e-12
    ).squeeze()

    logits = self.model(perturbed_input)
    target_logit = logits[0, target_class]
    other_logits = torch.cat(
        [logits[0, :target_class], logits[0, target_class + 1:]])
    margin = other_logits.max() - target_logit
    return distortion - F.relu(margin)

  def _run_pga(
      self,
      input_tensor: torch.Tensor,
      baseline_input: torch.Tensor,
      target_class: int,
      path_alphas: torch.Tensor,
      generator: torch.Generator,
  ) -> torch.Tensor:
    """Search for perturbations that increase attribution change.

    Args:
        input_tensor: Unperturbed input tensor of shape [1, C, H, W]
        baseline_input: Baseline tensor of shape [1, C, H, W]
        target_class: Class index the attributions are computed for
        path_alphas: Interpolation coefficients of the full path
        generator: Random generator used for initialization and subsampling

    Returns:
        Tensor of perturbations of shape [num_restarts, C, H, W].
    """
    radius = 0.1
    perturbations = []
    largest_gradient_norm = 0.0
    attack_task = self.progress.add_task(
        "[magenta]  Attack", total=self.num_restarts * self.num_iterations)
    # The fused flash/mem-efficient attention kernels define no double
    # backward, so attention runs on the math backend during the attack
    with sdpa_kernel(SDPBackend.MATH), \
            softplus_smoothed(self.model, 10.0):
      for restart in range(self.num_restarts):
        # Same perturbation law as robust.py at the same radius
        noise = (2 * torch.rand(
            input_tensor.shape, generator=generator,
            device=input_tensor.device
        ) - 1) * radius
        # The first restart starts close to the input, the rest anywhere in
        # the law's typical range
        perturbation = noise * (0.01 if restart == 0 else 1.0)

        for iteration in range(self.num_iterations):
          arrival_alphas, previous_alphas = self._sample_path_points(
              path_alphas, generator)
          perturbation = perturbation.detach().requires_grad_(True)
          reference = self._surrogate_attribution(
              input_tensor, baseline_input, target_class, arrival_alphas,
              previous_alphas, create_graph=False
          ).detach()
          objective = self._attack_objective(
              perturbation, input_tensor, baseline_input, target_class,
              reference, arrival_alphas, previous_alphas
          )
          gradient = torch.autograd.grad(objective, perturbation)[0]

          with torch.no_grad():
            # Cosine-decayed sign-gradient step, clamped back into the box
            decay = 0.55 + 0.45 * np.cos(
                np.pi * iteration / max(self.num_iterations - 1, 1))
            step = 0.2 * decay * radius
            gradient_norm = torch.linalg.vector_norm(gradient)
            largest_gradient_norm = max(
                largest_gradient_norm, float(gradient_norm))
            perturbation = perturbation + step * gradient.sign()
            perturbation = perturbation.clamp(-radius, radius)
          self.progress.update(
              attack_task, advance=1,
              description=f"[magenta]  Attack (restart "
              f"{restart + 1}/{self.num_restarts})")
        perturbations.append(perturbation.detach())
    self.progress.remove_task(attack_task)

    if largest_gradient_norm == 0.0:
      # A weighting that has saturated (e.g. a softmax collapsed to the
      # uniform distribution in float32) makes the attribution locally
      # constant, so the ascent has no signal and the reported value is not a
      # meaningful worst case
      logger.warning(
          "[bold yellow]Attack objective had a zero gradient at every "
          "iteration; the attribution is locally constant, so the reported "
          "shift is not an informative worst case[/bold yellow]"
      )
    return torch.cat(perturbations, dim=0)

  def _enforce_prediction(
      self,
      input_tensor: torch.Tensor,
      perturbations: torch.Tensor,
      target_class: int,
  ) -> tuple[torch.Tensor, int]:
    """Adjust perturbations to preserve the original prediction.

    Args:
        input_tensor: Unperturbed input tensor of shape [1, C, H, W]
        perturbations: Perturbation tensor of shape [num_deltas, C, H, W]
        target_class: Class index that must be preserved

    Returns:
        Tuple of (feasible perturbations, number of perturbations shrunk).
    """
    feasible = perturbations.clone()
    num_shrunk = 0
    for _ in range(10):
      with torch.inference_mode():
        predictions = self.model(input_tensor + feasible)
      flipped = predictions.argmax(dim=-1) != target_class
      if not flipped.any():
        break
      num_shrunk = max(num_shrunk, int(flipped.sum().item()))
      feasible[flipped] = feasible[flipped] * 0.5
    else:
      with torch.inference_mode():
        predictions = self.model(input_tensor + feasible)
      flipped = predictions.argmax(dim=-1) != target_class
      if flipped.any():
        logger.debug(
            f"{int(flipped.sum().item())} perturbation(s) still change the "
            "prediction after 10 halvings; zeroing them"
        )
        feasible[flipped] = 0.0
    return feasible, num_shrunk

  def compute_aggregate_metrics(self) -> dict[str, float | np.floating[Any]]:
    """Summarize adversarial stability across examples."""
    return {
        key: np.nanmedian(np.stack(self.aggregate_metrics[key], axis=0))
        for key in ('adv_lipschitz', 'ais', 'adv_sensitivity')
    }

  def compute_metric_per_batch(
      self,
      keys: list[str],
      examples: dict[str, Any],
      attributions: torch.Tensor,
      original_predictions: torch.Tensor,
  ) -> dict[str, np.ndarray]:
    """Compute adversarial sensitivity, Lipschitz, and input stability scores.

    Args:
        keys: List of example keys
        examples: Batch of input examples (dict of tensors)
        attributions: Corresponding attributions for the examples (tensor)
        original_predictions: Model predictions on the unperturbed examples
    Returns:
        AS (``adv_sensitivity``), ALC (``adv_lipschitz``), and AIS (``ais``).
    """
    start_idx = keys[0].split('_')[-1]
    end_idx = keys[-1].split('_')[-1]

    # move to CPU to save GPU memory
    examples = {k: v.cpu() if torch.is_tensor(v) else v
                for k, v in examples.items()}
    torch.cuda.empty_cache()
    attributions = attributions.cpu()
    original_predictions = original_predictions.cpu()
    pred_classes = original_predictions.argmax(dim=-1)

    # Compute adversarial robustness metrics for each example in the batch
    bsz = attributions.shape[0]
    adv_lipschitz_scores = []
    ais_scores = []
    adv_sensitivity_scores = []
    batch_bar = self.progress.add_task(
        description=f"[green]Computing Adversarial Robustness - Examples "
        f"{start_idx} to {end_idx}",
        total=bsz
    )
    for i in range(bsz):
      example_i = index_into_collated_batch(examples, i)
      pred_class_i = pred_classes[i]

      # Find the perturbations; seed by example index so the attack is
      # reproducible across runs and independent between examples
      generator = torch.Generator(device=self.device)
      generator.manual_seed(
          self.seed + int(keys[i].split('_')[-1]) * self.num_restarts)
      input_i = example_i[self.input_key].unsqueeze(0).to(self.device)
      baseline_i = get_baseline(
          input_i.squeeze(0).detach(), self.baseline_type,
          **self.baseline_kwargs
      ).unsqueeze(0).to(self.device)
      path_alphas = torch.arange(
          self.num_steps + 1, device=input_i.device) / self.num_steps
      perturbations = self._run_pga(
          input_i, baseline_i, int(pred_class_i), path_alphas,
          generator
      )
      stage_task = self.progress.add_task(
          "[magenta]  Feasibility backtrack", total=None)
      perturbations, num_shrunk = self._enforce_prediction(
          input_i, perturbations, int(pred_class_i))
      self.progress.remove_task(stage_task)
      if num_shrunk:
        logger.debug(
            f"Example {keys[i]}: shrank {num_shrunk} perturbation(s) to "
            f"preserve the prediction"
        )
      perturbations = perturbations.cpu()
      torch.cuda.empty_cache()

      # Evaluate the perturbations with the real method on the
      # unmodified network, so the attack surrogate cannot affect the
      # reported numbers
      perturbation_dataset = AdversarialPerturbationDataset(
          example_i, self.input_key, perturbations)
      stage_task = self.progress.add_task(
          f"[magenta]  Re-evaluating real method "
          f"({perturbations.shape[0]} perturbed runs)", total=None)
      perturbation_outputs = self.method.compute_attributions_for_dataset(
          perturbation_dataset, skip_transform=True, skip_progress_init=True
      )
      self.progress.remove_task(stage_task)
      perturbation_attributions = torch.stack(
          [out["attributions"] for out in perturbation_outputs], dim=0)
      orig_attr = attributions[i]
      orig_input = example_i[self.input_key]

      # Compute differences
      delta_attr = perturbation_attributions - orig_attr.unsqueeze(0)
      delta_input = perturbations

      # Adversarial Lipschitz Constant (ALC)
      attr_norm = torch.linalg.vector_norm(
          delta_attr.reshape(delta_attr.shape[0], -1), ord=2, dim=-1
      )
      input_norm = torch.linalg.vector_norm(
          delta_input.reshape(delta_input.shape[0], -1), ord=2, dim=1
      )
      norm_ratios = attr_norm / (input_norm + 1e-15)
      adv_lipschitz = torch.max(norm_ratios).item()
      adv_lipschitz_scores.append(adv_lipschitz)

      # Adversarial Input Stability (AIS)
      attr_ratio = attr_norm / (torch.linalg.vector_norm(
          orig_attr.reshape(1, -1), ord=2, dim=-1
      ) + 1e-15)
      input_ratio = input_norm / (torch.linalg.vector_norm(
          orig_input.reshape(1, -1), ord=2, dim=-1
      ) + 1e-15)
      ais_ratios = attr_ratio / (input_ratio + 1e-15)
      ais = torch.max(ais_ratios).item()

      ais_scores.append(ais)

      # Adversarial Sensitivity (AS)
      adv_sensitivity = torch.max(attr_norm).item()
      adv_sensitivity_scores.append(adv_sensitivity)
      self.progress.update(batch_bar, advance=1)
    self.progress.remove_task(batch_bar)

    return {
        "adv_lipschitz": np.array(adv_lipschitz_scores),
        "ais": np.array(ais_scores),
        "adv_sensitivity": np.array(adv_sensitivity_scores),
    }
