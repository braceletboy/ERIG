"""Guided Integrated Gradients (GIG).

References:
- Kapishnikov A, Venugopalan S, Avci B, Wedin B, Terry M, Bolukbasi T. Guided
  Integrated Gradients: An adaptive path method for removing noise.
  In Proceedings of the IEEE/CVF conference on computer vision and pattern
  recognition 2021 (pp. 5050-5058).

Implementation adapted from:
- https://github.com/PAIR-code/saliency/
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

import os
import json
import h5py
import logging

import torch
from torch.utils.data import DataLoader

from erig.methods.method import Method
from erig.methods.utils import (
    get_baseline,
    custom_collate_fn,
    index_into_collated_batch,
    move_collated_batch_to_device,
    convert_example_tensor_to_list,
    move_example_to_device,
    detach_example,
    squeeze_example
)

SAFETY_COUNTER = 9999  # to avoid infinite loops in Guided IG


logger = logging.getLogger("rich")


def restrict_to_alpha_interval(
    current_inputs: torch.Tensor,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    alpha_min: float | torch.Tensor,
    alpha_max: float | torch.Tensor,
) -> torch.Tensor:
  """Pull features that have fallen behind the segment back to its start.

  Guided IG restricts each step to an interval of the straight-line path,
  expressed as a fraction alpha in [0, 1] of the way from baseline to input.
  Any feature whose alpha has fallen below ``alpha_min`` is moved up to the
  point corresponding to ``alpha_min``.

  The comparison has to happen in alpha space, not in raw feature values.
  ``alpha`` runs 0 -> 1 whatever the sign of ``inputs - baselines``, whereas
  the corresponding feature values run downwards wherever an input sits below
  its baseline. Bounds derived from ``alpha_min`` and ``alpha_max`` therefore
  invert for every such feature, and ``torch.clamp`` resolves an inverted
  range to its upper argument, teleporting the feature to the far end of the
  path.

  Mirrors ``x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]`` in the
  reference implementation (PAIR-code/saliency, ``core/guided_ig.py``), which
  is likewise one-sided: nothing is pushed back down to ``alpha_max``. Features
  that have reached the segment end are excluded later, by assigning them
  infinite gradient.

  Args:
      current_inputs: Current point on the path, shape (batch, ...)
      inputs: Path endpoint, same shape
      baselines: Path start, same shape
      alpha_min: Lower end of this segment's alpha interval
      alpha_max: Upper end, used for features where input equals baseline and
          alpha is undefined -- those can never lag, matching the reference's
          ``x_alpha[np.isnan(x_alpha)] = alpha_max``

  Returns:
      ``current_inputs`` with lagging features moved up to the segment start
  """
  deltas = inputs - baselines
  alphas = torch.where(
      deltas != 0,
      (current_inputs - baselines) / torch.where(
          deltas != 0, deltas, torch.ones_like(deltas)),
      torch.full_like(deltas, alpha_max),
  )
  segment_mins = baselines + alpha_min * deltas
  return torch.where(alphas < alpha_min, segment_mins, current_inputs)


class GuidedIG(Method):
  """Compute attributions along the Guided IG path."""

  def __init__(self, config: dict[str, Any]) -> None:
    """
    Initialize the GIG method.

    Args:
        config: Configuration dictionary
    """
    super().__init__("gig", config)

    # GIG-specific initialization
    compute_config = config["compute_config"]
    self.baseline = compute_config.get("gig_baseline", "zero")
    self.num_segments = compute_config.get("gig_num_segments", 50)
    self.percentile = compute_config.get("gig_percentile", 0.25)
    self.max_alpha_dist = compute_config.get("gig_max_alpha_dist", 1.0)

  def compute_attributions_for_dataset(
      self,
      tensor_dataset: (
          Dataset[dict[str, Any]] | HFDataset | Sequence[dict[str, Any]]
      ),
      skip_transform: bool = False,
      skip_progress_init: bool = False,
  ) -> list[dict[str, Any]]:
    tensor_dataloader = torch.utils.data.DataLoader(
        tensor_dataset, batch_size=self.batch_size, shuffle=False,
        collate_fn=lambda list_batch: list_batch  # Keep structure as list
    )

    self.model.eval()

    outputs = []

    if not skip_progress_init:
      self.progress.start()

    batches_bar = self.progress.add_task(
        description="[green]Tensor Batches", total=len(tensor_dataloader),
    )

    for batch_idx, list_batch in enumerate(tensor_dataloader):
      if not skip_transform:  # Transform batch
        list_batch = [self.input_transform(example) for example in list_batch]

      # Convert list of examples to proper batch format
      batch = custom_collate_fn(list_batch)
      batch = move_collated_batch_to_device(batch, self.device)

      # Use batch processing method
      batch_outputs = self._compute_attrs_per_batch(batch_idx, batch)

      # Extend outputs with all results from the batch
      for output in batch_outputs:
        outputs.append({
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in output.items()
        })

      self.progress.update(batches_bar, advance=1)

    if skip_progress_init:  # batches are not the main task
      self.progress.remove_task(batches_bar)

    if not skip_progress_init:
      self.progress.stop()

    return outputs

  def compute_attributions(self) -> None:
    """Generate and save attribution scores for the tokenized dataset.

    The generated attributions will be saved in an HDF5 file. The structure of
    the file will be as follows:
    - /example_{save_idx}/
        - example: Original input example
        - attributions: Attribution scores for the example
    """
    self.model.eval()

    logger.info(f"[bold red]{self.name.upper()} uses batched attribution "
                f"computation because of its sequential nature.[/bold red]")
    logger.info(f"Starting prediction on {len(self.dataset)} examples")

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        self.dataset,
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=lambda list_batch: list_batch,  # Keep structure as list
        num_workers=min(os.cpu_count(), 24),
    )

    with h5py.File(
        os.path.join(self.output_dir, "attributions.hdf5"), 'w', libver='latest'
    ) as root_group:
      root_group.attrs["config"] = json.dumps(self.config)
      root_group.attrs["num_samples"] = len(self.dataset)

      with self.progress:
        batches_bar = self.progress.add_task(
            description="[green]Batches", total=len(dataloader),
        )

        save_idx = 0
        for batch_idx, list_batch in enumerate(dataloader):
          list_batch = [self.input_transform(example) for example in list_batch]

          batch = custom_collate_fn(list_batch)
          batch = move_collated_batch_to_device(batch, self.device)
          batch_outputs = self._compute_attrs_per_batch(batch_idx, batch)

          # Save results for each example in the batch
          writing_bar = self.progress.add_task(
              description=f"[blue]Writing Batch {batch_idx+1} Results",
              total=len(batch_outputs),
          )
          for idx, output in enumerate(batch_outputs):
            # Skip examples that were filtered out
            if output["attributions"] is None:
              continue

            # Create a group for this example
            example_group = root_group.create_group(f"example_{save_idx}")

            # Convert example to list format for JSON serialization
            example_list = convert_example_tensor_to_list(
                move_example_to_device(detach_example(
                    squeeze_example(list_batch[idx], dim=0)
                ), 'cpu')
            )
            example_group.create_dataset(
                "example", data=json.dumps(example_list))

            # Store attributions, original predictions and labels (if available)
            for key, value in output.items():
              if value is None:
                continue
              example_group.create_dataset(
                  key, data=(value.detach().cpu().numpy()
                             if torch.is_tensor(value) else value)
              )

            self.progress.update(writing_bar, advance=1)
            save_idx += 1

          self.progress.remove_task(writing_bar)
          self.progress.update(batches_bar, advance=1)

  def _compute_attrs_per_example(
      self,
      example_idx: int,
      example: dict[str, Any],
  ) -> dict[str, Any]:
    """
    Compute GIG attributions for a given example.

    Args:
        example_idx: Index of the example
        example: Input example
    Returns:
        Attributions dictionary
    """
    single_batch = custom_collate_fn([squeeze_example(example)])
    single_batch = move_collated_batch_to_device(single_batch, self.device)
    batch_outputs = self._compute_attrs_per_batch(example_idx, single_batch)
    output = batch_outputs[0]
    if output["attributions"] is not None:  # Add batch dim for consistency
      output["attributions"] = output["attributions"].unsqueeze(0)
    return output  # Return the single example's output

  def _compute_attrs_per_batch(
      self,
      batch_idx: int,
      batch: dict[str, Any],
  ) -> list[dict[str, Any]]:
    """
    Compute GIG attributions for a batch of examples.

    Args:
        batch_idx: Index of the batch
        batch: Batched input examples (tensors with batch dimension)

    Returns:
        List of dictionaries containing attributions, predictions, and labels
    """
    # Compute original predictions for the entire batch
    with torch.inference_mode():
      logits = self.model(batch[self.input_key])
      original_preds = torch.softmax(logits, dim=-1)

    original_pred_classes = torch.argmax(original_preds, dim=-1)

    # Store original batch size, predictions, and labels for all examples
    original_batch_size = batch[self.input_key].size(0)
    original_labels = batch.get("label", None)

    if batch.get("label") is not None:
      # Skip examples where the model prediction doesn't match the label
      correct_mask = (original_pred_classes == batch["label"])
      if not correct_mask.all():
        logger.debug(f"Skipping {(~correct_mask).sum().item()} examples due to "
                     f"prediction mismatch.")
        batch = index_into_collated_batch(batch, correct_mask)
        original_pred_classes = original_pred_classes[correct_mask]
    else:
      correct_mask = torch.ones(original_batch_size, dtype=torch.bool)

    # Get baseline inputs for the entire batch
    baseline_inputs = get_baseline(batch[self.input_key], self.baseline)
    baseline_inputs = baseline_inputs.to(self.device)

    batch_size = batch[self.input_key].size(0)
    batch_results = []

    if batch_size == 0:
      for idx in range(original_batch_size):
        batch_results.append({
            "attributions": None,
            "original_example_idx": idx + batch_idx * original_batch_size,
            "original_prediction": original_preds[idx],
            "label": (original_labels[idx].item()
                      if original_labels is not None else None),
        })
      return batch_results

    total_l1_distances = torch.abs(
        batch[self.input_key] - baseline_inputs
    ).view(batch_size, -1).sum(dim=-1)

    gig_bar = self.progress.add_task(
        description=f"[cyan]Batch {batch_idx+1} - GIG segments",
        total=self.num_segments,
    )
    gig = torch.zeros_like(batch[self.input_key])
    current_batch = batch.copy()
    current_inputs = baseline_inputs  # Initialize current inputs to baseline
    for i in range(self.num_segments):
      # Detach from previous computation graphs for clean gradient computation
      current_batch[self.input_key] = current_inputs.detach()
      current_batch[self.input_key].requires_grad_(True)

      current_logits = self.model(current_batch[self.input_key])
      current_preds = torch.softmax(current_logits, dim=-1)

      # Compute loss - cross sample gradients are zero - so sum is okay
      loss = torch.sum(
          current_preds[torch.arange(batch_size), original_pred_classes]
      )

      # Backward pass - compute gradients for the segment
      # All points in the segment will use the same gradients
      segment_gradients = torch.autograd.grad(
          outputs=loss,
          inputs=current_batch[self.input_key],
          create_graph=False
      )[0]  # use autograd instead of pytorch backward to save memory
      current_batch[self.input_key].requires_grad_(False)

      alpha_max = min(
          ((i + 1) / self.num_segments) + self.max_alpha_dist, 1.0
      )
      alpha_min = max(
          ((i + 1) / self.num_segments) - self.max_alpha_dist, 0.0
      )
      segment_maxs = baseline_inputs + alpha_max * (
          batch[self.input_key] - baseline_inputs)
      target_l1_distances = total_l1_distances * (
          1 - ((i + 1) / self.num_segments)
      )

      segment_bar = self.progress.add_task(
          description=f"[magenta]Segment {i+1}", total=None
      )
      gamma = torch.full((batch_size,), torch.inf).to(self.device)
      while_stop_mask = torch.zeros(
          batch_size, dtype=torch.bool).to(self.device)
      iteration_count = 0  # Local while loop counter

      # Only loop for each input till the reduction in L1 distance due to the
      # weak partials set is greater than the current L1 distance minus the
      # target L1 distance
      while (gamma > 1.0).any():
        gamma_mask = gamma <= 1.0
        prev_inputs = current_inputs.clone()

        # Pull features that have fallen behind back to the segment start
        current_inputs = restrict_to_alpha_interval(
            current_inputs, batch[self.input_key],
            baseline_inputs, alpha_min, alpha_max
        )

        current_l1_distance_to_inputs = torch.abs(
            current_inputs - batch[self.input_key]
        ).view(batch_size, -1).sum(dim=-1)

        prev_while_stop_mask = while_stop_mask.clone()

        while_stop_mask = gamma_mask | while_stop_mask | torch.isclose(
            current_l1_distance_to_inputs, target_l1_distances,
            rtol=1e-9, atol=1e-9
        )

        # For newly stopped inputs, add the final attributions
        newly_stopped = torch.logical_and(
            while_stop_mask, ~prev_while_stop_mask
        )
        if newly_stopped.any():
          gig[newly_stopped] += (
              current_inputs[newly_stopped] - prev_inputs[newly_stopped]
          ) * segment_gradients[newly_stopped]

        unstopped_indices = ~while_stop_mask
        num_unstopped = unstopped_indices.sum().item()
        if num_unstopped == 0:
          break

        # Advanced indexing gives a copy, so we can modify it safely
        unstopped_gradients_copy = segment_gradients[unstopped_indices]
        unstopped_current_inputs = current_inputs[unstopped_indices]
        unstopped_segment_maxs = segment_maxs[unstopped_indices]

        # Set gradients to inf where input equals max
        mask_at_max = (unstopped_current_inputs == unstopped_segment_maxs)
        unstopped_gradients_copy[mask_at_max] = torch.inf
        threshold = torch.quantile(
            unstopped_gradients_copy.abs().view(num_unstopped, -1),
            self.percentile, dim=-1, interpolation='lower', keepdim=True
        )
        weak_partials_derivatives_mask = torch.logical_and(
            unstopped_gradients_copy.abs().view(num_unstopped, -1) <= threshold,
            unstopped_gradients_copy.view(num_unstopped, -1) != torch.inf,
        ).view_as(unstopped_gradients_copy)

        # Compute drop in L1 distance to segment_maxs if we moved current inputs
        # fully along the weak partials derivatives towards segment_maxs
        l1_distance_drop = (
            (unstopped_current_inputs - unstopped_segment_maxs).abs() *
            weak_partials_derivatives_mask.float()
        ).view(num_unstopped, -1).sum(dim=-1)

        # Gamma: fraction of the distance we can move along weak partials
        # derivatives before reaching segment end - it is inversely proportional
        # to the drop in L1 distance to the input
        gamma_unstopped = torch.where(
            l1_distance_drop > 0,
            (
                (current_l1_distance_to_inputs[unstopped_indices] -
                 target_l1_distances[unstopped_indices]) /
                l1_distance_drop
            ),
            torch.inf
        )

        if (gamma_unstopped <= 0).any():
          raise ValueError(
              f"Non-positive gamma encountered in batch {batch_idx}, "
              f"segment {i}, iteration {iteration_count}. "
              f"This indicates an error in the implementation."
          )

        # Reshape gamma to broadcast per sample against [n, C, H, W] tensors
        gamma_broadcast = gamma_unstopped.view(
            -1, *([1] * (len(unstopped_current_inputs.shape) - 1))
        )

        # Move input along weak partials derivatives by gamma; computed on
        # the full tensors because masked 1-D indexing cannot broadcast the
        # per-sample gammas
        moved_inputs = torch.where(
            gamma_broadcast > 1.0,
            unstopped_segment_maxs,
            unstopped_current_inputs +
            gamma_broadcast * (unstopped_segment_maxs -
                               unstopped_current_inputs),
        )
        unstopped_current_inputs = torch.where(
            weak_partials_derivatives_mask, moved_inputs,
            unstopped_current_inputs)

        # Update
        current_inputs[unstopped_indices] = unstopped_current_inputs
        gamma[unstopped_indices] = gamma_unstopped
        gig[unstopped_indices] += (
            current_inputs[unstopped_indices] - prev_inputs[unstopped_indices]
        ) * segment_gradients[unstopped_indices]
        self.progress.update(segment_bar, advance=1)

        # While loop counter safety check
        iteration_count += 1
        if iteration_count > SAFETY_COUNTER:
          logger.warning(
              f"Safety counter exhausted for batch {batch_idx}, segment {i}. "
              f"Some attributions may be incorrect."
          )
          break

      # Complete the progress bar for this segment
      self.progress.remove_task(segment_bar)
      self.progress.update(gig_bar, advance=1)
    self.progress.remove_task(gig_bar)

    # Prepare results for all examples in the original batch
    filtered_idx = 0
    for idx in range(original_batch_size):
      if correct_mask[idx]:
        attributions = gig[filtered_idx]
        filtered_idx += 1
      else:
        attributions = None

      result = {
          "attributions": attributions,
          "original_example_idx": idx + batch_idx * original_batch_size,
          "original_prediction": original_preds[idx],
          "label": (original_labels[idx].item()
                    if original_labels is not None else None),
      }

      batch_results.append(result)

    return batch_results
