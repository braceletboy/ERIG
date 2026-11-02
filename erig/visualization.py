"""Plot attribution maps using the paper's visualization settings."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch


def _as_batch(values: torch.Tensor) -> torch.Tensor:
  """Detach a CHW or BCHW image tensor for plotting on the CPU."""
  if (not isinstance(values, torch.Tensor) or values.ndim not in (3, 4)
      or not values.is_floating_point() or not values.numel()):
    raise ValueError('Expected a nonempty floating-point CHW or BCHW tensor')
  if not torch.isfinite(values).all():
    raise ValueError('Images and attributions must contain only finite values')
  dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
  values = values.detach().to(device='cpu', dtype=dtype)
  return values.unsqueeze(0) if values.ndim == 3 else values


def _render_attribution(attribution: np.ndarray) -> np.ndarray:
  """Render channel magnitudes with the paper's cumulative-mass scaling."""
  # Adapted from Visualizations.ipynb in the research notebooks. Its scaling
  # follows the Integrated Gradients visualization library credited in README.
  magnitude = np.abs(attribution).sum(axis=0)
  if not magnitude.any():
    return np.zeros_like(magnitude)
  ordered = np.sort(magnitude.flatten())[::-1]
  cumulative = 100.0 * np.cumsum(ordered) / magnitude.sum()
  upper = ordered[np.flatnonzero(cumulative >= 5.0)[0]]
  lower = ordered[np.flatnonzero(cumulative >= 95.0)[0]]
  # A uniform or single-level map has no contrast range to rescale.
  if upper == lower:
    return (magnitude >= upper).astype(magnitude.dtype)
  rendered = 0.9 * (magnitude-lower) / (upper-lower) + 0.1
  rendered *= rendered >= 0.1
  return np.clip(rendered, 0.0, 1.0)


def visualize_attributions(
    attributions: torch.Tensor | Mapping[str, torch.Tensor],
    images: torch.Tensor | None = None,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    cmap: str = 'gray',
) -> Figure:
  """Plot attribution magnitudes, optionally beside their source images.

  Args:
      attributions: CHW or BCHW tensor, or a mapping of method names to tensors.
      images: Corresponding CHW or BCHW images in [0, 1], or normalized images
          accompanied by their preprocessing mean and standard deviation.
      mean: Per-channel mean used to normalize the source images.
      std: Per-channel standard deviation used to normalize the source images.
      cmap: Matplotlib colormap; use 'gray' or 'inferno' for the notebook styles.

  Returns:
      A figure with one row per image and one column per attribution method,
      plus a source-image column when images are supplied. Use savefig to save it.

  Raises:
      ValueError: If tensors are empty, nonfinite, or have incompatible shapes.
  """
  named = {'Attribution': attributions} if isinstance(attributions, torch.Tensor) else attributions
  if not isinstance(named, Mapping) or not named:
    raise ValueError('Provide an attribution tensor or a nonempty mapping of names to tensors')
  batches = {name: _as_batch(values) for name, values in named.items()}
  first = next(iter(batches.values()))
  rows, _, height, width = first.shape
  if any((values.shape[0], *values.shape[-2:]) != (rows, height, width)
         for values in batches.values()):
    raise ValueError('Attribution tensors must have matching batch and spatial dimensions')
  pixels = None
  if images is not None:
    pixels = _as_batch(images)
    channels = pixels.shape[1]
    if channels not in (1, 3) or (pixels.shape[0], *pixels.shape[-2:]) != (rows, height, width):
      raise ValueError('Source images must have one or three channels and match the attributions')
    for values, label in ((std, 'std'), (mean, 'mean')):
      if values is not None and (len(values) != channels or not np.isfinite(values).all()):
        raise ValueError(f'{label} must contain one finite value per image channel')
    if std is not None:
      scale = pixels.new_tensor(std).view(1, channels, 1, 1)
      if (scale <= 0).any():
        raise ValueError('std values must be positive')
      pixels = pixels * scale
    if mean is not None:
      pixels = pixels + pixels.new_tensor(mean).view(1, channels, 1, 1)
    pixels = (pixels.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
  elif mean is not None or std is not None:
    raise ValueError('mean and std require source images')
  offset = int(pixels is not None)
  columns = len(batches) + offset
  figure, axes = plt.subplots(rows, columns, squeeze=False,
                              figsize=(3.5*columns, 3.5*rows))
  for row in range(rows):
    if pixels is not None:
      source = pixels[row, ..., 0] if pixels.shape[-1] == 1 else pixels[row]
      axes[row, 0].imshow(source, cmap='gray', vmin=0, vmax=255)
      if row == 0:
        axes[row, 0].set_title('Original image', fontsize=22, fontweight='bold', pad=12)
    for column, (name, values) in enumerate(batches.items(), start=offset):
      axes[row, column].imshow(_render_attribution(values[row].numpy()),
                               cmap=cmap, vmin=0, vmax=1)
      if row == 0:
        axes[row, column].set_title(name, fontsize=22, fontweight='bold', pad=12)
    for axis in axes[row]:
      axis.axis('off')
  figure.tight_layout()
  return figure
