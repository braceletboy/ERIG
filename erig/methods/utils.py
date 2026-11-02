"""
Utility functions for explainability methods.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Sequence

import logging
import random

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageFilter

logger = logging.getLogger("rich")

# timm's prefix for a checkpoint hosted on the Hugging Face Hub rather than in
# timm's own registry, e.g. "hf-hub:nateraw/resnet50-oxford-iiit-pet".
TIMM_HUB_PREFIX = "hf-hub:"


# ---------------------------------------------------------------------------
# Model loading
#
# Two backends are supported. timm serves the official ImageNet checkpoints;
# transformers serves domain fine-tuned checkpoints pulled from the Hub. Each
# backend has a loader and a matching transform builder, so the attribution
# side (methods/method.py) and the metric side (metrics/metric.py) can share
# one implementation and cannot drift apart.
#
# Every builder returns ``(transform, spatial_transform)``:
#   transform         - produces the normalized pixel tensor the model expects
#   spatial_transform - applies the SAME geometry to a mode-"L" segmentation
#                       mask with nearest-neighbour resampling and no value
#                       rescaling, so integer label values survive intact and
#                       the mask lands on the same grid as the attribution map
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
  """Seed Python, NumPy, and PyTorch generators, including CUDA generators."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  logger.info(f"Global seed set to {seed} for reproducibility")


def get_device(config: dict[str, Any]) -> str | torch.device:
  """Use the configured device, otherwise prefer CUDA when available."""
  if config.get("device"):
    return config["device"]
  if torch.cuda.is_available():
    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    return "cuda"
  logger.info("CUDA not available. Using CPU")
  return "cpu"


def is_transformers_model(model_name: str) -> bool:
  """Whether ``model_name`` names a Hugging Face transformers checkpoint.

  Three naming forms are in use:

  - ``resnet50.a1_in1k``          -- a timm registry tag
  - ``hf-hub:org/name``           -- a timm checkpoint hosted on the Hub
  - ``org/name``                  -- a transformers checkpoint on the Hub

  Only the last is a transformers model. The ``hf-hub:`` prefix has to be
  checked first, since those names also contain a "/".
  """
  if model_name.startswith(TIMM_HUB_PREFIX):
    return False
  return "/" in model_name


class LogitsWrapper(nn.Module):
  """Expose a transformers classifier through the plain-tensor interface.

  ``AutoModelForImageClassification`` returns a ``ModelOutput`` -- the exact
  class varies by architecture (``ImageClassifierOutput`` for ViT/DeiT/BEiT,
  ``SwinImageClassifierOutput`` for Swin, ``ImageClassifierOutputWithNoAtt
  ention`` for ConvNeXt) -- while every method in this project calls
  ``self.model(x)`` and expects a logits tensor. All of them expose
  ``.logits``, so this adapter unwraps that and nothing else; gradients flow
  through it unchanged. Do not type-check against a specific output class.
  """

  def __init__(self, model: torch.nn.Module) -> None:
    super().__init__()
    self.model = model

  def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Return classifier logits while preserving the gradient graph."""
    return self.model(*args, **kwargs).logits


def load_timm_model(model_name: str) -> torch.nn.Module:
  """Load a pretrained timm classifier.

  Args:
      model_name: A timm model tag, e.g. ``resnet50.a1_in1k``

  Returns:
      The model, returning logits from a plain tensor input
  """
  model = timm.create_model(model_name, pretrained=True)
  # A registry checkpoint carries a "url"; a Hub-hosted one ("hf-hub:org/name")
  # carries "hf_hub_id" instead, and timm drops null keys entirely -- so
  # indexing "url" raises KeyError for every Hub checkpoint.
  pretrained_cfg = model.pretrained_cfg
  source = (pretrained_cfg.get('url')
            or pretrained_cfg.get('hf_hub_id')
            or model_name)
  logger.info(f"Model loaded from: {source}")
  logger.info(f"Model configuration: {pretrained_cfg}")
  return model


def load_transformers_model(model_name: str) -> torch.nn.Module:
  """Load a transformers image classifier, wrapped to return logits.

  The classification head is sized from the checkpoint's own config, so no
  explicit class count is passed here.

  Args:
      model_name: A Hugging Face Hub repo id, e.g. ``org/pets-vit``

  Returns:
      The model, returning logits from a plain tensor input
  """
  # Imported lazily so that timm-only runs do not require transformers.
  from transformers import AutoModelForImageClassification

  model = AutoModelForImageClassification.from_pretrained(model_name)
  logger.info(f"Model loaded from Hugging Face Hub: {model_name}")
  # num_labels is derived from id2label, so a checkpoint with a missing or
  # malformed label map silently yields the wrong head width. Log it: the
  # count must match the dataset, and the ordering must match its ClassLabel
  # order, since no label permutation is applied anywhere.
  logger.info(f"Model output classes: {model.config.num_labels}")
  logger.info(f"Model configuration: {model.config}")
  return LogitsWrapper(model)


def get_timm_transforms(
    model_name: str,
) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
  """Build the input and spatial transforms for a timm checkpoint.

  The data config is resolved from the pretrained tag alone, so this does not
  need the model to be built or its weights downloaded.

  Args:
      model_name: A timm model tag, e.g. ``resnet50.a1_in1k``

  Returns:
      Tuple ``(transform, spatial_transform)``
  """
  pretrained_cfg = None
  if not model_name.startswith(TIMM_HUB_PREFIX):
    registry_cfg = timm.get_pretrained_cfg(model_name)
    if registry_cfg is not None:
      pretrained_cfg = registry_cfg.to_dict()

  if pretrained_cfg is None:
    # Hub-hosted checkpoints are not in timm's registry, so their config only
    # exists on the Hub. Constructing the model reads it; pretrained=False
    # keeps that to the config fetch and skips the weights.
    pretrained_cfg = timm.create_model(
        model_name, pretrained=False).pretrained_cfg

  transform_config = timm.data.resolve_data_config(pretrained_cfg)
  transform = timm.data.create_transform(**transform_config)

  # `normalize=False` makes timm end the pipeline in MaybePILToTensor, which
  # keeps uint8 values unscaled -- ToTensor would divide by 255 and destroy
  # the label values.
  spatial_transform_config = transform_config.copy()
  spatial_transform_config.update(
      {'interpolation': 'nearest', 'normalize': False}
  )
  spatial_transform = timm.data.create_transform(**spatial_transform_config)

  return transform, spatial_transform


def get_transformers_transforms(
    model_name: str,
) -> tuple[Callable[[Image.Image], torch.Tensor],
           Callable[[Image.Image], torch.Tensor]]:
  """Build the input and spatial transforms for a transformers checkpoint.

  Images and masks use the checkpoint's processor so resizing, crop geometry
  and rounding match. Masks use nearest-neighbor resampling with rescaling
  and normalization disabled. Rebuilding these transforms with torchvision
  can shift masks by one pixel or use different crop dimensions.

  Args:
      model_name: A Hugging Face Hub repo id, e.g. ``org/pets-vit``

  Returns:
      Tuple ``(transform, spatial_transform)``
  """
  # Imported lazily so that timm-only runs do not require transformers.
  from transformers import AutoImageProcessor, PretrainedConfig
  from transformers.image_utils import PILImageResampling

  try:
    processor = AutoImageProcessor.from_pretrained(model_name)
  except OSError:
    # Repositories without preprocessor_config.json need the base model
    # processor. A model-type default can use the wrong normalization.
    # Read the raw config because AutoConfig replaces _name_or_path with
    # the requested repository name, losing the base-model reference.
    config_dict, _ = PretrainedConfig.get_config_dict(model_name)
    base_model = config_dict.get('_name_or_path')
    if not base_model or base_model == model_name:
      raise
    logger.warning(
        f"{model_name} ships no image processor; falling back to the "
        f"processor of its base model {base_model}"
    )
    processor = AutoImageProcessor.from_pretrained(base_model)

  def transform(image: Image.Image) -> torch.Tensor:
    """Preprocess an image into the classifier's input tensor."""
    return processor(image, return_tensors="pt")["pixel_values"][0]

  def spatial_transform(mask: Image.Image) -> torch.Tensor:
    # PILImageResampling.NEAREST maps to InterpolationMode.NEAREST_EXACT in
    # the torchvision-backed processors and to PIL NEAREST in the slow path.
    # With do_rescale and do_normalize off, the processor's
    # rescale_and_normalize step is a passthrough, so integer labels survive.
    """Apply the image geometry to a mask while preserving its labels."""
    return processor(
        mask,
        do_rescale=False,
        do_normalize=False,
        do_convert_rgb=False,
        resample=PILImageResampling.NEAREST,
        return_tensors="pt",
    )["pixel_values"][0]

  return transform, spatial_transform


def custom_collate_fn(list_batch: Sequence[Any]) -> Any:
  """Collates any type of data into a batch.

  Args:
      list_batch: List of batch elements to collate.

  Returns:
      Collated batch.
  """
  elem = list_batch[0]
  if isinstance(elem, torch.Tensor):
    return torch.stack(list_batch, 0)
  elif isinstance(elem, np.ndarray):
    return torch.stack([torch.as_tensor(b) for b in list_batch], 0)
  elif isinstance(elem, (int, float)):
    return torch.tensor(list_batch)
  elif isinstance(elem, dict):
    return {
        key: custom_collate_fn([d[key] for d in list_batch]) for key in elem
    }
  elif isinstance(elem, list):
    transposed = zip(*list_batch)
    return [custom_collate_fn(samples) for samples in transposed]
  elif isinstance(elem, tuple):
    transposed = zip(*list_batch)
    return tuple(custom_collate_fn(samples) for samples in transposed)
  elif hasattr(elem, '__iter__') and not isinstance(elem, (str, bytes, dict)):
    return [custom_collate_fn([b for b in list_batch])]
  elif hasattr(elem, 'keys') and hasattr(elem, '__getitem__'):
    return {
        key: custom_collate_fn([d[key] for d in list_batch])
        for key in elem.keys()
    }
  else:
    return list_batch  # Fallback: return as is


def index_into_collated_batch(
    collated_batch: Any,
    batch_index: int | list[int] | torch.Tensor,
) -> Any:
  """Indexes into a collated batch to retrieve one or more examples.

  Assumes that the first dimension of the tensors is the batch dimension.

  Args:
      collated_batch: Collated batch (output of custom_collate_fn).
      batch_index: Index or indices of the example(s) to retrieve.
                   Can be an integer, list/tensor of integers, or boolean mask.

  Returns:
      Single example or multiple examples from the collated batch.
  """
  if torch.is_tensor(batch_index) and batch_index.dtype == torch.bool:
    # Convert boolean mask to indices and get the first/batch index
    batch_index = batch_index.nonzero(as_tuple=True)[0]
  if torch.is_tensor(collated_batch):
    return collated_batch[batch_index]
  elif isinstance(collated_batch, np.ndarray):
    return collated_batch[batch_index]
  elif isinstance(collated_batch, dict):
    return {
        key: index_into_collated_batch(value, batch_index)
        for key, value in collated_batch.items()
    }
  elif isinstance(collated_batch, list):
    return [index_into_collated_batch(item, batch_index)
            for item in collated_batch]
  elif isinstance(collated_batch, tuple):
    return tuple(
        index_into_collated_batch(item, batch_index) for item in collated_batch
    )
  else:
    return collated_batch  # Fallback: return as is - no indexing


def move_collated_batch_to_device(
    collated_batch: Any,
    device: str | torch.device,
) -> Any:
  """Moves a collated batch to the specified device.

  Args:
      collated_batch: Collated batch (output of custom_collate_fn).
      device: Target device (e.g., 'cpu' or 'cuda').

  Returns:
      Collated batch on the specified device.
  """
  if torch.is_tensor(collated_batch):
    return collated_batch.to(device)
  elif isinstance(collated_batch, dict):
    return {
        key: move_collated_batch_to_device(value, device)
        for key, value in collated_batch.items()
    }
  elif isinstance(collated_batch, list):
    return [move_collated_batch_to_device(item, device)
            for item in collated_batch]
  elif isinstance(collated_batch, tuple):
    return tuple(
        move_collated_batch_to_device(item, device) for item in collated_batch
    )
  else:
    return collated_batch  # Fallback: return as is - cannot move to device


def move_example_to_device(
    example: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
  """Moves a single example to the specified device.

  Args:
      example: Single example (can be tensor, dict, list, etc.).
      device: Target device (e.g., 'cpu' or 'cuda').

  Returns:
      Example on the specified device.
  """
  if torch.is_tensor(example):
    return example.to(device)
  elif isinstance(example, dict):
    return {
        key: move_example_to_device(value, device)
        for key, value in example.items()
    }
  elif isinstance(example, list):
    return [move_example_to_device(item, device) for item in example]
  elif isinstance(example, tuple):
    return tuple(move_example_to_device(item, device) for item in example)
  else:
    return example  # Fallback: return as is - cannot move to device


def detach_example(example: dict[str, Any]) -> dict[str, Any]:
  """Detaches all tensors in an example from the computation graph.

  Args:
      example: Single example (can be tensor, dict, list, etc.).

  Returns:
      Example with all tensors detached.
  """
  if torch.is_tensor(example):
    return example.detach()
  elif isinstance(example, dict):
    return {
        key: detach_example(value)
        for key, value in example.items()
    }
  elif isinstance(example, list):
    return [detach_example(item) for item in example]
  elif isinstance(example, tuple):
    return tuple(detach_example(item) for item in example)
  else:
    return example  # Fallback: return as is - not a tensor


def convert_example_tensor_to_list(example: dict[str, Any]) -> dict[str, Any]:
  """Converts a single example's tensors to a list format.

  Args:
      example: Single example (can be tensor, dict, list, etc.).

  Returns:
      Example with tensors in list format.
  """
  if torch.is_tensor(example):
    return example.tolist()
  elif isinstance(example, dict):
    return {
        key: convert_example_tensor_to_list(value)
        for key, value in example.items()
    }
  elif isinstance(example, list):
    return [convert_example_tensor_to_list(item) for item in example]
  elif isinstance(example, tuple):
    return tuple(convert_example_tensor_to_list(item) for item in example)
  else:
    return example  # Fallback: return as is - cannot convert to list


def unsqueeze_example(example: dict[str, Any], dim: int = 0) -> dict[str, Any]:
  """Adds a dimension to a single example at the specified position.

  Args:
      example: Single example (can be tensor, dict, list, etc.).
      dim: Dimension position to add.

  Returns:
      Example with added dimension.
  """
  if torch.is_tensor(example):
    return example.unsqueeze(dim)
  elif isinstance(example, dict):
    return {
        key: unsqueeze_example(value, dim)
        for key, value in example.items()
    }
  elif isinstance(example, list):
    return [unsqueeze_example(item, dim) for item in example]
  elif isinstance(example, tuple):
    return tuple(unsqueeze_example(item, dim) for item in example)
  else:
    return example  # Fallback: return as is - cannot unsqueeze


def squeeze_example(example: dict[str, Any], dim: int = 0) -> dict[str, Any]:
  """Removes a dimension from a single example at the specified position.

  Args:
      example: Single example (can be tensor, dict, list, etc.).
      dim: Dimension position to remove.

  Returns:
      Example with removed dimension.
  """
  if torch.is_tensor(example):
    return example.squeeze(dim)
  elif isinstance(example, dict):
    return {
        key: squeeze_example(value, dim)
        for key, value in example.items()
    }
  elif isinstance(example, list):
    return [squeeze_example(item, dim) for item in example]
  elif isinstance(example, tuple):
    return tuple(squeeze_example(item, dim) for item in example)
  else:
    return example  # Fallback: return as is - cannot squeeze


def apply_pil_blur(args: tuple[np.ndarray, float]) -> Image.Image:
  """
  Standalone function to apply PIL Gaussian blur to a single image tensor.
  This is a module-level function to avoid pickling issues with multiprocessing.

  Args:
      args: Tuple of (img_array, sigma)
  Returns:
      Blurred PIL Image
  """

  img_array, sigma = args

  # Convert numpy array to PIL image
  # Transpose from (C, H, W) to (H, W, C) for PIL
  if img_array.shape[0] in [1, 3, 4]:  # Channels first
    img_array_hwc = np.transpose(img_array, (1, 2, 0))
  else:
    img_array_hwc = img_array

  # Squeeze if grayscale
  if len(img_array_hwc.shape) == 3 and img_array_hwc.shape[2] == 1:
    img_array_hwc = img_array_hwc.squeeze(2)

  # Convert to uint8 if needed
  if img_array_hwc.dtype in [np.float32, np.float64]:
    img_array_hwc = (img_array_hwc * 255).clip(0, 255).astype(np.uint8)
  elif img_array_hwc.dtype != np.uint8:
    img_array_hwc = img_array_hwc.astype(np.uint8)

  # Apply blur and return PIL Image
  img_pil = Image.fromarray(img_array_hwc)
  img_pil_blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

  return img_pil_blurred


def apply_gaussian_blur(
    img: torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
  """
  Apply Gaussian blur to an image tensor in its own (float) space.

  Unlike apply_pil_blur, this never leaves the tensor's value range, so it
  is safe for normalized inputs. The kernel follows 2*round(4*sigma) + 1,
  capped (odd) at the image's smaller spatial side; sigma == 0 returns the
  input unchanged. Mirrors the per-example blur of the BIG/BIDGI datasets.

  Args:
      img: Image tensor of shape [..., H, W]
      sigma: Gaussian standard deviation (float or 0-dim tensor)

  Returns:
      Blurred tensor with the same shape, dtype, and device.
  """
  if torch.is_tensor(sigma):
    sigma = float(sigma)
  if sigma == 0:
    return img

  kernel_size = int(2 * round(4 * sigma) + 1)
  max_kernel_size = min(img.shape[-2], img.shape[-1])
  if kernel_size >= max_kernel_size:
    kernel_size = max_kernel_size - 1 if max_kernel_size > 1 else 1
    if kernel_size % 2 == 0:
      kernel_size -= 1

  gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)
  return gaussian_blur(img)


def get_baseline(
    input_tensor: torch.Tensor,
    baseline_type: str = 'zero',
    **kwargs: Any,
) -> torch.Tensor:
  """
  Generate baseline input for explainability methods.

  All baselines are defined directly in the space of the input tensor.

  Args:
      input_tensor (torch.Tensor): Original input tensor, [C, H, W] or
          [B, C, H, W].
      baseline_type (str): Type of baseline to generate. Options: "zero"
          (the community-standard default), "ones" (all-ones constant),
          "uniform_noise" (alias "random": uniform noise image),
          "gaussian_noise" (standard-normal noise image),
          "gaussian_perturbation" (input plus Gaussian noise), "blur"
          (Gaussian-blurred input), "random_color" (per-channel random
          constant), "channel_mean" (per-image per-channel mean constant),
          "channel_median" (per-image per-channel median constant),
          "pixel_permutation" (global spatial shuffle of the input pixels),
          "window_permutation" (spatial shuffle within non-overlapping
          windows; edge windows are clipped when the window size does not
          divide the spatial dimensions), "scramble_magnitude" /
          "scramble_phase" (random
          permutation of the Fourier magnitude / phase spectrum),
          "torch_blur", "pil_blur". The random-color, noise, permutation,
          and scramble family follows Picard et al. (2025).
      **kwargs: Additional parameters for baseline generation.

  Additional Parameters:
      For "blur" baseline:
          sigma (float): Standard deviation for Gaussian blur. Default: 20.0
          kernel_size (int): Size of the Gaussian kernel. Default: 121
      For "gaussian_perturbation" baseline:
          gaussian_sigma (float): Noise standard deviation. Default: 0.1
      For "window_permutation" baseline:
          window_size (int): Side of the shuffling window. Default: 8
      For "torch_blur" baseline:
          sigma (float): Standard deviation for Gaussian blur. Default: 1.0
          kernel_size (int): Size of the Gaussian kernel. Default: (6*sigma + 1)

      For "pil_blur" baseline:
          sigma (float): Standard deviation for Gaussian blur. Default: 1.0

  Returns:
      torch.Tensor: Baseline input tensor.

  Raises:
      ValueError: If the baseline type is not recognized.
  """
  if baseline_type == "zero":
    # Zero baseline (the community-standard default)
    baseline = torch.zeros_like(input_tensor)
  elif baseline_type == "ones":
    # All-ones constant baseline
    baseline = torch.ones_like(input_tensor)
  elif baseline_type in ("uniform_noise", "random"):
    # Uniform noise image baseline
    baseline = torch.rand_like(input_tensor)
  elif baseline_type == "gaussian_perturbation":
    # Input plus Gaussian noise (Sturmfels et al., 2020)
    gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.1))
    baseline = input_tensor + torch.randn_like(input_tensor) * gaussian_sigma
  elif baseline_type == "blur":
    # Gaussian-blurred input baseline
    sigma = float(kwargs.get("sigma", 20.0))
    kernel_size = int(kwargs.get("kernel_size", 121))
    gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)
    baseline = gaussian_blur(input_tensor)
  elif baseline_type == "gaussian_noise":
    # Standard-normal noise image baseline (Picard et al., 2025)
    baseline = torch.randn_like(input_tensor)
  elif baseline_type == "random_color":
    # Per-channel random constant color, drawn per example
    # (Picard et al., 2025)
    color = torch.rand(
        *input_tensor.shape[:-2], 1, 1,
        device=input_tensor.device, dtype=input_tensor.dtype)
    baseline = torch.ones_like(input_tensor) * color
  elif baseline_type == "channel_mean":
    # Per-image per-channel mean constant (Picard et al., 2025)
    baseline = torch.ones_like(input_tensor) * input_tensor.mean(
        dim=(-2, -1), keepdim=True)
  elif baseline_type == "channel_median":
    # Per-image per-channel median constant (per-image variant of the
    # dataset-median baseline of Picard et al., 2025)
    height, width = input_tensor.shape[-2:]
    median = input_tensor.reshape(
        *input_tensor.shape[:-2], height * width
    ).median(dim=-1, keepdim=True).values.unsqueeze(-1)
    baseline = torch.ones_like(input_tensor) * median
  elif baseline_type == "pixel_permutation":
    # Global spatial shuffle of the input pixels, drawn per example and
    # shared across channels (Picard et al., 2025)
    num_channels, height, width = input_tensor.shape[-3:]
    flat = input_tensor.reshape(-1, num_channels, height * width)
    scores = torch.rand(
        flat.size(0), 1, height * width, device=input_tensor.device)
    idx = scores.argsort(dim=-1).expand_as(flat)
    baseline = flat.gather(-1, idx).reshape(input_tensor.shape)
  elif baseline_type == "window_permutation":
    # Spatial shuffle within non-overlapping windows, drawn per example and
    # shared across channels (Picard et al., 2025). Windows tile from the
    # top-left; edge windows are clipped when the window size does not
    # divide the spatial dimensions.
    window = int(kwargs.get("window_size", 8))
    orig_shape = input_tensor.shape
    num_channels, height, width = orig_shape[-3:]
    flat = input_tensor.reshape(-1, num_channels, height, width)

    def _shuffle_within_blocks(
        region: torch.Tensor,
        block_h: int,
        block_w: int,
    ) -> torch.Tensor:
      """Permute values within each (block_h, block_w) block of the region."""
      batch, channels, region_h, region_w = region.shape
      grid_h, grid_w = region_h // block_h, region_w // block_w
      blocks = region.reshape(
          batch, channels, grid_h, block_h, grid_w, block_w
      ).permute(0, 2, 4, 1, 3, 5).reshape(
          batch, grid_h * grid_w, channels, block_h * block_w)
      scores = torch.rand(
          batch, grid_h * grid_w, 1, block_h * block_w,
          device=region.device)
      idx = scores.argsort(dim=-1).expand_as(blocks)
      blocks = blocks.gather(-1, idx)
      return blocks.reshape(
          batch, grid_h, grid_w, channels, block_h, block_w
      ).permute(0, 3, 1, 4, 2, 5).reshape(
          batch, channels, region_h, region_w)

    main_h = (height // window) * window
    main_w = (width // window) * window
    rem_h = height - main_h
    rem_w = width - main_w

    baseline = flat.clone()
    if main_h and main_w:
      baseline[..., :main_h, :main_w] = _shuffle_within_blocks(
          flat[..., :main_h, :main_w], window, window)
    if main_h and rem_w:
      baseline[..., :main_h, main_w:] = _shuffle_within_blocks(
          flat[..., :main_h, main_w:], window, rem_w)
    if rem_h and main_w:
      baseline[..., main_h:, :main_w] = _shuffle_within_blocks(
          flat[..., main_h:, :main_w], rem_h, window)
    if rem_h and rem_w:
      baseline[..., main_h:, main_w:] = _shuffle_within_blocks(
          flat[..., main_h:, main_w:], rem_h, rem_w)
    baseline = baseline.reshape(orig_shape)
  elif baseline_type in ("scramble_magnitude", "scramble_phase"):
    # Random permutation of the Fourier magnitude or phase spectrum, drawn
    # per example and shared across channels (Picard et al., 2025)
    orig_shape = input_tensor.shape
    num_channels, height, width = orig_shape[-3:]
    flat = input_tensor.reshape(-1, num_channels, height, width)
    spectrum = torch.fft.fft2(flat)
    magnitude, phase = spectrum.abs(), spectrum.angle()
    scores = torch.rand(
        flat.size(0), 1, height * width, device=input_tensor.device)
    idx = scores.argsort(dim=-1)
    if baseline_type == "scramble_magnitude":
      flat_magnitude = magnitude.reshape(-1, num_channels, height * width)
      magnitude = flat_magnitude.gather(
          -1, idx.expand_as(flat_magnitude)
      ).reshape(-1, num_channels, height, width)
    else:
      flat_phase = phase.reshape(-1, num_channels, height * width)
      phase = flat_phase.gather(
          -1, idx.expand_as(flat_phase)
      ).reshape(-1, num_channels, height, width)
    baseline = torch.fft.ifft2(
        torch.polar(magnitude, phase)).real.reshape(orig_shape)
  elif baseline_type == "torch_blur":
    # Blurred baseline using torchvision
    sigma = float(kwargs.get("sigma", 1.0))
    kernel_size = int(kwargs.get("kernel_size", 6 * sigma + 1))
    gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)
    baseline = gaussian_blur(input_tensor)
  elif baseline_type == "pil_blur":
    # Blurred baseline using PIL with multiprocessing
    to_tensor = transforms.ToTensor()
    sigma = float(kwargs.get("sigma", 1.0))
    if input_tensor.dim() == 4:  # Batch of images
      # Convert tensors to numpy arrays for multiprocessing
      device = input_tensor.device
      dtype = input_tensor.dtype
      img_arrays = [input_tensor[i].detach().cpu().numpy()
                    for i in range(input_tensor.size(0))]

      # Prepare arguments for multiprocessing
      args_list = [(img_array, sigma) for img_array in img_arrays]

      # Use multiprocessing pool to blur images in parallel
      num_workers = min(cpu_count(), len(img_arrays), 24)
      with Pool(num_workers) as pool:
        blurred_pil_images = pool.map(apply_pil_blur, args_list)

      # Convert PIL images back to tensors and stack
      baseline_list = [to_tensor(pil_img).to(device=device, dtype=dtype)
                       for pil_img in blurred_pil_images]
      baseline = torch.stack(baseline_list, dim=0)
    elif input_tensor.dim() == 3:  # Single image
      device = input_tensor.device
      dtype = input_tensor.dtype
      img_array = input_tensor.detach().cpu().numpy()
      blurred_pil_image = apply_pil_blur((img_array, sigma))
      baseline = to_tensor(blurred_pil_image).to(device=device, dtype=dtype)
    else:
      raise ValueError("Input tensor must be 3D or 4D for PIL blur baseline.")
  else:
    raise ValueError(f"Unsupported baseline type: {baseline_type}")
  return baseline
