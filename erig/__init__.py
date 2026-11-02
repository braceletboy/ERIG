"""Entropy-Regularized Integrated Gradients and image attribution experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from erig.methods.method import Method
from erig.methods.method_config import get_method_config
from erig.methods.meta_method_config import get_meta_method_config
from erig.visualization import visualize_attributions

__version__ = '0.1.0'
__all__ = ['create_method', 'visualize_attributions']

METHODS = ('erig', 'ig', 'idgi', 'big', 'gig', 'smoothgrad', 'aerig', 'serig')


def create_method(
    model: torch.nn.Module,
    method: str = 'erig',
    *,
    steps: int = 200,
    batch_size: int = 1,
    temperature: float = 3.0,
    baseline: str = 'zero',
    options: dict[str, Any] | None = None,
) -> Method:
  """Create an attribution method for a supplied PyTorch classifier.

  Args:
      model: Evaluation-mode PyTorch classifier returning a logits tensor.
      method: Attribution method name.
      steps: Number of path intervals (segments for GIG).
      batch_size: Maximum number of inputs processed together; one suits small GPUs.
      temperature: ERIG temperature.
      baseline: Reference construction in the input tensor's coordinates.
      options: Additional method-prefixed parameters, e.g. big_max_sigma.

  Returns:
      A Method subclass accepting CHW or BCHW tensors of preprocessed inputs.
  """
  if method not in METHODS:
    raise ValueError(f'Unknown method {method!r}; choose from {METHODS}')
  if steps < 1 or batch_size < 1 or temperature <= 0:
    raise ValueError('steps, batch_size and temperature must be positive')

  meta = method == 'smoothgrad'
  config = (get_meta_method_config if meta else get_method_config)(
      method, 'attributions')
  cls = config.pop('class')
  cc = config['compute_config']
  cc.update({f'{method}_num_steps': steps, f'{method}_baseline': baseline})
  if method == 'gig':
    cc['gig_num_segments'] = steps
  if method in ('erig', 'aerig', 'serig'):
    cc[f'{method}_exp_temperature'] = temperature
  supplied = deepcopy(options or {})
  cc.update(supplied)
  config.update(_model=model, eval_batch_size=batch_size,
                no_progress_bar=True, input_key='image')
  return cls(config)
