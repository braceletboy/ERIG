"""Default parameters for SmoothGrad and its base method."""

from __future__ import annotations

from typing import Any
from erig.methods.method import Method
import os
from copy import deepcopy
from erig.methods.smoothgrad import SmoothGrad
from erig.methods.data_config import get_data_config


META_METHOD_CONFIGS = {
    'smoothgrad': {
        'class': SmoothGrad,
        'model_name': 'eva02',
        'dataset_name': 'mini-imagenet',
        'input_key': 'image',
        'num_samples': 1000,
        'base_dir': os.path.join(
            os.path.expandvars('$DATASETS'), 'ERIG', 'attributions'
        ),
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'skip_confirmation': True,
        'no_progress_bar': False,
        'compute_config': {
            'smoothgrad_num_samples': 50,
            'smoothgrad_noise_level': 0.1,
            'smoothgrad_base_method': 'vg',
            'smoothgrad_base_num_steps': 20,
            'smoothgrad_base_method_compute_config': {},
        },
    },
}


def get_meta_method_config(
    method_name: str, dataset_name: str
) -> dict[str, Any]:
  """
  Get meta-method configuration by name.

  Args:
      method_name: Name of the meta-method
      dataset_name: Name of the dataset

  Returns:
      Dictionary containing meta-method configuration

  Raises:
      ValueError: If the meta-method or dataset name is not supported
  """
  if method_name not in META_METHOD_CONFIGS:
    raise ValueError(
        f'Meta-method {method_name} not supported. '
        f'Available: {list(META_METHOD_CONFIGS.keys())}'
    )
  config = deepcopy(META_METHOD_CONFIGS[method_name])
  config.update({'method_name': method_name, 'dataset_name': dataset_name,
                 'dataset_config': get_data_config(dataset_name)})
  return config


def list_available_meta_methods() -> list[str]:
  """Get list of available meta-methods."""
  return list(META_METHOD_CONFIGS.keys())


def add_meta_method_config(method_name: str, config: dict[str, Any]) -> None:
  """
  Add a new meta-method configuration.

  Args:
      method_name: Name of the meta-method
      config: Configuration dictionary
  """
  META_METHOD_CONFIGS[method_name] = config


def get_meta_method_class(method_name: str) -> type[Method]:
  """
  Get the class of the meta-method by name.

  Args:
      method_name: Name of the meta-method

  Returns:
      Class of the meta-method

  Raises:
      ValueError: If method_name is not supported
  """
  if method_name not in META_METHOD_CONFIGS:
    raise ValueError(
        f'Meta-method {method_name} not supported. '
        f'Available: {list(META_METHOD_CONFIGS.keys())}'
    )
  return META_METHOD_CONFIGS[method_name]['class']
