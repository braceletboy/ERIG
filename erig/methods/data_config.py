"""Dataset identifiers, splits and input columns."""

from __future__ import annotations

from typing import Any
from copy import deepcopy


DATA_CONFIGS = {
    'imagenet-1k-wds': {
        'dataset_name': 'timm/imagenet-1k-wds',
        'task_name': None,
        'validation_key': 'validation',
        'data_files': {
            'validation': 'imagenet1k-validation-*.tar',
        },
    },
    'imagenet-s': {
        'dataset_name': 'braceletboy/imagenet-s',
        'task_name': None,
        'validation_key': 'validation',
    },
    'oxford-iiit-pet': {
        'dataset_name': 'dpdl-benchmark/oxford_iiit_pet',
        'task_name': None,
        'validation_key': 'test',
    },
}


def get_data_config(dataset_name: str) -> dict[str, Any]:
  """Get a dataset configuration or the mode for already-provided inputs.

  Args:
      dataset_name: Registered dataset name or ``attributions``.

  Returns:
      An independent configuration dictionary. ``attributions`` requires no
      dataset download or split configuration.

  Raises:
      ValueError: If the dataset name is unsupported.
  """
  if dataset_name == 'attributions':
    return {'dataset_name': 'attributions'}
  if dataset_name not in DATA_CONFIGS:
    raise ValueError(
        f'Dataset {dataset_name} not supported. Available: '
        f'{list(DATA_CONFIGS)} or attributions')
  return deepcopy(DATA_CONFIGS[dataset_name])


def list_available_datasets() -> list[str]:
  """Get list of available datasets."""
  return list(DATA_CONFIGS.keys())


def add_data_config(dataset_name: str, config: dict[str, Any]) -> None:
  """
Add a new dataset configuration.

Args:
    dataset_name: Name of the dataset
    config: Configuration dictionary
"""
  DATA_CONFIGS[dataset_name] = config
