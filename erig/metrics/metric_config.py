"""Default parameters and classes for evaluation metrics."""

from __future__ import annotations

from typing import Any
import os
from copy import deepcopy
from erig.metrics.isc import ISC
from erig.metrics.aic import AIC
from erig.metrics.linv import LINV
from erig.metrics.complex import Complexity
from erig.metrics.robust import Robust
from erig.metrics.arobust import ARobust
from erig.metrics.local import Local
from erig.metrics.deletion import DEL


METRIC_CONFIGS = {
    'isc': {
        'class': ISC,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'isc_num_partitions': 10,
        },
    },
    'aic': {
        'class': AIC,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'aic_num_partitions': 10,
        },
    },
    'linv': {
        'class': LINV,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'linv_batch_size': 512,
        },
    },
    'complex': {
        'class': Complexity,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {},
    },
    'robust': {
        'class': Robust,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'robust_num_perturbations': 50,
            'robust_max_examples': 100,
        },
    },
    'arobust': {
        'class': ARobust,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'arobust_num_restarts': 3,
            'arobust_num_iterations': 100,
            'arobust_num_path_points': 20,
            'arobust_max_examples': 100,
        },
    },
    'local': {
        'class': Local,
        'model_name': 'eva02',
        'dataset_name': 'braceletboy/imagenet-s',
        'input_key': 'image',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {'local_batch_size': 128},
    },
    'del': {
        'class': DEL,
        'model_name': 'eva02',
        'dataset_name': 'timm/imagenet-1k-wds',
        'input_key': 'jpg',
        'output_dir': None,
        'seed': 42,
        'log_level': 'INFO',
        'compile': False,
        'eval_batch_size': 128,
        'max_examples': -1,
        'compute_config': {
            'del_num_workers': min(os.cpu_count(), 32),
        },
    },
}


def get_metric_config(metric_name: str) -> dict[str, Any]:
  """
Get metric configuration by name.

Args:
    metric_name: Name of the metric

Returns:
    Dictionary containing metric configuration

Raises:
    ValueError: If metric_name is not supported
"""
  if metric_name not in METRIC_CONFIGS:
    raise ValueError(
        f'Metric {metric_name} not supported. Available: {list(METRIC_CONFIGS.keys())}')
  config = deepcopy(METRIC_CONFIGS[metric_name])
  return config


def list_available_metrics() -> list[str]:
  """Get list of available metrics."""
  return list(METRIC_CONFIGS.keys())


def add_metric_config(metric_name: str, config: dict[str, Any]) -> None:
  """
Add a new metric configuration.

Args:
    metric_name: Name of the metric
    config: Configuration dictionary
"""
  METRIC_CONFIGS[metric_name] = config
