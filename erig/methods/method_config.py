"""Default parameters and classes for attribution methods."""

from __future__ import annotations

from typing import Any
from erig.methods.method import Method
import os
from copy import deepcopy
from erig.methods.big import BlurIntegratedGradients
from erig.methods.ig import IntegratedGradients
from erig.methods.erig import EntropyRegularizedIntegratedGradients
from erig.methods.serig import SignedERIG
from erig.methods.aerig import AdaptiveERIG
from erig.methods.idgi import ImportantDirectionGradientIntegration
from erig.methods.gig import GuidedIG
from erig.methods.vg import VanillaGradients
from erig.methods.data_config import get_data_config


METHOD_CONFIGS = {
    'ig': {
        'class': IntegratedGradients,
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
            'ig_baseline': 'zero',
            'ig_num_steps': 200,
        },
    },
    'erig': {
        'class': EntropyRegularizedIntegratedGradients,
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
            'erig_baseline': 'zero',
            'erig_num_steps': 200,
            'erig_exp_temperature': 3.0,
        },
    },
    'serig': {
        'class': SignedERIG,
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
            'serig_baseline': 'zero',
            'serig_num_steps': 200,
            'serig_exp_temperature': 3.0,
        },
    },
    'aerig': {
        'class': AdaptiveERIG,
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
            'aerig_baseline': 'zero',
            'aerig_num_steps': 200,
            'aerig_tau_schedule': 'linear',
            'aerig_tau_start': 3.0,
            'aerig_tau_end': 3.0,
        },
    },
    'idgi': {
        'class': ImportantDirectionGradientIntegration,
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
            'idgi_baseline': 'zero',
            'idgi_num_steps': 200,
        },
    },
    'big': {
        'class': BlurIntegratedGradients,
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
            'big_num_steps': 200,
            'big_max_sigma': 50.0,
            'big_use_sqrt': False,
            'big_gaussian_grad_step': 0.01,
        },
    },
    'gig': {
        'class': GuidedIG,
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
            'gig_baseline': 'zero',
            'gig_num_segments': 200,
            'gig_percentile': 0.25,
            'gig_max_alpha_dist': 1.0,
        },
    },
    'vg': {
        'class': VanillaGradients,
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
        'compute_config': {},
    },
}


def get_method_config(method_name: str, dataset_name: str) -> dict[str, Any]:
  """
  Get method configuration by name.

  Args:
      method_name: Name of the method
      dataset_name: Name of the dataset

  Returns:
      Dictionary containing method configuration

  Raises:
      ValueError: If the method or dataset name is not supported
  """
  if method_name not in METHOD_CONFIGS:
    raise ValueError(
        f'Method {method_name} not supported. '
        f'Available: {list(METHOD_CONFIGS.keys())}'
    )
  config = deepcopy(METHOD_CONFIGS[method_name])
  config.update({'method_name': method_name, 'dataset_name': dataset_name,
                 'dataset_config': get_data_config(dataset_name)})
  return config


def list_available_methods() -> list[str]:
  """Get list of available methods."""
  return list(METHOD_CONFIGS.keys())


def add_method_config(method_name: str, config: dict[str, Any]) -> None:
  """
  Add a new method configuration.

  Args:
      method_name: Name of the method
      config: Configuration dictionary
  """
  METHOD_CONFIGS[method_name] = config


def get_method_class(method_name: str) -> type[Method]:
  """
  Get the class of the method by name.

  Args:
      method_name: Name of the method

  Returns:
      Class of the method

  Raises:
      ValueError: If method_name is not supported
  """
  if method_name not in METHOD_CONFIGS:
    raise ValueError(
        f'Method {method_name} not supported. '
        f'Available: {list(METHOD_CONFIGS.keys())}'
    )
  return METHOD_CONFIGS[method_name]['class']
