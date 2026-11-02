"""Small generated inputs and an isolated, offline test environment."""
import os
import tempfile

_CACHE = tempfile.TemporaryDirectory(prefix='erig-tests-')
os.environ.update(HF_HOME=_CACHE.name, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')

import pytest
import torch


@pytest.fixture(scope='session', autouse=True)
def offline_environment():
  previous_threads = torch.get_num_threads()
  torch.set_num_threads(2)
  yield
  torch.set_num_threads(previous_threads)
  _CACHE.cleanup()


@pytest.fixture
def problem():
  torch.manual_seed(17)
  model = torch.nn.Sequential(
      torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Softplus(),
      torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)
  ).eval()
  images = torch.rand(1, 3, 32, 32)
  return model, images
