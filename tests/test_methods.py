"""Check the public API and the attribution equations on a known classifier."""
from copy import deepcopy
import os

import pytest
import torch
from erig import METHODS, create_method


@pytest.mark.parametrize('name', METHODS)
def test_method_api(name, problem):
  model, images = problem
  options = {'smoothgrad_num_samples': 2, 'smoothgrad_base_num_steps': 3,
             'big_max_sigma': 1.0}
  original = deepcopy(options)
  result = create_method(model, name, steps=4, options=options).attribute(images)
  assert result.shape == images.shape and torch.isfinite(result).all()
  assert not result.requires_grad
  assert options == original
  assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('name', ['erig', 'ig'])
def test_attributions_match_analytic_probability_gradients(name):
  # Logits (a·x, 0) give a sigmoid probability with an analytic gradient.
  weights = torch.linspace(-0.3, 0.5, 12)
  model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2, bias=False))
  with torch.no_grad():
    model[1].weight.copy_(torch.stack((weights, torch.zeros_like(weights))))
  images = torch.linspace(0.1, 0.9, 12).reshape(1, 3, 2, 2)
  steps, temperature = 5, 0.07
  probabilities = torch.sigmoid(torch.linspace(0, 1, steps+1) * (weights * images.flatten()).sum())
  gradients = probabilities[:, None] * (1-probabilities[:, None]) * weights
  if name == 'erig':
    expected = (torch.softmax(gradients[1:].square()/temperature, dim=1)
                * probabilities.diff()[:, None]).sum(0)
  else:
    # IG retains the research implementation's inclusive endpoints and 1/K scale.
    expected = gradients.sum(0) * images.flatten() / steps
  for batch_size in (1, 3):
    actual = create_method(model, name, steps=steps, temperature=temperature,
                           batch_size=batch_size).attribute(images)
    torch.testing.assert_close(actual.flatten(), expected, rtol=1e-5, atol=1e-7)
  if name == 'erig':
    torch.testing.assert_close(actual.sum(), probabilities[-1]-probabilities[0])


def test_tensor_api_preserves_inputs_and_training_mode(problem):
  model, images = problem
  model.train()
  image = images[0].clone().requires_grad_()
  before = image.detach().clone()
  method = create_method(model, steps=3)
  actual = method.attribute(image)
  expected = method.attribute(images)
  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(image, before)
  assert image.grad is None and image.requires_grad and model.training
  with pytest.raises(ValueError, match='nonempty floating-point CHW or BCHW'):
    method.attribute(torch.empty(0, 3, 32, 32))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('ERIG_GPU_TEST') != '1', reason='Set ERIG_GPU_TEST=1')
def test_cuda_attribution_matches_cpu(problem, tmp_path):
  if not torch.cuda.is_available():
    pytest.skip('CUDA is unavailable')
  model, images = problem
  expected = create_method(model, steps=4).attribute(images)
  actual = create_method(deepcopy(model).cuda(), steps=4).attribute(images)
  assert actual.is_cuda
  torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-7)
  import matplotlib.pyplot as plt
  from erig import visualize_attributions
  figure = visualize_attributions(actual, images=images)
  try:
    figure.savefig(tmp_path/'cuda.png')
    assert (tmp_path/'cuda.png').stat().st_size > 0
  finally:
    plt.close(figure)
