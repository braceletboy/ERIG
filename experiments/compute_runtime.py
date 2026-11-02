"""Measure attribution wall time with warmup and CUDA synchronization."""

from __future__ import annotations

import argparse
from typing import Any
import json
from rich import get_console
import time
import timm
import torch
from erig import create_method
from experiments.common import CONFIG_HELP, read_config, expand, print_plan
from erig.methods.model_config import get_model_id
from erig.methods.utils import (
    is_transformers_model, load_timm_model,
    load_transformers_model, get_transformers_transforms
)
from PIL import Image


def measure_runtime(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    method: str,
    steps: int,
    repeats: int = 5,
    warmup: int = 2,
    batch_size: int = 1,
    options: dict[str, Any] | None = None,
) -> dict[str, str | int | float | list[float] | list[int]]:
  """Measure attribution runtime after warming up the model.

  Args:
      model: Classifier returning logits for preprocessed inputs.
      inputs: Batch of images to attribute.
      method: Attribution method name.
      steps: Number of path intervals or Guided IG segments.
      repeats: Number of timed evaluations.
      warmup: Number of untimed evaluations before measurement.
      batch_size: Maximum number of model inputs processed together.

  Returns:
      Individual and mean runtimes in seconds, with workload settings.

  Raises:
      ValueError: If repeats is not positive or warmup is negative."""
  if repeats < 1 or warmup < 0:
    raise ValueError('Invalid repeat or warmup count')
  attribution_method = create_method(
      model, method, steps=steps, batch_size=batch_size, options=options)
  times = []
  for i in range(warmup+repeats):
    if inputs.is_cuda:
      torch.cuda.synchronize()
    start = time.perf_counter()
    attribution_method.attribute(inputs)
    if inputs.is_cuda:
      torch.cuda.synchronize()
    if i >= warmup:
      times.append(time.perf_counter()-start)
  return dict(method=method,
              steps=steps,
              seconds=times,
              mean_seconds=sum(times)/len(times),
              eval_batch_size=batch_size,
              batch_size=len(inputs),
              input_shape=list(inputs.shape),
              device=str(inputs.device))


def main() -> None:
  """Measure attribution runtimes specified in a YAML configuration."""
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CONFIG_HELP + '\n\nPreview example:\n'
          '  pipenv run python -m experiments.compute_runtime --config configs/runtime.yaml --plan')
  p.add_argument('--config', required=True,
                 help='YAML configuration; shared defaults load automatically')
  p.add_argument('--plan', action='store_true',
                 help='Preview runtime settings without loading models or data')
  args = p.parse_args()
  runs = list(expand(read_config(args.config)))
  if args.plan:
    print_plan(runs, None, stage='runtime')
    return
  loaded = None
  for _, config in runs:
    dataset, = config['datasets']
    model_name, = config['models']
    method_name, = config['methods']
    device = config['device']
    selection = (dataset, model_name, device, config['seed'])
    if selection != loaded:
      if loaded is not None:
        del model, inputs
      torch.manual_seed(config['seed'])
      model_id = get_model_id(model_name, dataset)
      if is_transformers_model(model_id):
        model = load_transformers_model(model_id).eval().to(device)
        transform, _ = get_transformers_transforms(model_id)
        shape = transform(Image.new('RGB', (512, 512))).shape
      else:
        model = load_timm_model(model_id).eval().to(device)
        shape = timm.data.resolve_model_data_config(model)['input_size']
      inputs = torch.rand(1, *shape, device=device)
      loaded = selection
    options = config['method_options']
    step_key = f'{method_name}_num_segments' if method_name == 'gig' else f'{method_name}_num_steps'
    result = measure_runtime(
        model, inputs, method_name, options.get(step_key, 200),
        repeats=config['runtime']['repeats'], warmup=config['runtime']['warmup'],
        batch_size=config['eval_batch_size'], options=options)
    get_console().print(
        json.dumps(dict(model=model_name, dataset=dataset, **result)),
        markup=False, highlight=False, soft_wrap=True)


if __name__ == '__main__':
  main()
