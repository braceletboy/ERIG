"""Compute evaluation metrics for a completed attribution run."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence
from os import PathLike
import argparse
import json
import torch
from erig.metrics.metric_config import get_metric_config
from pathlib import Path
from experiments.common import (
    CONFIG_HELP, read_config, setup_logging, write_json, expand, print_plan
)


def run_metrics(
    config: dict[str, Any],
    output: str | PathLike[str],
    names: Sequence[str] | None = None,
    *,
    debug: bool = False,
) -> None:
  """Evaluate saved attributions and record the metric settings.

  Args:
      config: Expanded settings for one model, dataset, and method.
      output: Run directory containing attributions and ``config.json``.
      names: Metrics to compute; defaults to the configured metric list.
      debug: Save the run log and enable debug messages.

  Raises:
      ValueError: If required inputs are missing, settings disagree with the
          attribution run, a metric is unsupported, or outputs already exist."""
  output = Path(output)
  if (not (output/'attributions.hdf5').is_file()
          or not (output/'config.json').is_file()):
    raise ValueError(
        'Saved attributions and config.json are required for metrics')
  saved = json.loads((output/'config.json').read_text())
  attribution_config = saved['attributions']
  model_name, = config['models']
  method_name, = config['methods']
  dataset, = config['datasets']
  for key, value in (('dataset_name', dataset), ('model_name', model_name),
                     ('method_name', method_name)):
    if value != attribution_config[key]:
      raise ValueError(
          f'Metric request disagrees with attribution configuration: {key}')
  for key, default in (
      ('seed', 42), ('num_samples', 1000), ('input_key', 'image')
  ):
    if config.get(key, default) != attribution_config[key]:
      raise ValueError(
          f'Metric request disagrees with attribution configuration: {key}')
  method_options = dict(config.get('method_defaults', {}).get(method_name, {}))
  method_options.update(config.get('method_options', {}))
  for supplied, stored in (
      (method_options, attribution_config['compute_config']),
      (config.get('dataset_options', {}), attribution_config['dataset_config'])
  ):
    if any(stored.get(key) != value for key, value in supplied.items()):
      raise ValueError(
          'Metric request disagrees with attribution method or dataset options'
      )
  metric_names = list(dict.fromkeys(names or config['metrics']))
  for name in metric_names:
    if name not in ('isc', 'aic', 'del', 'local', 'robust',
                    'arobust', 'linv', 'complex'):
      raise ValueError(f'Unsupported metric {name}')
    if (
        name == 'arobust'
        and attribution_config['method_name'] not in ('ig', 'idgi', 'erig')
    ):
      raise ValueError('arobust supports ig, idgi and erig only')
    path = output/f'{name}.hdf5'
    if path.exists() or name in saved['metrics']:
      raise ValueError(
          f'Output already exists for {name}. Use a new output directory.')
  setup_logging(output, debug=debug)
  for name in metric_names:
    resolved = get_metric_config(name)
    cls = resolved.pop('class')
    resolved.update(
        model_name=attribution_config['model_name'],
        dataset_name=attribution_config['dataset_name'],
        input_key=attribution_config['input_key'],
        attributions_dir=str(output.resolve()),
        output_dir=str(output.resolve()),
        seed=attribution_config['seed'],
        device=config.get('device', 'cpu'),
        metric_batch_size=config.get(
            'gradient_free_metric_batch_size', config.get('eval_batch_size', 1)
        ),
        eval_batch_size=config.get('eval_batch_size', 1), num_workers=0
    )
    if name in ('robust', 'arobust'):
      resolved['metric_batch_size'] = config.get(
          'gradient_metric_batch_size', resolved['metric_batch_size']
      )
    resolved['compute_config'].update(
        config.get('metric_defaults', {}).get(name, {}))
    saved['metrics'][name] = resolved
    write_json(output/'config.json', saved)
    metric = cls(resolved)
    metric.compute_metrics()
    del metric
    if torch.cuda.is_available():
      torch.cuda.empty_cache()


def main() -> None:
  """Evaluate the metrics selected by command-line arguments."""
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CONFIG_HELP + '\n\nPreview example:\n'
          '  pipenv run python -m experiments.compute_metrics --config configs/custom.yaml --output outputs/custom --plan')
  p.add_argument('--config', required=True,
                 help='Experiment configuration file with metric options')
  p.add_argument('--output', required=True,
                 help='Root directory containing the selected experiment folders')
  p.add_argument('--plan', action='store_true',
                 help='Preview metric settings without loading models or writing outputs')
  p.add_argument('--debug', action='store_true',
                 help='Save run.log and enable debug logging')
  args = p.parse_args()
  config = read_config(args.config)
  if args.plan:
    print_plan(list(expand(config)), args.output, stage='metrics')
    return
  for name, run_config in expand(config):
    run_metrics(run_config, Path(args.output)/name, debug=args.debug)


if __name__ == '__main__':
  main()
