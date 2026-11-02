"""Configuration loading, JSON output, and experiment logging."""

from __future__ import annotations

from typing import Any
from collections.abc import Iterator, Sequence
from copy import deepcopy
from os import PathLike
import json
import logging
import os
from pathlib import Path
import yaml
from rich import box, get_console
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


CONFIG_HELP = """Configuration:
  --config is required. configs/defaults.yaml loads automatically, followed by
  inherited YAML files (extends), then the selected file. Nested settings merge;
  lists replace inherited lists. An empty mapping ({}) clears its defaults.
  Put datasets, models, methods, metrics, device, and batch sizes in YAML.
  Select configs/main.yaml for the main experiments, or edit
  configs/custom.yaml for your own run. No experiment config loads implicitly."""


def _merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> None:
  """Merge nested settings, replacing lists, scalars, and explicit empty maps."""
  for key, value in overrides.items():
    if isinstance(value, dict) and value and isinstance(base.get(key), dict):
      _merge_settings(base[key], value)
    else:
      base[key] = deepcopy(value)


def read_config(path: str | PathLike[str]) -> dict[str, Any]:
  """Load shared defaults, inherited YAML files, and the selected settings.

  Args:
      path: YAML configuration file; relative extends paths use its directory.

  Returns:
      Merged settings, with the selected file taking precedence.

  Raises:
      ValueError: If a file is not a mapping or inheritance contains a cycle.
  """
  defaults = Path(__file__).resolve().parents[1] / 'configs/defaults.yaml'
  config = yaml.safe_load(defaults.read_text())
  layers = []
  seen = set()
  path = Path(path).resolve()
  while path != defaults:
    if path in seen:
      raise ValueError(f'Circular configuration inheritance: {path}')
    seen.add(path)
    layer = yaml.safe_load(path.read_text())
    if not isinstance(layer, dict):
      raise ValueError(f'Configuration must be a mapping: {path}')
    parent = layer.pop('extends', None)
    layers.append(layer)
    if not parent:
      break
    path = (path.parent / parent).resolve()
  for layer in reversed(layers):
    _merge_settings(config, layer)
  return config


def expand(config: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
  """Expand dataset, model, method, and variant selections into runs.

  Args:
      config: Suite settings with dataset defaults. An explicit models list
          overrides the models for each selected dataset.

  Yields:
      Pairs of relative output paths and settings for each run.
  """
  for key in ('methods', 'datasets'):
    values = config.get(key)
    if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values):
      raise ValueError(f'{key} must be a nonempty list of names')
  for dataset in config['datasets']:
    selected = deepcopy(config.get('dataset_defaults', {}).get(dataset, {}))
    models = config.get('models', selected.get('models'))
    if not isinstance(models, list) or not models or not all(
            isinstance(model, str) and model for model in models):
      raise ValueError(f'models must be a nonempty list of names for {dataset}')
    for model in models:
      for method in config['methods']:
        for variant in config.get('variants', [{'name': 'default'}]):
          run = deepcopy(selected)
          run.update(run.pop('batch_sizes', {}).get(model, {}))
          run.update(deepcopy(config))
          for key in ('variants', 'dataset_defaults'):
            run.pop(key, None)
          run.update(models=[model], methods=[method], datasets=[dataset])
          defaults = deepcopy(run.pop('method_defaults', {}).get(method, {}))
          defaults.update(run.get('method_options', {}))
          defaults.update(variant.get('method_options', {}))
          run['method_options'] = {
              key: value for key, value in defaults.items()
              if key.startswith(method+'_')
          }
          slug = model.replace('/', '_').replace(':', '_')
          yield f'{slug}/{method}/{variant["name"]}/{dataset}', run


def print_plan(
    runs: Sequence[tuple[str, dict[str, Any]]],
    output: str | PathLike[str] | None,
    stage: str = 'all',
) -> None:
  """Display planned runs and their settings without executing them.

  Args:
      runs: Expanded run names and configurations, including YAML overrides.
      output: Root directory for experiment folders, or None for runtime output.
      stage: Preview attributions, metrics, both, or runtime measurements.
  """
  console = get_console()
  console.rule('[bold cyan]Experiment plan')
  stage_label = {'all': 'Attributions and metrics',
                 'attributions': 'Attributions',
                 'metrics': 'Metrics', 'runtime': 'Runtime measurements'}[stage]
  noun = 'run' if len(runs) == 1 else 'runs'
  console.print(f'[bold]{len(runs)} {noun}[/bold] · {stage_label}')
  if output is not None:
    console.print(Text(f'Output root: {output}', style='dim'))
  for index, (name, config) in enumerate(runs, 1):
    table = Table(title=Text(f'{index}. {name}', style='bold'),
                  show_header=False, box=box.SIMPLE, expand=True)
    table.add_column('Setting', style='cyan', no_wrap=True)
    table.add_column('Value', overflow='fold')
    if stage == 'runtime':
      table.add_row('Input', '1 generated image')
    else:
      samples = config.get('num_samples', 1000)
      table.add_row('Samples', str(samples) if samples > 0 else 'All')
    table.add_row('Seed', str(config.get('seed', 42)))
    table.add_row('Device', Text(config.get('device', 'cpu')))
    eval_batch = config.get('eval_batch_size', 1)
    table.add_row('Evaluation batch', str(eval_batch))
    if stage in ('all', 'metrics'):
      metric_batch = config.get('gradient_free_metric_batch_size', eval_batch)
      table.add_row('Gradient-free metric batch', str(metric_batch))
      table.add_row('Gradient metric batch', str(
          config.get('gradient_metric_batch_size', metric_batch)))
      table.add_row('Metrics', Text(', '.join(config.get('metrics', []))))
    if output is not None:
      table.add_row('Output', Text(str(Path(output)/name)))
    if stage == 'runtime':
      table.add_row('Warmup', str(config['runtime']['warmup']))
      table.add_row('Timed repeats', str(config['runtime']['repeats']))
    for label, options in [
        ('Method options', config.get('method_options', {})),
        ('Dataset options', config.get('dataset_options', {}))
    ]:
      if options:
        table.add_row(label, Syntax(
            yaml.safe_dump(options, sort_keys=False).rstrip(),
            'yaml', background_color='default', word_wrap=True
        ))
    if stage in ('all', 'metrics'):
      options = {metric: dict(config.get('metric_defaults', {}).get(metric, {}))
                 for metric in config.get('metrics', [])}
      options = {metric: values for metric, values in options.items() if values}
      if options:
        table.add_row('Metric options', Syntax(
            yaml.safe_dump(options, sort_keys=False).rstrip(),
            'yaml', background_color='default', word_wrap=True
        ))
    console.print(table)


def write_json(path: str | PathLike[str], data: Any) -> None:
  """Write JSON without leaving a partially written destination.

  Args:
      path: Destination file in an existing directory.
      data: JSON-serializable content."""
  path = Path(path)
  temp = path.with_suffix(path.suffix + '.tmp')
  temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
  os.replace(temp, path)


def setup_logging(
    directory: str | PathLike[str], *, debug: bool = False
) -> None:
  """Configure console output and optional debug logging.

  Args:
      directory: Existing run directory for the debug log.
      debug: Enable DEBUG messages and save them in ``run.log``."""
  handlers: list[logging.Handler] = [
      RichHandler(markup=True, rich_tracebacks=True)
  ]
  if debug:
    handlers.append(logging.FileHandler(Path(directory)/'run.log'))
  logging.basicConfig(
      level=logging.DEBUG if debug else logging.INFO,
      format='%(message)s',
      datefmt='%m/%d/%Y %H:%M:%S',
      handlers=handlers,
      force=True
  )
