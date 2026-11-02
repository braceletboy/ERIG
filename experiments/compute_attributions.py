"""Compute attributions for one configuration and save them as HDF5."""

from __future__ import annotations

from typing import Any
from os import PathLike
import argparse
from pathlib import Path

import h5py
from erig import METHODS
from erig.methods.method_config import get_method_config
from erig.methods.meta_method_config import get_meta_method_config
from experiments.common import (
    CONFIG_HELP, read_config, write_json, setup_logging,
    expand, print_plan
)


def run_attributions(
    config: dict[str, Any], output: str | PathLike[str], *, debug: bool = False
) -> None:
  """Compute and save attributions for one experiment.

  Args:
      config: Expanded settings for one model, dataset, and method.
      output: Directory for attributions and resolved settings.
      debug: Save sample indices and enable debug logging.

  Raises:
      ValueError: If outputs already exist, the method is unsupported, or no
          correctly classified examples produce attributions."""
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  for path in (output/'attributions.hdf5', output/'config.json',
               output/'samples.json', output/'accepted_samples.json'):
    if path.exists():
      raise ValueError(
          f'Output already exists: {path}. Use a new output directory.')
  model_name, = config['models']
  name, = config['methods']
  dataset, = config['datasets']
  setup_logging(output, debug=debug)
  if name not in METHODS:
    raise ValueError(f'Unsupported method: {name}')
  meta = name == 'smoothgrad'
  resolved = (get_meta_method_config if meta
              else get_method_config)(name, dataset)
  cls = resolved.pop('class')
  resolved.update(model_name=model_name,
                  dataset_name=dataset,
                  input_key=config.get('input_key', 'image'),
                  num_samples=config.get('num_samples', 1000),
                  eval_batch_size=config.get('eval_batch_size', 1),
                  seed=config.get('seed', 42),
                  device=config.get('device', 'cpu'),
                  output_dir=str(output.resolve()),
                  no_progress_bar=True, compile=False)
  resolved['dataset_config'].update(config.get('dataset_options', {}))
  options = dict(config.get('method_defaults', {}).get(name, {}))
  options.update(config.get('method_options', {}))
  resolved['compute_config'].update(options)
  write_json(output/'config.json', {'attributions': resolved, 'metrics': {}})
  method = cls(resolved)
  if debug:
    write_json(
        output/'samples.json',
        dict(dataset=dataset,
             indices=method.sample_indices,
             seed=resolved['seed'])
    )
  method.compute_attributions()
  with h5py.File(output/'attributions.hdf5', 'r') as f:
    if not len(f):
      raise ValueError(
          'No correctly classified samples produced attributions; '
          'run is incomplete'
      )
    if debug:
      accepted = [int(f[k]['original_example_idx'][()]) for k in f]
      write_json(output/'accepted_samples.json', dict(
          selected_positions=accepted,
          dataset_indices=[method.sample_indices[i] for i in accepted]
      ))


def main() -> None:
  """Run or preview an attribution experiment from command-line arguments."""
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CONFIG_HELP + '\n\nPreview example:\n'
          '  pipenv run python -m experiments.compute_attributions --config configs/custom.yaml --output outputs/custom --plan')
  p.add_argument('--config', required=True,
                 help='Experiment configuration file; '
                      'supports extends')
  p.add_argument('--output', required=True,
                 help='Root directory for experiment folders; '
                      'existing outputs are preserved')
  p.add_argument('--plan', action='store_true',
                 help='Preview runs and settings in Rich tables without '
                      'loading models or data, or writing outputs')
  p.add_argument('--debug', action='store_true',
                 help='Save sample indices and run.log, and enable debug logs')
  args = p.parse_args()
  config = read_config(args.config)
  if args.plan:
    print_plan(list(expand(config)), args.output, stage='attributions')
    return
  for name, run_config in expand(config):
    run_attributions(run_config, Path(args.output)/name, debug=args.debug)


if __name__ == '__main__':
  main()
