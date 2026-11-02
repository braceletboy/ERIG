"""Run the dataset, model, method, and variant combinations in a config."""

from __future__ import annotations

import argparse
import torch
from experiments.compute_attributions import run_attributions
from experiments.compute_metrics import run_metrics
from pathlib import Path
from experiments.common import CONFIG_HELP, read_config, expand, print_plan


def main() -> None:
  """Run or preview the experiment suite selected on the command line."""
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CONFIG_HELP + '\n\nPreview example:\n'
          '  pipenv run python -m experiments.run --config configs/custom.yaml --output outputs/custom --plan')
  p.add_argument('--config', required=True,
                 help='Suite configuration file defining datasets, '
                      'models, methods, and variants')
  p.add_argument('--output', required=True,
                 help='Root directory for the individual experiment folders')
  p.add_argument('--plan', action='store_true',
                 help='Preview runs and settings in Rich tables without '
                      'loading models or data, or writing outputs')
  p.add_argument(
      '--debug', action='store_true',
      help='Save sample indices and run.log, and enable debug logging'
  )
  p.add_argument(
      '--stage', choices=['all', 'attributions', 'metrics'], default='all',
      help='Compute attributions, metrics, or both (default: %(default)s)'
  )
  a = p.parse_args()
  config = read_config(a.config)
  planned = []
  for name, config in expand(config):
    folder = Path(a.output)/name
    if a.plan:
      planned.append((name, config))
      continue
    if a.stage in ['all', 'attributions']:
      run_attributions(config, folder, debug=a.debug)
    if a.stage in ['all', 'metrics']:
      run_metrics(config, folder, debug=a.debug)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  if a.plan:
    print_plan(planned, a.output, a.stage)


if __name__ == '__main__':
  main()
