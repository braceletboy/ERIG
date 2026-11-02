"""Check configuration precedence and the public experiment commands."""
import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from experiments.common import read_config, expand

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_defaults_overrides_and_main_selections(tmp_path):
  main = dict(expand(read_config(ROOT/'configs/main.yaml')))
  assert len(main) == 102
  assert all(run['metrics'] == ['local'] for run in main.values()
             if run['datasets'] == ['imagenet-s'])
  parent = tmp_path/'base.yaml'
  parent.write_text(yaml.safe_dump({
      'datasets': ['imagenet-s'], 'models': ['vit'], 'methods': ['erig'],
      'method_defaults': {'erig': {'erig_exp_temperature': 0.3}},
  }))
  child = tmp_path/'custom.yaml'
  child.write_text(yaml.safe_dump({
      'extends': 'base.yaml', 'eval_batch_size': 2,
      'method_defaults': {'erig': {'erig_num_steps': 4}},
  }))
  _, selected = next(expand(read_config(child)))
  assert selected['eval_batch_size'] == 2
  assert selected['method_options'] == {
      'erig_baseline': 'zero', 'erig_exp_temperature': 0.3, 'erig_num_steps': 4}
  assert selected['input_key'] == 'image' and selected['metrics'] == ['local']
  assert dict(expand(read_config(ROOT/'configs/main.yaml'))) == main


@pytest.mark.parametrize('command', ['run', 'compute_attributions', 'compute_metrics', 'compute_runtime'])
def test_cli_requires_yaml_and_can_preview(command, tmp_path, monkeypatch, capsys):
  module = importlib.import_module(f'experiments.{command}')
  monkeypatch.setattr('sys.argv', [command])
  with pytest.raises(SystemExit) as missing:
    module.main()
  assert missing.value.code == 2 and '--config' in capsys.readouterr().err
  monkeypatch.setattr('sys.argv', [command, '--help'])
  with pytest.raises(SystemExit) as help_result:
    module.main()
  assert help_result.value.code == 0
  assert 'configs/defaults.yaml loads automatically' in capsys.readouterr().out
  output = tmp_path/'runs'
  args = [command, '--config', str(ROOT/'configs/custom.yaml')]
  if command != 'compute_runtime':
    args += ['--output', str(output)]
  monkeypatch.setattr('sys.argv', args + ['--models', 'vit'])
  with pytest.raises(SystemExit) as removed:
    module.main()
  assert removed.value.code == 2
  monkeypatch.setattr('sys.argv', args + ['--plan'])
  module.main()
  assert '1 run' in capsys.readouterr().out
  assert not output.exists()


def test_runtime_excludes_warmup_from_measurements(monkeypatch):
  from experiments import compute_runtime
  method = Mock()
  monkeypatch.setattr(compute_runtime, 'create_method', Mock(return_value=method))
  clock = iter([0, 1, 2, 5, 6, 10])
  monkeypatch.setattr(compute_runtime.time, 'perf_counter', lambda: next(clock))
  result = compute_runtime.measure_runtime(
      torch.nn.Identity(), torch.zeros(1, 3, 8, 8), 'erig', 4, repeats=2, warmup=2)
  assert method.attribute.call_count == 4
  assert result['seconds'] == [3, 4] and result['mean_seconds'] == 3.5
