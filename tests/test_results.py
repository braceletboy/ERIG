"""Exercise saved attributions, every metric group, and notebook exports."""
import json
from pathlib import Path

from datasets import Dataset
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from erig.methods.method import Method
from erig.metrics.metric import Metric
from experiments import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(('metric', 'scores'), [
    ('isc', {'isc'}),
    ('aic', {'aic'}),
    ('del', {'deletion_auc'}),
    ('local', {'pixel_accuracy', 'pixel_f1', 'iou'}),
    ('robust', {'max_sensitivity', 'local_lipschitz', 'ris'}),
    ('arobust', {'adv_sensitivity', 'adv_lipschitz', 'ais'}),
    ('linv', {'linv_conv', 'linv_pca'}),
    ('complex', {'entropy_complexity', 'histogram_entropy', 'jpeg_ratio',
                 'jpeg_rmse', 'gini_sparsity'}),
])
def test_experiment_saves_metrics_and_filters_mismatches(metric, scores, problem, tmp_path, monkeypatch):
  model, images = problem
  if metric == 'aic':
    images = torch.nn.functional.interpolate(images, size=(192, 192))
  target = model(images).argmax(-1).item()
  mask = torch.full((1, *images.shape[-2:]), 2)
  mask[:, 8:24, 8:24] = 1
  dataset = Dataset.from_dict({
      'image': [images[0].tolist()]*2, 'mask': [mask.tolist()]*2,
      'label': [target, (target+1) % 3],
  })
  monkeypatch.setattr('erig.methods.method.load_dataset', lambda *args, **kwargs: dataset)
  def transform(example):
    return {key: torch.tensor(value) if isinstance(value, list) else value
            for key, value in example.items()}
  monkeypatch.setattr(Method, '_get_model_and_input_transform', lambda self: (model, transform))
  monkeypatch.setattr(Metric, '_get_model', lambda self: model)
  monkeypatch.setattr('erig.methods.utils.cpu_count', lambda: 1)
  config = {
      'datasets': ['oxford-iiit-pet'], 'models': ['vit'], 'methods': ['erig'],
      'metrics': [metric], 'num_samples': 2, 'device': 'cpu', 'eval_batch_size': 2,
      'gradient_free_metric_batch_size': 1, 'gradient_metric_batch_size': 1,
      'method_options': {'erig_num_steps': 4},
      'metric_defaults': {
          'isc': {'isc_num_partitions': 2}, 'aic': {'aic_num_partitions': 3},
          'del': {'del_num_workers': 1}, 'local': {'local_batch_size': 1},
          'linv': {'linv_batch_size': 1}, 'robust': {'robust_num_perturbations': 2},
          'arobust': {'arobust_num_iterations': 2, 'arobust_num_restarts': 1,
                     'arobust_num_path_points': 3},
      },
  }
  path = tmp_path/'experiment.yaml'
  path.write_text(yaml.safe_dump(config))
  output = tmp_path/'runs'
  monkeypatch.setattr('sys.argv', ['run', '--config', str(path), '--output', str(output), '--debug'])
  run.main()
  folder = output/'vit/erig/default/oxford-iiit-pet'
  with h5py.File(folder/'attributions.hdf5') as saved:
    assert len(saved) == 1
    attrs = saved[next(iter(saved))]['attributions'][()]
    assert attrs.shape == images.shape[1:] and np.isfinite(attrs).all()
  assert json.loads((folder/'accepted_samples.json').read_text())['dataset_indices'] == [0]
  settings = json.loads((folder/'config.json').read_text())
  assert settings['attributions']['compute_config']['erig_num_steps'] == 4
  assert set(settings['metrics']) == {metric}
  with h5py.File(folder/f'{metric}.hdf5') as saved:
    assert set(saved['aggregate']) == scores
    assert len(saved['individual']) == 1
    assert any(np.isfinite(saved['aggregate'][key][()]) for key in scores)
  protected = {p: p.read_bytes() for p in folder.iterdir() if p.suffix in ('.json', '.hdf5')}
  with pytest.raises(ValueError, match='Output already exists'):
    run.main()
  assert all(path.read_bytes() == content for path, content in protected.items())


def test_notebook_reads_saved_values_and_exports_only_on_request(tmp_path):
  notebook = json.loads((ROOT/'notebooks/Results.ipynb').read_text())
  code = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
          if cell['cell_type'] == 'code'}
  folder = tmp_path/'runs'/'example'
  folder.mkdir(parents=True)
  with h5py.File(folder/'linv.hdf5', 'w') as saved:
    saved['aggregate/linv_conv'] = 3.125
    saved['aggregate/linv_pca'] = float('nan')
    saved['individual/example_0/linv_conv'] = 99.0
    saved['aggregate/curve'] = [0.1, 0.2]
  with h5py.File(folder/'attributions.hdf5', 'w') as saved:
    saved['example_0/attributions'] = [1.0, 2.0]
  namespace = {}
  exec(code['settings'], namespace)
  namespace['root'] = tmp_path/'runs'
  exec(code['load-results'], namespace)
  expected = pd.DataFrame([
      ['example', 'linv', 'linv_conv', 3.125],
      ['example', 'linv', 'linv_pca', float('nan')],
  ], columns=['run', 'metric', 'statistic', 'value'])
  pd.testing.assert_frame_equal(namespace['scores'], expected, check_exact=True)
  exec(code['export-results'], namespace)
  assert not list(tmp_path.rglob('*.csv'))
  destination = tmp_path/'export'/'scores.csv'
  namespace['csv_path'] = destination
  exec(code['export-results'], namespace)
  pd.testing.assert_frame_equal(pd.read_csv(destination), expected, check_exact=True)


def test_visualization_renders_comparisons_without_changing_inputs(tmp_path):
  import matplotlib.pyplot as plt
  from erig import visualize_attributions
  attribution = torch.tensor([[[0., -1.], [1., -1.]], [[0., 0.], [-1., 2.]]])
  batch = torch.stack((attribution, torch.ones_like(attribution))).requires_grad_()
  normalized = torch.stack((torch.full((3, 2, 2), -1.), torch.ones(3, 2, 2)))
  original = batch.detach().clone()
  figure = visualize_attributions(
      {'ERIG': batch, 'Zero': torch.zeros_like(batch)}, images=normalized,
      mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
  try:
    assert len(figure.axes) == 6
    np.testing.assert_allclose(figure.axes[1].images[0].get_array(), [[0, 0.1], [0.55, 1]])
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), np.full((2, 2, 3), 63))
    np.testing.assert_array_equal(figure.axes[3].images[0].get_array(), np.full((2, 2, 3), 191))
    for index, value in ((2, 0), (4, 1), (5, 0)):
      np.testing.assert_array_equal(figure.axes[index].images[0].get_array(), np.full((2, 2), value))
    torch.testing.assert_close(batch, original)
    assert batch.grad is None and batch.requires_grad
    figure.savefig(tmp_path/'comparison.svg')
    assert (tmp_path/'comparison.svg').stat().st_size > 0
  finally:
    plt.close(figure)
  with pytest.raises(ValueError, match='match the attributions'):
    visualize_attributions(attribution, images=normalized)
