# ERIG

**Entropy-Regularized Integrated Gradients for Stable Feature Attribution**

Rukmangadh Sai Myana, Sumit Kumar Jha, and Yanzhao Wu · WACV 2027

PyTorch implementation and experiments for ImageNet, ImageNet-S, and Oxford-IIIT
Pet. See the [presentation](docs/index.html) for the method and results.

## Installation

Use Python 3.12 and Pipenv. From the repository root:

```bash
pipenv sync
pipenv run python -m examples.explain_image
```

The example compares IG, IDGI, BIG, GIG, SmoothGrad, and ERIG on the bundled
[Indian cobra photograph](examples/indian_cobra.jpg) using pretrained MobileNetV4. It saves
[examples/indian_cobra_comparison.png](examples/indian_cobra_comparison.png) and uses CUDA when available. Model weights download
on the first run; no dataset download is needed. The image is ImageNet validation
sample `ILSVRC2012_val_00038480`, also shown in the paper's appendix.

The locked environment uses CUDA 13.0; GPU execution requires a compatible
NVIDIA driver.

## Explain an image

Pass a classifier that returns logits and inputs preprocessed for that model:

```python
from erig import create_method, visualize_attributions

model.eval()
method = create_method(model, method="erig", steps=200, temperature=3.0)
attributions = method.attribute(inputs)  # CHW or BCHW input; BCHW output
figure = visualize_attributions(attributions)
figure.savefig("attributions.png")
```

Inputs move to the model's device automatically. Attributions explain the
predicted class's softmax probability. Use `batch_size` to control model input
batching. Supported methods are `erig`, `ig`, `idgi`, `big`, `gig`, `smoothgrad`,
`aerig`, and `serig`.

To show source images alongside the maps, pass `images=inputs` to
`visualize_attributions`, with `mean` and `std` if the inputs are normalized.
A dictionary of named attribution tensors produces a method-comparison grid.

## Run experiments

Choose a YAML configuration and preview it before running:

```bash
pipenv run python -m experiments.run \
  --config configs/custom.yaml --output outputs/custom --plan
```

Omit `--plan` to execute. Dataset and checkpoint downloads use the Hugging Face
cache; set `HF_HOME` to choose its location.

| Configuration | Purpose |
|---|---|
| [main.yaml](configs/main.yaml) | Main ImageNet, ImageNet-S, and Pets experiments |
| [custom.yaml](configs/custom.yaml) | Editable two-image example |
| [ablations/](configs/ablations/) | Temperature, baselines, and other ablation studies |
| [runtime.yaml](configs/runtime.yaml) | Attribution runtime measurements |

`--config` is required. [defaults.yaml](configs/defaults.yaml) loads automatically,
followed by inherited files (`extends`) and your selected YAML. Nested settings
merge; lists replace inherited lists; `{}` clears a mapping.

Set datasets, models, methods, metrics, devices, and batch sizes in YAML. If
`models` is omitted, each dataset selects its configured models. ImageNet-S
defaults to localization metrics only. Explicit batch sizes override model presets.

Use `--stage attributions` or `--stage metrics` to run one stage, and `--help`
for all operational arguments. Each run saves its resolved configuration and
HDF5 results under `<output>/<model>/<method>/<variant>/<dataset>/`. Existing
results are protected from overwriting.

For runtime measurements:

```bash
pipenv run python -m experiments.compute_runtime \
  --config configs/runtime.yaml --plan
```

Omit `--plan` to measure runtime on generated inputs with a pretrained model.

## Export and plot results

Open [Results.ipynb](notebooks/Results.ipynb), set `root` to your results folder,
and select a statistic to plot. It reads saved HDF5 aggregates directly.
For optional CSV export, set `csv_path` to a destination and run the final cell.

## Presentation

Host the [docs/](docs/) folder on GitHub Pages. Figures, styles, and math fonts
are bundled locally. After editing an equation's `data-tex` in `docs/index.html`,
regenerate its typesetting:

```bash
node docs/render_presentation_math.cjs
```

## License and attribution

Original project code is licensed under [Apache-2.0](LICENSE). Incorporated code,
model weights, and datasets retain their own licenses. Source comments credit
[PAIR-code/saliency](https://github.com/PAIR-code/saliency),
[TMLR-IDGI](https://github.com/ShreeSinghi/TMLR-IDGI), and the
[Integrated Gradients visualization library](https://github.com/ankurtaly/Integrated-Gradients/blob/master/VisualizationLibrary/visualization_lib.py).
KaTeX is bundled under its [MIT license](docs/assets/vendor/katex/LICENSE).
Citation metadata is in [CITATION.cff](CITATION.cff).
