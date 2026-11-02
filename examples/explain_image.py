"""Compare the paper's attribution methods on the bundled Indian cobra image."""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import timm
import torch
from rich import print

from erig import create_method, visualize_attributions
from erig.methods.model_config import get_model_id


def main() -> None:
  """Save a comparison using MobileNetV4 and the main method settings."""
  torch.manual_seed(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = timm.create_model(
      get_model_id('mobilenetv4', 'imagenet-1k-wds'), pretrained=True
  ).eval().to(device)
  preprocessing = timm.data.resolve_model_data_config(model)
  transform = timm.data.create_transform(**preprocessing, is_training=False)
  with Image.open(Path(__file__).with_name('indian_cobra.jpg')) as source:
    image = transform(source.convert('RGB')).unsqueeze(0).to(device)

  with torch.no_grad():
    probabilities = model(image).softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
  print(f'MobileNetV4 on {device}: class {prediction.item()} '
        f'(Indian cobra = 63), '
        f'probability {confidence.item():.3f}')

  attributions = {}
  for name, title in (
      ('ig', 'IG'), ('idgi', 'IDGI'), ('big', 'BIG'), ('gig', 'GIG'),
      ('smoothgrad', 'SmoothGrad'), ('erig', 'ERIG'),
  ):
    print(f'Computing {title}…')
    method = create_method(model, method=name, steps=200, batch_size=16)
    attributions[title] = method.attribute(image).cpu()

  figure = visualize_attributions(
      attributions, images=image,
      mean=preprocessing['mean'], std=preprocessing['std'])
  output = Path(__file__).resolve().with_name('indian_cobra_comparison.png')
  figure.savefig(output, dpi=200, bbox_inches='tight')
  plt.close(figure)
  print(f'Saved visualization to {output}')


if __name__ == '__main__':
  main()
