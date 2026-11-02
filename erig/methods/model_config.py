"""Checkpoint identifiers for the models used in the paper."""

IMAGENET_MODELS = {
    'resnet101': 'resnet101.a2_in1k',
    'mobilenetv4': 'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
    'efficientnet': 'efficientnet_b4.ra2_in1k',
    'vit': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
    'eva02': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'deit3': 'deit3_large_patch16_384.fb_in22k_ft_in1k',
}
PETS_MODELS = {
    'resnet50': 'hf-hub:nateraw/resnet50-oxford-iiit-pet',
    'mobilenetv2': 'jayasurya0131/animal-detection-vit',
    'convnext': 'anonauthors/oxford_pet-ConvNeXt-base',
    'vit': 'anonauthors/oxford_pet-ViT-b32',
    'dinov2': 'mike323zyf/dinov2-small-finetuned-oxford',
}


def get_model_id(model_name: str, dataset_name: str) -> str:
  """Resolve a short model name for the selected dataset.

  Args:
      model_name: Model name, such as ``vit`` or ``resnet50``.
      dataset_name: Dataset used for evaluation.

  Returns:
      The checkpoint identifier used by timm or transformers.

  Raises:
      ValueError: If the dataset or model is unsupported.
  """
  if dataset_name in ('oxford-iiit-pet', 'dpdl-benchmark/oxford_iiit_pet'):
    models = PETS_MODELS
  elif dataset_name in ('imagenet-1k-wds', 'timm/imagenet-1k-wds',
                        'imagenet-s', 'braceletboy/imagenet-s'):
    models = IMAGENET_MODELS
    if model_name == 'resnet18':
      return 'resnet18.a1_in1k'
  else:
    raise ValueError(f'Unsupported dataset: {dataset_name}')
  if model_name not in models:
    available = list(models)
    if models is IMAGENET_MODELS:
      available.append('resnet18')
    raise ValueError(
        f'Unknown model {model_name!r} for {dataset_name}; '
        f'choose from {available}'
    )
  return models[model_name]
