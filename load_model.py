import torch.hub
from settings import TORCH_CACHE_DIR

torch.hub.set_dir(TORCH_CACHE_DIR)

from torchvision import models
m = models.resnet34(weights="IMAGENET1K_V1")

def get_model():
    return m