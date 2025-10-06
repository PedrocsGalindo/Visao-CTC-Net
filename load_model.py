import torch.hub
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env
TORCH_CACHE_DIR = os.getenv("TORCH_CACHE_DIR")
torch.hub.set_dir(TORCH_CACHE_DIR)

from torchvision import models
m = models.resnet34(weights="IMAGENET1K_V1")

def get_model():
    return m