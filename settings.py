from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()  
TORCH_CACHE_DIR = os.getenv("TORCH_CACHE_DIR")
SYS_TOKEN = os.getenv("SYS_TOKEN")

BASE = Path(__file__).parent.resolve()
