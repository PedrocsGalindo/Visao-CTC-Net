from dotenv import load_dotenv
import os

load_dotenv()  
TORCH_CACHE_DIR = os.getenv("TORCH_CACHE_DIR")
SYS_TOKEN = os.getenv("SYS_TOKEN")