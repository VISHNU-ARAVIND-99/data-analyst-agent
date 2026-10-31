import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent
CHARTS_DIR = BASE_DIR / "assets" / "charts"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama & Model Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")

# Execution Sandbox Configuration
MAX_RETRIES = 3

