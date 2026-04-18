"""
Configuration for Baseline1
"""
import os

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Config
# Large Model (GPT-4/Deepseek level)
LARGE_MODEL_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LARGE_MODEL_API_BASE = os.getenv("DEEPSEEK_API_BASE")
LARGE_MODEL_NAME = "glm-4.7"

# Small Model (Qwen level)
SMALL_MODEL_API_KEY = os.getenv("SMALL_MODEL_API_KEY")
SMALL_MODEL_API_BASE = os.getenv("SMALL_MODEL_API_BASE")
SMALL_MODEL_NAME = "Qwen/Qwen3.5-4B"

# Code Generator (Deepseek level)
CODE_GENERATOR_API_KEY = os.getenv("PPINFRA_API_KEY")
CODE_GENERATOR_API_BASE = os.getenv("PPINFRA_API_BASE")
CODE_GENERATOR_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"

# Evaluation Model (Usually a strong model like GPT-4 or similar)
EVALUATION_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
EVALUATION_LLM_API_BASE = os.getenv("OPENROUTER_API_BASE")
EVALUATION_LLM_MODEL = "google/gemini-3-flash-preview"

# Google Custom Search Engine API configuration
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")

# HuggingFace Token for Dataset
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "lmsys/lmsys-chat-1m"
DATASET_SPLIT = "train"
SAMPLE_SIZE_Multi = 300 # Default sample size for testing
SAMPLE_SIZE_Single = 100 # Default sample size for testing

# Hybrid Strategy Config
CONFIDENCE_THRESHOLD = 0.9 # Threshold for small model confidence

# Config tools
TOOLS_CONFIG = {
    "web_search": {
        "enabled": True,
        "description": "Search the Internet for the latest information"
    },
    "calculator": {
        "enabled": True,
        "description": "Perform mathematical calculations"
    },
    "get_current_date": {
        "enabled": True,
        "description": "Get the current date and time"
    },
    "code_generator": {
        "enabled": True,
        "description": "Generate, modify, or explain programming code"
    }
}

EMBEDDING_API_KEY = os.getenv("SMALL_MODEL_API_KEY")
EMBEDDING_API_BASE = os.getenv("SMALL_MODEL_EMBEDDING_BASE")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_USE_LOCAL = False
EMBEDDING_GPU_ID = 0
