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
LARGE_MODEL_NAME = "deepseek-v3.2"

# Small Model (Qwen level)
SMALL_MODEL_API_KEY = os.getenv("SMALL_MODEL_API_KEY")
SMALL_MODEL_API_BASE = os.getenv("SMALL_MODEL_API_BASE")
SMALL_MODEL_NAME = "Qwen/Qwen3-1.7B"
# SMALL_MODEL_NAME = "Qwen/Qwen3.5-2B"
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

# Serper API Key for alternative web search
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # FIXME: Manual entry required

# HuggingFace Token for Dataset
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "lmsys/lmsys-chat-1m"
DATASET_SPLIT = "train"
SAMPLE_SIZE_Multi = 300 # Default sample size for testing
SAMPLE_SIZE_Single = 100 # Default sample size for testing

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

# Hybrid Strategy Config
SIMILARITY_THRESHOLD_1 = 0.9 # Threshold to return original trace
SIMILARITY_THRESHOLD_2 = 0.6 # Threshold to return experience
FAILURE_SCORE_THRESHOLD = 4.0 # Threshold below which an experience needs reflection

# Innovation Toggles (创新点开关)
# 创新点1：使用分层的轨迹检索
# 为True时：sim > T1 -> 原始轨迹; T2 < sim <= T1 -> 抽象经验; sim <= T2 -> 大模型
# 为False时：sim > T2 -> 抽象经验; sim <= T2 -> 大模型
HIERARCHICAL_TRAJECTORY_RETRIEVAL = True

# 创新点2：工具记忆库
# 为True时：通过检索相似度，如果大于阈值，不执行工具而直接返回库中结果
# 为False时：不使用工具库，直接真实调用API
TOOL_MEMORY_LIBRARY = True

# 创新点3：失败经验更新
# 为True时：如果发现经验效果不好，调用大模型对该经验进行修改或追加注意事项
# 为False时：生成的经验是静止的，不进行修改
FAILURE_EXPERIENCE_UPDATE = True

TOOL_SIMILARITY_THRESHOLD = 0.9 # Threshold for tool input similarity

# Embedding Settings
EMBEDDING_API_KEY = os.getenv("SMALL_MODEL_API_KEY")
EMBEDDING_API_BASE = os.getenv("SMALL_MODEL_EMBEDDING_BASE")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# EMBEDDING_MODEL = "qwen/qwen3-embedding-0.6b"
EMBEDDING_USE_LOCAL = False
EMBEDDING_GPU_ID = 0