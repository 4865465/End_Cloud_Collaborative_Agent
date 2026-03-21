# Baseline1 Project

## Overview
This project implements a React Agent system with support for three execution modes:
1. **Large Model Only**: Uses a powerful LLM (e.g., Deepseek/GPT-4) to process tasks.
2. **Small Model Only**: Uses a smaller, efficient LLM (e.g., Qwen) to process tasks.
3. **Hybrid Mode**: Start with the Small Model; if its confidence is high (> threshold), return the result. Otherwise, fallback to the Large Model.

This project is built independently and uses the LMSYS Chat 1M dataset.

## Project Structure
- `config.py`: Configuration for API keys, models, and thresholds.
- `main.py`: Entry point and pipeline orchestration.
- `agent.py`: Implementation of the React Agent.
- `llm_client.py`: Client for interacting with LLM APIs.
- `dataset_utils.py`: Utilities for loading and processing the dataset.
- `tools.py`: Search and Calculator tools for the agent.

## Setup
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install requests datasets tqdm python-dotenv
   ```
3. Configure `config.py` with your API keys and endpoints. The project currently defaults to a template configuration.

## Usage

### Run with Small Model Only
```bash
python3 main.py --mode small_only --sample_size 50
```

### Run with Large Model Only
```bash
python3 main.py --mode large_only --sample_size 50
```

### Run in Hybrid Mode
```bash
python3 main.py --mode hybrid --sample_size 50
```

## Features
- **Queue-driven Processing**: The pipeline saves progress incrementally to `results_{mode}.jsonl` files. If interrupted, run the command again to resume from where it left off.
- **Resilient Data Loading**: Falls back to mock data if the Hugging Face token is missing or the dataset is inaccessible.
- **Tools**: Includes Web Search and Calculator capabilities for the agent.
